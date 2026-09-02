"""One-node EP8 serving smoke with explicit run-scoped native and host collectors."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import sys
import time

from enroot_run_config import prepare
from evidence import atomic, metric, utcnow
from infra_node import allocated_run, read_inventory


LABEL = 'native-qwen-serving-v1'


def native_collector(run, code):
    phase = run.phase('02-qwen-serving-native-collector')
    rc, _, _ = phase.command(['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root),
        '--limit-s', '2050', '--ib-backend', 'perfquery', '--stream-label', LABEL,
        '--role', 'rollout-serving-validation', '--lustre-backend', 'host-debugfs-pod'], timeout=2070)
    phase.finish('fail' if rc else 'ok', failure_summary='Serving native collector failed.' if rc else None, refresh=False)
    return rc


def container_command(run, code):
    root = run.root / 'images/qwen-serving-runtime-v1'
    root.mkdir(exist_ok=False)
    env = prepare(root)
    os.environ.update({k: v for k, v in env.items() if k.startswith('ENROOT_')})
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    nccl = run.root / 'telemetry/nccl/qwen-serving-v1'
    nccl.mkdir(parents=True, exist_ok=False)
    envs = {'NVIDIA_VISIBLE_DEVICES': 'all', 'PYTHONDONTWRITEBYTECODE': '1',
            'SLURM_JOB_ID': os.environ['SLURM_JOB_ID'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET,GRAPH',
            'NCCL_DEBUG_FILE': '/run-artifacts/telemetry/nccl/qwen-serving-v1/nccl.%h.%p.log'}
    model = run.root / 'models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0'
    command = ['enroot', 'start', '--net', '--pid', '--ipc', '--rw']
    for key, value in envs.items():
        command += ['--env', key + '=' + value]
    for source, target, options in [(run.root, '/run-artifacts', 'bind,rw,x-create=dir'),
                                   (code, '/ptx', 'bind,ro,x-create=dir'), (model, '/model', 'bind,ro,x-create=dir')]:
        command += ['--mount', str(source) + ':' + target + ':none:' + options]
    command += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
                'python3', '/ptx/qwen_serving_probe.py', '--run-dir', '/run-artifacts']
    return command


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    phase = run.phase('02-qwen-serving-orchestration')
    code = Path(__file__).resolve().parent
    errors, results, collector = [], [], None
    marker = run.root / 'control/qwen-serving-job.json'
    atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            if read_inventory(run, 'qwen-serving-start'):
                raise RuntimeError('Fresh physical/Slurm allocation inventory did not reconcile.')
            host = socket.gethostname()
            host_stats = run.root / 'telemetry/lustre-qwen-serving-v1' / host / 'lustre.jsonl.partial'
            if not host_stats.is_file():
                raise RuntimeError('Read-only host Lustre collector is not ready; no model load started.')
            collector = pool.submit(native_collector, run, code)
            deadline = time.monotonic() + 30
            while not (run.root / 'telemetry' / LABEL / host / 'nvidia-smi.jsonl.partial').exists():
                if collector.done() or time.monotonic() > deadline:
                    raise RuntimeError('Native collector not ready; no model load started.')
                time.sleep(0.25)
            rc, _, _ = phase.command(container_command(run, code), timeout=2000)
            if rc:
                errors.append('EP8 serving validation exited with ' + str(rc))
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
        finally:
            atomic(run.root / 'control' / (LABEL + '-telemetry.stop'), {'time': utcnow()})
            atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': False})
            atomic(run.root / 'control/qwen-serving-lustre.stop', {'time': utcnow()})
            if collector is not None and collector.result():
                errors.append('Native collector exited with an error.')
            if read_inventory(run, 'qwen-serving-end'):
                errors.append('Final allocation inventory failed.')
    phase.finish('fail' if errors else 'ok', results=results, failure_summary='; '.join(errors) or None,
        metadata={'findings': errors, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                  'scope': 'Single-node EP8 MTP on/off correctness and observability smoke, not a performance comparison or RL run.',
                  'artifacts': ['telemetry/' + LABEL, 'telemetry/lustre-qwen-serving-v1', 'telemetry/nccl/qwen-serving-v1']}, refresh=False)
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
