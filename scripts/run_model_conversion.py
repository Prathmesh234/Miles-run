"""One-node pinned model conversion with allocation reconciliation and telemetry."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import sys
import time

from enroot_run_config import prepare
from evidence import atomic, sha256, utcnow
from infra_node import allocated_run, read_inventory
from model_conversion import MILES_SHA


def verify_sources(run, source_version=1, revision=MILES_SHA):
    source = run.root / f'provenance/training-source-v{source_version}'
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['source_git_sha'] != revision:
        raise ValueError('Materialized Miles source revision differs from conversion pin.')
    for item in manifest['files']:
        path = source / item['path']
        if path.is_symlink() or sha256(path) != item['sha256']:
            raise ValueError('Materialized Miles source changed: ' + item['path'])
    return source / 'miles'


def container_command(run, code, attempt, source):
    root = run.root / f'images/model-conversion-runtime-v{attempt}'
    root.mkdir(exist_ok=False)
    env = prepare(root)
    os.environ.update({key: value for key, value in env.items() if key.startswith('ENROOT_')})
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    nccl = run.root / f'telemetry/nccl/model-conversion-v{attempt}'
    nccl.mkdir(parents=True, exist_ok=False)
    envs = {'NVIDIA_VISIBLE_DEVICES': 'all', 'PYTHONDONTWRITEBYTECODE': '1',
        'SLURM_JOB_ID': os.environ['SLURM_JOB_ID'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'PYTHONPATH': '/miles-source:/root/Megatron-LM', 'OMP_NUM_THREADS': '1',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET,GRAPH',
        'GLOO_SOCKET_IFNAME': 'lo', 'NCCL_SOCKET_IFNAME': 'lo',
        'NCCL_DEBUG_FILE': f'/run-artifacts/telemetry/nccl/model-conversion-v{attempt}/nccl.%h.%p.log'}
    model = run.root / 'models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0'
    command = ['enroot', 'start', '--net', '--pid', '--ipc', '--rw']
    for key, value in envs.items():
        command += ['--env', key + '=' + value]
    for path, target, options in [(run.root, '/run-artifacts', 'bind,rw,x-create=dir'),
        (code, '/ptx', 'bind,ro,x-create=dir'), (model, '/model', 'bind,ro,x-create=dir'),
        (source, '/miles-source', 'bind,ro,x-create=dir')]:
        command += ['--mount', str(path) + ':' + target + ':none:' + options]
    command += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
        'python3', '/ptx/model_conversion.py', '--run-dir', '/run-artifacts', '--attempt', str(attempt)]
    return command


def native_collector(run, code, attempt):
    phase = run.phase(f'02-model-conversion-native-collector-v{attempt}')
    rc, _, _ = phase.command(['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root),
        '--limit-s', '2600', '--ib-backend', 'perfquery', '--stream-label', f'native-model-conversion-v{attempt}',
        '--role', 'checkpoint-conversion', '--lustre-backend', 'host-debugfs-pod'], timeout=2620)
    phase.finish('fail' if rc else 'ok', failure_summary='Conversion native collector failed.' if rc else None, refresh=False)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    phase = run.phase(f'02-model-conversion-orchestration-v{args.attempt}')
    code = Path(__file__).resolve().parent
    errors, collector = [], None
    label = f'native-model-conversion-v{args.attempt}'
    marker = run.root / f'control/model-conversion-job-v{args.attempt}.json'
    atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            if read_inventory(run, f'model-conversion-start-v{args.attempt}'):
                raise RuntimeError('Fresh physical/Slurm inventory did not reconcile.')
            source = verify_sources(run)
            host = socket.gethostname()
            stats = run.root / f'telemetry/lustre-model-conversion-v{args.attempt}' / host / 'lustre.jsonl.partial'
            if not stats.is_file():
                raise RuntimeError('Host Lustre collector is not ready; no conversion started.')
            collector = pool.submit(native_collector, run, code, args.attempt)
            deadline = time.monotonic() + 30
            while not (run.root / 'telemetry' / label / host / 'nvidia-smi.jsonl.partial').exists():
                if collector.done() or time.monotonic() > deadline:
                    raise RuntimeError('Native collector is not ready; no conversion started.')
                time.sleep(.25)
            rc, _, _ = phase.command(container_command(run, code, args.attempt, source), timeout=2500)
            if rc:
                errors.append('Conversion container exited with ' + str(rc))
        except Exception as exc:
            errors.append(str(exc))
        finally:
            atomic(run.root / 'control' / (label + '-telemetry.stop'), {'time': utcnow()})
            atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': False})
            atomic(run.root / f'control/model-conversion-lustre-v{args.attempt}.stop', {'time': utcnow()})
            if collector is not None and collector.result():
                errors.append('Native collector exited with an error.')
            if read_inventory(run, f'model-conversion-end-v{args.attempt}'):
                errors.append('Final allocation inventory failed.')
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None,
        metadata={'findings': errors, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                  'scope': 'One-node model conversion, not GRPO training or validated checkpoint parity.',
                  'artifacts': [f'tests/02-model-conversion-child-v{args.attempt}', 'telemetry/' + label,
                                f'telemetry/lustre-model-conversion-v{args.attempt}']}, refresh=False)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
