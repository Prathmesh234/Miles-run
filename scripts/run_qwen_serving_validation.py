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
from evidence import atomic, metric, sha256, utcnow
from infra_node import allocated_run, read_inventory
from qwen_serving_probe import attempt_suffix


def native_collector(run, code, attempt):
    phase = run.phase('02-qwen-serving-native-collector' + attempt_suffix(attempt))
    rc, _, _ = phase.command(['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root),
        '--limit-s', '2050', '--ib-backend', 'perfquery', '--stream-label', f'native-qwen-serving-v{attempt}',
        '--role', 'rollout-serving-validation', '--lustre-backend', 'host-debugfs-pod'], timeout=2070)
    phase.finish('fail' if rc else 'ok', failure_summary='Serving native collector failed.' if rc else None, refresh=False)
    return rc


def model_path(run, code):
    spec_path = code / 'candidate.json'
    if not spec_path.exists():
        return run.root / 'models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0'
    spec = json.loads(spec_path.read_text())
    relative = Path(spec['model_relpath'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'models':
        raise ValueError('Candidate must be inside the run models directory.')
    model = run.root / relative
    if model.is_symlink() or not model.resolve().is_relative_to(run.root.resolve()):
        raise ValueError('Candidate path escapes the run.')
    marker = json.loads((model / 'CONVERSION_COMPLETE.json').read_text())
    if marker['checksums_sha256'] != spec['checksums_sha256'] or sha256(model / 'checksums.sha256') != spec['checksums_sha256']:
        raise ValueError('Candidate completion/checksum pin differs.')
    for line in (model / 'checksums.sha256').read_text().splitlines():
        digest, filename = line.split('  ', 1)
        if Path(filename).name != filename or (model / filename).is_symlink() or sha256(model / filename) != digest:
            raise ValueError('Candidate file checksum differs: ' + filename)
    if json.loads((model / 'config.json').read_text())['quantization_config']['quant_method'] != 'mxfp8':
        raise ValueError('Only the qualified MXFP8 candidate is accepted.')
    return model


def container_command(run, code, attempt):
    root = run.root / f'images/qwen-serving-runtime-v{attempt}'
    root.mkdir(exist_ok=False)
    env = prepare(root)
    os.environ.update({k: v for k, v in env.items() if k.startswith('ENROOT_')})
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    nccl = run.root / f'telemetry/nccl/qwen-serving-v{attempt}'
    nccl.mkdir(parents=True, exist_ok=False)
    envs = {'NVIDIA_VISIBLE_DEVICES': 'all', 'PYTHONDONTWRITEBYTECODE': '1',
            'SLURM_JOB_ID': os.environ['SLURM_JOB_ID'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET,GRAPH',
            'NCCL_DEBUG_FILE': f'/run-artifacts/telemetry/nccl/qwen-serving-v{attempt}/nccl.%h.%p.log'}
    model = model_path(run, code)
    command = ['enroot', 'start', '--net', '--pid', '--ipc', '--rw']
    for key, value in envs.items():
        command += ['--env', key + '=' + value]
    for source, target, options in [(run.root, '/run-artifacts', 'bind,rw,x-create=dir'),
                                   (code, '/ptx', 'bind,ro,x-create=dir'), (model, '/model', 'bind,ro,x-create=dir')]:
        command += ['--mount', str(source) + ':' + target + ':none:' + options]
    command += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
                'python3', '/ptx/qwen_serving_probe.py', '--run-dir', '/run-artifacts', '--attempt', str(attempt)]
    return command


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    phase = run.phase('02-qwen-serving-orchestration' + attempt_suffix(args.attempt))
    code = Path(__file__).resolve().parent
    errors, results, collector = [], [], None
    label = f'native-qwen-serving-v{args.attempt}'
    marker = run.root / f'control/qwen-serving-job-v{args.attempt}.json'
    atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            if read_inventory(run, 'qwen-serving-start' + attempt_suffix(args.attempt)):
                raise RuntimeError('Fresh physical/Slurm allocation inventory did not reconcile.')
            host = socket.gethostname()
            host_stats = run.root / f'telemetry/lustre-qwen-serving-v{args.attempt}' / host / 'lustre.jsonl.partial'
            if not host_stats.is_file():
                raise RuntimeError('Read-only host Lustre collector is not ready; no model load started.')
            collector = pool.submit(native_collector, run, code, args.attempt)
            deadline = time.monotonic() + 30
            while not (run.root / 'telemetry' / label / host / 'nvidia-smi.jsonl.partial').exists():
                if collector.done() or time.monotonic() > deadline:
                    raise RuntimeError('Native collector not ready; no model load started.')
                time.sleep(0.25)
            rc, _, _ = phase.command(container_command(run, code, args.attempt), timeout=2000)
            if rc:
                errors.append('EP8 serving validation exited with ' + str(rc))
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
        finally:
            atomic(run.root / 'control' / (label + '-telemetry.stop'), {'time': utcnow()})
            atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': False})
            atomic(run.root / f'control/qwen-serving-lustre-v{args.attempt}.stop', {'time': utcnow()})
            if collector is not None and collector.result():
                errors.append('Native collector exited with an error.')
            if read_inventory(run, 'qwen-serving-end' + attempt_suffix(args.attempt)):
                errors.append('Final allocation inventory failed.')
    phase.finish('fail' if errors else 'ok', results=results, failure_summary='; '.join(errors) or None,
        metadata={'findings': errors, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                  'scope': 'Single-node EP8 MTP on/off correctness and observability smoke, not a performance comparison or RL run.',
                  'artifacts': ['telemetry/' + label, f'telemetry/lustre-qwen-serving-v{args.attempt}',
                                f'telemetry/nccl/qwen-serving-v{args.attempt}']}, refresh=False)
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
