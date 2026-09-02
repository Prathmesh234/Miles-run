"""Whole-node, telemetry-covered CPU checkpoint parity in the pinned GPU image."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from enroot_run_config import prepare
from evidence import atomic, utcnow
from infra_node import allocated_run, read_inventory
from run_model_conversion import verify_sources


def container_command(run, code, attempt, source):
    root = run.root / f'images/checkpoint-parity-runtime-v{attempt}'
    root.mkdir(exist_ok=False)
    env = prepare(root)
    os.environ.update({key: value for key, value in env.items() if key.startswith('ENROOT_')})
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    envs = {'NVIDIA_VISIBLE_DEVICES': 'all', 'PYTHONDONTWRITEBYTECODE': '1',
        'SLURM_JOB_ID': os.environ['SLURM_JOB_ID'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'PYTHONPATH': '/miles-source:/root/Megatron-LM', 'OMP_NUM_THREADS': '8', 'CUDA_DEVICE_MAX_CONNECTIONS': '1'}
    command = ['enroot', 'start', '--net', '--pid', '--ipc', '--rw']
    for key, value in envs.items():
        command += ['--env', key + '=' + value]
    model = run.root / 'models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0'
    checkpoint = run.root / 'models/qwen3.6-35b-a3b-torch-dist-v2'
    for path, target, mode in [(run.root, '/run-artifacts', 'rw'), (code, '/ptx', 'ro'),
                              (model, '/model', 'ro'), (checkpoint, '/checkpoint', 'ro'),
                              (source, '/miles-source', 'ro')]:
        command += ['--mount', str(path) + ':' + target + ':none:bind,' + mode + ',x-create=dir']
    return command + [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
        'python3', '/ptx/checkpoint_parity.py', '--run-dir', '/run-artifacts', '--attempt', str(attempt)]


def native_collector(run, code, attempt):
    phase = run.phase(f'02-checkpoint-parity-native-collector-v{attempt}')
    rc, _, _ = phase.command(['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root),
        '--limit-s', '1700', '--ib-backend', 'perfquery', '--stream-label', f'native-checkpoint-parity-v{attempt}',
        '--role', 'checkpoint-parity', '--lustre-backend', 'host-debugfs-pod'], timeout=1720)
    phase.finish('fail' if rc else 'ok', failure_summary='Parity native collector failed.' if rc else None, refresh=False)
    return rc


def run_child(phase, command, attempt, stage='checkpoint-parity', execution_limit_s=1500):
    run = phase.run
    started, timestamp = time.monotonic(), utcnow()
    stdout, stderr = phase.path / 'logs/container.out', phase.path / 'logs/container.err'
    timed_out, stop_reason, code = False, None, 1
    with stdout.open('x') as out, stderr.open('x') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            while child.poll() is None:
                if time.monotonic() - started > execution_limit_s:
                    timed_out, stop_reason = True, f'{stage} exceeded its {execution_limit_s}s execution cap.'
                    break
                if shutil.disk_usage(run.root).free < 128 * 1024**3:
                    stop_reason = f'{stage} free-space reserve reached 128 GiB.'
                    break
                if (run.root / f'control/{stage}-v{attempt}.stop').exists():
                    stop_reason = f'Explicit {stage} stop marker received.'
                    break
                time.sleep(1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            code = 124 if timed_out else child.returncode
            phase.commands.append({'argv': command, 'started_at': timestamp, 'ended_at': utcnow(),
                'duration_s': time.monotonic() - started, 'exit_code': code, 'timeout': timed_out,
                'stdout': str(stdout.relative_to(run.root)), 'stderr': str(stderr.relative_to(run.root))})
            atomic(phase.path / 'logs/commands.json', phase.commands)
    if stop_reason:
        raise RuntimeError(stop_reason)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    phase = run.phase(f'02-checkpoint-parity-orchestration-v{args.attempt}')
    code = Path(__file__).resolve().parent
    errors, collector = [], None
    label = f'native-checkpoint-parity-v{args.attempt}'
    marker = run.root / f'control/checkpoint-parity-job-v{args.attempt}.json'
    atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            if read_inventory(run, f'checkpoint-parity-start-v{args.attempt}'):
                raise RuntimeError('Fresh physical/Slurm inventory did not reconcile.')
            source = verify_sources(run)
            host = socket.gethostname()
            stats = run.root / f'telemetry/lustre-checkpoint-parity-v{args.attempt}' / host / 'lustre.jsonl.partial'
            if not stats.is_file():
                raise RuntimeError('Host Lustre collector is not ready; no parity started.')
            collector = pool.submit(native_collector, run, code, args.attempt)
            deadline = time.monotonic() + 30
            while not (run.root / 'telemetry' / label / host / 'nvidia-smi.jsonl.partial').exists():
                if collector.done() or time.monotonic() > deadline:
                    raise RuntimeError('Native collector is not ready; no parity started.')
                time.sleep(.25)
            if run_child(phase, container_command(run, code, args.attempt, source), args.attempt):
                errors.append('Parity container returned a failure. Preserve inputs and inspect child evidence.')
        except Exception as exc:
            errors.append(str(exc))
        finally:
            atomic(run.root / 'control' / (label + '-telemetry.stop'), {'time': utcnow()})
            atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': False})
            atomic(run.root / f'control/checkpoint-parity-lustre-v{args.attempt}.stop', {'time': utcnow()})
            if collector is not None and collector.result():
                errors.append('Native collector exited with an error.')
            if read_inventory(run, f'checkpoint-parity-end-v{args.attempt}'):
                errors.append('Final allocation inventory failed.')
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None,
        metadata={'findings': errors, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                  'scope': 'CPU weight parity, not training or checkpoint resume.',
                  'artifacts': [f'tests/02-checkpoint-parity-child-v{args.attempt}', 'telemetry/' + label,
                                f'telemetry/lustre-checkpoint-parity-v{args.attempt}']}, refresh=False)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
