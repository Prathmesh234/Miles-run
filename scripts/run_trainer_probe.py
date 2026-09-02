"""Run one whole-node EP8 model probe with inventory and native/Lustre telemetry."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import socket
import time

from enroot_run_config import prepare
from evidence import atomic, utcnow
from infra_node import allocated_run, read_inventory
from run_checkpoint_parity import run_child
from run_model_conversion import verify_sources
from trainer_probe import MILES_SHA


def container_command(run, code, attempt, source):
    root = run.root / f'images/trainer-probe-runtime-v{attempt}'
    root.mkdir(exist_ok=False)
    environment = prepare(root)
    os.environ.update({k: v for k, v in environment.items() if k.startswith('ENROOT_')})
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    nccl = run.root / f'telemetry/nccl/trainer-probe-v{attempt}'
    nccl.mkdir(parents=True, exist_ok=False)
    envs = {'NVIDIA_VISIBLE_DEVICES': 'all', 'PYTHONDONTWRITEBYTECODE': '1',
        'SLURM_JOB_ID': os.environ['SLURM_JOB_ID'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'PYTHONPATH': '/miles-source:/root/Megatron-LM', 'OMP_NUM_THREADS': '1',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '0',
        'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET,GRAPH,COLL',
        'GLOO_SOCKET_IFNAME': 'lo', 'NCCL_SOCKET_IFNAME': 'lo',
        'TORCH_NCCL_TRACE_BUFFER_SIZE': '4096', 'TORCH_NCCL_DUMP_ON_TIMEOUT': '1',
        'NCCL_DEBUG_FILE': f'/run-artifacts/telemetry/nccl/trainer-probe-v{attempt}/nccl.%h.%p.log',
        'TRITON_CACHE_DIR': f'/run-artifacts/cache/trainer-probe-v{attempt}/triton',
        'TORCHINDUCTOR_CACHE_DIR': f'/run-artifacts/cache/trainer-probe-v{attempt}/inductor'}
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
        'python3', '/ptx/trainer_probe.py', '--run-dir', '/run-artifacts', '--attempt', str(attempt)]


def native_collector(run, code, attempt):
    phase = run.phase(f'02-trainer-probe-native-collector-v{attempt}')
    rc, _, _ = phase.command(['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root),
        '--limit-s', '1250', '--ib-backend', 'perfquery', '--stream-label', f'native-trainer-probe-v{attempt}',
        '--role', 'trainer-probe', '--lustre-backend', 'host-debugfs-pod'], timeout=1270)
    phase.finish('fail' if rc else 'ok', failure_summary='Trainer-probe native collector failed.' if rc else None, refresh=False)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    phase = run.phase(f'02-trainer-probe-orchestration-v{args.attempt}')
    code = Path(__file__).resolve().parent
    errors, collector = [], None
    label = f'native-trainer-probe-v{args.attempt}'
    marker = run.root / f'control/trainer-probe-job-v{args.attempt}.json'
    atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            if read_inventory(run, f'trainer-probe-start-v{args.attempt}'):
                raise RuntimeError('Fresh physical/Slurm inventory did not reconcile.')
            source = verify_sources(run, source_version=2, revision=MILES_SHA)
            host = socket.gethostname()
            if not (run.root / f'telemetry/lustre-trainer-probe-v{args.attempt}' / host / 'lustre.jsonl.partial').is_file():
                raise RuntimeError('Host Lustre collector is not ready; no model execution started.')
            collector = pool.submit(native_collector, run, code, args.attempt)
            deadline = time.monotonic() + 30
            while not (run.root / 'telemetry' / label / host / 'nvidia-smi.jsonl.partial').exists():
                if collector.done() or time.monotonic() > deadline:
                    raise RuntimeError('Native collector is not ready; no model execution started.')
                time.sleep(.25)
            if run_child(phase, container_command(run, code, args.attempt, source), args.attempt,
                         stage='trainer-probe', execution_limit_s=1050):
                errors.append('Trainer-probe container failed; preserve inputs and per-rank evidence.')
        except Exception as exc:
            errors.append(str(exc))
        finally:
            atomic(run.root / 'control' / (label + '-telemetry.stop'), {'time': utcnow()})
            atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': False})
            atomic(run.root / f'control/trainer-probe-lustre-v{args.attempt}.stop', {'time': utcnow()})
            if collector is not None and collector.result():
                errors.append('Native collector exited with an error.')
            if read_inventory(run, f'trainer-probe-end-v{args.attempt}'):
                errors.append('Final allocation inventory failed.')
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None,
        metadata={'findings': errors, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                  'scope': 'One-node EP8 checkpoint reshard and diagnostic forward/backward. No optimizer or GRPO execution.',
                  'artifacts': [f'tests/02-trainer-probe-child-v{args.attempt}', 'telemetry/' + label,
                                f'telemetry/lustre-trainer-probe-v{args.attempt}']}, refresh=False)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
