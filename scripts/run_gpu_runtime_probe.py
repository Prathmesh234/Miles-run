"""Run one eight-GPU visibility check in an exclusive Slurm allocation."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

from enroot_run_config import prepare
from evidence import atomic, metric
from infra_node import allocated_run, read_inventory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    phase = run.phase('02-enroot-eight-gpu-runtime')
    if read_inventory(run, 'container-start'):
        phase.finish('fail', failure_summary='Native allocation inventory failed; no container GPU operation.', refresh=False)
        return 1
    root = run.root / 'images/runtime-gpu-probe-v1'
    errors, results = [], []
    try:
        root.mkdir(exist_ok=False)
        env = prepare(root)
        os.environ.update({k: v for k, v in env.items() if k.startswith('ENROOT_')})
        os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
        gpus = json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']
        uuids = sorted(g['uuid'] for g in gpus if g['hostname'] == socket.gethostname())
        code = Path(__file__).resolve().parent
        rc, out, _ = phase.command(['enroot', 'start', '--net', '--pid', '--ipc', '--env',
            'NVIDIA_VISIBLE_DEVICES=all', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--mount',
            str(code) + ':/ptx:none:bind,ro,x-create=dir', str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
            'python3', '/ptx/probe_container_gpus.py', '--expected-uuids', json.dumps(uuids)], timeout=150)
        if rc:
            errors.append('Container GPU visibility/BF16 probe failed: ' + str(rc))
        else:
            records = [json.loads(line.removeprefix('PTX_GPU_PROBE=')) for line in out.splitlines() if line.startswith('PTX_GPU_PROBE=')]
            if len(records) != 1:
                raise ValueError('Expected one framed GPU probe result.')
            atomic(phase.path / 'gpu-probe.json', records[0])
            results.append(metric('container_gpu_count', len(records[0]['gpus']), 'count', socket.gethostname()))
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    if read_inventory(run, 'container-end'):
        errors.append('Final native inventory failed.')
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None, results=results,
        metadata={'findings': errors, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                  'scope': 'B200 container validation only; no Qwen serving or training.',
                  'artifacts': [str(root.relative_to(run.root))]}, refresh=False)
    print(json.dumps({'status': 'fail' if errors else 'ok', 'findings': errors}), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
