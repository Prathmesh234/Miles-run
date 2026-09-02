"""Bounded storage load for the read-only host Lustre collector validation."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

from evidence import atomic, metric, sha256
from infra_controller import HOSTS, srun
from infra_node import allocated_run, read_inventory


def node(run):
    host = socket.gethostname()
    if read_inventory(run, 'lustre-start'):
        return 1
    phase = run.phase('01-lustre-collector-storage-' + host)
    directory = run.root / 'scratch/lustre-telemetry-v1' / host
    directory.mkdir(parents=True, exist_ok=False)
    data = directory / 'fio.bin'
    command = ['fio', '--name=lustre-collector-validation', '--filename=' + str(data), '--size=2g',
        '--rw=write', '--bs=1m', '--ioengine=psync', '--iodepth=1', '--numjobs=1', '--direct=1',
        '--verify=crc32c', '--do_verify=1', '--verify_fatal=1', '--end_fsync=1', '--output-format=json']
    rc, out, _ = phase.command(command, timeout=90)
    errors, results = [], []
    if rc:
        errors.append('fio failed with exit code ' + str(rc))
    else:
        try:
            for job in json.loads(out)['jobs']:
                if job['error']:
                    errors.append('fio reported a data or I/O error.')
                for direction in ('read', 'write'):
                    for name, unit in [('io_bytes', 'B'), ('runtime', 'ms'), ('bw_bytes', 'B/s')]:
                        results.append(metric(direction + '_' + name, job[direction][name], unit, host))
        except (ValueError, KeyError) as exc:
            errors.append(str(exc))
    phase.finish('fail' if errors else 'ok', results=results, failure_summary='; '.join(errors) or None,
        metadata={'scope': '2 GiB per-node collector validation; not a storage performance benchmark.',
                  'data_sha256': sha256(data) if data.exists() else None,
                  'artifacts': [str(data.relative_to(run.root))]}, refresh=False)
    final = read_inventory(run, 'lustre-end')
    return int(bool(errors) or bool(final))


def controller(run):
    phase = run.phase('01-lustre-collector-load-dispatch')
    marker = run.root / 'control/lustre-validation-job.json'
    atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': True})
    deadline = time.monotonic() + 30
    while not all((run.root / 'telemetry/lustre-host-validation-v1' / host / 'lustre.jsonl.partial').exists() for host in HOSTS):
        if time.monotonic() > deadline:
            phase.finish('fail', failure_summary='Host collectors were not ready on all four nodes; no storage load.', refresh=False)
            atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': False})
            return 1
        time.sleep(0.25)
    rc, _, _ = phase.command(srun(HOSTS) + ['python3', str(Path(__file__).resolve()),
                            '--run-dir', str(run.root), '--node'], timeout=150)
    atomic(marker, {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': False})
    phase.finish('fail' if rc else 'ok', failure_summary='Storage collector validation load failed.' if rc else None,
                 metadata={'slurm_job_id': os.environ['SLURM_JOB_ID'], 'scope': 'Collector validation load only.'}, refresh=False)
    return rc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--node', action='store_true')
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    sys.exit(node(run) if args.node else controller(run))
