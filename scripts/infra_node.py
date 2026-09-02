"""Bounded native preflight actions inside an exclusive Slurm allocation."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

from evidence import Run, atomic, metric, sha256


def allocated_run(root):
    root = Path(root).resolve()
    if not root.is_relative_to(Path('/shared/posttrainingx/runs')):
        raise ValueError('Cluster writes must stay in the PostTrainingX run tree.')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('This action requires a Slurm allocation.')
    mount = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(root)], text=True).strip()
    if mount != 'lustre':
        raise ValueError('The run directory must be on Lustre.')
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise ValueError('The preflight requires at least 10 GiB free.')
    return Run(root)


def read_inventory(run, label):
    host = socket.gethostname()
    phase = run.phase(f'01-allocation-{label}-{host}')
    fields = 'index,uuid,pci.bus_id,name,memory.total,memory.free,memory.used'
    code, out, _ = phase.command(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'])
    errors = []
    rows = list(csv.reader(out.splitlines(), skipinitialspace=True)) if not code else []
    expected = json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']
    expected = {g['uuid'] for g in expected if g['hostname'] == host}
    actual = {r[1] for r in rows if len(r) == 7}
    if len(actual) != 8 or actual != expected:
        errors.append('Physical GPU UUIDs differ from the frozen eight-GPU inventory.')
    for argv in [
        ['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '--json'],
        ['scontrol', 'show', 'node', host, '--json'],
        ['nvidia-smi', '-q', '-x'], ['nvidia-smi', 'topo', '-m'],
        ['nvidia-smi', 'nvlink', '-s'], ['nvidia-smi', 'nvlink', '-e'],
        ['ldd', '/usr/local/bin/all_reduce_perf'], ['srun', '--version'],
        ['fio', '--version'], ['python3', '--version'],
    ]:
        c, text, _ = phase.command(argv, timeout=20)
        if c:
            errors.append('Inventory command failed: ' + ' '.join(argv))
        if argv[:3] == ['scontrol', 'show', 'node'] and not c:
            nodes = json.loads(text)['nodes']
            if len(nodes) != 1 or not re.search(r'(^|,)gpu:8(?:\(|,|$)', nodes[0].get('gres', '')):
                errors.append('Slurm GRES does not describe exactly eight GPUs.')
    gpu_env = {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_JOB_NODELIST',
               'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES')}
    hashes = {p: sha256(p) for p in ['/usr/local/bin/all_reduce_perf',
              '/opt/nccl-tests/lib/libnccl.so.2', '/opt/nccl-tests/lib/libcudart.so.13']}
    if hashes['/usr/local/bin/all_reduce_perf'] != '0005f6c040df8bec6fd1c4780b7ac4f5f2dde11469449e5d8b4f4c33a6282f78':
        errors.append('The native all-reduce binary differs from the inspected binary.')
    phase.finish('fail' if errors else 'ok',
                 results=[metric('physical_gpu_count', len(actual), 'count', host)],
                 metadata={'hostname': host, 'environment': gpu_env, 'binary_hashes': hashes,
                           'gpu_uuids': sorted(actual), 'findings': errors},
                 failure_summary='; '.join(errors) or None, refresh=False)
    return int(bool(errors))


def storage(run):
    host = socket.gethostname()
    phase = run.phase(f'01-storage-smoke-{host}')
    data = run.root / 'scratch' / host
    data.mkdir(parents=True, exist_ok=False)
    filename = data / 'fio-checkpoint.bin'
    argv = ['fio', '--name=posttrainingx-checkpoint-smoke', '--filename=' + str(filename),
            '--size=256m', '--rw=write', '--bs=1m', '--ioengine=psync', '--iodepth=1',
            '--numjobs=1', '--direct=1', '--verify=crc32c', '--do_verify=1',
            '--verify_fatal=1', '--end_fsync=1', '--output-format=json']
    code, out, _ = phase.command(argv, timeout=90)
    errors, results = [], []
    parsed = None
    if code:
        errors.append('The bounded write, fsync, or verification command failed.')
    else:
        try:
            parsed = json.loads(out)
            for job in parsed['jobs']:
                if job['error']:
                    errors.append('fio reported a job error.')
                for direction in ['read', 'write']:
                    for key, unit in [('bw_bytes', 'B/s'), ('iops', 'IOPS'), ('io_bytes', 'B'), ('runtime', 'ms')]:
                        results.append(metric(f'checkpoint_smoke_{direction}_{key}', job[direction][key], unit, host))
        except (ValueError, KeyError) as exc:
            errors.append('fio output could not be normalized: ' + str(exc))
    checksum = sha256(filename) if filename.is_file() else None
    phase.finish('fail' if errors else 'ok', results=results,
                 metadata={'scope': '256 MiB native path smoke, not a model checkpoint or saturated storage benchmark',
                           'fio': parsed, 'data_relpath': str(filename.relative_to(run.root)),
                           'data_sha256': checksum, 'artifacts': [str(filename.relative_to(run.root))]},
                 failure_summary='; '.join(errors) or None, refresh=False)
    return int(bool(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('action', choices=['start-inventory', 'end-inventory', 'storage'])
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    if args.action == 'storage':
        return storage(run)
    return read_inventory(run, args.action.split('-')[0])


if __name__ == '__main__':
    sys.exit(main())
