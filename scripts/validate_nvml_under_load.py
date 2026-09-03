"""Bounded, read-only-counter qualification around a node-local EP8 all-reduce.

This validates collection, not communication performance or checkpoint-load
reliability. The subsequent GRPO attempt must qualify actual checkpoint load.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

from evidence import atomic, metric, sha256, utcnow
from infra_controller import parse_nccl
from infra_node import allocated_run, read_inventory
from telemetry_health import assert_healthy, require_healthy


TEARDOWN_PROBE = '''import json,os,time,torch
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
if os.environ.get('PTX_PROBE_NCCL')=='1':
 import torch.distributed as dist
 dist.init_process_group(backend='nccl',device_id=torch.device('cuda',rank))
 control=torch.ones(1024,device='cuda')
 dist.all_reduce(control);assert torch.all(control==8).item()
free,total=torch.cuda.mem_get_info()
assert free >= 96*1024**3, 'Require 96 GiB free HBM before bounded 64 GiB allocation'
print(json.dumps({'event':'before_allocate','rank':rank,'monotonic_s':time.monotonic(),
 'free_bytes':free,'total_bytes':total,'torch':torch.__version__}),flush=True)
payload=torch.empty(64*1024**3,dtype=torch.uint8,device='cuda')
payload.zero_();torch.cuda.synchronize()
assert payload[0].item()==0 and payload[-1].item()==0
print(json.dumps({'event':'allocated','rank':rank,'monotonic_s':time.monotonic(),
 'allocated_bytes':torch.cuda.memory_allocated()}),flush=True)
time.sleep(10)
print(json.dumps({'event':'exit_with_live_context','rank':rank,'monotonic_s':time.monotonic()}),flush=True)
'''


def teardown_command(run, code, label, host, profile):
    from enroot_run_config import prepare
    runtime = run.root / 'images' / label / host
    runtime.mkdir(parents=True, exist_ok=False)
    env = prepare(runtime)
    env['NVIDIA_VISIBLE_DEVICES'] = 'all'
    command = ['enroot', 'start', '--pid', '--ipc', '--rw',
        '--env', 'NVIDIA_VISIBLE_DEVICES=all', '--env', 'OMP_NUM_THREADS=1',
        '--env', 'PTX_PROBE_NCCL=' + ('1' if profile == 'nccl-context-teardown' else '0'),
        '--env', 'NCCL_NVLS_ENABLE=0', '--env', 'NCCL_DEBUG=INFO',
        '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--mount', str(code) + ':/ptx:none:bind,ro,x-create=dir',
        str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
        'python3', '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=8',
        '--master-addr=127.0.0.1', '--master-port=29687', '/ptx/teardown_probe.py']
    return command, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--load-profile', choices=['all-reduce', 'context-teardown', 'nccl-context-teardown'], default='all-reduce')
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    host, job = socket.gethostname(), os.environ['SLURM_JOB_ID']
    label = f'nvml-qualification-v{args.attempt}'
    phase = run.phase('01-' + label + '-' + host)
    code = Path(__file__).resolve().parent
    directory = run.root / 'telemetry' / label / host
    findings, children, handles, results = [], [], [], []
    def spawn(argv, name, env=None):
        out, err = phase.path / 'logs' / (name + '.out'), phase.path / 'logs' / (name + '.err')
        streams = [out.open('x'), err.open('x')]
        handles.extend(streams)
        atomic(phase.path / (name + '.command.json'), dict(argv=argv, time=utcnow()))
        p = subprocess.Popen(argv, stdout=streams[0], stderr=streams[1], start_new_session=True, env=env)
        children.append(p)
        return p
    try:
        if read_inventory(run, label + '-start'):
            raise RuntimeError('Allocation GPU reconciliation failed.')
        collector = spawn(['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root),
            '--gpu-backend', 'nvml', '--nvml-binding', str(code / 'pynvml.py'), '--ib-backend', 'perfquery',
            '--lustre-backend', 'host-debugfs-pod', '--stream-label', label, '--limit-s', '240',
            '--stop-marker', 'control/' + label + '-' + host + '-telemetry.stop'], 'collector')
        deadline = time.monotonic() + 35
        while not assert_healthy(directory, host, job):
            if collector.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Collector did not become ready.')
            time.sleep(0.25)
        if args.load_profile != 'all-reduce':
            command, env = teardown_command(run, code, label, host, args.load_profile)
            load = spawn(command, 'load', env)
        else:
            load = spawn(['/usr/local/bin/all_reduce_perf', '-b', '512M', '-e', '512M', '-g', '8',
                          '-n', '2000', '-w', '20', '-c', '1'], 'load')
        deadline, exited = time.monotonic() + 180, None
        while exited is None or time.monotonic() - exited < 15:
            if load.poll() is not None and exited is None:
                exited = time.monotonic()
            if any((run.root / 'control').glob(label + '-failure-*.json')):
                raise RuntimeError('A peer collector qualification failed.')
            require_healthy(directory, host, job)
            if collector.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Collector stopped or qualification load exceeded 180 seconds.')
            time.sleep(0.25)
        if load.returncode:
            raise RuntimeError('Qualification load failed: ' + str(load.returncode))
        if args.load_profile == 'all-reduce':
            results.extend(parse_nccl((phase.path / 'logs/load.out').read_text(), 1))
    except Exception as exc:
        findings.append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
        atomic(run.root / 'control' / (label + '-failure-' + host + '.json'), {'failure': str(exc), 'time': utcnow()})
    finally:
        atomic(run.root / 'control' / (label + '-' + host + '-telemetry.stop'), {'time': utcnow()})
        for p in reversed(children):
            if p.poll() is None:
                if findings and p is not collector:
                    os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait(timeout=5)
            if p.returncode:
                findings.append('Owned child exited with code ' + str(p.returncode))
        for handle in handles:
            handle.close()
        if read_inventory(run, label + '-end'):
            findings.append('Final GPU inventory reconciliation failed.')
    try:
        proof = json.loads((directory / 'nvml-validation.json').read_text())
        if not proof['cli_counter_bracket_passed'] or proof['nvlink_counter_identities'] != 288:
            findings.append('NVML CLI parity proof is incomplete.')
        for name in ('nvidia-smi', 'nvlink', 'infiniband', 'cpu-memory-numa', 'lustre'):
            rows = [json.loads(line) for line in (directory / (name + '.jsonl')).read_text().splitlines()]
            stamps = sorted({row['monotonic_s'] for row in rows})
            gap = max((b-a for a,b in zip(stamps, stamps[1:])), default=0)
            if len(stamps) < 10 or gap > 3 or any(row.get('metric') == 'collector_error' for row in rows):
                findings.append(name + ': missing samples, >3 second gap, or collector errors.')
            results += [metric(name + '_records', len(rows), 'count', host), metric(name + '_max_gap', gap, 's', host)]
            if name == 'nvidia-smi':
                util = [row['value'] for row in rows if row['metric'] == 'utilization.gpu']
                if not util or max(util) <= 0:
                    findings.append('No GPU activity observed during the all-reduce load.')
        if (directory / 'failure.json').exists():
            findings.append('Sticky collector failure marker exists.')
    except Exception as exc:
        findings.append('Final telemetry audit: ' + str(exc))
    logs = [{'path': str(p.relative_to(run.root)), 'sha256': sha256(p)}
            for p in sorted((phase.path / 'logs').glob('*')) if p.is_file()]
    result = dict(findings=findings, hostname=host, slurm_job_id=job, scope=__doc__,
        load_profile=args.load_profile, artifacts=[str(directory.relative_to(run.root))], raw_logs=logs)
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if findings else 'ok', results=results, metadata=result,
        failure_summary='; '.join(findings) or None, refresh=False)
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
