"""Run a bounded native preflight, retaining failed gates and every raw stream."""
import argparse
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

from evidence import atomic, metric
from infra_node import allocated_run


HOSTS = [f'gpu-nodes-{i}' for i in range(4)]


def srun(hosts, mpi='none'):
    return ['srun', '--overlap', '--exact', '--mpi=' + mpi, '--cpu-bind=none',
            '--kill-on-bad-exit=1', '--nodes=' + str(len(hosts)),
            '--ntasks=' + str(len(hosts)), '--ntasks-per-node=1', '--cpus-per-task=8',
            '--nodelist=' + ','.join(hosts)]


def parse_nccl(text, nodes):
    results, row_count = [], 0
    if not re.search(r'Out of bounds values\s*:\s*0\s+OK', text):
        raise ValueError('NCCL did not report a successful correctness check.')
    for line in text.splitlines():
        cells = line.split()
        if not cells or not cells[0].isdigit():
            continue
        if len(cells) != 13:
            continue
        size = int(cells[0])
        for offset, mode in [(5, 'out_of_place'), (9, 'in_place')]:
            latency, algo, bus = map(float, cells[offset:offset+3])
            wrong = int(cells[offset+3])
            if wrong != 0:
                raise ValueError(f'NCCL reported {wrong} incorrect values.')
            labels = dict(nodes=nodes, message_bytes=size, mode=mode, dtype=cells[2],
                          operation='all_reduce', ranks=nodes*8)
            results += [metric('nccl_latency', latency, 'us', **labels),
                        metric('nccl_algorithm_bandwidth', algo, 'GB/s', **labels),
                        metric('nccl_bus_bandwidth', bus, 'GB/s', **labels)]
        row_count += 1
    if not row_count:
        raise ValueError('NCCL returned no parseable measurement rows.')
    return results


def run_step(run, name, argv, timeout, metadata=None, parser=None):
    phase = run.phase(name)
    code, out, _ = phase.command(argv, timeout=timeout)
    errors, results = [], []
    if code:
        errors.append(f'The command exited with status {code}.')
    elif parser:
        try:
            results = parser(out)
        except (ValueError, IndexError) as exc:
            errors.append(str(exc))
    phase.finish('fail' if errors else 'ok', results=results, metadata=metadata or {},
                 failure_summary='; '.join(errors) or None, exit_code=code or 1, refresh=False)
    return not errors


def verify_telemetry(run):
    phase = run.phase('01-native-telemetry-coverage')
    errors, results = [], []
    expected = json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']
    for host in HOSTS:
        root = run.root / 'telemetry/native' / host
        try:
            rows = [json.loads(x) for x in (root / 'nvidia-smi.jsonl').read_text().splitlines()]
            links = [json.loads(x) for x in (root / 'nvlink.jsonl').read_text().splitlines()]
        except (OSError, ValueError) as exc:
            errors.append(f'{host}: {exc}')
            continue
        collection_errors = [r for r in rows + links if r['metric'] == 'collector_error']
        if collection_errors:
            errors.append(f'{host}: {len(collection_errors)} GPU/NVLink collector errors.')
        for gpu in [g for g in expected if g['hostname'] == host]:
            uuid = gpu['uuid']
            samples = [r['value'] for r in rows if r.get('gpu_uuid') == uuid and r['metric'] == 'utilization.gpu']
            if not samples or max(samples) <= 0:
                errors.append(f'{uuid}: no nonzero GPU utilization sample during preflight.')
            else:
                results.append(dict(metric('observed_gpu_utilization_max', max(samples), '%', host), gpu_uuid=uuid))
            for link in range(18):
                samples = [r['value'] for r in links if r.get('gpu_uuid') == uuid and
                           r.get('link') == link and r['metric'] == 'nvlink_data_tx_bytes_total']
                if len(samples) < 2 or samples[-1] <= samples[0]:
                    errors.append(f'{uuid}/link{link}: no positive observed NVLink transmit-counter change.')
        for name in ['infiniband', 'cpu-memory-numa', 'lustre']:
            p = root / (name + '.jsonl')
            if not p.is_file() or p.stat().st_size == 0:
                errors.append(f'{host}: missing {name} samples.')
    # Complete streams are merged only after all node writers have exited.
    for name in ['nvidia-smi', 'nvlink', 'infiniband', 'cpu-memory-numa', 'lustre']:
        atomic(run.root / 'telemetry' / (name + '.jsonl'), ''.join(
            p.read_text() for p in sorted((run.root / 'telemetry/native').glob('*/' + name + '.jsonl'))))
    phase.finish('fail' if errors else 'ok', results=results,
                 metadata={'findings': errors, 'scope': 'Native GPU/NVLink load visibility and stream presence. Not full RL telemetry coverage.',
                           'artifacts': ['telemetry/native']}, failure_summary='; '.join(errors) or None, refresh=False)
    return not errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    code = Path(__file__).resolve().parent
    (run.root / 'control').mkdir(exist_ok=True)
    phase = run.phase('01-native-orchestration')
    errors = []
    child = lambda action: ['python3', str(code / 'infra_node.py'), '--run-dir', str(run.root), action]
    os.environ.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,NET,COLL,ENV',
                      NCCL_SOCKET_IFNAME='eth0', NCCL_SOCKET_FAMILY='AF_INET',
                      OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1',
                      OMPI_MCA_btl_tcp_if_include='eth0')
    collector = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    load_phases = ['01-native-allreduce-' + host for host in HOSTS] + ['01-native-allreduce-four-nodes', '01-native-storage-dispatch']
    try:
        status, _, _ = phase.command(srun(HOSTS) + child('start-inventory'), timeout=120)
        if status:
            raise RuntimeError('Allocation inventory failed. No load was started.')
        collector = executor.submit(run_step, run, '01-native-telemetry-processes', srun(HOSTS) +
            ['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root), '--limit-s', '840'], 855)
        deadline = time.monotonic() + 30
        while not all((run.root / 'telemetry/native' / h / 'nvidia-smi.jsonl.partial').exists() for h in HOSTS):
            if collector.done() or time.monotonic() > deadline:
                raise RuntimeError('Collectors did not start on all four nodes. No load was started.')
            time.sleep(0.25)
        for hosts in [[h] for h in HOSTS] + [HOSTS]:
            name = '01-native-allreduce-' + (hosts[0] if len(hosts) == 1 else 'four-nodes')
            debug = run.root / 'telemetry' / 'nccl' / name
            debug.mkdir(parents=True, exist_ok=False)
            os.environ['NCCL_DEBUG_FILE'] = str(debug / 'nccl.%h.%p.log')
            argv = srun(hosts, mpi='pmix') + ['/usr/local/bin/all_reduce_perf',
                    '-b', '8M', '-e', '512M', '-f', '8', '-g', '8', '-n', '2000', '-w', '20', '-c', '1']
            if not run_step(run, name, argv, 120,
                            metadata={'hosts': hosts, 'scope': 'Native all-reduce preflight; one MPI process controls eight GPUs per host.',
                                      'environment': {k: v for k, v in os.environ.items() if k.startswith('NCCL_') or k.startswith('OMPI_')},
                                      'artifacts': [str(debug.relative_to(run.root))]},
                            parser=lambda out, n=len(hosts): parse_nccl(out, n)):
                raise RuntimeError(name + ' failed. Remaining load phases were stopped.')
        if not run_step(run, '01-native-storage-dispatch', srun(HOSTS) + child('storage'), 120):
            raise RuntimeError('The storage smoke failed.')
    except Exception as exc:
        errors.append(str(exc))
        atomic(phase.path / 'logs' / 'exception.txt', traceback.format_exc())
    finally:
        atomic(run.root / 'control/native-telemetry.stop', {'time': dt.datetime.now(dt.timezone.utc).isoformat()})
        if collector is not None:
            try:
                if not collector.result(timeout=30):
                    errors.append('Telemetry collector processes failed.')
            except Exception as exc:
                errors.append('Telemetry shutdown failed: ' + str(exc))
        executor.shutdown(wait=True)
        for name in load_phases:
            if not (run.root / 'tests' / name).exists():
                run.phase(name).finish('skip', reason='prerequisite_gate_failed', metadata={'findings': errors}, refresh=False)
        status, _, _ = phase.command(srun(HOSTS) + child('end-inventory'), timeout=120)
        if status:
            errors.append('Final allocation inventory failed.')
        if collector is not None and not verify_telemetry(run):
            errors.append('Native telemetry coverage gate failed.')
        phase.finish('fail' if errors else 'ok', metadata={'findings': errors, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                         'hosts': HOSTS, 'scope': 'Bounded native preflight only. CollectiveX, full NCCL suite, serving, and training remain pending.'},
                     failure_summary='; '.join(errors) or None)
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
