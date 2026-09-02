"""Explicit recovery test for the unavailable sysfs IB counter backend.

This is collector validation, not an eligible throughput benchmark. Existing
failed phases are retained; missing Lustre/DCGM coverage is not waived.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
import traceback

from evidence import atomic, metric, utcnow
from infra_controller import HOSTS, parse_nccl, srun
from infra_node import allocated_run


LABEL = 'native-perfquery-validation-v1'


def validate_records(rows, host):
    errors, results = [], []
    if any(r['metric'] == 'collector_error' for r in rows):
        errors.append(host + ': collector errors')
    ports = sorted({(r['hca'], r['hca_port']) for r in rows if 'hca' in r})
    if len(ports) != 8:
        errors.append(host + ': expected eight observed training ports')
    for hca, port in ports:
        for name in ['PortXmitData', 'PortRcvData']:
            samples = [r for r in rows if (r.get('hca'), r.get('hca_port')) == (hca, port)
                       and r['metric'] == name]
            values = [r['value'] for r in samples]
            if len(values) < 3 or any(b < a for a, b in zip(values, values[1:])):
                errors.append(f'{host}/{hca}/{port}/{name}: missing samples or counter reset')
            elif values[-1] <= values[0]:
                errors.append(f'{host}/{hca}/{port}/{name}: no traffic observed')
            else:
                results.append(metric(name + '_delta', values[-1] - values[0], 'B', host,
                                      hca=hca, hca_port=port, samples=len(samples)))
    return errors, results


def command_gate(run, name, argv, timeout, parser=None):
    phase = run.phase(name)
    code, out, _ = phase.command(argv, timeout=timeout)
    results, error = [], None
    if code:
        error = 'Command failed: ' + str(code)
    elif parser:
        try:
            results = parser(out)
        except ValueError as exc:
            error = str(exc)
    phase.finish('fail' if error else 'ok', results=results, failure_summary=error,
                 exit_code=code or 1, refresh=False)
    return not error


def run_validation(run):
    code_root = Path(__file__).resolve().parent
    phase = run.phase('01-perfquery-load-validation')
    errors, results = [], []
    stop = run.root / 'control' / (LABEL + '-telemetry.stop')
    if stop.exists():
        phase.finish('fail', failure_summary='Immutable recovery label already has a stop marker.', refresh=False)
        return 1
    os.environ.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,NET,COLL,ENV',
                      NCCL_SOCKET_IFNAME='eth0', NCCL_SOCKET_FAMILY='AF_INET',
                      OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1',
                      OMPI_MCA_btl_tcp_if_include='eth0')
    inventory = ('import sys; sys.path.insert(0,sys.argv[1]); '
                 'from infra_node import allocated_run,read_inventory; '
                 'sys.exit(read_inventory(allocated_run(sys.argv[2]),sys.argv[3]))')
    collector = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            rc, _, _ = phase.command(srun(HOSTS) + ['python3', '-c', inventory, str(code_root),
                                    str(run.root), 'perfquery-start'], timeout=120)
            if rc:
                raise RuntimeError('Fresh allocation inventory failed; load not started.')
            collector = pool.submit(command_gate, run, '01-perfquery-load-collectors',
                srun(HOSTS) + ['python3', str(code_root / 'telemetry_native.py'), '--run-dir', str(run.root),
                '--ib-backend', 'perfquery', '--stream-label', LABEL, '--limit-s', '150'], 165)
            deadline = time.monotonic() + 30
            while not all((run.root / 'telemetry' / LABEL / h / 'infiniband.jsonl.partial').exists() for h in HOSTS):
                if collector.done() or time.monotonic() > deadline:
                    raise RuntimeError('All eight-rail collectors did not become ready; no load started.')
                time.sleep(0.25)
            debug = run.root / 'telemetry/nccl/perfquery-validation-v1'
            debug.mkdir(parents=True, exist_ok=False)
            os.environ['NCCL_DEBUG_FILE'] = str(debug / 'nccl.%h.%p.log')
            if not command_gate(run, '01-perfquery-load-allreduce', srun(HOSTS, mpi='pmix') +
                ['/usr/local/bin/all_reduce_perf', '-b', '512M', '-e', '512M', '-g', '8',
                 '-n', '2000', '-w', '20', '-c', '1'], 90, lambda out: parse_nccl(out, 4)):
                raise RuntimeError('Bounded load failed; stop and preserve evidence.')
        except Exception as exc:
            errors.append(str(exc))
            atomic(phase.path / 'logs/exception.txt', traceback.format_exc())
        finally:
            atomic(stop, {'time': utcnow()})
            if collector is not None and not collector.result():
                errors.append('Telemetry process failed.')
            rc, _, _ = phase.command(srun(HOSTS) + ['python3', '-c', inventory, str(code_root),
                                    str(run.root), 'perfquery-end'], timeout=120)
            if rc:
                errors.append('Final allocation inventory failed.')
    for host in HOSTS:
        try:
            path = run.root / 'telemetry' / LABEL / host / 'infiniband.jsonl'
            findings, metrics = validate_records([json.loads(x) for x in path.read_text().splitlines()], host)
            errors.extend(findings)
            results.extend(metrics)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(host + ': ' + str(exc))
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None, results=results,
        metadata={'scope': 'Explicit perfquery collector recovery validation only, not full telemetry or throughput benchmark.',
                  'findings': errors, 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hosts': HOSTS,
                  'concurrent_activity': 'Run-scoped pinned-image Enroot import may still be active.',
                  'artifacts': ['telemetry/' + LABEL, 'telemetry/nccl/perfquery-validation-v1']}, refresh=False)
    return int(bool(errors))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    sys.exit(run_validation(allocated_run(args.run_dir)))
