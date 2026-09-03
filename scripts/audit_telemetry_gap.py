"""Locate collector gaps in retained evidence; do not infer hardware causality."""
import argparse
import inspect
import json

from evidence import Run, atomic, metric


def inspect_gap(root, label, host):
    import datetime as dt
    import hashlib
    import json
    from pathlib import Path

    root = Path(root)
    directory = root / 'telemetry' / label / host
    sources = {}

    def rows(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                yield json.loads(line)
        sources[str(path.relative_to(root))] = digest.hexdigest()

    ticks = [row for row in rows(directory / 'cpu-memory-numa.jsonl')
             if row.get('metric') == 'collection_duration']
    if len(ticks) < 2:
        raise ValueError('Insufficient finalized collector ticks.')
    pairs = list(zip(ticks, ticks[1:]))
    before, after = max(pairs, key=lambda pair: pair[1]['monotonic_s'] - pair[0]['monotonic_s'])
    tick = before['monotonic_s']
    fields = {}
    for row in rows(directory / 'nvlink.jsonl'):
        if row['monotonic_s'] != tick:
            continue
        if row.get('metric') == 'collector_error':
            raise ValueError('NVLink failure at selected tick; inspect original error.')
        gpu = fields.setdefault(row['gpu_uuid'], dict(first_timestamp_us=row['nvml_sample_timestamp_us'],
            last_timestamp_us=row['nvml_sample_timestamp_us'], max_query_latency_us=0))
        gpu['first_timestamp_us'] = min(gpu['first_timestamp_us'], row['nvml_sample_timestamp_us'])
        gpu['last_timestamp_us'] = max(gpu['last_timestamp_us'], row['nvml_sample_timestamp_us'])
        gpu['max_query_latency_us'] = max(gpu['max_query_latency_us'], row['nvml_query_latency_us'])
    epoch = dt.datetime.fromisoformat(before['time'].replace('Z', '+00:00')).timestamp()
    for gpu in fields.values():
        gpu['first_field_after_loop_start_s'] = gpu['first_timestamp_us'] / 1e6 - epoch
    pma = [dict(time=row['time'], hca=row['hca'], duration_s=row['duration_s'],
                exit_code=row.get('exit_code'), error=row.get('error'))
           for row in rows(directory / 'raw-infiniband.jsonl')
           if tick <= row['monotonic_s'] < after['monotonic_s']]
    lustre_times = {}
    for row in rows(root / 'telemetry' / ('lustre-' + label) / host / 'lustre.jsonl'):
        lustre_times[row['monotonic_s']] = row['time']
    ordered = sorted(lustre_times)
    return dict(hostname=host, gap_start=before['time'], gap_end=after['time'],
        max_sample_gap_s=after['monotonic_s'] - tick,
        collection_duration_before_flush_s=before['value'], nvml_fields=fields, pma=pma,
        host_lustre_max_gap_s=max(b-a for a, b in zip(ordered, ordered[1:])),
        source_sha256=sources,
        limitation='GPU getters were not individually timed. Field timestamps constrain the stall location but do not prove which API, scheduler, or driver operation caused it.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--training-attempt', type=int, required=True)
    parser.add_argument('--audit-attempt', type=int, required=True)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-telemetry-gap-v{args.training_attempt}-a{args.audit_attempt}')
    program = inspect.getsource(inspect_gap) + '\nimport json,sys\n'
    program += "print(json.dumps([inspect_gap(sys.argv[1], sys.argv[2], 'gpu-nodes-'+str(i)) for i in range(4)]))\n"
    atomic(phase.path / 'inspect-remote.py', program)
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program, remote,
        f'sync-grpo-v{args.training_attempt}'], timeout=55)
    if rc:
        phase.finish('fail', failure_summary='Gap inspection failed; retain original evidence.', refresh=False)
        return 1
    nodes = json.loads(out)
    failed = [row['hostname'] for row in nodes if row['max_sample_gap_s'] > 12]
    result = dict(schema_version=1, nodes=nodes, heartbeat_limit_s=12,
        findings=['Heartbeat sampling gap exceeds 12 seconds on ' + host for host in failed],
        scope='Retrospective failure localization, not a new telemetry qualification or hardware diagnosis.')
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if failed else 'ok', metadata=result,
        results=[metric('max_sample_gap', row['max_sample_gap_s'], 's', row['hostname']) for row in nodes],
        failure_summary='; '.join(result['findings']) or None, refresh=False)
    print(json.dumps({row['hostname']: dict(gap_s=row['max_sample_gap_s'],
        collection_s=row['collection_duration_before_flush_s'],
        first_nvml_field_delay_s=min(v['first_field_after_loop_start_s'] for v in row['nvml_fields'].values())) for row in nodes}))
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
