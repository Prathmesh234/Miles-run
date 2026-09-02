"""Read-only, post-allocation audit. Exit zero is never a quality claim."""
import argparse
import inspect
import json

from evidence import Run, atomic, metric


def audit_remote(root, attempt, job):
    import collections
    import hashlib
    import json
    from pathlib import Path
    import subprocess

    root = Path(root)
    label = f'sync-grpo-v{attempt}'
    accounting = subprocess.check_output([
        'sacct', '-j', str(job), '--noheader', '--parsable2',
        '--format=JobID,State,ExitCode,Elapsed,Start,End,NodeList'], text=True)
    rows = [line.split('|') for line in accounting.splitlines() if line.split('|')[0] == str(job)]
    terminal = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'}
    if len(rows) != 1 or rows[0][1].split()[0] not in terminal:
        raise ValueError('Allocation is not unambiguously terminal; do not finalize its audit.')
    findings = []
    data = {'schema_version': 1, 'slurm_job_id': str(job), 'attempt': attempt,
            'slurm_accounting': accounting, 'slurm_state': rows[0][1], 'slurm_exit_code': rows[0][2],
            'findings': findings, 'coverage': [], 'nodes': [], 'artifacts': []}

    def record(path):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or linked artifact: ' + str(path))
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        row = {'path': str(path.relative_to(root)), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
        data['artifacts'].append(row)
        return row

    if rows[0][1:3] != ['COMPLETED', '0:0']:
        findings.append('Slurm allocation failed; no training success inferred.')
    train = root / 'training' / label
    config_path = root / f'provenance/sync-grpo-code-v{attempt}/launch.json'
    config = json.loads(config_path.read_text())
    record(config_path)
    data['provenance'] = {key: config[key] for key in ('root_sha', 'miles_sha', 'layout', 'task_ids', 'optimizer_steps_requested')}
    for name in ('driver.finished.json', 'training-command.json', 'ray-placement.json'):
        path = train / name
        if path.is_file():
            data[name.removesuffix('.json')] = json.loads(path.read_text())
            record(path)
        else:
            findings.append('Missing finalized ' + name)
    bundles = data.get('ray-placement', {}).get('bundles', [])
    if len(bundles) != 32 or len({b['gpu_uuid'] for b in bundles}) != 32:
        findings.append('Ray bundle placement does not contain 32 unique GPU UUIDs.')
    for node in config['host_map']['nodes']:
        host = node['hostname']
        expected = set(node['gpu_uuids'])
        placed = [b for b in bundles if b['hostname'] == host]
        if {b['gpu_uuid'] for b in placed} != expected or any(b['role'] != node['role'] for b in placed):
            findings.append('Ray bundle placement differs from frozen host map: ' + host)
        directory = root / 'tests' / ('02-' + label + '-' + host)
        phases = list(directory.glob('*.values.json')) + list(directory.glob('*.failed.json'))
        node_row = {'hostname': host, 'role': node['role'], 'phase_finalized': len(phases) == 1}
        if len(phases) == 1:
            node_row['phase'] = json.loads(phases[0].read_text())
            record(phases[0])
        else:
            findings.append('Node phase interrupted before structured finalization: ' + host)
        for path in directory.glob('exception.txt'):
            record(path)
        data['nodes'].append(node_row)
        paths = [root / 'telemetry' / label / host / (name + '.jsonl')
                 for name in ('nvidia-smi', 'nvlink', 'infiniband', 'cpu-memory-numa', 'lustre')]
        paths += [root / 'telemetry' / ('lustre-' + label) / host / 'lustre.jsonl']
        for path in paths:
            if not path.is_file():
                findings.append('Missing finalized telemetry: ' + str(path.relative_to(root)))
                continue
            digest = hashlib.sha256()
            count = errors = 0
            times, uuids, metrics = set(), set(), collections.Counter()
            first = last = None
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line)
                    row = json.loads(line)
                    count += 1
                    times.add(row['monotonic_s'])
                    first = min(first, row['time']) if first else row['time']
                    last = max(last, row['time']) if last else row['time']
                    if row['metric'] == 'collector_error':
                        errors += 1
                        continue
                    metrics[row['metric']] += 1
                    if path.name == 'nvidia-smi.jsonl':
                        uuids.add(row['gpu_uuid'])
            ordered = sorted(times)
            intervals = [b - a for a, b in zip(ordered, ordered[1:])]
            data['coverage'].append({'path': str(path.relative_to(root)), 'sha256': digest.hexdigest(),
                'hostname': host, 'role': node['role'], 'records': count, 'collector_errors': errors,
                'sample_times': len(times), 'max_interval_s': max(intervals) if intervals else None,
                'first_time': first, 'last_time': last, 'metrics': dict(metrics)})
            if not count or errors:
                findings.append('Empty telemetry or collector errors: ' + str(path.relative_to(root)))
            if path.name == 'nvidia-smi.jsonl' and uuids != expected:
                findings.append('GPU telemetry does not cover the frozen eight UUIDs: ' + host)
    for path in [train / 'logs/gpu-nodes-0/miles.out', train / 'logs/gpu-nodes-0/miles.err',
                 root / f'provenance/{label}-{job}.out', root / f'provenance/{label}-{job}.err']:
        if path.is_file():
            record(path)
    data['scope'] = 'Terminal allocation, Ray bundle placement, node-finalization and native telemetry audit only.'
    data['unverified'] = ['GRPO trajectory accounting, token/logprob correctness and optimizer execution',
        'Resumable optimizer/scheduler/RNG/data/buffer/policy-version state',
        'Complete actor/engine identity beyond placement probe actors',
        'Full required telemetry, including RL pipeline, SGLang and DCGM coverage',
        'Held-out quality improvement and controlled role-split benchmark']
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--job-id', type=int, required=True)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-sync-grpo-result-audit-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    program = inspect.getsource(audit_remote) + '\nimport json,sys\nprint(json.dumps(audit_remote(sys.argv[1], int(sys.argv[2]), sys.argv[3])))\n'
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program, remote, str(args.attempt), str(args.job_id)], timeout=300)
    data = json.loads(out) if not rc else {'findings': ['Read-only terminal audit failed; inspect retained stderr.']}
    atomic(phase.path / 'audit.json', data)
    phase.finish('fail' if data['findings'] else 'ok', metadata=data,
        results=[metric('finalized_telemetry_streams', len(data.get('coverage', [])), 'count')],
        failure_summary='; '.join(data['findings']) or None, refresh=False)
    print(json.dumps({k: data[k] for k in ('slurm_job_id', 'slurm_state', 'findings') if k in data}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
