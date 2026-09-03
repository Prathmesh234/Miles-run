"""Independently reconcile terminal replay receipts, comparison rows and inventory."""
import argparse
import inspect
import json

from evidence import Run, atomic, metric


def canonical_gpu_uuid(value):
    import uuid
    return str(uuid.UUID(value.removeprefix('GPU-')))


def audit_comparisons(rows, summary):
    import collections
    import math

    findings, components = [], collections.defaultdict(lambda: dict(leaves=0, failed=0, tensor_bytes=0))
    by_path = {row['path']: row for row in rows}
    if len(by_path) != len(rows):
        findings.append('Duplicate comparison paths.')
    required_failures = excluded = excluded_bytes = tensor_bytes = 0
    for row in rows:
        parts = row['path'].split('/')
        if len(parts) < 2 or parts[0] != 'state' or type(row.get('equal')) is not bool:
            findings.append('Malformed comparison identity/equality.')
            continue
        component = components[parts[1]]
        component['leaves'] += 1
        size = row.get('bytes', 0)
        tensor_bytes += size
        component['tensor_bytes'] += size
        required = row.get('required_for_resume', True)
        if type(required) is not bool:
            findings.append('Malformed logical-state requirement.')
        if not required:
            marker = by_path.get(row['path'].rsplit('/', 1)[0] + '/padding', {})
            valid = (row['path'].startswith('state/optimizer/') and '/param_state/' in row['path']
                and row.get('actual_type') == row.get('expected_type') == 'Tensor'
                and row.get('exclusion_reason') == 'native_optimizer_padding_not_restored'
                and marker.get('equal') is True and marker.get('actual') is True and marker.get('expected') is True)
            if not valid:
                findings.append('Invalid padding exclusion: ' + row['path'])
            excluded += 1
            excluded_bytes += size
        if row['equal'] and row.get('actual_type') == 'Tensor':
            if not row.get('actual_sha256') or row.get('actual_sha256') != row.get('expected_sha256') or row.get('finite') is False:
                findings.append('Invalid tensor equality: ' + row['path'])
        if row['equal'] and row.get('actual_type') == 'type':
            if not row.get('actual_class') or row.get('actual_class') != row.get('expected_class'):
                findings.append('Invalid class equality: ' + row['path'])
        if not row['equal']:
            component['failed'] += 1
            required_failures += int(required)
    expected_components = {'model', 'optimizer', 'opt_param_scheduler', 'rng_state', 'iteration'}
    if set(components) != expected_components or not all(components[k]['tensor_bytes'] > 0 for k in ('model', 'optimizer', 'rng_state')):
        findings.append('Missing required checkpoint components or tensor coverage.')
    expected = dict(leaves=len(rows), failed_leaves=required_failures, tensor_bytes=tensor_bytes,
                    excluded_padding_leaves=excluded, excluded_padding_bytes=excluded_bytes)
    if any(summary.get(key) != value for key, value in expected.items()):
        findings.append('Comparison summary differs from raw rows.')
    if summary.get('comparison_contract') != 'logical_state_v2_exact_nonpadding_tensors_and_class_identity':
        findings.append('Unexpected comparison contract.')
    if required_failures or summary.get('missing_keys') or summary.get('unexpected_keys') or summary.get('findings'):
        findings.append('Logical checkpoint comparison did not pass.')
    duration = summary.get('duration_s')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        findings.append('Invalid comparison duration.')
    return dict(findings=findings, components=dict(components), counts=expected)


def audit_tree(root, attempt, job):
    import hashlib
    import json
    import math
    from pathlib import Path
    import subprocess

    root = Path(root)
    accounting = subprocess.check_output(['sacct', '-j', str(job), '--noheader', '--parsable2',
        '--format=JobID,State,ExitCode,Elapsed'], text=True)
    parent = [line.split('|') for line in accounting.splitlines() if line.split('|')[0] == str(job)]
    if len(parent) != 1 or parent[0][1] in ('PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING'):
        raise ValueError('Job is not unambiguously terminal; no final audit performed.')
    findings = [] if parent[0][1:3] == ['COMPLETED', '0:0'] else ['Slurm replay did not complete with exit zero.']
    config = json.loads((root / f'provenance/resume-replay-code-v{attempt}/launch.json').read_text())
    output = root / f'training/resume-replay-v{attempt}'
    artifacts, ranks = {}, []

    def read(path):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Required regular artifact missing: ' + str(path.relative_to(root)))
        raw = path.read_bytes()
        artifacts[str(path.relative_to(root))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw

    samples = {}
    for replica in ('a', 'b'):
        seen_samples = []
        for rank in range(16):
            folder = output / replica / f'rank-{rank:02d}'
            row_findings = []
            try:
                receipt = json.loads(read(folder / 'result.json'))
                host = next(h for h in config['hosts'] if h['replica'] == replica and h['node_rank'] == rank // 8)
                if (receipt['rank'] != rank or receipt['replica'] != replica or str(receipt['slurm_job_id']) != str(job)
                        or receipt['hostname'] != host['hostname']
                        or canonical_gpu_uuid(receipt.get('gpu_uuid', '')) not in {canonical_gpu_uuid(u) for u in host['gpu_uuids']}):
                    row_findings.append('Rank, allocation or GPU identity mismatch.')
                if receipt['optimizer_steps'] != 1 or receipt['findings']:
                    row_findings.append('Rank did not finish exactly one clean optimizer step.')
                for key in ('payload_identity_before', 'payload_identity_after'):
                    if receipt.get(key) != config['payload_stat']:
                        row_findings.append('Input identity evidence missing or changed: ' + key)
                source = config['dump_root'] + f'/train_data/1_{rank}.pt'
                if receipt.get('input_sha256') != config['small_inputs'][source]:
                    row_findings.append('Frozen trainer input checksum mismatch.')
                if receipt.get('topology') != dict(tp=1, pp=1, cp=1, ep=8, etp=1, dense_dp=16, expert_dp=2):
                    row_findings.append('Unexpected model/optimizer parallel topology.')
                for key in ('load_duration_s', 'step_duration_s'):
                    value = receipt.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        row_findings.append('Missing or invalid duration: ' + key)
                comparisons = {}
                for label, iteration in [('loaded-state', 0), ('next-state', 1)]:
                    summary = json.loads(read(folder / (label + '.json')))
                    rows = [json.loads(line) for line in read(folder / (label + '.jsonl')).splitlines()]
                    proof = audit_comparisons(rows, summary)
                    if summary.get('iteration') != iteration or summary.get('label') != label:
                        row_findings.append('Comparison iteration identity differs.')
                    row_findings.extend(label + ': ' + item for item in proof['findings'])
                    comparisons[label] = dict(summary=summary, counts=proof['counts'], components=proof['components'])
                selected = receipt.get('sample_indices', [])
                if len(selected) != 1 or any(type(value) is not int for value in selected):
                    row_findings.append('Expected one frozen sample per rank.')
                seen_samples.extend(selected)
                ranks.append(dict(replica=replica, rank=rank, hostname=host['hostname'], gpu_uuid=canonical_gpu_uuid(receipt['gpu_uuid']),
                    load_duration_s=receipt.get('load_duration_s'), step_duration_s=receipt.get('step_duration_s'),
                    comparisons=comparisons, findings=row_findings))
            except (ValueError, KeyError, OSError) as exc:
                row_findings.append(str(exc))
            findings.extend(f'{replica}/{rank}: {item}' for item in row_findings)
        samples[replica] = seen_samples
        if len(seen_samples) != 16 or len(set(seen_samples)) != 16:
            findings.append('Frozen sample accounting incomplete for replica ' + replica)
    if samples['a'] != samples['b']:
        findings.append('The two replicas did not consume identical ordered samples.')
    if len({r['gpu_uuid'] for r in ranks}) != 32:
        findings.append('Expected32 distinct physical GPU receipts.')
    for host in config['hosts']:
        for stem in [f'02-resume-replay-v{attempt}-', f'01-allocation-resume-replay-v{attempt}-start-',
                     f'01-allocation-resume-replay-v{attempt}-end-']:
            name = stem + host['hostname']
            try:
                phase = json.loads(read(root / 'tests' / name / (name + '.values.json')))
                if phase['status'] != 'ok':
                    findings.append('Node or inventory phase did not pass: ' + name)
            except (ValueError, KeyError, OSError) as exc:
                findings.append(str(exc))
    return dict(schema_version=1, slurm_job_id=job, accounting=accounting, findings=findings, ranks=ranks,
        samples=samples, artifacts=artifacts,
        scope='Frozen native trainer reload and one-update comparison only. No async buffer, policy activation, held-out quality or complete telemetry claim.')


def main(args):
    run = Run(args.run_dir)
    phase = run.phase(f'02-resume-replay-result-audit-v{args.attempt}-a{args.audit_attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    source = '\n'.join(inspect.getsource(f) for f in (canonical_gpu_uuid, audit_comparisons, audit_tree))
    source += '\nimport json,sys\nprint(json.dumps(audit_tree(sys.argv[1],int(sys.argv[2]),int(sys.argv[3])),allow_nan=False))\n'
    atomic(phase.path / 'audit.py', source)
    try:
        rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
            'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', source, remote, str(args.attempt), str(args.job_id)], timeout=180)
        if rc:
            raise ValueError('Read-only replay audit failed; inspect retained stderr.')
        result = json.loads(out)
        atomic(phase.path / 'result.json', result)
        values = [metric(key, row[key], 's', row['hostname'], replica=row['replica'], rank=row['rank'])
                  for row in result['ranks'] for key in ('load_duration_s', 'step_duration_s') if row[key] is not None]
        phase.finish('fail' if result['findings'] else 'ok', metadata=result, results=values,
                     failure_summary='; '.join(result['findings']) or None, refresh=False)
        print(json.dumps(dict(slurm_job_id=args.job_id, findings=result['findings'], ranks=len(result['ranks']))))
        return int(bool(result['findings']))
    except Exception as exc:
        phase.finish('fail', failure_summary=str(exc), refresh=False)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--audit-attempt', type=int, default=1)
    raise SystemExit(main(parser.parse_args()))
