"""Explain collector stalls from finalized per-API evidence, without relaxing gates."""
import argparse
import inspect
import json
import re

from evidence import Run, atomic, metric
from summarize_native import percentile, summary


def analyze_calls(rows, hostname, job_id, expected_uuids):
    import datetime as dt
    import math

    by_api, by_gpu, errors = {}, {}, []
    for row in rows:
        if (row['hostname'] != hostname or str(row['slurm_job_id']) != str(job_id)
                or row['gpu_uuid'] not in expected_uuids or row['metric'] != 'nvml_api_duration'
                or row['source'] != 'persistent-nvml' or row['unit'] != 's'):
            raise ValueError('NVML call identity or metric contract differs.')
        stamp = dt.datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
        if stamp.tzinfo is None or any(not math.isfinite(row[k]) or row[k] < 0 for k in ('value', 'monotonic_s')):
            raise ValueError('Invalid NVML call timing; do not replace it with zero.')
        by_api.setdefault(row['api'], []).append(row['value'])
        by_gpu.setdefault((row['gpu_uuid'], row['api']), []).append(row['value'])
        if row['error'] is not None:
            errors.append(row)
    if not rows or {row['gpu_uuid'] for row in rows} != set(expected_uuids):
        raise ValueError('NVML call evidence does not cover every expected GPU.')
    findings = []
    if errors:
        findings.append(f'{hostname}: {len(errors)} NVML calls raised errors.')
    if max(row['value'] for row in rows) > 12:
        findings.append(hostname + ': an NVML call exceeded the unchanged 12s health deadline.')
    return dict(hostname=hostname, call_count=len(rows), error_count=len(errors), errors=errors,
        findings=findings, first_call=min(row['time'] for row in rows), last_call=max(row['time'] for row in rows),
        by_api=[dict(api=api, statistics=summary(values)) for api, values in sorted(by_api.items())],
        by_gpu=[dict(gpu_uuid=gpu, api=api, statistics=summary(values)) for (gpu, api), values in sorted(by_gpu.items())],
        slowest_calls=sorted(rows, key=lambda row: row['value'], reverse=True)[:10])


def audit_remote(root_text, label, job_id):
    import hashlib
    import json
    from pathlib import Path
    import subprocess

    root = Path(root_text)
    accounting = subprocess.check_output(['sacct', '-X', '-j', str(job_id), '--noheader', '--parsable2',
        '--format=JobID,State,ExitCode'], text=True)
    jobs = [line.split('|') for line in accounting.splitlines() if line.split('|')[0] == str(job_id)]
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'}
    if len(jobs) != 1 or jobs[0][1].split()[0] not in terminal:
        raise ValueError('Call-duration audit requires a terminal allocation.')
    inventory = json.loads((root / 'inventory/gpu.values.json').read_text())['gpus']
    nodes, findings, sources = [], [], {}
    for index in range(4):
        host = f'gpu-nodes-{index}'
        path = root / 'telemetry' / label / host / 'nvml-api.jsonl'
        if path.is_symlink() or not path.is_file():
            findings.append(host + ': no finalized API timing stream.')
            continue
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Finalized NVML evidence changed during audit.')
        expected = {gpu['uuid'] for gpu in inventory if gpu['hostname'] == host}
        if len(expected) != 8:
            raise ValueError('Inventory does not define a whole eight-GPU node.')
        node = analyze_calls([json.loads(line) for line in raw.splitlines()], host, job_id, expected)
        nodes.append(node)
        findings.extend(node['findings'])
        sources[str(path.relative_to(root))] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return dict(schema_version=1, slurm_job_id=str(job_id), slurm_accounting=accounting,
        stream_label=label, nodes=nodes, findings=findings, source_sha256=sources,
        scope='Observed NVML call latency only. Slurm outcome, collector continuity, GPU performance and hardware causality require separate evidence; a passing API audit does not qualify the training run.')


def render(data):
    lines = [f"# Job {data['slurm_job_id']}: NVML call latency", '', data['scope'], '',
        '| Node | API | Calls | Min | Mean | Median | p90 | p95 | p99 | Max | CV |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields = ('n', 'min', 'mean', 'median', 'p90', 'p95', 'p99', 'max', 'coefficient_of_variation')
    for node in data['nodes']:
        for row in node['by_api']:
            cells = [node['hostname'], row['api']] + [row['statistics'][key] for key in fields]
            lines.append('| ' + ' | '.join('undefined' if value is None else f'{value:.6g}' if isinstance(value, float) else str(value) for value in cells) + ' |')
    lines += ['', 'All latency values are seconds. JSON retains per-GPU distributions and the ten slowest calls per node.', '', '## Findings', '']
    lines += ['- ' + finding for finding in data['findings']] or ['No API timing finding. This is not a full telemetry pass.']
    lines += ['', '## Raw evidence', '']
    lines += [f"- `{path}`; SHA256 `{row['sha256']}`." for path, row in data['source_sha256'].items()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--stream-label', required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', args.stream_label):
        parser.error('Invalid stream label.')
    run = Run(args.run_dir)
    phase = run.phase(f'01-nvml-call-audit-job{args.job_id}-v{args.attempt}')
    program = 'import math,statistics\n' + '\n'.join(inspect.getsource(fn) for fn in (percentile, summary, analyze_calls, audit_remote))
    program += '\nimport json,sys\nprint(json.dumps(audit_remote(sys.argv[1],sys.argv[2],sys.argv[3]),allow_nan=False))\n'
    atomic(phase.path / 'audit-remote.py', program)
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program,
        '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name, args.stream_label, str(args.job_id)], timeout=55)
    if rc:
        phase.finish('fail', failure_summary='NVML evidence audit failed; see raw command.', refresh=False)
        return 1
    data = json.loads(out)
    atomic(phase.path / 'result.json', data)
    atomic(phase.path / 'call-latency.md', render(data))
    values = [metric('nvml_call_duration_' + key, value, 's' if key not in ('n', 'coefficient_of_variation') else 'count' if key == 'n' else 'ratio',
              node=node['hostname'], api=row['api']) for node in data['nodes'] for row in node['by_api']
              for key, value in row['statistics'].items() if value is not None]
    phase.finish('fail' if data['findings'] else 'ok', metadata={'scope': data['scope'], 'source_sha256': data['source_sha256']},
        results=values, failure_summary='; '.join(data['findings']) or None, refresh=False)
    print(json.dumps(dict(job_id=args.job_id, calls=sum(n['call_count'] for n in data['nodes']), findings=data['findings'])))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
