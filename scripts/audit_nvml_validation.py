"""Fetch a compact qualification result; keep raw GPU evidence on shared storage."""
import argparse
import inspect
import json

from evidence import Run, atomic, markdown


def audit_remote(root_text, attempt, job):
    import hashlib
    import json
    from pathlib import Path
    import subprocess

    root = Path(root_text)
    label = f'nvml-qualification-v{attempt}'
    accounting = subprocess.check_output(['sacct', '-j', str(job), '--noheader', '--parsable2',
        '--format=JobID,State,ExitCode,Elapsed,NodeList'], text=True)
    jobs = [line.split('|') for line in accounting.splitlines() if line.split('|')[0] == str(job)]
    if len(jobs) != 1 or jobs[0][1].split()[0] not in ('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL'):
        raise ValueError('Job is not unambiguously terminal; no final result emitted.')
    findings, nodes = [], []
    if jobs[0][1:3] != ['COMPLETED', '0:0']:
        findings.append('Slurm qualification did not complete successfully.')
    for index in range(4):
        host = f'gpu-nodes-{index}'
        phase = root / 'tests' / ('01-' + label + '-' + host)
        finals = list(phase.glob('*.values.json')) + list(phase.glob('*.failed.json'))
        if len(finals) != 1:
            findings.append(host + ': missing or ambiguous phase result.')
            continue
        data = json.loads(finals[0].read_text())
        node = {'hostname': host, 'phase': data, 'path': str(finals[0].relative_to(root)),
                'sha256': hashlib.sha256(finals[0].read_bytes()).hexdigest()}
        for name in ('failure.json', 'nvml-validation.json', 'heartbeat.json'):
            path = root / 'telemetry' / label / host / name
            if path.is_file():
                node[name] = json.loads(path.read_text())
        if data['status'] != 'ok':
            findings.append(host + ': ' + data['failure_summary'])
        nodes.append(node)
    return dict(slurm_job_id=str(job), attempt=attempt, findings=findings, nodes=nodes,
        slurm_accounting=accounting, scope='GPU/NVLink collector qualification, not full training telemetry or quality.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--job-id', type=int, required=True)
    a = ap.parse_args()
    run = Run(a.run_dir)
    phase = run.phase(f'01-nvml-result-audit-v{a.attempt}')
    program = inspect.getsource(audit_remote) + '\nimport sys\nprint(json.dumps(audit_remote(sys.argv[1], int(sys.argv[2]), sys.argv[3])))\n'
    # json is imported inside the function too, but needed by the final serializer.
    program = 'import json\n' + program
    atomic(phase.path / 'audit-remote.py', program)
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', a.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program,
        '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name, str(a.attempt), str(a.job_id)], timeout=45)
    data = json.loads(out) if not rc else {'findings': ['Read-only qualification audit failed; inspect logs.']}
    atomic(phase.path / 'result.json', data)
    phase.finish('fail' if data['findings'] else 'ok', metadata=data,
        failure_summary='; '.join(data['findings']) or None, refresh=False)
    print(json.dumps({'job_id': a.job_id, 'findings': data['findings']}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
