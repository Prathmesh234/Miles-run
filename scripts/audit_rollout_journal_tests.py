"""Verify terminal native CPU lifecycle tests against their Slurm/JUnit receipts."""
import argparse
import inspect
import json

from evidence import Run, atomic


def inspect_remote(root_text, attempt, job):
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import xml.etree.ElementTree as ET

    root = Path(root_text)
    phase = root / 'tests' / f'02-rollout-journal-native-tests-v{attempt}'
    accounting = subprocess.check_output(['sacct', '-j', job, '-X', '-n', '-P',
        '-o', 'JobID,State,ExitCode'], text=True)
    rows = [line.split('|') for line in accounting.splitlines() if line.split('|')[0] == job]
    if len(rows) != 1 or rows[0][1] in ('RUNNING', 'PENDING', 'COMPLETING'):
        raise ValueError('Native test allocation is not unambiguously terminal.')
    result = json.loads((phase / 'result.json').read_text())
    xml = phase / 'results.xml'
    counts = {key: sum(int(s.attrib[key]) for s in ET.parse(xml).getroot().findall('testsuite'))
              for key in ('tests', 'failures', 'errors', 'skipped')}
    findings = []
    if rows[0][1:3] != ['COMPLETED', '0:0']:
        findings.append('Slurm native test allocation did not complete successfully.')
    if (str(result['slurm_job_id']) != job or result['exit_code'] != 0 or result['counts'] != counts
            or not counts['tests'] or any(counts[k] for k in ('failures', 'errors', 'skipped'))):
        findings.append('Native test result and JUnit success/counts do not reconcile.')
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [phase / 'result.json', xml]}
    return dict(findings=findings, result=result, slurm_accounting=accounting, source_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--audit-attempt', type=int, default=1)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-rollout-journal-native-audit-v{args.attempt}-a{args.audit_attempt}')
    receipt = json.loads((run.root / f'tests/02-rollout-journal-test-submission-v{args.attempt}/submission.json').read_text())
    program = inspect.getsource(inspect_remote) + '\nimport json,sys\nprint(json.dumps(inspect_remote(sys.argv[1],int(sys.argv[2]),sys.argv[3])))\n'
    atomic(phase.path / 'audit-remote.py', program)
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '--request-timeout=30s',
        '-n', 'slurm', 'exec', 'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program,
        '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name,
        str(args.attempt), str(receipt['slurm_job_id'])], timeout=45)
    result = json.loads(out) if not rc else {'findings': ['Native test evidence could not be read.']}
    if not rc and result['result']['miles_revision'] != receipt['miles_revision']:
        result['findings'].append('Tested Miles revision differs from submission.')
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if result['findings'] else 'ok', metadata=result,
        failure_summary='; '.join(result['findings']) or None, refresh=False)
    print(json.dumps({'job': receipt['slurm_job_id'], 'findings': result['findings'],
                      'counts': result.get('result', {}).get('counts')}))
    return int(bool(result['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
