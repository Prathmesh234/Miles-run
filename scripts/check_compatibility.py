"""Resolve isolated dependency plans without installing or modifying either stack."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from evidence import Run, atomic, metric, sha256


CASES = {
    'verifiers-range': 'verifiers>=0.2.0,<0.2.1\nverifiers[harbor]==0.3.1\n',
    'openai-pin': 'openai==2.6.1\nverifiers[harbor]==0.3.1\n',
    'agents-range': 'openai-agents<0.5\nverifiers[harbor]==0.3.1\n',
    'offline-evaluation': 'verifiers[harbor]==0.3.1\nharbor==0.21.0\n',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--uv', default=shutil.which('uv') or str(Path.home() / '.local/bin/uv'))
    ap.add_argument('--exclude-newer', default='2026-09-02T00:00:00Z')
    args = ap.parse_args()
    run = Run(args.run_dir)
    for name, requirements in CASES.items():
        phase = run.phase('00-compatibility-' + name)
        source = phase.path / 'requirements.in'
        target = phase.path / 'requirements.lock'
        atomic(source, requirements)
        phase.command([args.uv, '--version'])
        code, _, stderr = phase.command([
            args.uv, 'pip', 'compile', str(source), '--python-version', '3.12.11',
            '--python-platform', 'x86_64-unknown-linux-gnu', '--generate-hashes',
            '--exclude-newer', args.exclude_newer, '--index-url', 'https://pypi.org/simple',
            '--output-file', str(target),
        ], timeout=180)
        expected_conflict = name != 'offline-evaluation'
        unsatisfiable = code == 1 and 'No solution found' in stderr
        metadata = {
            'python': '3.12.11', 'platform': 'x86_64-unknown-linux-gnu',
            'exclude_newer': args.exclude_newer, 'requirements_sha256': sha256(source),
            'expected_unsatisfiable': expected_conflict, 'unsatisfiable': unsatisfiable,
            'installed': False, 'optimizer_enabled': False,
            'artifacts': [str(source.relative_to(run.root))],
            'scope': 'Dependency resolution only; not a runtime or trajectory compatibility test.',
        }
        if target.exists():
            metadata['lock_sha256'] = sha256(target)
            metadata['artifacts'].append(str(target.relative_to(run.root)))
        if expected_conflict:
            phase.finish('fail', metadata=metadata,
                failure_summary=(
                    'The strict online combination is unsatisfiable, as expected. No package was installed.'
                    if unsatisfiable else
                    'The expected dependency conflict was not proven. Inspect the resolver output.'
                ))
        else:
            phase.finish('fail' if code else 'ok', metadata=metadata,
                failure_summary='Offline evaluation dependency resolution failed.' if code else None,
                results=[] if code else [metric('dependency_lock_created', 1, 'bool')])
        print(json.dumps({'case': name, 'exit_code': code,
                          'expected_conflict_proven': expected_conflict and unsatisfiable}), flush=True)
        if (expected_conflict and not unsatisfiable) or (not expected_conflict and code):
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
