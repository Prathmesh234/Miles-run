"""Stage an upstream-baseline diagnostic without changing the failed test bundle."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from evidence import Run, atomic, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry


BASE = '0709889b2848f293b5575d50aa3340fa4de5a20d'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    miles = repo / 'vendor/miles'
    phase = run.phase('00-miles-baseline-test-staging')
    original = json.loads((run.root / 'provenance/linux-launch-tests-v1/manifest.json').read_text())
    code, changed, _ = phase.command(['git', '-C', str(miles), 'diff', '--name-only', BASE, original['source_revision']])
    if code:
        phase.finish('fail', failure_summary='Unable to enumerate the committed source delta.')
        return 1
    delta = {}
    for name in changed.splitlines():
        exists = subprocess.run(['git', '-C', str(miles), 'cat-file', '-e', BASE + ':' + name], capture_output=True)
        delta[name] = (base64.b64encode(subprocess.check_output(
            ['git', '-C', str(miles), 'show', BASE + ':' + name])).decode() if exists.returncode == 0 else None)
    local = run.root / 'provenance/linux-launch-tests-baseline-v1'
    local.mkdir(exist_ok=False)
    atomic(local / 'baseline-delta.json', delta)
    manifest = dict(original, baseline_revision=BASE,
        baseline_delta_sha256=sha256(local / 'baseline-delta.json'),
        test_paths=['tests/fast/launch_scripts', 'tests/manual/launch_scripts', 'tests/fast/utils/test_command_utils.py'],
        scope='Diagnostic upstream-baseline comparison only. Does not repair or supersede the failed patched suite.')
    atomic(local / 'manifest.json', manifest)
    frozen = (run.root / 'tests/00-miles-linux-launch-suite/logs/container/packages.freeze.txt').read_text()
    atomic(local / 'requirements.txt', '\n'.join(x for x in frozen.splitlines() if not x.startswith('torch==')) + '\n')
    atomic(local / 'runner.py', (repo / 'scripts/linux_launch_tests.py').read_text())
    files = {'provenance/' + local.name + '/' + p.name: entry(p.read_bytes()) for p in local.iterdir()}
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '--request-timeout=30s', '-n', 'slurm',
              'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    common = {'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}
    for payload in batches(common, files):
        code, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if code:
            phase.finish('fail', failure_summary='Baseline diagnostic staging failed; preserve partial evidence.')
            return 1
    phase.finish('ok', metadata=manifest)
    print(json.dumps(manifest))
    return 0


if __name__ == '__main__':
    sys.exit(main())
