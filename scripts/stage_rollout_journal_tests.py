"""Test the committed native rollout journal in the pinned Miles image, with zero GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import traceback
import xml.etree.ElementTree as ET

from enroot_run_config import prepare
from evidence import Run, atomic, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry


TESTS = ['tests/fast/rollout/inference_rollout/test_lifecycle_attempt.py',
         'tests/fast/rollout/inference_rollout/test_sample_completion_backfill.py',
         'tests/fast/rollout/test_fully_async_rollout.py',
         'tests/fast/rollout/test_checkpointed_data_source.py']
PROBE = '''import importlib.metadata,json,os,sys,torch
sys.path.insert(0,'/miles-source');os.chdir('/miles-source')
assert torch.cuda.device_count()==0
import pytest
print(json.dumps({'cuda_device_count':torch.cuda.device_count(),'torch':torch.__version__,
 'pytest':pytest.__version__,'pytest_asyncio':importlib.metadata.version('pytest-asyncio')}),flush=True)
tests=json.loads(sys.argv[1])
raise SystemExit(pytest.main(['--noconftest','-p','asyncio','-p','no:cacheprovider',
 '-o','asyncio_mode=auto','--basetemp=/test-artifacts/tmp','--junitxml=/test-artifacts/results.xml',
 '-q','--tb=short',*tests]))
'''


def worker(run, attempt):
    phase = run.phase(f'02-rollout-journal-native-tests-v{attempt}')
    try:
        return run_tests(run, phase, attempt)
    except Exception as exc:
        atomic(phase.path / 'exception.txt', traceback.format_exc())
        phase.finish('fail', failure_summary=str(exc), refresh=False)
        return 1


def run_tests(run, phase, attempt):
    code = Path(__file__).resolve().parent
    manifest = json.loads((code / 'launch.json').read_text())
    source = (run.root / manifest['miles_source']).parent
    source_manifest = json.loads((source / 'manifest.json').read_text())
    if source_manifest['source_git_sha'] != manifest['miles_sha']:
        raise ValueError('Runtime source revision mismatch.')
    for row in source_manifest['files']:
        path = source / row['path']
        if path.is_symlink() or sha256(path) != row['sha256']:
            raise ValueError('Runtime source file checksum mismatch.')
    runtime = run.root / f'images/rollout-journal-native-tests-v{attempt}'
    runtime.mkdir(exist_ok=False)
    env = prepare(runtime)
    env['NVIDIA_VISIBLE_DEVICES'] = 'void'
    command = ['enroot', 'start', '--pid', '--ipc', '--rw', '--env', 'NVIDIA_VISIBLE_DEVICES=void',
        '--env', 'CUDA_VISIBLE_DEVICES=', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'PYTHONPATH=/miles-source:/root/Megatron-LM', '--env', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
        '--env', 'HF_HUB_OFFLINE=1', '--env', 'TRANSFORMERS_OFFLINE=1']
    for source, target, mode in [(run.root / manifest['miles_source'], '/miles-source', 'ro'),
                                 (phase.path, '/test-artifacts', 'rw')]:
        command += ['--mount', str(source) + ':' + target + ':none:bind,' + mode + ',x-create=dir']
    command += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
                'python3', '-c', PROBE, json.dumps(TESTS)]
    previous = os.environ.copy()
    os.environ.update(env)
    try:
        rc, _, _ = phase.command(command, timeout=240)
    finally:
        os.environ.clear()
        os.environ.update(previous)
    xml = phase.path / 'results.xml'
    suites = ET.parse(xml).getroot().findall('testsuite') if xml.exists() else []
    counts = {key: sum(int(suite.attrib[key]) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    okay = rc == 0 and counts['tests'] > 0 and not any(counts[key] for key in ('failures', 'errors', 'skipped'))
    result = dict(miles_revision=manifest['miles_sha'], slurm_job_id=os.environ['SLURM_JOB_ID'],
        exit_code=rc, counts=counts,
        source_manifest_sha256=sha256((run.root / manifest['miles_source']).parent / 'manifest.json'),
        scope='Native CPU scheduler, journal, cancellation, stale-buffer and cursor checkpoint tests; no model, optimizer or online trajectory qualification.')
    atomic(phase.path / 'result.json', result)
    phase.finish('ok' if okay else 'fail', metadata=result,
        failure_summary=None if okay else 'Native rollout tests failed or skipped; do not enable the candidate.', refresh=False)
    return int(not okay)


def stage(args):
    # Staging helpers are local-only; do not require Kubernetes/publisher
    # dependencies inside the CPU allocation that runs the frozen tests.
    from stage_grpo import MATERIALIZE, PARENT_MILES
    repo = Path(__file__).resolve().parents[1]
    miles = repo / 'vendor/miles'
    for path in (repo, miles):
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Commit both source trees before staging immutable tests.')
    revision = subprocess.check_output(['git', '-C', str(miles), 'rev-parse', 'HEAD'], text=True).strip()
    run = Run(args.run_dir)
    phase = run.phase(f'02-rollout-journal-test-submission-v{args.attempt}')
    prefix = f'provenance/rollout-journal-tests-code-v{args.attempt}/'
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    config = dict(miles_sha=revision, miles_parent=PARENT_MILES,
                  miles_source=f'provenance/rollout-journal-source-v{args.attempt}/miles')
    files = {prefix + 'launch.json': entry(json.dumps(config).encode())}
    changed = subprocess.check_output(['git', '-C', str(miles), 'diff', '--name-only', PARENT_MILES, revision], text=True).splitlines()
    delta = {}
    for name in changed:
        content = subprocess.check_output(['git', '-C', str(miles), 'show', revision + ':' + name])
        delta[name] = hashlib.sha256(content).hexdigest()
        files[prefix + 'miles-delta/' + name] = entry(content)
    files[prefix + 'miles-delta.json'] = entry(json.dumps(delta).encode())
    for name in ['stage_rollout_journal_tests.py', 'enroot_run_config.py', 'evidence.py', 'submit_native_preflight.py']:
        files[prefix + name] = entry((repo / 'scripts' / name).read_bytes())
    command = ['python3', remote + '/' + prefix + 'stage_rollout_journal_tests.py', '--worker',
               '--run-dir', remote, '--attempt', str(args.attempt)]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n').encode())
    worker_cmd = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    try:
        payloads = list(batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files, limit=128*1024))
    except ValueError as exc:
        phase.finish('fail', failure_summary='Source payload validation failed before upload: ' + str(exc), refresh=False)
        return 1
    for payload in payloads:
        rc, _, _ = phase.command(worker_cmd + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if rc:
            phase.finish('fail', failure_summary='Test staging failed; no job submitted.', refresh=False)
            return 1
    rc, _, _ = phase.command(worker_cmd + ['python3', '-c', MATERIALIZE, remote, prefix], timeout=60)
    if rc:
        phase.finish('fail', failure_summary='Candidate source materialization failed.', refresh=False)
        return 1
    rc, out, _ = phase.command(worker_cmd + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=1',
        '--nodelist=gpu-nodes-0', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--gpus=0', '--time=00:05:00',
        '--no-requeue', '--job-name=ptx-journal-tests-v' + str(args.attempt), '--chdir=' + remote,
        '--output=' + remote + '/provenance/journal-tests-v' + str(args.attempt) + '-%j.out',
        '--error=' + remote + '/provenance/journal-tests-v' + str(args.attempt) + '-%j.err',
        remote + '/' + prefix + 'submit.sbatch'], timeout=45)
    job = out.strip().split(';')[0]
    okay = rc == 0 and job.isdigit()
    receipt = dict(slurm_job_id=job, miles_revision=revision, requested_gpus=0)
    atomic(phase.path / 'submission.json', receipt)
    phase.finish('ok' if okay else 'fail', metadata=receipt,
        failure_summary=None if okay else 'Ambiguous submission; inspect the queue before retrying.', refresh=False)
    print(json.dumps(receipt), flush=True)
    return int(not okay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig')
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not args.worker and not args.kubeconfig:
        parser.error('--kubeconfig is required for submission')
    raise SystemExit(worker(Run(args.run_dir), args.attempt) if args.worker else stage(args))
