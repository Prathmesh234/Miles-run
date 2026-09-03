"""Queue an opt-in data-source checkpoint test in the pinned Miles image.

Uses zero GPUs and waits until the specified training allocation is terminal.
No running source, model, environment, or optimizer configuration is changed.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from enroot_run_config import prepare
from evidence import Run, atomic, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry

PARENT = '977bdee2fa4486349728b3768d504129bfa56848'
MODULE = 'miles/rollout/checkpointed_data_source.py'
TEST = 'tests/fast/rollout/test_checkpointed_data_source.py'
PROBE = '''import importlib.util,json,sys,torch
assert torch.cuda.device_count()==0, 'CPU-only checkpoint tests must not expose GPUs'
name='miles.rollout.checkpointed_data_source'
spec=importlib.util.spec_from_file_location(name,'/candidate/checkpointed_data_source.py')
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
import pytest
print(json.dumps({'cuda_device_count':torch.cuda.device_count(),'torch':torch.__version__}),flush=True)
raise SystemExit(pytest.main(['--noconftest','/candidate/test_checkpointed_data_source.py',
 '--basetemp=/test-artifacts/tmp','--junitxml=/test-artifacts/results.xml','-q','--tb=short']))
'''


def worker(run, attempt):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('A CPU Slurm allocation is required.')
    code = Path(__file__).resolve().parent
    manifest = json.loads((code / 'manifest.json').read_text())
    phase = run.phase(f'02-data-source-resume-test-v{attempt}')
    for name, checksum in manifest['files'].items():
        if sha256(code / name) != checksum:
            raise ValueError('Candidate file hash mismatch.')
    parent = run.root / 'provenance/sync-grpo-source-v10'
    if json.loads((parent / 'manifest.json').read_text())['source_git_sha'] != PARENT:
        raise ValueError('Unexpected parent Miles source.')
    runtime = run.root / f'images/data-source-resume-test-v{attempt}'
    runtime.mkdir(exist_ok=False)
    env = prepare(runtime)
    env['NVIDIA_VISIBLE_DEVICES'] = 'void'
    cmd = ['enroot', 'start', '--pid', '--ipc', '--rw', '--env', 'NVIDIA_VISIBLE_DEVICES=void',
        '--env', 'CUDA_VISIBLE_DEVICES=', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'PYTHONPATH=/miles-source:/root/Megatron-LM', '--env', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
        '--env', 'HF_HUB_OFFLINE=1', '--env', 'TRANSFORMERS_OFFLINE=1']
    for source, target, mode in [(parent / 'miles', '/miles-source', 'ro'), (code, '/candidate', 'ro'),
                                  (phase.path, '/test-artifacts', 'rw')]:
        cmd += ['--mount', str(source) + ':' + target + ':none:bind,' + mode + ',x-create=dir']
    cmd += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'), 'python3', '-c', PROBE]
    # prepare() configures this process's child through environment variables.
    # Keep the command invocation in Phase so stdout/stderr and timeouts survive.
    previous = os.environ.copy()
    os.environ.update(env)
    try:
        rc, _, _ = phase.command(cmd, timeout=180)
    finally:
        os.environ.clear()
        os.environ.update(previous)
    result = dict(manifest, slurm_job_id=os.environ['SLURM_JOB_ID'], exit_code=rc,
        scope='Native Sample/cursor/recycled-buffer CPU round trip only; not model, optimizer, async queue, or policy-version resume.')
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if rc else 'ok', metadata=result,
        failure_summary='Pinned CPU checkpoint test failed; candidate remains disabled.' if rc else None, refresh=False)
    return rc


def stage(args):
    repo = Path(__file__).resolve().parents[1]
    miles = repo / 'vendor/miles'
    for path in (repo, miles):
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Commit candidate and test runner before staging.')
    revision = subprocess.check_output(['git', '-C', str(miles), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(miles), 'diff', '--name-only', PARENT, revision], text=True).splitlines()
    if sorted(changed) != sorted([MODULE, TEST]):
        raise ValueError('Read-only overlay test requires exactly the two candidate additions.')
    run = Run(args.run_dir)
    phase = run.phase(f'02-data-source-resume-test-submission-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/data-source-resume-test-v{args.attempt}/'
    content = {Path(MODULE).name: (miles / MODULE).read_bytes(), Path(TEST).name: (miles / TEST).read_bytes()}
    for name in ('stage_data_source_resume_test.py', 'enroot_run_config.py', 'evidence.py', 'submit_native_preflight.py'):
        content[name] = (repo / 'scripts' / name).read_bytes()
    import hashlib
    manifest = dict(miles_revision=revision, parent_revision=PARENT,
        files={name: hashlib.sha256(data).hexdigest() for name, data in content.items()},
        scope='Opt-in source checkpoint candidate; not enabled in the GRPO recipe.')
    content['manifest.json'] = (json.dumps(manifest, sort_keys=True) + '\n').encode()
    cmd = ['python3', remote + '/' + prefix + 'stage_data_source_resume_test.py', '--worker',
        '--run-dir', remote, '--attempt', str(args.attempt)]
    content['submit.sbatch'] = ('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(cmd) + '\n').encode()
    files = {prefix + name: entry(data) for name, data in content.items()}
    command = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    for payload in batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files):
        rc, _, _ = phase.command(command + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if rc:
            phase.finish('fail', failure_summary='Candidate staging failed; no test submitted.', refresh=False)
            return 1
    rc, out, _ = phase.command(command + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=1',
        '--nodelist=gpu-nodes-0', '--ntasks=1', '--cpus-per-task=2', '--mem=8G', '--gpus=0', '--time=00:04:00',
        '--dependency=afterany:' + str(args.after_job), '--no-requeue', '--job-name=ptx-source-resume-v' + str(args.attempt),
        '--chdir=' + remote, '--output=' + remote + '/provenance/source-resume-v' + str(args.attempt) + '-%j.out',
        '--error=' + remote + '/provenance/source-resume-v' + str(args.attempt) + '-%j.err',
        remote + '/' + prefix + 'submit.sbatch'], timeout=45)
    job = out.strip().split(';')[0]
    okay = not rc and job.isdigit()
    receipt = dict(slurm_job_id=job, afterany_job=args.after_job, miles_revision=revision, requested_gpus=0)
    atomic(phase.path / 'submission.json', receipt)
    phase.finish('ok' if okay else 'fail', metadata=receipt, refresh=False,
        failure_summary=None if okay else 'Ambiguous submission; inspect the job name before retrying.')
    print(json.dumps(receipt), flush=True)
    return int(not okay)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig')
    ap.add_argument('--attempt', required=True, type=int)
    ap.add_argument('--after-job', type=int)
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    if not args.worker and (not args.after_job or not args.kubeconfig):
        ap.error('--after-job and --kubeconfig are required for submission')
    raise SystemExit(worker(Run(args.run_dir), args.attempt) if args.worker else stage(args))
