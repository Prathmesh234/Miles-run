"""Freeze and execute native checkpoint-probe tests in a zero-GPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import traceback

from enroot_run_config import prepare
from evidence import Run, atomic, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry

MILES_SHA = '3db148a3fec7afb87a8c6275027ae274a7122a19'
MILES_SOURCE = 'provenance/sync-grpo-source-v14/miles'


def worker(run, attempt):
    phase = run.phase(f'02-resume-native-cpu-test-v{attempt}')
    try:
        if not os.environ.get('SLURM_JOB_ID'):
            raise ValueError('A CPU Slurm allocation is required.')
        code = Path(__file__).resolve().parent
        manifest = json.loads((code / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha256(code / name) != digest:
                raise ValueError('Frozen probe source differs: ' + name)
        source = (run.root / manifest['miles_source']).parent
        native = json.loads((source / 'manifest.json').read_text())
        if native['source_git_sha'] != manifest['miles_sha']:
            raise ValueError('Pinned Miles revision differs.')
        for row in native['files']:
            path = source / row['path']
            if path.is_symlink() or sha256(path) != row['sha256']:
                raise ValueError('Pinned Miles source checksum differs.')
        runtime = run.root / f'images/resume-native-cpu-test-v{attempt}'
        runtime.mkdir(exist_ok=False)
        env = prepare(runtime)
        env.update(NVIDIA_VISIBLE_DEVICES='none', NVIDIA_DRIVER_CAPABILITIES='compute,utility')
        command = ['enroot', 'start', '--pid', '--ipc', '--rw']
        for key, value in dict(NVIDIA_VISIBLE_DEVICES='none', CUDA_VISIBLE_DEVICES='',
            NVIDIA_DRIVER_CAPABILITIES='compute,utility', PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH='/ptx:/miles-source:/root/Megatron-LM', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1').items():
            command += ['--env', key + '=' + value]
        for src, dst, mode in [(code, '/ptx', 'ro'), (source / 'miles', '/miles-source', 'ro'),
                               (phase.path, '/test-artifacts', 'rw')]:
            command += ['--mount', str(src) + ':' + dst + ':none:bind,' + mode + ',x-create=dir']
        command += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'), 'python3',
                    '/ptx/resume_checkpoint_probe.py', '--cpu-self-test', '/test-artifacts/fixture']
        previous = os.environ.copy()
        os.environ.update(env)
        try:
            rc, _, _ = phase.command(command, timeout=240)
        finally:
            os.environ.clear()
            os.environ.update(previous)
        result_file = phase.path / 'fixture/result.json'
        fixture = json.loads(result_file.read_text()) if result_file.exists() else None
        okay = (rc == 0 and fixture is not None and fixture['status'] == 'ok' and fixture['checks'] == 14
                and fixture['native_padding_filter_verified'] and fixture['native_class_roundtrip_verified']
                and fixture['deterministic_control_verified'] and fixture['native_shard_attribution_verified'])
        result = dict(manifest, slurm_job_id=os.environ['SLURM_JOB_ID'], exit_code=rc,
                      fixture=fixture, findings=[] if okay else ['Native CPU checkpoint fixture failed.'])
        atomic(phase.path / 'result.json', result)
        phase.finish('ok' if okay else 'fail', metadata=result, refresh=False,
                     failure_summary=None if okay else result['findings'][0])
        return int(not okay)
    except Exception as exc:
        atomic(phase.path / 'exception.txt', traceback.format_exc())
        phase.finish('fail', failure_summary=str(exc), refresh=False)
        return 1


def stage(args):
    repo = Path(__file__).resolve().parents[1]
    for path in (repo, repo / 'vendor/miles'):
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Commit both source trees before staging immutable tests.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo / 'vendor/miles'), 'rev-parse', 'HEAD'], text=True).strip() != MILES_SHA:
        raise ValueError('Unexpected Miles revision.')
    run = Run(args.run_dir)
    phase = run.phase(f'02-resume-native-cpu-submission-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/resume-native-cpu-code-v{args.attempt}/'
    content = {name: (repo / 'scripts' / name).read_bytes() for name in (
        'stage_resume_cpu_test.py', 'resume_checkpoint_probe.py', 'resume_replay_controls.py', 'enroot_run_config.py',
        'evidence.py', 'submit_native_preflight.py')}
    manifest = dict(source_git_sha=revision, miles_sha=MILES_SHA, miles_source=MILES_SOURCE,
                    files={name: hashlib.sha256(data).hexdigest() for name, data in content.items()})
    content['manifest.json'] = (json.dumps(manifest, sort_keys=True) + '\n').encode()
    command = ['python3', remote + '/' + prefix + 'stage_resume_cpu_test.py', '--worker',
               '--run-dir', remote, '--attempt', str(args.attempt)]
    content['submit.sbatch'] = ('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n').encode()
    files = {prefix + name: entry(data) for name, data in content.items()}
    kube = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    try:
        for payload in batches(dict(root=remote, create=False, manifest_sha256=sha256(run.root / 'run.json')), files):
            rc, _, _ = phase.command(kube + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
            if rc:
                raise ValueError('Immutable source upload failed; no job submitted.')
        rc, out, _ = phase.command(kube + ['sbatch', '--parsable', '--partition=gpu-nodes',
            '--nodes=1', '--nodelist=gpu-nodes-0', '--ntasks=1', '--cpus-per-task=4', '--mem=16G',
            '--gpus=0', '--time=00:05:00', '--no-requeue', '--job-name=ptx-resume-cpu-v' + str(args.attempt),
            '--chdir=' + remote, '--output=' + remote + '/' + prefix + 'slurm-%j.out',
            '--error=' + remote + '/' + prefix + 'slurm-%j.err', remote + '/' + prefix + 'submit.sbatch'], timeout=45)
        job = out.strip().split(';')[0]
        if rc or not job.isdigit():
            raise ValueError('Ambiguous submission; inspect the queue before retrying.')
        receipt = dict(slurm_job_id=job, source_git_sha=revision, miles_revision=MILES_SHA, requested_gpus=0)
        atomic(phase.path / 'submission.json', receipt)
        phase.finish('ok', metadata=receipt, refresh=False)
        print(json.dumps(receipt), flush=True)
        return 0
    except Exception as exc:
        atomic(phase.path / 'exception.txt', traceback.format_exc())
        phase.finish('fail', failure_summary=str(exc), refresh=False)
        return 1


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
