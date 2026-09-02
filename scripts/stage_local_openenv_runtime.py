"""Transfer immutable OpenEnv sources/lock and submit CPU-only controller build."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from evidence import Run, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit controller build code before staging.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    openenv = '7b212b00b3a8881b463bc1bf630a79194ef837c4'
    archive = gzip.compress(subprocess.check_output(['git', '-C', str(repo / 'vendor/openenv'), 'archive', openenv,
                                                    'src/openenv', 'envs/tbench2_env']), mtime=0)
    lock = (repo / 'locks/openenv-server-py312-linux.lock').read_bytes()
    phase = run.phase(f'02-local-openenv-image-submission-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/local-openenv-image-code-v{args.attempt}/'
    files = {prefix + name: entry((repo / 'scripts' / name).read_bytes())
             for name in ['evidence.py', 'build_local_openenv_runtime.py']}
    chunks = [archive[i:i+32*1024] for i in range(0, len(archive), 32*1024)]
    for i, chunk in enumerate(chunks):
        files[prefix + f'archive-parts/{i:04d}'] = entry(chunk)
    files[prefix + 'server.lock'] = entry(lock)
    manifest = {'openenv_revision': openenv, 'root_revision': revision,
                'archive_sha256': hashlib.sha256(archive).hexdigest(), 'archive_parts': len(chunks),
                'files': {'server.lock': hashlib.sha256(lock).hexdigest(),
                          **{f'archive-parts/{i:04d}': hashlib.sha256(chunk).hexdigest() for i, chunk in enumerate(chunks)}}}
    files[prefix + 'input-manifest.json'] = entry(json.dumps(manifest, sort_keys=True).encode())
    argv = ['python3', remote + '/' + prefix + 'build_local_openenv_runtime.py', '--run-dir', remote, '--attempt', str(args.attempt)]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(argv) + '\n').encode())
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    try:
        payloads = list(batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files, limit=64*1024))
    except Exception as exc:
        phase.finish('fail', failure_summary='Controller upload planning failed: ' + str(exc))
        return 1
    for payload in payloads:
        rc, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if rc:
            phase.finish('fail', failure_summary='Controller source staging failed; no build submitted.')
            return 1
    rc, text, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=1',
        '--nodelist=gpu-nodes-0', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--gpus=0',
        '--time=00:15:00', '--no-requeue', '--job-name=ptx-openenv-build-' + run.root.name + f'-v{args.attempt}',
        '--chdir=' + remote, '--output=' + remote + f'/provenance/local-openenv-build-v{args.attempt}-%j.out',
        '--error=' + remote + f'/provenance/local-openenv-build-v{args.attempt}-%j.err', remote + '/' + prefix + 'submit.sbatch'], timeout=45)
    job = text.strip().split(';')[0]
    okay = not rc and job.isdigit()
    phase.finish('ok' if okay else 'fail', failure_summary=None if okay else 'Submission ambiguous; inspect exact job name before retry.',
                 metadata={'slurm_job_id': job, **manifest, 'scope': 'CPU-only controller-image preparation, not training.'})
    print(json.dumps({'slurm_job_id': job, 'source_revision': revision}), flush=True)
    return int(not okay)


if __name__ == '__main__':
    raise SystemExit(main())
