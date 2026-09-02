"""Submit bounded live sandbox qualification; never reserve GPUs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from evidence import Run, atomic, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit sources before staging.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    run = Run(a.run_dir)
    phase = run.phase(f'02-local-env-validation-submission-v{a.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/local-env-validation-code-v{a.attempt}/'
    names = ['evidence.py', 'local_file_env.py', 'local_openenv_client.py', 'local_openenv_app.py', 'validate_local_file_env.py']
    files = {prefix + name: entry((repo / 'scripts' / name).read_bytes()) for name in names}
    files[prefix + 'source-revision.txt'] = entry((revision + '\n').encode())
    image = 'sha256:1a93e04935e2b6a3948bea8cef77ebe99c07939eb685bd7fcf69226ad18b14a8'
    cmd = ['docker', 'run', '--rm', '--runtime=runc', '--network=host', '--cpus=4', '--memory=8g',
           '--pids-limit=512', '--cap-drop=ALL', '--security-opt=no-new-privileges',
           '--label=posttrainingx.run=' + run.root.name, '--label=posttrainingx.role=validation-controller',
           '--name=ptx-env-validation-' + run.root.name + '-v' + str(a.attempt),
           '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'PYTHONPATH=/ptx:/opt/openenv/src:/opt/openenv/envs',
           '-v', remote + ':' + remote + ':rw', '-v', remote + '/' + prefix.rstrip('/') + ':/ptx:ro',
           '-v', '/var/run/docker.sock:/var/run/docker.sock', image,
           'timeout', '--signal=TERM', '--kill-after=30s', '600s', 'python3', '/ptx/validate_local_file_env.py',
           '--run-dir', remote, '--images-manifest', remote + '/environments/local-file-runtime-v3/images.json',
           '--attempt', str(a.attempt)]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(cmd) + '\n').encode())
    worker = ['kubectl', '--kubeconfig', a.kubeconfig, '-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    payloads = list(batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files, limit=64*1024))
    for payload in payloads:
        rc, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if rc:
            phase.finish('fail', failure_summary='Validation source staging failed; no execution submitted.')
            return 1
    rc, out, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=1',
        '--nodelist=gpu-nodes-0', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--gpus=0',
        '--time=00:12:00', '--no-requeue', '--job-name=ptx-local-validation-' + run.root.name + '-v' + str(a.attempt),
        '--chdir=' + remote, '--output=' + remote + f'/provenance/local-validation-v{a.attempt}-%j.out',
        '--error=' + remote + f'/provenance/local-validation-v{a.attempt}-%j.err', remote + '/' + prefix + 'submit.sbatch'], timeout=45)
    job = out.strip().split(';')[0]
    okay = not rc and job.isdigit()
    result = {'slurm_job_id': job, 'source_git_sha': revision, 'scope': 'CPU-only sandbox qualification; no model or optimizer.'}
    atomic(phase.path / 'submission.json', result)
    phase.finish('ok' if okay else 'fail', failure_summary=None if okay else 'Ambiguous submission; inspect unique job name before retry.', metadata=result)
    print(json.dumps(result), flush=True)
    return int(not okay)


if __name__ == '__main__':
    raise SystemExit(main())
