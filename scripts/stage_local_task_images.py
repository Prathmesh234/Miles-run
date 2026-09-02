"""Stage committed local-task image builder and submit CPU-only preparation."""
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
    args = ap.parse_args()
    run = Run(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit image-builder sources before submission.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    phase = run.phase(f'02-local-task-images-submission-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/local-task-images-code-v{args.attempt}/'
    files = {prefix + name: entry((repo / 'scripts' / name).read_bytes())
             for name in ['evidence.py', 'prepare_local_task_images.py']}
    files[prefix + 'terminal-lego-subset.json'] = entry((repo / 'locks/terminal-lego-subset.json').read_bytes())
    files[prefix + 'source-revision.txt'] = entry((revision + '\n').encode())
    argv = ['python3', remote + '/' + prefix + 'prepare_local_task_images.py', '--run-dir', remote,
            '--attempt', str(args.attempt)]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(argv) + '\n').encode())
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    for payload in batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files, limit=64*1024):
        rc, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if rc:
            phase.finish('fail', failure_summary='Builder staging failed; no workload submitted.')
            return 1
    rc, out, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=1',
        '--nodelist=gpu-nodes-0', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--gpus=0',
        '--time=00:35:00', '--no-requeue', '--job-name=ptx-local-env-images-' + run.root.name + f'-v{args.attempt}',
        '--chdir=' + remote, '--output=' + remote + f'/provenance/local-task-images-v{args.attempt}-%j.out',
        '--error=' + remote + f'/provenance/local-task-images-v{args.attempt}-%j.err', remote + '/' + prefix + 'submit.sbatch'], timeout=45)
    job = out.strip().split(';')[0]
    okay = not rc and job.isdigit()
    data = {'slurm_job_id': job, 'source_git_sha': revision,
            'scope': 'CPU-only environment image preparation; zero GPU reservation, no model or optimizer execution.'}
    atomic(phase.path / 'submission.json', data)
    phase.finish('ok' if okay else 'fail', failure_summary=None if okay else 'Ambiguous submission: inspect job name before retrying.', metadata=data)
    print(json.dumps(data), flush=True)
    return int(not okay)


if __name__ == '__main__':
    raise SystemExit(main())
