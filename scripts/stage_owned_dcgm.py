"""Queue the bounded run-owned DCGM probe after a specific GPU allocation."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from evidence import Run, atomic, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--after-job', required=True, type=int)
    parser.add_argument('--attempt', default=1, type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit the probe before submission.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    run = Run(args.run_dir)
    phase = run.phase(f'01-owned-dcgm-submission-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/owned-dcgm-code-v{args.attempt}/'
    files = {prefix + name: entry((repo / 'scripts' / name).read_bytes())
             for name in ('probe_owned_dcgm.py', 'evidence.py')}
    command = ['python3', remote + '/' + prefix + 'probe_owned_dcgm.py',
               '--run-dir', remote, '--attempt', str(args.attempt)]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n').encode())
    files[prefix + 'revision.txt'] = entry((revision + '\n').encode())
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    for payload in batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files):
        rc, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if rc:
            phase.finish('fail', failure_summary='Probe staging failed; no allocation submitted.', refresh=False)
            return 1
    rc, out, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=1',
        '--nodelist=gpu-nodes-3', '--ntasks=1', '--cpus-per-task=2', '--mem=4G', '--gpus=0', '--time=00:03:00',
        '--dependency=afterany:' + str(args.after_job), '--no-requeue',
        '--job-name=ptx-owned-dcgm-' + run.root.name + f'-v{args.attempt}', '--chdir=' + remote,
        '--output=' + remote + f'/provenance/owned-dcgm-v{args.attempt}-%j.out',
        '--error=' + remote + f'/provenance/owned-dcgm-v{args.attempt}-%j.err', remote + '/' + prefix + 'submit.sbatch'], timeout=45)
    job = out.strip().split(';')[0]
    okay = not rc and job.isdigit()
    receipt = {'slurm_job_id': job, 'root_sha': revision, 'afterany_job': args.after_job,
               'scope': 'CPU-only, separate run-owned DCGM read-only probe on node3; no shared daemon restart or GPU diagnostic.'}
    atomic(phase.path / 'submission.json', receipt)
    phase.finish('ok' if okay else 'fail', metadata=receipt,
                 failure_summary=None if okay else 'Ambiguous submission; inspect job name before retry.', refresh=False)
    print(json.dumps(receipt))
    return int(not okay)


if __name__ == '__main__':
    raise SystemExit(main())
