"""Queue an isolated CPU evaluation-image build after a specific live allocation."""
import argparse
import hashlib
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
    ap.add_argument('--after-job', type=int, required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit evaluation-image sources before submission.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    run = Run(args.run_dir)
    phase = run.phase(f'02-offline-evaluation-image-submission-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/offline-evaluation-code-v{args.attempt}/'
    lock = (repo / 'locks/offline-verifiers-py312-linux.lock').read_bytes()
    chunks = [lock[i:i+32*1024] for i in range(0,len(lock),32*1024)]
    files = {prefix + name:entry((repo / 'scripts' / name).read_bytes()) for name in ('evidence.py','build_offline_evaluation.py')}
    parts = []
    for i, chunk in enumerate(chunks):
        name = f'lock-parts/{i:04d}'
        files[prefix + name] = entry(chunk)
        parts.append({'path':name,'sha256':hashlib.sha256(chunk).hexdigest()})
    manifest = {'root_revision':revision,'lock_sha256':hashlib.sha256(lock).hexdigest(),'lock_parts':parts}
    files[prefix + 'input-manifest.json'] = entry(json.dumps(manifest).encode())
    command = ['python3',remote + '/' + prefix + 'build_offline_evaluation.py','--run-dir',remote,'--attempt',str(args.attempt)]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n').encode())
    worker = ['kubectl','--kubeconfig',args.kubeconfig,'-n','slurm','exec','-i','slurm-worker-gpu-nodes-0','--']
    for payload in batches({'root':remote,'create':False,'manifest_sha256':sha256(run.root / 'run.json')},files,limit=64*1024):
        rc,_,_ = phase.command(worker + ['python3','-c',BOOTSTRAP],stdin=payload,timeout=45)
        if rc:
            phase.finish('fail',failure_summary='Offline build staging failed.',refresh=False)
            return 1
    rc,out,_ = phase.command(worker + ['sbatch','--parsable','--partition=gpu-nodes','--nodes=1',
        '--nodelist=gpu-nodes-0','--ntasks=1','--cpus-per-task=4','--mem=16G','--gpus=0','--time=00:20:00',
        '--dependency=afterany:' + str(args.after_job),'--no-requeue',
        '--job-name=ptx-offline-eval-build-' + run.root.name + f'-v{args.attempt}',
        '--chdir=' + remote,'--output=' + remote + f'/provenance/offline-eval-build-v{args.attempt}-%j.out',
        '--error=' + remote + f'/provenance/offline-eval-build-v{args.attempt}-%j.err',remote + '/' + prefix + 'submit.sbatch'],timeout=45)
    job = out.strip().split(';')[0]
    okay = not rc and job.isdigit()
    receipt = {'slurm_job_id':job,'root_revision':revision,'afterany_job':args.after_job,'gpus':0,
               'scope':'CPU-only offline evaluator image build; no task execution or online training dependency changes.'}
    atomic(phase.path / 'submission.json',receipt)
    phase.finish('ok' if okay else 'fail',failure_summary=None if okay else 'Ambiguous submission; inspect job name before retry.',metadata=receipt,refresh=False)
    print(json.dumps(receipt))
    return int(not okay)


if __name__ == '__main__':
    raise SystemExit(main())
