"""Submit one immutable, bounded four-node collector-recovery validation."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from evidence import Run, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit before submitting a recovery validation.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = 'provenance/perfquery-load-code-v1/'
    k = ['kubectl', '--kubeconfig', str(Path(args.kubeconfig).resolve()), '--request-timeout=45s']
    worker = k + ['-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    phase = run.phase('01-perfquery-load-submission')
    code, out, _ = phase.command(k + ['get', 'nodes', '-o', 'json'])
    expected = {r['kubernetes_node'] for r in json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']}
    ready = {n['metadata']['name'] for n in json.loads(out)['items'] if n['metadata']['name'] in expected
             and n['status'].get('allocatable', {}).get('nvidia.com/gpu') == '8'
             and any(c['type'] == 'Ready' and c['status'] == 'True' for c in n['status']['conditions'])} if not code else set()
    if ready != expected or len(ready) != 4:
        phase.finish('fail', failure_summary='Kubernetes does not reconcile to the frozen 32-GPU inventory.')
        return 1
    code, queue, _ = phase.command(worker + ['squeue', '--noheader', '--format=%i %j %T %D'])
    if code or queue.strip():
        phase.finish('fail', failure_summary='Queue nonempty or unreadable; no workload displaced.')
        return 1
    names = ['evidence.py', 'infra_node.py', 'infra_controller.py', 'fabric_probe.py',
             'telemetry_native.py', 'validate_fabric_under_load.py']
    files = {prefix + name: entry((repo / 'scripts' / name).read_bytes()) for name in names}
    files[prefix + 'source-revision.txt'] = entry((revision + '\n').encode())
    command = ['python3', remote + '/' + prefix + 'validate_fabric_under_load.py', '--run-dir', remote]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n').encode())
    for payload in batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files, limit=128*1024):
        code, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], timeout=45, stdin=payload)
        if code:
            phase.finish('fail', failure_summary='Immutable staging failed; inspect remote files before retrying.')
            return 1
    code, out, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=4',
        '--nodelist=gpu-nodes-[0-3]', '--ntasks-per-node=1', '--cpus-per-task=16', '--gres=gpu:8',
        '--exclusive', '--time=00:06:00', '--no-requeue', '--job-name=ptx-perfquery-' + run.root.name,
        '--chdir=' + remote, '--output=' + remote + '/provenance/perfquery-slurm-%j.out',
        '--error=' + remote + '/provenance/perfquery-slurm-%j.err', remote + '/' + prefix + 'submit.sbatch'])
    job = out.strip().split(';')[0]
    if code or not job.isdigit():
        phase.finish('fail', failure_summary='Ambiguous Slurm submission; inspect unique job name before any retry.')
        return 1
    phase.finish('ok', metadata={'slurm_job_id': job, 'source_git_sha': revision,
                                'scope': 'Submission receipt only; not a passed collector or load gate.'})
    print(json.dumps({'slurm_job_id': job, 'source_git_sha': revision}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
