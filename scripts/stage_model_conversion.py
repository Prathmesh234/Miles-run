"""Submit a source-pinned, telemetry-covered conversion in one whole-node allocation."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from evidence import Run, atomic, sha256
from probe_host_lustre import pod_manifest
from submit_native_preflight import BOOTSTRAP, batches, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit the conversion implementation before submission.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/model-conversion-code-v{args.attempt}/'
    phase = run.phase(f'02-model-conversion-submission-v{args.attempt}')
    k = ['kubectl', '--kubeconfig', str(Path(args.kubeconfig).resolve()), '-n', 'slurm']
    worker = k + ['exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    rc, out, _ = phase.command(worker + ['squeue', '--noheader', '--format=%i %j %T %D'])
    if rc or out.strip():
        phase.finish('fail', failure_summary='Queue nonempty or unreadable; no existing workload displaced.')
        return 1
    rc, out, _ = phase.command(k + ['get', 'node', 'b200-nodepool-ac23753e6cfa', '-o', 'json'])
    status = json.loads(out).get('status', {}) if not rc else {}
    if status.get('allocatable', {}).get('nvidia.com/gpu') != '8' or not any(
        c['type'] == 'Ready' and c['status'] == 'True' for c in status.get('conditions', [])):
        phase.finish('fail', failure_summary='Kubernetes does not report eight ready GPUs.')
        return 1
    names = ['evidence.py', 'infra_node.py', 'enroot_run_config.py', 'fabric_probe.py', 'telemetry_native.py',
             'telemetry_lustre_host.py', 'model_conversion.py', 'run_model_conversion.py', 'rendezvous_probe.py']
    files = {prefix + name: entry((repo / 'scripts' / name).read_bytes()) for name in names}
    files[prefix + 'source-revision.txt'] = entry((revision + '\n').encode())
    command = ['python3', remote + '/' + prefix + 'run_model_conversion.py', '--run-dir', remote,
               '--attempt', str(args.attempt)]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n').encode())
    for payload in batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files, limit=64*1024):
        rc, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], timeout=45, stdin=payload)
        if rc:
            phase.finish('fail', failure_summary='Immutable code staging failed. Inspect partial artifacts before retrying.')
            return 1
    name = 'ptx-convert-lustre-' + run.root.name + '-v' + str(args.attempt)
    pod = pod_manifest(name, 'b200-nodepool-ac23753e6cfa', run.root.name)
    pod['metadata']['labels']['component'] = 'model-conversion-lustre'
    pod['spec']['activeDeadlineSeconds'] = 2800
    container = pod['spec']['containers'][0]
    container['command'] = ['python3', '/run-artifacts/' + prefix + 'telemetry_lustre_host.py', '--run-dir', '/run-artifacts',
        '--hostname', 'gpu-nodes-0', '--duration-s', '2750', '--stream-label', f'lustre-model-conversion-v{args.attempt}',
        '--job-marker', f'control/model-conversion-job-v{args.attempt}.json',
        '--stop-marker', f'control/model-conversion-lustre-v{args.attempt}.stop', '--role', 'checkpoint-conversion']
    container['volumeMounts'] = [{'name': 'host-lustre', 'mountPath': '/host-lustre', 'readOnly': True},
        {'name': 'run-artifacts', 'mountPath': '/run-artifacts', 'subPath': 'posttrainingx/runs/vultr-b200-slurm/' + run.root.name}]
    pod['spec']['volumes'] = [{'name': 'host-lustre', 'hostPath': {'path': '/sys/kernel/debug/lustre/llite', 'type': 'Directory'}},
                            {'name': 'run-artifacts', 'persistentVolumeClaim': {'claimName': 'slurm-shared'}}]
    atomic(phase.path / 'lustre-pod.json', pod)
    rc, _, _ = phase.command(k + ['create', '-f', '-'], stdin=json.dumps(pod), timeout=45)
    if not rc:
        rc, _, _ = phase.command(k + ['wait', '--for=condition=Ready', 'pod/' + name, '--timeout=60s'], timeout=70)
    if rc:
        phase.finish('fail', failure_summary='Host collector did not become ready; no GPU job submitted.')
        return 1
    rc, out, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=1',
        '--nodelist=gpu-nodes-0', '--ntasks-per-node=1', '--cpus-per-task=32', '--gres=gpu:8', '--exclusive',
        '--time=00:45:00', '--no-requeue', '--job-name=ptx-model-convert-' + run.root.name + '-v' + str(args.attempt),
        '--chdir=' + remote, '--output=' + remote + f'/provenance/model-conversion-v{args.attempt}-slurm-%j.out',
        '--error=' + remote + f'/provenance/model-conversion-v{args.attempt}-slurm-%j.err', remote + '/' + prefix + 'submit.sbatch'])
    job = out.strip().split(';')[0]
    okay = not rc and job.isdigit()
    phase.finish('ok' if okay else 'fail', failure_summary=None if okay else 'Submission ambiguous; inspect unique job name before retry.',
        metadata={'slurm_job_id': job, 'source_git_sha': revision, 'collector_pod': name,
                  'scope': 'Submission receipt only; not successful conversion or training.'})
    print(json.dumps({'slurm_job_id': job, 'collector_pod': name, 'source_git_sha': revision}), flush=True)
    return int(not okay)


if __name__ == '__main__':
    raise SystemExit(main())
