"""Submit a short four-node, whole-node collector qualification without model changes."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from evidence import Run, atomic, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry
from telemetry_nvml import BINDING_SHA256
from validate_nvml_under_load import TEARDOWN_PROBE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--load-profile', choices=['all-reduce', 'context-teardown'], default='all-reduce')
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit before submitting immutable qualification code.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    run = Run(a.run_dir)
    label = f'nvml-qualification-v{a.attempt}'
    phase = run.phase('01-' + label + '-submission')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = 'provenance/' + label + '-code/'
    worker = ['kubectl', '--kubeconfig', a.kubeconfig, '-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', a.kubeconfig, 'get', 'nodes', '-o', 'json'], timeout=30)
    expected = {row['kubernetes_node'] for row in json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']}
    ready = {row['metadata']['name'] for row in json.loads(out)['items'] if
        row['status'].get('allocatable', {}).get('nvidia.com/gpu') == '8' and
        any(c['type'] == 'Ready' and c['status'] == 'True' for c in row['status'].get('conditions', []))} if not rc else set()
    if len(expected) != 4 or not expected <= ready:
        phase.finish('fail', failure_summary='Kubernetes does not reconcile to the frozen 32 GPUs.', refresh=False)
        return 1
    rc, out, _ = phase.command(worker + ['squeue', '--noheader', '--format=%i %T %j'], timeout=30)
    if rc or out.strip():
        phase.finish('fail', failure_summary='Queue not empty or unreadable; no workload displaced.', refresh=False)
        return 1
    binding = run.root / 'tests/01-pinned-nvml-bindings-v1/pynvml.py'
    if sha256(binding) != BINDING_SHA256:
        raise ValueError('NVML binding differs from the pinned image.')
    names = ['evidence.py', 'infra_node.py', 'infra_controller.py', 'fabric_probe.py',
             'telemetry_native.py', 'telemetry_nvml.py', 'telemetry_health.py', 'validate_nvml_under_load.py',
             'enroot_run_config.py']
    files = {prefix + name: entry((repo / 'scripts' / name).read_bytes()) for name in names}
    files[prefix + 'pynvml.py'] = entry(binding.read_bytes())
    files[prefix + 'source-revision.txt'] = entry((revision + '\n').encode())
    files[prefix + 'teardown_probe.py'] = entry(TEARDOWN_PROBE.encode())
    command = ['srun', '--kill-on-bad-exit=0', '--nodes=4', '--ntasks=4', '--ntasks-per-node=1',
        '--gpus-per-node=8', 'python3', remote + '/' + prefix + 'validate_nvml_under_load.py',
        '--run-dir', remote, '--attempt', str(a.attempt), '--load-profile', a.load_profile]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n').encode())
    for payload in batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files, limit=128*1024):
        rc, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if rc:
            phase.finish('fail', failure_summary='Code staging failed; no job submitted.', refresh=False)
            return 1
    rc, out, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=4',
        '--nodelist=gpu-nodes-[0-3]', '--ntasks-per-node=1', '--cpus-per-task=8', '--gres=gpu:8', '--exclusive',
        '--time=00:05:00', '--no-requeue', '--job-name=ptx-' + label, '--chdir=' + remote,
        '--output=' + remote + '/provenance/' + label + '-%j.out',
        '--error=' + remote + '/provenance/' + label + '-%j.err', remote + '/' + prefix + 'submit.sbatch'], timeout=45)
    job = out.strip().split(';')[0]
    okay = not rc and job.isdigit()
    receipt = dict(slurm_job_id=job, source_sha=revision, gpus=32, load_profile=a.load_profile,
        scope='Collector qualification submission only; zero optimizer steps.')
    atomic(phase.path / 'submission.json', receipt)
    phase.finish('ok' if okay else 'fail', metadata=receipt, refresh=False,
        failure_summary=None if okay else 'Ambiguous submission; inspect the queue before retry.')
    print(json.dumps(receipt), flush=True)
    return int(not okay)


if __name__ == '__main__':
    raise SystemExit(main())
