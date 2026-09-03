"""Submit a bounded 32-GPU checkpoint replay, with no new checkpoint payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import traceback

from evidence import Run, atomic, sha256
from probe_host_lustre import pod_manifest
from stage_resume_cpu_test import MILES_SHA, MILES_SOURCE
from submit_native_preflight import BOOTSTRAP, batches, entry
from telemetry_nvml import BINDING_SHA256
from resume_replay_controls import DETERMINISTIC_ENV


FREEZE = '''import hashlib,json,pathlib,shutil,sys
r=pathlib.Path(sys.argv[1]);a=int(sys.argv[2]);small={};payloads={}
gate=r/f'tests/02-resume-native-cpu-test-v{a}/result.json'
g=json.loads(gate.read_text())
if g['findings'] or g['fixture']['status']!='ok' or g['fixture']['cuda_device_count']!=0:raise ValueError('CPU gate failed')
for step in (0,1):
 d=r/'training/sync-grpo-v14/checkpoints'/f'iter_{step:07d}'
 files=sorted(d.glob('*.distcp'))
 if len(files)!=16:raise ValueError('Expected 16 native checkpoint shards')
 for p in files:
  if p.is_symlink():raise ValueError('Checkpoint symlink refused')
  s=p.stat();payloads[str(p.relative_to(r))]=dict(bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns)
 for name in ('.metadata','metadata.json'):
  p=d/name;small[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
for rank in range(16):
 p=r/'training/sync-grpo-v14/dump_details/train_data'/f'1_{rank}.pt'
 if p.is_symlink() or not 0<p.stat().st_size<16*1024**2:raise ValueError('Invalid saved rank input')
 small[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
if shutil.disk_usage(r).free<256*1024**3:raise ValueError('Free-space guard failed')
source=r/'provenance/sync-grpo-source-v14'
m=json.loads((source/'manifest.json').read_text())
for row in m['files']:
 p=source/row['path']
 if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Miles source differs')
print(json.dumps(dict(small_inputs=small,payload_stat=payloads,cpu_gate=g,cpu_gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
 miles_source_sha=m['source_git_sha'],source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
 free_bytes=shutil.disk_usage(r).free,payload_hash_scope='Payload file identity/size frozen; per-tensor hashes are produced by the strict native replay, not whole-file input hashes.')))
'''


def require_same_inputs(current, original):
    """A diagnostic retry cannot silently adopt changed checkpoint evidence."""
    for key in ('small_inputs', 'payload_stat', 'miles_source_sha', 'source_manifest_sha256'):
        if current[key] != original[key]:
            raise ValueError('Retry inputs differ from the first frozen replay: ' + key)


def stage(args):
    repo = Path(__file__).resolve().parents[1]
    for path in (repo, repo / 'vendor/miles'):
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Commit both source trees before submission.')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo / 'vendor/miles'), 'rev-parse', 'HEAD'], text=True).strip() != MILES_SHA:
        raise ValueError('Unexpected Miles source revision.')
    run = Run(args.run_dir)
    phase = run.phase(f'02-resume-replay-submission-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/resume-replay-code-v{args.attempt}/'
    label = f'resume-replay-v{args.attempt}'
    kube = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm']
    worker = kube + ['exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    pods = []
    submitted = False
    try:
        rc, out, _ = phase.command(worker + ['squeue', '--noheader', '--format=%i %j %T %D'], timeout=30)
        if rc or out.strip():
            raise ValueError('Queue not empty or unreadable; no job displaced.')
        rc, out, _ = phase.command(worker + ['python3', '-c', FREEZE, remote, str(args.cpu_test_attempt)], timeout=90)
        if rc:
            raise ValueError('Read-only input or CPU qualification gate failed.')
        freeze = json.loads(out)
        if freeze['miles_source_sha'] != MILES_SHA or any(
            freeze['cpu_gate']['files'].get(name) != sha256(repo / 'scripts' / name)
            for name in ('resume_checkpoint_probe.py', 'resume_replay_controls.py')):
            raise ValueError('Native CPU gate did not test this exact probe source.')
        if args.attempt > 1:
            original = json.loads((run.root / 'tests/02-resume-replay-submission-v1/frozen-inputs.json').read_text())
            require_same_inputs(freeze, original)
        atomic(phase.path / 'frozen-inputs.json', freeze)
        inventory = json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']
        hosts = []
        for index in range(4):
            hostname = f'gpu-nodes-{index}'
            physical = [g for g in inventory if g['hostname'] == hostname]
            if len(physical) != 8:
                raise ValueError('Frozen physical inventory is not eight GPUs per host.')
            node = physical[0]['kubernetes_node']
            rc, out, _ = phase.command(kube + ['get', 'node', node, '-o', 'json'], timeout=30)
            status = json.loads(out)['status'] if not rc else {}
            if status.get('allocatable', {}).get('nvidia.com/gpu') != '8' or not any(
                c['type'] == 'Ready' and c['status'] == 'True' for c in status.get('conditions', [])):
                raise ValueError('Kubernetes GPU inventory failed reconciliation.')
            rc, out, _ = phase.command(kube + ['exec', 'slurm-worker-' + hostname, '--', 'ip', '-j', '-4', 'addr', 'show', 'dev', 'eth0'], timeout=30)
            if rc:
                raise ValueError('Worker network identity unavailable.')
            hosts.append(dict(hostname=hostname, kubernetes_node=node, ip=json.loads(out)[0]['addr_info'][0]['local'],
                              gpu_uuids=[g['uuid'] for g in physical], replica='a' if index < 2 else 'b', node_rank=index % 2))
        for index, host in enumerate(hosts):
            host['master_ip'] = hosts[index // 2 * 2]['ip']
        names = ['resume_checkpoint_probe.py', 'resume_replay_controls.py', 'resume_probe_node.py', 'evidence.py', 'infra_node.py',
            'enroot_run_config.py', 'fabric_probe.py', 'container_fabric_probe.py', 'telemetry_native.py',
            'telemetry_health.py', 'telemetry_nvml.py', 'telemetry_lustre_host.py']
        content = {name: (repo / 'scripts' / name).read_bytes() for name in names}
        binding = run.root / 'tests/01-pinned-nvml-bindings-v1/pynvml.py'
        if sha256(binding) != BINDING_SHA256:
            raise ValueError('Pinned NVML binding differs.')
        content['pynvml.py'] = binding.read_bytes()
        content['load-root/latest_checkpointed_iteration.txt'] = b'0\n'
        content['load-root/iter_0000000/.mount-target'] = b'Replaced by read-only bind mount of job167 checkpoint zero.\n'
        config = dict(schema_version=1, source_git_sha=revision, miles_sha=MILES_SHA, miles_source=MILES_SOURCE,
            execution_profile=args.execution_profile,
            deterministic_environment=DETERMINISTIC_ENV if args.execution_profile == 'deterministic' else {},
            hosts=hosts, hf_model='models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0',
            checkpoint_root='training/sync-grpo-v14/checkpoints', dump_root='training/sync-grpo-v14/dump_details',
            small_inputs=freeze['small_inputs'], payload_stat=freeze['payload_stat'],
            cpu_gate_sha256=freeze['cpu_gate_sha256'], image_digest='sha256:59a11219eae0defc6594ec678fafe4e897c16904263223f79968cd3e0209a502',
            code_files={name: hashlib.sha256(data).hexdigest() for name, data in content.items()},
            telemetry_health_contract='Fail on errors, identity mismatch, or host-local heartbeat older than 12 seconds.',
            scope='Two independent 16-rank replay replicas; all32 reload gate before one update each. Explicit execution profile; original checkpoint equality remains mandatory. No serving, fresh trajectory or async-resume claim.')
        atomic(phase.path / 'launch.json', config)
        content['launch.json'] = (json.dumps(config, sort_keys=True) + '\n').encode()
        command = ['srun', '--kill-on-bad-exit=0', '--nodes=4', '--ntasks=4', '--ntasks-per-node=1',
            '--cpus-per-task=32', '--gpus-per-node=8', 'python3', remote + '/' + prefix + 'resume_probe_node.py',
            '--run-dir', remote, '--attempt', str(args.attempt)]
        content['submit.sbatch'] = ('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n').encode()
        files = {prefix + name: entry(data) for name, data in content.items()}
        for payload in batches(dict(root=remote, create=False, manifest_sha256=sha256(run.root / 'run.json')), files):
            rc, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
            if rc:
                raise ValueError('Immutable replay staging failed; no job submitted.')
        for index, row in enumerate(hosts):
            name = 'ptx-resume-lustre-' + run.root.name + '-v' + str(args.attempt) + '-' + str(index)
            pod = pod_manifest(name, row['kubernetes_node'], run.root.name)
            pod['spec']['activeDeadlineSeconds'] = 1900
            container = pod['spec']['containers'][0]
            container['command'] = ['python3', '/run-artifacts/' + prefix + 'telemetry_lustre_host.py', '--run-dir', '/run-artifacts',
                '--hostname', row['hostname'], '--duration-s', '1860', '--stream-label', 'lustre-' + label,
                '--job-marker', 'control/' + label + '-job.json',
                '--stop-marker', 'control/' + label + '-' + row['hostname'] + '-lustre.stop', '--role', 'trainer']
            container['volumeMounts'] = [{'name': 'host-lustre', 'mountPath': '/host-lustre', 'readOnly': True},
                {'name': 'run-artifacts', 'mountPath': '/run-artifacts', 'subPath': 'posttrainingx/runs/vultr-b200-slurm/' + run.root.name}]
            pod['spec']['volumes'] = [{'name': 'host-lustre', 'hostPath': {'path': '/sys/kernel/debug/lustre/llite', 'type': 'Directory'}},
                {'name': 'run-artifacts', 'persistentVolumeClaim': {'claimName': 'slurm-shared'}}]
            atomic(phase.path / ('lustre-pod-' + str(index) + '.json'), pod)
            # Exact unique names only; no cleanup of shared or preexisting pods.
            rc, _, _ = phase.command(kube + ['create', '-f', '-'], stdin=json.dumps(pod), timeout=45)
            if rc:
                raise ValueError('Lustre pod creation failed; inspect this exact name before retrying.')
            pods.append(name)
            rc, _, _ = phase.command(kube + ['wait', '--for=condition=Ready', 'pod/' + name, '--timeout=45s'], timeout=55)
            if rc:
                raise ValueError('Lustre collector is not ready; no replay submitted.')
        rc, out, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=4',
            '--nodelist=gpu-nodes-[0-3]', '--ntasks-per-node=1', '--cpus-per-task=32', '--gres=gpu:8', '--exclusive',
            '--time=00:30:00', '--no-requeue', '--job-name=ptx-resume-' + (
                'deterministic' if args.execution_profile == 'deterministic' else 'replay') + '-v' + str(args.attempt),
            '--chdir=' + remote, '--output=' + remote + '/' + prefix + 'slurm-%j.out',
            '--error=' + remote + '/' + prefix + 'slurm-%j.err', remote + '/' + prefix + 'submit.sbatch'], timeout=45)
        # Do not clean live collectors after an ambiguous submission.
        submitted = True
        job = out.strip().split(';')[0]
        if rc or not job.isdigit():
            raise ValueError('Ambiguous Slurm submission; inspect the job name before any retry.')
        receipt = dict(slurm_job_id=job, source_git_sha=revision, miles_sha=MILES_SHA, gpus=32, replicas=2,
                       execution_profile=args.execution_profile,
                       optimizer_steps_per_replica=1, telemetry_stream=label, payload_writes=False)
        atomic(phase.path / 'submission.json', receipt)
        phase.finish('ok', metadata=receipt, refresh=False)
        print(json.dumps(receipt), flush=True)
        return 0
    except Exception as exc:
        atomic(phase.path / 'exception.txt', traceback.format_exc())
        if not submitted:
            for pod in pods:
                phase.command(kube + ['delete', 'pod', pod, '--wait=false'], timeout=30)
        phase.finish('fail', failure_summary=str(exc), refresh=False)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--cpu-test-attempt', type=int, required=True)
    parser.add_argument('--execution-profile', choices=('original', 'deterministic'), default='original')
    raise SystemExit(stage(parser.parse_args()))
