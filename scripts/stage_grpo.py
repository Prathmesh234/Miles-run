"""Stage pinned sources and submit real four-node synchronous GRPO validation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from evidence import Run, atomic, sha256
from probe_host_lustre import pod_manifest
from submit_native_preflight import BOOTSTRAP, batches, entry

PARENT_MILES = '346946ae870be97e9cb6f4e8b7214c7fcf66c041'
BASE_MILES = '0709889b2848f293b5575d50aa3340fa4de5a20d'

MATERIALIZE = r'''
import hashlib,json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); code=root/sys.argv[2]
config=json.loads((code/'launch.json').read_text())
old=root/'provenance/training-source-v2'; manifest=json.loads((old/'manifest.json').read_text())
if manifest['source_git_sha']!=config['miles_parent']: raise ValueError('Unexpected parent source revision.')
for row in manifest['files']:
 path=old/row['path']
 if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
  raise ValueError('Parent source hash changed: '+row['path'])
target=root/config['miles_source']
target.parent.mkdir(parents=True,exist_ok=False)
shutil.copytree(old/'miles',target)
delta=json.loads((code/'miles-delta.json').read_text())
for name,expected in delta.items():
 source=code/'miles-delta'/name
 if pathlib.Path(name).is_absolute() or '..' in pathlib.Path(name).parts: raise ValueError('Unsafe source path.')
 if hashlib.sha256(source.read_bytes()).hexdigest()!=expected: raise ValueError('Delta checksum changed.')
 dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
rows=[]
for path in sorted(target.rglob('*')):
 if path.is_symlink(): raise ValueError('Source symlink refused.')
 if path.is_file():rows.append({'path':str(path.relative_to(target.parent)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
result={'source_git_sha':config['miles_sha'],'parent_git_sha':config['miles_parent'],'files':rows,'delta':delta}
with (target.parent/'manifest.json').open('x') as output:json.dump(result,output,sort_keys=True,indent=2)
print(json.dumps({'miles_sha':config['miles_sha'],'files':len(rows),'delta_files':len(delta)}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    ap.add_argument('--environment-attempt', type=int, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    miles = repo / 'vendor/miles'
    for repository in (repo, miles):
        if subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Commit implementation before submission: ' + str(repository))
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    miles_sha = subprocess.check_output(['git', '-C', str(miles), 'rev-parse', 'HEAD'], text=True).strip()
    run = Run(a.run_dir)
    phase = run.phase(f'02-sync-grpo-submission-v{a.attempt}')
    tito_gate = run.root / 'tests/02-tito-candidate-validation-v1/result.json'
    tito_proof = json.loads(tito_gate.read_text())
    if (tito_proof['findings'] or tito_proof.get('samples_unchanged') != 32
            or len(tito_proof.get('negative_controls', [])) != 7
            or not all(case['rejected'] for case in tito_proof['negative_controls'])
            or sha256(miles / 'examples/experimental/openenv/posttrainingx_local_agent.py') != tito_proof['candidate_sha256']):
        phase.finish('fail', failure_summary='Exact candidate tokenization replay gate has not passed; no allocation submitted.', refresh=False)
        return 1
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    prefix = f'provenance/sync-grpo-code-v{a.attempt}/'
    label = f'sync-grpo-v{a.attempt}'
    k = ['kubectl', '--kubeconfig', a.kubeconfig, '-n', 'slurm']
    worker = k + ['exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    gate = f'tests/02-local-file-runtime-validation-v{a.environment_attempt}/result.json'
    rc, out, _ = phase.command(worker + ['cat', remote + '/' + gate], timeout=30)
    if rc or json.loads(out)['findings'] or len(json.loads(out)['cases']) < 10:
        phase.finish('fail', failure_summary='Live local environment gate has not passed; no training allocation submitted.')
        return 1
    rc, out, _ = phase.command(worker + ['squeue', '--noheader', '--format=%i %j %T %D'], timeout=30)
    if rc or out.strip():
        phase.finish('fail', failure_summary='Queue not empty or unreadable; no workload displaced.')
        return 1
    inventory = json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']
    host_map = {'schema_version': 1, 'nodes': []}
    for index in range(4):
        host = f'gpu-nodes-{index}'
        physical = [g for g in inventory if g['hostname'] == host]
        rc, out, _ = phase.command(k + ['get', 'node', physical[0]['kubernetes_node'], '-o', 'json'], timeout=30)
        status = json.loads(out)['status'] if not rc else {}
        if status.get('allocatable', {}).get('nvidia.com/gpu') != '8' or not any(c['type']=='Ready' and c['status']=='True' for c in status.get('conditions', [])):
            phase.finish('fail', failure_summary='Kubernetes GPU inventory no longer reconciles.')
            return 1
        rc, out, _ = phase.command(k + ['exec', 'slurm-worker-' + host, '--', 'ip', '-j', '-4', 'addr', 'show', 'dev', 'eth0'], timeout=30)
        if rc:
            phase.finish('fail', failure_summary='Worker network identity unavailable.')
            return 1
        ip = json.loads(out)[0]['addr_info'][0]['local']
        host_map['nodes'].append({'hostname': host, 'ray_node_ip': ip, 'kubernetes_node': physical[0]['kubernetes_node'],
            'role': 'trainer' if index < 2 else 'rollout', 'gpu_uuids': [g['uuid'] for g in physical]})
    config = {'schema_version': 1, 'root_sha': revision, 'miles_sha': miles_sha, 'miles_parent': PARENT_MILES,
        'miles_source': f'provenance/sync-grpo-source-v{a.attempt}/miles', 'host_map': host_map,
        'environment_gate': gate, 'images_manifest': 'environments/local-file-runtime-v3/images.json',
        'controller_image': 'sha256:1a93e04935e2b6a3948bea8cef77ebe99c07939eb685bd7fcf69226ad18b14a8',
        'hf_model': 'models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0',
        'converted_model': 'models/qwen3.6-35b-a3b-torch-dist-v2', 'network_interface': 'eth0',
        'ray_port': 19379, 'dashboard_port': 18265, 'env_port': 18243,
        'task_ids': ['task_06652', 'task_14118', 'task_10753', 'task_09467'],
        'layout': '2t2r', 'optimizer_steps_requested': 3, 'group_size': 8, 'global_batch_size': 16,
        'sglang_moe_runner_backend': 'triton', 'verify_initial_weight_broadcast': True,
        'tito_comparison_contract': 'qwen36_length_limited_final_message_v1',
        'tito_candidate_validation_sha256': sha256(tito_gate),
        'backend_change_reason': 'Job 137 rejected BF16 broadcast into auto-selected FlashInfer packed expert weights; pinned-loader CPU reproduction retained.',
        'scope': 'Initial synchronous GRPO validation, not the final 400-step hill climb or placement benchmark.'}
    atomic(phase.path / 'launch.json', config)
    names = ['evidence.py', 'infra_node.py', 'enroot_run_config.py', 'fabric_probe.py', 'telemetry_native.py',
             'telemetry_lustre_host.py', 'grpo_node.py', 'grpo_container.py', 'local_file_env.py', 'local_openenv_app.py']
    files = {prefix + name: entry((repo / 'scripts' / name).read_bytes()) for name in names}
    files[prefix + 'launch.json'] = entry((json.dumps(config, indent=2) + '\n').encode())
    files[prefix + 'tito-validation.json'] = entry(tito_gate.read_bytes())
    files[prefix + 'host-map.json'] = entry((json.dumps(host_map, indent=2) + '\n').encode())
    system = ('You are an autonomous terminal agent solving a Linux task. You will be given the task instruction, '
              'then interact with a real Linux shell. On each turn respond with EXACTLY ONE shell command inside '
              'a single ```bash code block and nothing else. Inspect the environment, make the required changes, '
              'and verify your work. When the task is complete, reply TASK_COMPLETE with no code block.')
    prompts = ''.join(json.dumps({'prompt': [{'role': 'system', 'content': system}], 'metadata': {'task_id': task,
        'dataset': 'Terminal-Lego', 'revision': '9c197f1c2e87b64cc316b1a5bfcef57b584929f0'}}) + '\n' for task in config['task_ids'])
    files[prefix + 'train.prompts.jsonl'] = entry(prompts.encode())
    delta = {}
    changed = subprocess.check_output(['git', '-C', str(miles), 'diff', '--name-only', PARENT_MILES, miles_sha], text=True).splitlines()
    for name in changed:
        content = subprocess.check_output(['git', '-C', str(miles), 'show', miles_sha + ':' + name])
        delta[name] = hashlib.sha256(content).hexdigest()
        files[prefix + 'miles-delta/' + name] = entry(content)
    files[prefix + 'miles-delta.json'] = entry(json.dumps(delta, indent=2).encode())
    patch = subprocess.check_output(['git', '-C', str(miles), 'format-patch', '--stdout', BASE_MILES + '..' + miles_sha])
    files[prefix + 'miles.patch'] = entry(patch)
    # Node-local failure markers stop peer children without interrupting their
    # bounded final inventory and structured failure-report writes.
    launch = ['srun', '--kill-on-bad-exit=0', '--nodes=4', '--ntasks=4', '--ntasks-per-node=1', '--cpus-per-task=32',
              '--gpus-per-node=8', 'python3', remote + '/' + prefix + 'grpo_node.py', '--run-dir', remote, '--attempt', str(a.attempt)]
    files[prefix + 'submit.sbatch'] = entry(('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(launch) + '\n').encode())
    payloads = list(batches({'root': remote, 'create': False, 'manifest_sha256': sha256(run.root / 'run.json')}, files, limit=64*1024))
    for payload in payloads:
        rc, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if rc:
            phase.finish('fail', failure_summary='Pinned GRPO source staging failed; no workload submitted.')
            return 1
    rc, _, _ = phase.command(worker + ['python3', '-c', MATERIALIZE, remote, prefix], timeout=90)
    if rc:
        phase.finish('fail', failure_summary='Pinned Miles materialization failed; no workload submitted.')
        return 1
    for index, row in enumerate(host_map['nodes']):
        name = 'ptx-grpo-lustre-' + run.root.name + '-v' + str(a.attempt) + '-' + str(index)
        pod = pod_manifest(name, row['kubernetes_node'], run.root.name)
        pod['spec']['activeDeadlineSeconds'] = 5500
        container = pod['spec']['containers'][0]
        container['command'] = ['python3', '/run-artifacts/' + prefix + 'telemetry_lustre_host.py', '--run-dir', '/run-artifacts',
            '--hostname', row['hostname'], '--duration-s', '5460', '--stream-label', 'lustre-' + label,
            '--job-marker', 'control/' + label + '-job.json', '--stop-marker', 'control/' + label + '-lustre.stop', '--role', row['role']]
        container['volumeMounts'] = [{'name': 'host-lustre', 'mountPath': '/host-lustre', 'readOnly': True},
            {'name': 'run-artifacts', 'mountPath': '/run-artifacts', 'subPath': 'posttrainingx/runs/vultr-b200-slurm/' + run.root.name}]
        pod['spec']['volumes'] = [{'name': 'host-lustre', 'hostPath': {'path': '/sys/kernel/debug/lustre/llite', 'type': 'Directory'}},
            {'name': 'run-artifacts', 'persistentVolumeClaim': {'claimName': 'slurm-shared'}}]
        atomic(phase.path / ('lustre-pod-' + str(index) + '.json'), pod)
        rc, _, _ = phase.command(k + ['create', '-f', '-'], stdin=json.dumps(pod), timeout=45)
        if not rc:
            rc, _, _ = phase.command(k + ['wait', '--for=condition=Ready', 'pod/' + name, '--timeout=45s'], timeout=55)
        if rc:
            phase.finish('fail', failure_summary='Host Lustre telemetry pod not ready; no training submitted.')
            return 1
    rc, out, _ = phase.command(worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=4',
        '--nodelist=gpu-nodes-[0-3]', '--ntasks-per-node=1', '--cpus-per-task=32', '--gres=gpu:8', '--exclusive',
        '--time=01:30:00', '--no-requeue', '--job-name=ptx-sync-grpo-' + run.root.name + '-v' + str(a.attempt),
        '--chdir=' + remote, '--output=' + remote + f'/provenance/sync-grpo-v{a.attempt}-%j.out',
        '--error=' + remote + f'/provenance/sync-grpo-v{a.attempt}-%j.err', remote + '/' + prefix + 'submit.sbatch'], timeout=45)
    job = out.strip().split(';')[0]
    okay = not rc and job.isdigit()
    receipt = {'slurm_job_id': job, 'root_sha': revision, 'miles_sha': miles_sha, 'layout': '2t2r',
               'gpus': 32, 'optimizer_steps_requested': 3, 'scope': 'Actual GRPO job submitted; optimizer execution not yet verified.'}
    atomic(phase.path / 'submission.json', receipt)
    phase.finish('ok' if okay else 'fail', failure_summary=None if okay else 'Ambiguous submission; inspect job name before retry.', metadata=receipt)
    print(json.dumps(receipt), flush=True)
    return int(not okay)


if __name__ == '__main__':
    raise SystemExit(main())
