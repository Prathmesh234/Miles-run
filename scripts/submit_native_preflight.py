"""Stage immutable code and submit one bounded four-node Slurm preflight."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from evidence import Run, atomic


BOOTSTRAP = r'''
import base64,hashlib,json,os,re,shutil,subprocess,sys,tempfile
from pathlib import Path
p=json.load(sys.stdin)
root=Path(p['root'])
if not re.fullmatch(r'/shared/posttrainingx/runs/vultr-b200-slurm/[0-9]{8}-[0-9]{6}-[a-z0-9]+',str(root)):
 raise ValueError('Invalid remote run path.')
if subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T','/shared'],text=True).strip()!='lustre':
 raise ValueError('/shared is not Lustre.')
if shutil.disk_usage('/shared').free < 10*1024**3:
 raise ValueError('Less than 10 GiB free.')
for parent in [root]+list(root.parents):
 if parent.is_symlink(): raise ValueError('Symlink in run path.')
if p['create']:
 root.mkdir(parents=True,exist_ok=False)
elif not (root/'run.json').is_file():
 raise ValueError('Existing run manifest required.')
for rel,entry in p['files'].items():
 path=root/rel
 if Path(rel).is_absolute() or '..' in Path(rel).parts:
  raise ValueError('Unsafe artifact path.')
 if not p['create'] and not rel.startswith('tests/01-native-submission/'):
  raise ValueError('Follow-up upload is restricted to submission evidence.')
 for parent in [path]+list(path.parents):
  if parent.is_symlink(): raise ValueError('Symlink in artifact path.')
 data=base64.b64decode(entry['data'],validate=True)
 if hashlib.sha256(data).hexdigest()!=entry['sha256']:
  raise ValueError('Payload checksum mismatch.')
 if path.exists(): raise ValueError('Refusing to replace an existing staged file.')
 path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
 with os.fdopen(fd,'wb') as f:
  f.write(data); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
for name in ['inventory','tests','telemetry','rl','reports','provenance','control']:
 (root/name).mkdir(exist_ok=True)
print(json.dumps({'root':str(root),'files':len(p['files']),'free_bytes':shutil.disk_usage(root).free}))
'''


def entry(data):
    return {'data': base64.b64encode(data).decode(), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    workspace = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(['git', '-C', str(workspace), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(workspace), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise ValueError('Commit code before cluster submission.')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    manifest = json.loads((run.root / 'run.json').read_text())
    manifest['metadata']['native_preflight'] = {'source_git_sha': revision, 'remote_run_dir': remote,
        'partition': 'gpu-nodes', 'nodes': 4, 'gpus_per_node': 8, 'walltime_minutes': 15,
        'scope': 'Bounded native inventory, all-reduce, telemetry, and storage smoke.'}
    atomic(run.root / 'run.json', manifest)
    phase = run.phase('01-native-submission')
    k = ['kubectl', '--kubeconfig', str(Path(args.kubeconfig).resolve()), '--request-timeout=45s']
    worker = k + ['-n', 'slurm', 'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    code, out, _ = phase.command(k + ['get', 'nodes', '-o', 'json'])
    nodes = json.loads(out)['items'] if not code else []
    expected = {g['kubernetes_node'] for g in json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']}
    ready = {n['metadata']['name'] for n in nodes if n['metadata']['name'] in expected and
             int(n['status'].get('allocatable', {}).get('nvidia.com/gpu', 0)) == 8 and
             any(c['type'] == 'Ready' and c['status'] == 'True' for c in n['status']['conditions'])}
    if ready != expected or len(ready) != 4:
        phase.finish('fail', failure_summary='Current Kubernetes inventory does not reconcile to the frozen four nodes.')
        return 1
    code, out, _ = phase.command(worker + ['squeue', '--noheader', '--format=%i,%j,%u,%T,%D,%N'])
    if code or out.strip():
        phase.finish('fail', failure_summary='Queue is nonempty or unreadable. No shared workload was displaced.')
        return 1
    files = {}
    for path in sorted(run.root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Evidence symlinks are not permitted.')
        if path.is_file() and phase.path not in path.parents:
            files[str(path.relative_to(run.root))] = entry(path.read_bytes())
    for path in sorted((workspace / 'scripts').glob('*.py')):
        files['provenance/native-code/' + path.name] = entry(path.read_bytes())
    shell = '#!/bin/bash\nset -euo pipefail\nexec python3 ' + shlex.quote(remote + '/provenance/native-code/infra_controller.py') + ' --run-dir ' + shlex.quote(remote) + '\n'
    files['provenance/native-preflight.sbatch'] = entry(shell.encode())
    files['provenance/native-source-revision.txt'] = entry((revision + '\n').encode())
    atomic(run.root / 'provenance/native-bootstrap.py', BOOTSTRAP)
    code, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], timeout=120,
                               stdin=json.dumps({'root': remote, 'create': True, 'files': files}))
    if code:
        phase.finish('fail', failure_summary='Run staging failed. Inspect the remote directory before any retry.')
        return 1
    argv = worker + ['sbatch', '--parsable', '--partition=gpu-nodes', '--nodes=4',
           '--nodelist=gpu-nodes-[0-3]', '--ntasks-per-node=1', '--cpus-per-task=16',
           '--gres=gpu:8', '--exclusive', '--time=00:15:00', '--no-requeue',
           '--job-name=ptx-native-' + run.root.name, '--chdir=' + remote,
           '--output=' + remote + '/provenance/slurm-%j.out',
           '--error=' + remote + '/provenance/slurm-%j.err', remote + '/provenance/native-preflight.sbatch']
    code, out, _ = phase.command(argv, timeout=45)
    job_id = out.strip().split(';')[0]
    if code or not job_id.isdigit():
        phase.finish('fail', failure_summary='Submission response is failed or ambiguous. Inspect squeue and sacct by unique job name; do not resubmit automatically.')
        return 1
    phase.finish('ok', metadata={'slurm_job_id': job_id, 'remote_run_dir': remote, 'source_git_sha': revision,
                                'scope': 'Submission only. Execution and measurement gates are not yet proven.'})
    # Copy the now-complete submission phase without changing any staged artifact.
    complete = {str(p.relative_to(run.root)): entry(p.read_bytes()) for p in phase.path.rglob('*') if p.is_file()}
    result = subprocess.run(worker + ['python3', '-c', BOOTSTRAP], text=True, capture_output=True, timeout=45,
                            input=json.dumps({'root': remote, 'create': False, 'files': complete}))
    atomic(run.root / 'provenance/submission-evidence-mirror.json',
           {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'slurm_job_id': job_id})
    run.refresh()
    print(json.dumps({'slurm_job_id': job_id, 'remote_run_dir': remote, 'mirror_exit_code': result.returncode}))
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
