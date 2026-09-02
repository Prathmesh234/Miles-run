"""Build a source-pinned CPU test bundle and stage it without large exec streams."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evidence import Run, atomic
from submit_native_preflight import BOOTSTRAP, batches, entry


IMAGE = 'python@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    miles = repo / 'vendor/miles'
    phase = run.phase('00-miles-linux-test-staging')
    code, revision, _ = phase.command(['git', '-C', str(miles), 'rev-parse', 'HEAD'])
    dirty = subprocess.check_output(['git', '-C', str(miles), 'status', '--porcelain'], text=True)
    if code or dirty.strip():
        phase.finish('fail', failure_summary='Miles must be committed before building the test bundle.')
        return 1
    local = run.root / 'provenance/linux-launch-tests-v1'
    local.mkdir(exist_ok=False)
    tar_path = local / 'miles.tar'
    code, _, _ = phase.command(['git', '-C', str(miles), 'archive', '--format=tar', '--prefix=miles/',
                                '-o', str(tar_path) + '.partial', revision.strip()])
    if code:
        phase.finish('fail', failure_summary='The pinned source archive could not be created.')
        return 1
    os.replace(str(tar_path) + '.partial', tar_path)
    compressed = gzip.compress(tar_path.read_bytes(), mtime=0)
    checksum = hashlib.sha256(compressed).hexdigest()
    files = {}
    prefix = 'provenance/linux-launch-tests-v1/'
    chunk_size = 128 * 1024
    parts = [compressed[i:i+chunk_size] for i in range(0, len(compressed), chunk_size)]
    for index, part in enumerate(parts):
        files[prefix + f'parts/{index:04d}'] = entry(part)
    uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
    requirement_text = subprocess.check_output(
        [uv, 'pip', 'freeze', '--python', str(repo / '.venv-launch-tests/bin/python')], text=True)
    requirement_text = '\n'.join(line for line in requirement_text.splitlines() if not line.startswith('torch==')) + '\n'
    manifest = {'schema_version': 1, 'source_revision': revision.strip(), 'source_archive_sha256': checksum,
                'source_archive_bytes': len(compressed), 'chunks': len(parts), 'image': IMAGE,
                'python': '3.12.11', 'scope': 'CPU launcher tests only; no policy, environment, or optimizer execution.'}
    atomic(local / 'manifest.json', manifest)
    files[prefix + 'manifest.json'] = entry((local / 'manifest.json').read_bytes())
    files[prefix + 'requirements.txt'] = entry(requirement_text.encode())
    files[prefix + 'runner.py'] = entry((repo / 'scripts/linux_launch_tests.py').read_bytes())
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '--request-timeout=30s', '-n', 'slurm',
              'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    common = {'root': remote, 'create': False,
              'manifest_sha256': hashlib.sha256((run.root / 'run.json').read_bytes()).hexdigest()}
    for payload in batches(common, files):
        code, _, _ = phase.command(worker + ['python3', '-c', BOOTSTRAP], stdin=payload, timeout=45)
        if code:
            phase.finish('fail', failure_summary='CPU test staging failed. Inspect partial artifacts before any retry.')
            return 1
    assemble = """import hashlib,json,os
from pathlib import Path
p=Path(__import__('sys').argv[1]); m=json.loads((p/'manifest.json').read_text())
target=p/'miles.tar.gz'
if target.exists():raise ValueError('Refusing to overwrite source archive.')
h=hashlib.sha256()
with (p/'miles.tar.gz.partial').open('xb') as f:
 for i in range(m['chunks']):
  data=(p/'parts'/f'{i:04d}').read_bytes();h.update(data);f.write(data)
 f.flush();os.fsync(f.fileno())
if h.hexdigest()!=m['source_archive_sha256']:raise ValueError('Assembled source checksum mismatch.')
os.replace(p/'miles.tar.gz.partial',target)
print(json.dumps(m))
"""
    code, _, _ = phase.command(worker + ['python3', '-c', assemble, remote + '/' + prefix], timeout=45)
    phase.finish('fail' if code else 'ok', failure_summary='Source assembly failed.' if code else None,
                 metadata=dict(manifest, remote_input_dir=remote + '/' + prefix))
    print(json.dumps(manifest))
    return code


if __name__ == '__main__':
    sys.exit(main())
