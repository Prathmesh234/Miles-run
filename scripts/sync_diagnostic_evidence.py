"""Append diagnostic evidence to the existing shared run without overwriting files."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from evidence import Run, atomic, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry


PREFIXES = ('00-miles-', '00-compatibility-', '00-dependency-', '00-campaign-',
            '01-live-dcgm-', '01-fabric-', '01-full-native-', '01-native-infrastructure-',
            '01-native-report-link-', '01-native-telemetry-export')

INSPECT = '''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);names=json.load(sys.stdin);result={}
for name in names:
 p=root/name
 if p.exists():
  if p.is_symlink() or not p.is_file():raise ValueError('Invalid existing evidence path: '+name)
  result[name]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(result))
'''

INSTALL = '''import hashlib,io,json,os,sys,tarfile,tempfile
from pathlib import Path
root=Path(sys.argv[1]);bundle=root/sys.argv[2];manifest=json.loads((bundle/'manifest.json').read_text())
payload=b''.join((bundle/'parts'/f'{i:04d}').read_bytes() for i in range(manifest['parts']))
if hashlib.sha256(payload).hexdigest()!=manifest['archive_sha256']:raise ValueError('Archive checksum mismatch.')
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:
 members=t.getmembers()
 if sorted(x.name for x in members)!=sorted(manifest['files']):raise ValueError('Archive manifest mismatch.')
 for member in members:
  path=root/member.name
  if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:raise ValueError('Unsafe archive member.')
  if path.exists() or any(p.is_symlink() for p in [path]+list(path.parents)):raise ValueError('Refusing to replace existing evidence.')
  if hashlib.sha256(t.extractfile(member).read()).hexdigest()!=manifest['files'][member.name]:raise ValueError('Member checksum mismatch.')
 for member in members:
  path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True)
  fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
  with os.fdopen(fd,'wb') as f:f.write(t.extractfile(member).read());f.flush();os.fsync(f.fileno())
  os.link(tmp,path);os.unlink(tmp)
print(json.dumps({'installed_files':len(members),'archive_sha256':manifest['archive_sha256']}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('00-diagnostic-evidence-sync')
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        phase.finish('fail', failure_summary='Commit the diagnostic code before freezing it.')
        return 1
    source_name = 'provenance/campaign-source-' + revision + '.tar.gz'
    source = gzip.compress(subprocess.check_output(['git', '-C', str(repo), 'archive', revision]), mtime=0)
    source_path = run.root / source_name
    with source_path.open('xb') as f:
        f.write(source)
        f.flush()
        os.fsync(f.fileno())
    selected = [p for p in (run.root/'tests').rglob('*') if p.is_file()
                and p.relative_to(run.root/'tests').parts[0].startswith(PREFIXES)]
    selected += [p for p in (run.root/'reports').rglob('*') if p.is_file()]
    selected += [source_path, run.root/'provenance/miles-patch-verification.tar.gz']
    files = {str(p.relative_to(run.root)): p for p in selected}
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '--request-timeout=0', '-n', 'slurm',
              'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    code, out, _ = phase.command(worker + ['python3', '-c', INSPECT, remote],
                                stdin=json.dumps(sorted(files)), timeout=60)
    if code:
        phase.finish('fail', failure_summary='Shared evidence inspection failed before upload.')
        return 1
    existing = json.loads(out)
    conflicts = [name for name, checksum in existing.items() if sha256(files[name]) != checksum]
    if conflicts:
        phase.finish('fail', failure_summary='Shared evidence conflicts; nothing was overwritten.', metadata={'conflicts': conflicts})
        return 1
    new = {name: p for name, p in files.items() if name not in existing}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz', dereference=False) as tar:
        for name, p in sorted(new.items()):
            if p.is_symlink():
                raise ValueError('Evidence symlinks are forbidden.')
            tar.add(p, arcname=name, recursive=False)
    payload = buffer.getvalue()
    chunks = [payload[i:i+128*1024] for i in range(0, len(payload), 128*1024)]
    prefix = 'provenance/diagnostic-evidence-sync-v1/'
    manifest = {'schema_version': 1, 'source_revision': revision,
                'archive_sha256': hashlib.sha256(payload).hexdigest(), 'parts': len(chunks),
                'files': {name: sha256(p) for name, p in new.items()},
                'existing_identical_files': len(existing)}
    atomic(phase.path/'transfer-manifest.json', manifest)
    staged = {prefix+f'parts/{i:04d}': entry(chunk) for i, chunk in enumerate(chunks)}
    staged[prefix+'manifest.json'] = entry(json.dumps(manifest, sort_keys=True).encode())
    common = {'root': remote, 'create': False, 'manifest_sha256': sha256(run.root/'run.json')}
    plan = list(batches(common, staged))
    for index, encoded in enumerate(plan):
        code, _, _ = phase.command(worker+['python3', '-c', BOOTSTRAP], stdin=encoded, timeout=45)
        if code:
            phase.finish('fail', failure_summary='Diagnostic evidence upload failed; preserve partial archive chunks.', metadata={'upload_index': index})
            return 1
    code, _, _ = phase.command(worker+['python3', '-c', INSTALL, remote, prefix], timeout=90)
    phase.finish('fail' if code else 'ok', failure_summary='Diagnostic evidence installation failed.' if code else None,
        metadata={'source_revision': revision, 'installed_files': len(new),
                  'existing_identical_files': len(existing), 'archive_sha256': manifest['archive_sha256'],
                  'scope': 'Diagnostic phases and reports appended without overwrite; shared summary refresh is separate.'})
    print(json.dumps({'installed_files': len(new), 'exit_code': code, 'archive_bytes': len(payload)}))
    return code


if __name__ == '__main__':
    sys.exit(main())
