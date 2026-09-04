"""Download a finished run's raw evidence; leave model/checkpoint shards on /shared.

The streamed archive carries per-file SHA-256 hashes. It never modifies the
remote run, starts a container, removes data, or extracts an unsafe tar member.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


REMOTE = r'''
import hashlib,io,json,pathlib,sys,tarfile
p=pathlib.Path(sys.argv[1]); assert p.parent.parent.parent==pathlib.Path('/shared/clustermax-campaigns')
assert p.name.startswith('job-') and (p/'exit-code.txt').exists(), 'Run must finish before archival'
files=[]; skipped=[]
for path in sorted(p.rglob('*')):
 if not path.is_file() or path.is_symlink():continue
 rel=path.relative_to(p)
 if rel.parts[0] in ['checkpoints','checkpoints_critic']:
  skipped.append({'path':str(rel),'bytes':path.stat().st_size});continue
 files.append((path,str(rel)))
for folder in ['preflight','analysis-source']:
 for path in sorted((p.parent.parent/folder).rglob('*')):
  if path.is_file() and not path.is_symlink():files.append((path,'campaign-'+folder+'/'+str(path.relative_to(p.parent.parent/folder))))
manifest={'remote_run':str(p),'included':[],'checkpoint_files_retained_remote':skipped}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for path,name in files:
  digest=hashlib.sha256()
  with path.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
  manifest['included'].append({'path':name,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
  tar.add(path,arcname=name,recursive=False)
 data=json.dumps(manifest,indent=2).encode(); info=tarfile.TarInfo('archive-manifest.json');info.size=len(data)
 tar.addfile(info,io.BytesIO(data))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('remote_run')
    parser.add_argument('destination',type=Path)
    parser.add_argument('--kubeconfig',default=str(Path.home()/'.kube/vultr-vke.yaml'))
    args = parser.parse_args()
    assert not args.destination.exists(), 'Use a new local archive directory'
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.destination.parent).free > 2*1024**3
    archive = args.destination.with_suffix('.tar.gz')
    assert not archive.exists()
    command = ['kubectl','--kubeconfig',args.kubeconfig,'exec','-i','-n','slurm',
               'slurm-worker-gpu-nodes-0','-c','slurmd','--','python3','-c',REMOTE,args.remote_run]
    with archive.open('xb') as stream:
        process = subprocess.run(command,stdout=stream,stderr=subprocess.PIPE)
    if process.returncode:
        archive.rename(archive.with_name(archive.name+'.failed'))
        raise RuntimeError(process.stderr.decode())
    args.destination.mkdir()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            path=Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            target=args.destination/path;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:shutil.copyfileobj(tar.extractfile(member),stream)
    manifest=json.loads((args.destination/'archive-manifest.json').read_text())
    for row in manifest['included']:
        path=args.destination/row['path']
        assert path.stat().st_size==row['bytes']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==row['sha256'],row['path']
    print(json.dumps({'directory':str(args.destination),'files_verified':len(manifest['included']),
                     'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))


if __name__=='__main__':
    main()
