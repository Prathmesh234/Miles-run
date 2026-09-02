"""Materialize pinned Miles runtime sources from the already verified CPU-test archive."""
import argparse
import json
from pathlib import Path

from evidence import Run, atomic


REMOTE = r'''
import hashlib,json,pathlib,shutil,sys,tarfile
root=pathlib.Path(sys.argv[1]).resolve()
if not root.is_relative_to('/shared/posttrainingx/runs/vultr-b200-slurm'):
 raise ValueError('Destination outside campaign run scope')
sys.path.insert(0,str(root/'provenance/qwen-serving-code-v3'))
from evidence import Run,atomic,sha256
run=Run(root);phase=run.phase('00-pinned-training-source-materialization')
try:
 source=root/'provenance/linux-launch-tests-v1'
 manifest=json.loads((source/'manifest.json').read_text())
 if manifest['source_revision']!='b61dbe83ee815412b72c84ed367ffd329d7922d4':
  raise ValueError('Archive revision differs from committed campaign Miles')
 archive=source/'miles.tar.gz'
 if sha256(archive)!=manifest['source_archive_sha256']:
  raise ValueError('Pinned source archive checksum mismatch')
 if shutil.disk_usage(root).free < 2*1024**3:
  raise ValueError('Source preparation requires 2 GiB free')
 target=root/'provenance/training-source-v1'
 target.mkdir(exist_ok=False)
 omitted=[]
 with tarfile.open(archive) as tar:
  members=tar.getmembers()
  if len(members)>10000 or sum(m.size for m in members)>64*1024**2:
   raise ValueError('Unexpected archive size or member count')
  selected=[]
  for member in members:
   path=pathlib.PurePosixPath(member.name)
   if path.is_absolute() or '..' in path.parts or path.parts[0]!='miles':
    raise ValueError('Unsafe source archive member')
   if member.name=='miles/.agents/skills' and member.issym():
    omitted.append({'member':member.name,'link_target':member.linkname,
     'reason':'Non-runtime agent skill alias omitted; evidence bundles prohibit symlinks'})
    continue
   if not (member.isfile() or member.isdir()):
    raise ValueError('Unexpected non-regular source member')
   selected.append(member)
  tar.extractall(target,members=selected,filter='data')
 files=[]
 for path in sorted((target/'miles').rglob('*')):
  if path.is_symlink():raise ValueError('Symlink survived source materialization')
  if path.is_file():files.append({'path':str(path.relative_to(target)),
   'bytes':path.stat().st_size,'sha256':sha256(path)})
 receipt={'source_git_sha':manifest['source_revision'],'source_archive_sha256':manifest['source_archive_sha256'],
  'source_root':str(target/'miles'),'files':files,'omitted_nonruntime_members':omitted,
  'scope':'Pinned runtime source files only; no package install, model conversion, policy or optimizer execution'}
 atomic(target/'manifest.json',receipt)
 phase.finish('ok',metadata={'source_git_sha':receipt['source_git_sha'],'file_count':len(files),
  'manifest_sha256':sha256(target/'manifest.json'),'artifacts':[str(target.relative_to(root))],
  'omitted_nonruntime_members':omitted},refresh=False)
 print(json.dumps({'source_root':receipt['source_root'],'file_count':len(files),
  'manifest_sha256':sha256(target/'manifest.json'),'source_git_sha':receipt['source_git_sha']}))
except Exception as exc:
 phase.finish('fail',failure_summary=str(exc),refresh=False)
 raise
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('00-pinned-training-source-materialization-controller')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', str(Path(args.kubeconfig).resolve()), '-n', 'slurm',
        'exec', 'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', REMOTE, remote], timeout=90)
    receipt = json.loads(out) if not rc else {}
    if not rc:
        atomic(run.root / 'provenance/training-source-materialization.json', receipt)
    phase.finish('ok' if not rc else 'fail', failure_summary='Source materialization failed; preserve partial directory.' if rc else None,
                 metadata=receipt)
    print(json.dumps(receipt))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
