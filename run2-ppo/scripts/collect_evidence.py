"""Copy bounded text evidence into a NEW snapshot; never modify the remote run.

Usage: python3 collect_evidence.py REMOTE_CAMPAIGN JOB_ID DESTINATION
Weights/checkpoints, binary rollout dumps, caches and credentials are not copied.
Full remote evidence remains available. JSONL captures end at a complete line.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

REMOTE = r'''
import datetime,hashlib,io,json,pathlib,subprocess,sys,tarfile
campaign=pathlib.Path(sys.argv[1]);job=int(sys.argv[2])
assert campaign.parent==pathlib.Path('/shared/clustermax-campaigns')
run=campaign/'runs'/f'job-{job}'
ext={'.json','.jsonl','.log','.txt','.out','.err','.py','.yaml','.patch','.sbatch','.md'}
roots=[(run,''),(campaign/'preflight','preflight'),(campaign/'code','submitted-scripts')]
manifest={'job_id':job,'campaign':str(campaign),'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'complete':(run/'exit-code.txt').exists(),'included':[],'excluded':[]}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for root,prefix in roots:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or p.is_symlink():continue
   rel=p.relative_to(root)
   if p.suffix not in ext or any(x.startswith(('checkpoints','harness-cache','__pycache__')) for x in rel.parts):
    manifest['excluded'].append(str(p));continue
   data=p.read_bytes()
   if p.suffix=='.jsonl' and data and not data.endswith(b'\n'):data=data[:data.rfind(b'\n')+1]
   name=str(pathlib.Path(prefix)/rel)
   info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
   manifest['included'].append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 for cmd,name in [(['scontrol','show','job',str(job),'--oneliner'],'slurm-status.txt'),
                  (['sacct','-j',str(job),'--format=JobID,JobName,State,ExitCode,Start,End','-P'],'slurm-accounting-snapshot.txt')]:
  p=subprocess.run(cmd,capture_output=True);data=p.stdout+p.stderr
  info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
 data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('snapshot-manifest.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign'); ap.add_argument('job_id', type=int); ap.add_argument('destination', type=Path)
    ap.add_argument('--kubeconfig', default=str(Path.home()/'.kube/vultr-vke.yaml'))
    args = ap.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    archive = args.destination/'snapshot.tar.gz'
    argv = ['kubectl','--kubeconfig',args.kubeconfig,'exec','-i','-n','slurm',
            'slurm-worker-gpu-nodes-0','-c','slurmd','--','python3','-c',REMOTE,args.campaign,str(args.job_id)]
    with archive.open('xb') as f:
        p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.PIPE, timeout=180)
    if p.returncode:
        raise RuntimeError(p.stderr.decode())
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            target = args.destination/path; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f: f.write(tar.extractfile(member).read())
    manifest = json.loads((args.destination/'snapshot-manifest.json').read_text())
    for item in manifest['included']:
        assert hashlib.sha256((args.destination/item['path']).read_bytes()).hexdigest() == item['sha256']
    print(json.dumps({'destination':str(args.destination), 'files_verified':len(manifest['included']),
                      'complete':manifest['complete']}))


if __name__ == '__main__': main()
