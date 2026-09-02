"""Copy finalized native telemetry and verify against the frozen job checksums."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

from evidence import Run, sha256


NAMES = ['nvidia-smi.jsonl', 'nvlink.jsonl', 'infiniband.jsonl', 'cpu-memory-numa.jsonl', 'lustre.jsonl']
EXPORT = '''import base64,hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);names=json.loads(sys.argv[2])
files=[root/'telemetry'/n for n in names]
if any(p.is_symlink() or not p.is_file() for p in files):raise ValueError('Invalid telemetry file.')
if sum(p.stat().st_size for p in files)>150*1024**2:raise ValueError('Telemetry export exceeds 150 MiB guard.')
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz',dereference=False) as t:
 for p in files:t.add(p,arcname=p.name,recursive=False)
payload=buf.getvalue()
if len(payload)>16*1024**2:raise ValueError('Compressed export exceeds 16 MiB guard.')
print(json.dumps({'sha256':hashlib.sha256(payload).hexdigest(),'tar_gz_base64':base64.b64encode(payload).decode()}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('01-native-telemetry-export')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    code, out, _ = phase.command([
        'kubectl', '--kubeconfig', args.kubeconfig, '--request-timeout=0', '-n', 'slurm',
        'exec', 'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', EXPORT, remote, json.dumps(NAMES),
    ], timeout=90)
    if code:
        phase.finish('fail', failure_summary='Native telemetry export failed. No training depends on a partial export.')
        return 1
    try:
        response = json.loads(out)
        payload = base64.b64decode(response['tar_gz_base64'], validate=True)
        if hashlib.sha256(payload).hexdigest() != response['sha256']:
            raise ValueError('Export checksum mismatch.')
        frozen = {}
        for line in (run.root / 'provenance/shared-checksums-after-job-109.sha256').read_text().splitlines():
            checksum, name = line.split('  ', 1)
            frozen[name] = checksum
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
            members = tar.getmembers()
            if sorted(x.name for x in members) != sorted(NAMES) or any(not x.isfile() for x in members):
                raise ValueError('Unexpected archive members.')
            # Validate the complete transfer before creating any final files.
            files = {x.name: tar.extractfile(x).read() for x in members}
        for name, data in files.items():
            if hashlib.sha256(data).hexdigest() != frozen['telemetry/' + name]:
                raise ValueError('Frozen job checksum mismatch: ' + name)
            if (run.root / 'telemetry' / name).exists():
                raise ValueError('Refusing to overwrite telemetry: ' + name)
        if shutil.disk_usage(run.root).free < sum(len(v) for v in files.values()) + 128*1024**2:
            raise ValueError('Insufficient free space for telemetry plus the evidence reserve.')
        for name, data in files.items():
            target = run.root / 'telemetry' / name
            with Path(str(target) + '.partial').open('xb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(target) + '.partial', target)
    except Exception as exc:
        phase.finish('fail', failure_summary='Telemetry validation failed: ' + str(exc))
        return 1
    phase.finish('ok', metadata={
        'scope': 'Finalized native job 109 time series only; additional NCCL and per-node raw logs remain on shared storage.',
        'archive_sha256': response['sha256'],
        'files': {name: sha256(run.root / 'telemetry' / name) for name in NAMES},
        'artifacts': ['telemetry/' + name for name in NAMES],
    })
    print(json.dumps({'files': len(NAMES), 'uncompressed_bytes': sum(len(v) for v in files.values())}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
