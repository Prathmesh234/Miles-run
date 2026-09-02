"""Resume a preserved diagnostic archive using verified 64 KiB upload pieces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from evidence import Run, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry
from sync_diagnostic_evidence import INSTALL, INSPECT


ASSEMBLE = '''import hashlib,json,os,sys,tempfile
from pathlib import Path
p=Path(sys.argv[1]);plan=json.loads((p/'resume-plan.json').read_text())
for index,checksum in enumerate(plan['part_sha256']):
 path=p/'parts'/f'{index:04d}'
 if path.exists():
  if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('Existing part mismatch.')
  continue
 data=b''.join((p/plan.get('piece_dir','recovery-64k')/f'{index:04d}-{j}').read_bytes() for j in range(plan['piece_counts'][index]))
 if hashlib.sha256(data).hexdigest()!=checksum:raise ValueError('Recovered part checksum mismatch.')
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.link(tmp,path);os.unlink(tmp)
print(json.dumps({'verified_parts':len(plan['part_sha256'])}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--original-part-bytes', type=int, choices=[65536, 131072], default=131072)
    ap.add_argument('--piece-bytes', type=int, choices=[32768, 65536], default=65536)
    args = ap.parse_args()
    run = Run(args.run_dir)
    suffix = '' if args.tag == 'v1' else '-' + args.tag
    phase = run.phase(f'00-diagnostic-evidence-resume{suffix}-{args.attempt}')
    original_phase = '00-diagnostic-evidence-sync' + suffix
    manifest = json.loads((run.root/'tests'/original_phase/'transfer-manifest.json').read_text())
    archive = (run.root/'tests/00-diagnostic-archive-recovery/diagnostic-evidence.tar.gz' if args.tag == 'v1'
               else run.root/'tests'/original_phase/'diagnostic-evidence.tar.gz')
    if sha256(archive) != manifest['archive_sha256']:
        phase.finish('fail', failure_summary='Recovered archive does not match the original transfer manifest.')
        return 1
    payload = archive.read_bytes()
    parts = [payload[i:i+args.original_part_bytes] for i in range(0, len(payload), args.original_part_bytes)]
    if len(parts) != manifest['parts']:
        phase.finish('fail', failure_summary='Declared original chunk size does not match the frozen manifest.')
        return 1
    prefix = 'provenance/diagnostic-evidence-sync-' + args.tag + '/'
    piece_dir = f'recovery-{args.piece_bytes//1024}k'
    files = {prefix+'manifest.json': entry(json.dumps(manifest, sort_keys=True).encode())}
    part_hashes, piece_counts = [], []
    for index, part in enumerate(parts):
        part_hashes.append(hashlib.sha256(part).hexdigest())
        pieces = [part[i:i+args.piece_bytes] for i in range(0, len(part), args.piece_bytes)]
        piece_counts.append(len(pieces))
        for j, piece in enumerate(pieces):
            files[prefix+f'{piece_dir}/{index:04d}-{j}'] = entry(piece)
    plan = {'part_sha256': part_hashes, 'piece_counts': piece_counts}
    if piece_dir != 'recovery-64k':
        plan['piece_dir'] = piece_dir
    files[prefix+'resume-plan.json'] = entry(json.dumps(plan, sort_keys=True).encode())
    originals = {prefix+f'parts/{i:04d}': checksum for i, checksum in enumerate(part_hashes)}
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '--request-timeout=0', '-n', 'slurm',
              'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    code, out, _ = phase.command(worker+['python3', '-c', INSPECT, remote],
                                stdin=json.dumps(sorted(set(files) | set(originals))), timeout=60)
    if code:
        phase.finish('fail', failure_summary='Partial upload inspection failed. No retry was issued.')
        return 1
    existing = json.loads(out)
    expected = dict(originals, **{name: item['sha256'] for name, item in files.items()})
    if any(expected[name] != checksum for name, checksum in existing.items()):
        phase.finish('fail', failure_summary='A remote upload fragment differs from the frozen archive.')
        return 1
    for index in range(len(parts)):
        if prefix+f'parts/{index:04d}' in existing:
            for j in range(piece_counts[index]):
                files.pop(prefix+f'{piece_dir}/{index:04d}-{j}', None)
    files = {name: item for name, item in files.items() if name not in existing}
    common = {'root': remote, 'create': False, 'manifest_sha256': sha256(run.root/'run.json')}
    uploads = list(batches(common, files, limit=2*args.piece_bytes))
    for index, encoded in enumerate(uploads):
        code, _, _ = phase.command(worker+['python3', '-c', BOOTSTRAP], stdin=encoded, timeout=45)
        if code:
            phase.finish('fail', failure_summary='The bounded evidence upload failed; stop and reconcile before another attempt.',
                         metadata={'upload_index': index, 'upload_count': len(uploads)})
            return 1
    code, _, _ = phase.command(worker+['python3', '-c', ASSEMBLE, remote+'/'+prefix], timeout=60)
    if not code:
        code, _, _ = phase.command(worker+['python3', '-c', INSTALL, remote, prefix], timeout=90)
    phase.finish('fail' if code else 'ok', failure_summary='Archive assembly or installation failed.' if code else None,
        metadata={'original_failed_phase': original_phase,
                  'archive_sha256': manifest['archive_sha256'], 'files': len(manifest['files']),
                  'piece_bytes': args.piece_bytes, 'upload_calls': len(uploads),
                  'scope': 'Verified transfer resume only; original transport failure remains recorded.'})
    print(json.dumps({'exit_code': code, 'files': len(manifest['files']), 'upload_calls': len(uploads)}))
    return code


if __name__ == '__main__':
    sys.exit(main())
