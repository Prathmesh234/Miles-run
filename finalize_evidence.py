"""Read-only post-run hashes and scheduler evidence; writes only into its finished run.

Run on a cluster worker with the pinned harness Python after training has exited.
Hashing runs after the measured window so it cannot distort training storage IO.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def atomic(path, value):
    assert not path.exists(), f'Refusing to replace evidence: {path}'
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def hash_file(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    return {'path': str(path), 'bytes': before.st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    assert root.parent.parent.parent == Path('/shared/clustermax-campaigns')
    assert int((root/'exit-code.txt').read_text()) == 0, 'Only finalize a completed training process'
    assert shutil.disk_usage(root).free > 1024**3
    started = time.monotonic()
    old = Path('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030')
    model = Path('/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00/model-fetch/models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0')
    for name, folders in [('model-checksums', [model, old/'converted-model']),
                          ('checkpoint-checksums', [root/'checkpoints', root/'checkpoints_critic'])]:
        files = sorted(p for folder in folders for p in folder.rglob('*') if p.is_file())
        assert files and sum(p.stat().st_size for p in files) < 1024**4
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(hash_file, files))
        atomic(root/(name+'.json'), {'time':dt.datetime.now(dt.timezone.utc).isoformat(),
            'roots':list(map(str,folders)), 'algorithm':'sha256', 'files':results,
            'total_bytes':sum(r['bytes'] for r in results), 'elapsed_since_start_s':time.monotonic()-started})
        print(json.dumps({'artifact':name, 'files':len(results), 'bytes':sum(r['bytes'] for r in results)}), flush=True)
    commands = [['sacct','-j','191,192,193,194,195,'+root.name.removeprefix('job-'),
                 '--allocations','-P','--format=JobID,State,ExitCode,Start,End,Elapsed,AllocTRES'],
                ['squeue','-h','-o','%i|%T|%N'], ['scontrol','show','nodes','--oneliner'],
                ['df','-B1','/shared']]
    records=[]
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
        records.append({'argv':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    reference=json.loads((old/'baseline-preservation-reference.json').read_text())
    preservation=[]
    for name,expected in reference['hashes'].items():
        actual=hash_file(Path(name))
        preservation.append({**actual,'expected_sha256':expected,'unchanged':actual['sha256']==expected})
    assert all(r['unchanged'] for r in preservation), 'Baseline source/config hash changed; inspect evidence'
    atomic(root/'final-runtime.json', {'time':dt.datetime.now(dt.timezone.utc).isoformat(),
        'commands':records, 'node_inventory':'infra/*-after.json',
        'harness_python':sys.version,
        'harness_package_lock':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        'baseline_source_preservation':preservation,
        'weights_only':True, 'optimizer_rng_resume_tested':False})


if __name__ == '__main__':
    main()
