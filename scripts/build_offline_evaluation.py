"""Build a hash-locked Python3.12 Verifiers/Harbor image, separate from Miles."""
import argparse
import json
from pathlib import Path
import shutil
import traceback

from evidence import Run, atomic, sha256

BASE = 'python@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-offline-evaluation-image-v{args.attempt}')
    code = Path(__file__).parent
    parent = run.root / f'environments/offline-evaluation-v{args.attempt}'
    result = {'findings': [], 'scope': 'CPU-only hash-locked package installation/import gate, not TB2.1 task execution.',
              'base_image':BASE, 'online_miles_stack_modified':False}
    try:
        parent.mkdir(parents=True, mode=0o700, exist_ok=False)
        manifest = json.loads((code / 'input-manifest.json').read_text())
        pieces = []
        for item in manifest['lock_parts']:
            path = code / item['path']
            if sha256(path) != item['sha256']:
                raise ValueError('Offline lock fragment changed.')
            pieces.append(path.read_text())
        atomic(parent / 'requirements.lock', ''.join(pieces))
        if sha256(parent / 'requirements.lock') != manifest['lock_sha256']:
            raise ValueError('Reassembled offline lock changed.')
        atomic(parent / 'Dockerfile', 'FROM ' + BASE + '\n'
               'COPY requirements.lock /opt/requirements.lock\n'
               'RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: --require-hashes -r /opt/requirements.lock\n'
               'RUN python -m pip check && python -m pip freeze > /opt/packages.freeze.txt\n'
               'ENV PYTHONDONTWRITEBYTECODE=1\n')
        if min(shutil.disk_usage(run.root).free, shutil.disk_usage('/var/lib/docker').free) < 128 * 1024**3:
            raise ValueError('128GiB shared/local free-space reserve not met.')
        tag = 'posttrainingx-offline-eval/' + run.root.name + f':v{args.attempt}'
        rc, _, _ = phase.command(['docker', 'build', '--pull=false', '--network=default', '--label',
            'posttrainingx.run=' + run.root.name, '--tag', tag, str(parent)], timeout=900)
        if rc:
            raise RuntimeError('Pinned offline image build failed; no forced package installation or online changes.')
        rc, out, _ = phase.command(['docker', 'image', 'inspect', tag], timeout=30)
        if rc:
            raise RuntimeError('Offline image identity could not be verified.')
        image = json.loads(out)[0]
        probe = "import sys,json,verifiers,harbor;from importlib.metadata import version;from pathlib import Path;assert sys.version_info[:3]==(3,12,11);v={k:version(k) for k in ('verifiers','harbor','openai')};assert v=={'verifiers':'0.3.1','harbor':'0.21.0','openai':'2.54.0'};print(json.dumps({'python':sys.version,'packages':v,'freeze':Path('/opt/packages.freeze.txt').read_text()}))"
        rc, out, _ = phase.command(['docker', 'run', '--rm', '--runtime=runc', '--network=none', '--read-only',
            '--tmpfs=/tmp:rw,nosuid,nodev,size=128m', '--cpus=2', '--memory=4g', '--pids-limit=128', '--cap-drop=ALL',
            '--security-opt=no-new-privileges', '--label=posttrainingx.run=' + run.root.name,
            '-e', 'NVIDIA_VISIBLE_DEVICES=void', image['Id'], 'python', '-c', probe], timeout=120)
        if rc:
            raise RuntimeError('Offline import/version gate failed.')
        packages = json.loads(out.splitlines()[-1])
        result.update(image_id=image['Id'], tag=tag, manifest=manifest, probe=packages)
        atomic(parent / 'image.json', result)
        atomic(parent / 'packages.freeze.txt', packages['freeze'])
    except Exception as exc:
        result['findings'].append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if result['findings'] else 'ok', failure_summary='; '.join(result['findings']) or None,
                 metadata=result, refresh=False)
    print(json.dumps({'findings':result['findings'], 'image_id':result.get('image_id')}))
    return int(bool(result['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
