"""Build an isolated, pinned OpenEnv controller image without changing Miles."""
import argparse
import io
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import traceback

from evidence import Run, atomic, sha256

IMAGE = 'python@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-local-openenv-image-v{args.attempt}')
    code = Path(__file__).parent
    parent = run.root / f'environments/local-openenv-runtime-v{args.attempt}'
    result = {'findings': [], 'scope': 'Isolated CPU-only OpenEnv controller image; no policy or optimizer execution.'}
    try:
        parent.mkdir(parents=True, mode=0o700, exist_ok=False)
        manifest = json.loads((code / 'input-manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha256(code / name) != digest:
                raise ValueError('Controller build input changed: ' + name)
        source = parent / 'source'
        source.mkdir()
        payload = b''.join((code / f'archive-parts/{i:04d}').read_bytes() for i in range(manifest['archive_parts']))
        if hashlib.sha256(payload).hexdigest() != manifest['archive_sha256']:
            raise ValueError('Reassembled controller source archive changed.')
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe OpenEnv source archive member.')
            archive.extractall(source, filter='data')
        shutil.copyfile(code / 'server.lock', parent / 'server.lock')
        atomic(parent / 'Dockerfile', 'FROM ' + IMAGE + '\n'
               'COPY server.lock /opt/server.lock\n'
               'RUN python -m pip install --no-cache-dir --require-hashes -r /opt/server.lock\n'
               'COPY source /opt/openenv\n'
               'ENV PYTHONPATH=/opt/openenv/src:/opt/openenv/envs PYTHONDONTWRITEBYTECODE=1\n'
               'RUN python -c "from openenv.core.env_server.http_server import create_app; import docker; print(docker.__version__)"\n')
        if shutil.disk_usage(run.root).free < 128*1024**3 or shutil.disk_usage('/var/lib/docker').free < 128*1024**3:
            raise ValueError('Controller image build requires 128 GiB shared and local reserve.')
        tag = 'posttrainingx-openenv/' + run.root.name + f':v{args.attempt}'
        rc, _, _ = phase.command(['docker', 'build', '--pull=false', '--network=default', '--label',
                                 'posttrainingx.run=' + run.root.name, '--tag', tag, str(parent)], timeout=600)
        if rc:
            raise RuntimeError('Pinned OpenEnv image build failed; preserve build output.')
        rc, text, _ = phase.command(['docker', 'image', 'inspect', tag], timeout=30)
        if rc:
            raise RuntimeError('Controller image identity is unavailable.')
        info = json.loads(text)[0]
        result.update(image_id=info['Id'], tag=tag, inputs=manifest, parent_image=IMAGE)
        atomic(parent / 'image.json', result)
        # A successful build is not a running environment service.
    except Exception as exc:
        result['findings'].append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if result['findings'] else 'ok', failure_summary='; '.join(result['findings']) or None,
                 metadata=result, refresh=False)
    print(json.dumps(result), flush=True)
    return int(bool(result['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
