"""Download only manifest-listed model files, verify bytes, and preserve failures."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request

from evidence import Run, atomic, metric, utcnow


def validate_manifest(manifest):
    if manifest['model'] != 'Qwen/Qwen3.6-35B-A3B':
        raise ValueError('Unexpected model repository.')
    if not re.fullmatch('[0-9a-f]{40}', manifest['revision']):
        raise ValueError('An exact model revision is required.')
    names = set()
    for item in manifest['files']:
        name = item['path']
        if not name or name == '.' or Path(name).is_absolute() or '..' in Path(name).parts or name in names:
            raise ValueError('Unsafe or duplicate model file path.')
        if not isinstance(item['size'], int) or item['size'] < 0:
            raise ValueError('Every file needs an exact byte size.')
        checksum = item['lfs']['sha256'] if item.get('lfs') else item['git_blob_sha1']
        if not re.fullmatch('[0-9a-f]{64}' if item.get('lfs') else '[0-9a-f]{40}', checksum):
            raise ValueError('Every file needs a valid upstream content hash.')
        names.add(name)
    if not {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'} <= names:
        raise ValueError('Missing required model or tokenizer files.')


def download_file(item, base_url, destination, stop, opener=urllib.request.urlopen):
    if stop.is_set():
        raise RuntimeError('Download cancelled before request after another file failed.')
    target = destination / item['path']
    if target.exists() or target.is_symlink():
        raise ValueError('Refusing to replace a model file.')
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in target.parents):
        raise ValueError('Symlink in model destination.')
    partial = Path(str(target) + '.partial')
    url = base_url + '/' + urllib.parse.quote(item['path'], safe='/')
    started = utcnow()
    begin = time.monotonic()
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {item['size']}\0".encode())
    received = 0
    with opener(url, timeout=120) as response, partial.open('xb') as output:
        status = response.status
        if status != 200:
            raise ValueError(f'Unexpected HTTP status {status}.')
        while True:
            if stop.is_set():
                raise RuntimeError('Download cancelled after another file failed.')
            block = response.read(1024 * 1024)
            if not block:
                break
            received += len(block)
            if received > item['size']:
                raise ValueError('Model response exceeds its pinned byte size.')
            output.write(block)
            sha.update(block)
            blob.update(block)
        output.flush()
        os.fsync(output.fileno())
    expected = item['lfs']['sha256'] if item.get('lfs') else item['git_blob_sha1']
    actual = sha.hexdigest() if item.get('lfs') else blob.hexdigest()
    if received != item['size'] or actual != expected:
        raise ValueError('Pinned model size or content hash mismatch.')
    os.link(partial, target)
    partial.unlink()
    return {'path': item['path'], 'url': url, 'status': status, 'bytes': received,
            'sha256': sha.hexdigest(), 'git_blob_sha1': blob.hexdigest() if not item.get('lfs') else None,
            'started_at': started, 'ended_at': utcnow(), 'duration_s': time.monotonic() - begin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('00-qwen-model-download')
    manifest = json.loads(Path(args.manifest).read_text())
    try:
        validate_manifest(manifest)
        if not 1 <= args.workers <= 8:
            raise ValueError('Download concurrency must be between one and eight.')
        required = sum(x['size'] for x in manifest['files']) + 100*1024**3
        if shutil.disk_usage(run.root).free < required:
            raise ValueError('Insufficient space for model plus a 100 GiB reserve.')
        destination = run.root / 'models' / ('qwen3.6-35b-a3b-' + manifest['revision'])
        destination.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        phase.finish('fail', failure_summary=str(exc), refresh=False)
        raise
    stop = threading.Event()
    results, failures = [], []
    base_url = f"https://huggingface.co/{manifest['model']}/resolve/{manifest['revision']}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download_file, item, base_url, destination, stop): item
                   for item in manifest['files']}
        for future in concurrent.futures.as_completed(pending):
            item = pending[future]
            try:
                result = future.result()
                results.append(result)
                atomic(phase.path / 'logs' / (item['path'].replace('/', '_') + '.json'), result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                stop.set()
                # Do not serialize exception URLs: redirected CDN URLs can
                # contain signed query strings. The type and file identify it.
                failure = {'path': item['path'], 'error_type': type(exc).__name__,
                           'http_status': getattr(exc, 'code', None), 'time': utcnow()}
                failures.append(failure)
                atomic(phase.path / 'logs' / (item['path'].replace('/', '_') + '.failed.json'), failure)
                print(json.dumps(failure), flush=True)
    atomic(phase.path / 'files.json', {'revision': manifest['revision'], 'files': results, 'failures': failures})
    if not failures:
        atomic(destination / 'checksums.sha256', ''.join(
            x['sha256'] + '  ' + x['path'] + '\n' for x in sorted(results, key=lambda r: r['path'])))
    phase.finish('fail' if failures else 'ok',
        failure_summary='At least one model download failed; no retry, overwrite, or partial-file cleanup was attempted.' if failures else None,
        results=[metric('verified_model_files', len(results), 'count'),
                 metric('verified_download_bytes', sum(x['bytes'] for x in results), 'B')],
        metadata={'revision': manifest['revision'], 'model': manifest['model'], 'workers': args.workers,
                  'directory': str(destination), 'failed_files': failures,
                  'artifacts': [str((phase.path/'files.json').relative_to(run.root)),
                                str(destination.relative_to(run.root))],
                  'scope': 'Pinned source download and content verification, not storage benchmarking or model execution.'},
        refresh=False)
    print(json.dumps({'status': 'fail' if failures else 'ok', 'verified_files': len(results),
                      'directory': str(destination)}), flush=True)
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
