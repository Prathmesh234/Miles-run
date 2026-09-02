"""Run the complete Miles launcher checks inside a run-owned CPU container."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def command(argv, cwd=None, check=True):
    started = time.time()
    print(json.dumps({'argv': argv, 'cwd': cwd, 'started_unix_s': started}), flush=True)
    result = subprocess.run(argv, cwd=cwd)
    print(json.dumps({'exit_code': result.returncode, 'duration_s': time.time()-started}), flush=True)
    if check and result.returncode:
        raise SystemExit(result.returncode)
    return result.returncode


def main():
    manifest = json.loads(Path('/input/manifest.json').read_text())
    archive = Path('/source/miles.tar.gz')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['source_archive_sha256']:
        raise ValueError('Source archive checksum mismatch.')
    with tarfile.open(archive) as tar:
        tar.extractall('/work', filter='data')
    # A diagnostic baseline restores only the committed delta inside this
    # disposable CPU container. The uploaded source archive remains read-only.
    if 'baseline_delta_sha256' in manifest:
        delta_path = Path('/input/baseline-delta.json')
        if hashlib.sha256(delta_path.read_bytes()).hexdigest() != manifest['baseline_delta_sha256']:
            raise ValueError('Baseline delta checksum mismatch.')
        for relative, content in json.loads(delta_path.read_text()).items():
            target = Path('/work/miles') / relative
            if not target.resolve().is_relative_to('/work/miles') or target.is_symlink():
                raise ValueError('Invalid baseline delta path.')
            if content is None:
                target.unlink()
            else:
                target.write_bytes(base64.b64decode(content, validate=True))
    # This test environment is separate from both the GPU training image and
    # the offline Verifiers/Harbor environment. It never installs either stack.
    command([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir',
             '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.11.0+cpu'])
    command([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir',
             '-r', '/input/requirements.txt'])
    installed = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    Path('/artifacts/packages.freeze.txt.partial').write_text(installed)
    os.replace('/artifacts/packages.freeze.txt.partial', '/artifacts/packages.freeze.txt')
    command([sys.executable, '-m', 'pip', 'check'])
    test_paths = manifest.get('test_paths', [
        'tests/fast/launch_scripts', 'tests/manual/launch_scripts',
        'tests/fast/utils/test_command_utils.py', 'tests/fast/ray/test_host_placement.py'])
    code = command([sys.executable, '-m', 'pytest', '--noconftest', *test_paths,
             '--junitxml=/artifacts/results.xml.partial', '-q', '--tb=short'], cwd='/work/miles', check=False)
    if Path('/artifacts/results.xml.partial').exists():
        os.replace('/artifacts/results.xml.partial', '/artifacts/results.xml')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
