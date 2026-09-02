"""Import the GPU image with Enroot using only run-scoped paths and bounded work."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from evidence import Run, atomic, metric, sha256, utcnow


IMAGE = 'docker://registry-1.docker.io#radixark/miles@sha256:59a11219eae0defc6594ec678fafe4e897c16904263223f79968cd3e0209a502'


def guarded_import(root, output):
    env = os.environ.copy()
    for key, directory in {
        'ENROOT_CACHE_PATH': 'cache', 'ENROOT_DATA_PATH': 'data',
        'ENROOT_RUNTIME_PATH': 'runtime', 'ENROOT_TEMP_PATH': 'tmp',
        'ENROOT_CONFIG_PATH': 'config', 'TMPDIR': 'tmp',
    }.items():
        path = root / directory
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    env.update(ENROOT_MAX_PROCESSORS='8', ENROOT_MAX_CONNECTIONS='4',
               ENROOT_TRANSFER_RETRIES='0', ENROOT_CONNECT_TIMEOUT='30',
               ENROOT_TRANSFER_TIMEOUT='1500')
    print(json.dumps({'time': utcnow(), 'image': IMAGE, 'output': str(output),
        'environment': {k: env[k] for k in env if k.startswith('ENROOT_') and k in {
            'ENROOT_CACHE_PATH', 'ENROOT_DATA_PATH', 'ENROOT_RUNTIME_PATH', 'ENROOT_TEMP_PATH',
            'ENROOT_CONFIG_PATH', 'ENROOT_MAX_PROCESSORS', 'ENROOT_MAX_CONNECTIONS',
            'ENROOT_TRANSFER_RETRIES', 'ENROOT_CONNECT_TIMEOUT', 'ENROOT_TRANSFER_TIMEOUT'}}}), flush=True)
    process = subprocess.Popen(['enroot', 'import', '--output', str(output), IMAGE],
                               env=env, cwd=root, start_new_session=True)
    started = time.monotonic()
    with (root / 'import-progress.jsonl').open('x') as log:
        while process.poll() is None:
            free = shutil.disk_usage(root).free
            elapsed = time.monotonic() - started
            log.write(json.dumps({'time': utcnow(), 'monotonic_s': time.monotonic(),
                'pid': process.pid, 'elapsed_s': elapsed, 'free_bytes': free}) + '\n')
            log.flush()
            if free < 200*1024**3 or elapsed > 1800:
                reason = 'free_space_guard' if free < 200*1024**3 else 'walltime_guard'
                print(json.dumps({'stopping_owned_process_group': process.pid, 'reason': reason}), flush=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124 if reason == 'walltime_guard' else 28
            time.sleep(1)
    return process.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--guard-child', action='store_true')
    args = ap.parse_args()
    run = Run(args.run_dir)
    root = run.root / 'images/enroot-import-v1'
    output = root / 'miles-amd64.sqsh.partial'
    if args.guard_child:
        raise SystemExit(guarded_import(root, output))
    phase = run.phase('00-pinned-enroot-image-import')
    if shutil.disk_usage(run.root).free < 300*1024**3:
        phase.finish('fail', failure_summary='Enroot import requires 300 GiB initial free space.', refresh=False)
        return 1
    root.mkdir(parents=True, exist_ok=False)
    phase.command(['enroot', 'version'])
    versions = {name: sha256(name) for name in ['/usr/bin/enroot', '/usr/lib/enroot/docker.sh', '/usr/bin/mksquashfs']}
    code, _, _ = phase.command([sys.executable, str(Path(__file__).resolve()),
                               '--run-dir', str(run.root), '--guard-child'], timeout=1860)
    final = root / 'miles-amd64.sqsh'
    checksum = None
    if not code and output.is_file():
        checksum = sha256(output)
        os.link(output, final)
        output.unlink()
    elif not code:
        code = 1
    metadata = {'image': IMAGE, 'runtime_file_sha256': versions, 'sqsh_sha256': checksum,
        'output': str(final), 'scope': 'Pinned image preparation only; no GPU runtime, policy, or optimizer execution.',
        'artifacts': [str(root.relative_to(run.root))]}
    atomic(root / 'image-manifest.json', metadata)
    phase.finish('fail' if code else 'ok', failure_summary='Run-scoped Enroot import failed; preserve cache and partial image for diagnosis.' if code else None,
        metadata=metadata, results=[metric('sqsh_bytes', final.stat().st_size, 'B')] if checksum else [],
        exit_code=code, refresh=False)
    print(json.dumps({'exit_code': code, 'output': str(final), 'sha256': checksum}), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
