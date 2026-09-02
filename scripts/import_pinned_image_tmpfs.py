"""Retry the pinned Enroot import only after proving a private tmpfs overlay.

All paths remain below the current run. The temporary mount exists only in a
new child mount namespace; host and worker mount configuration is unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evidence import Run, atomic, metric, sha256
from import_pinned_image import IMAGE, guarded_import


def mount_temporary(root, probe):
    if os.readlink('/proc/self/ns/mnt') == os.environ['PTX_PARENT_MOUNT_NAMESPACE']:
        raise ValueError('Temporary mount must be in a distinct private mount namespace.')
    temporary = root / 'tmp'
    temporary.mkdir(exist_ok=False)
    subprocess.run(['mount', '-t', 'tmpfs', '-o', 'size=' + ('2G' if probe else '128G') + ',mode=0700,nosuid,nodev',
                    'posttrainingx-enroot-tmp', str(temporary)], check=True)
    info = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE,PROPAGATION', '-T', str(temporary)], text=True)
    if info.split() != ['tmpfs', 'private']:
        raise ValueError('Expected a private temporary filesystem: ' + info)
    print(json.dumps({'temporary_mount': str(temporary), 'type_propagation': info.strip(),
                      'mount_namespace': os.readlink('/proc/self/ns/mnt')}), flush=True)
    return temporary


def overlay_probe(root, temporary):
    for name in ('0', '1', 'rootfs'):
        (temporary / name).mkdir()
    (temporary / '1/test.txt').write_text('base\n')
    (temporary / '0/test.txt').write_text('overlay-success\n')
    output = root / 'overlay-probe.sqsh'
    env = dict(os.environ, MOUNTPOINT=str(temporary / 'rootfs'))
    subprocess.run(['enroot-mksquashovlfs', '0:1', str(output), '-all-root', '-no-progress', '-processors', '2'],
                   cwd=temporary, env=env, check=True, timeout=30)
    text = subprocess.check_output(['unsquashfs', '-cat', str(output), 'test.txt'], text=True, timeout=15)
    if text != 'overlay-success\n':
        raise ValueError('SquashFS content does not respect overlay ordering.')
    print(json.dumps({'probe_sha256': sha256(output), 'overlay_content_verified': True}), flush=True)
    return 0


def child(run, probe):
    root = run.root / 'images' / ('enroot-tmpfs-probe-v1' if probe else 'enroot-import-v2')
    temporary = mount_temporary(root, probe)
    if probe:
        return overlay_probe(root, temporary)
    cache = run.root / 'images/enroot-import-v1/cache'
    return guarded_import(root, root / 'miles-amd64.sqsh.partial', cache_path=cache,
                          temporary_reserve_bytes=8*1024**3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--namespace-child', action='store_true')
    ap.add_argument('--probe', action='store_true')
    args = ap.parse_args()
    run = Run(args.run_dir)
    if args.namespace_child:
        return child(run, args.probe)
    os.environ['PTX_PARENT_MOUNT_NAMESPACE'] = os.readlink('/proc/self/ns/mnt')
    for probe in (True, False):
        name = '00-enroot-private-tmpfs-overlay-probe' if probe else '00-pinned-enroot-tmpfs-image-import'
        phase = run.phase(name)
        root = run.root / 'images' / ('enroot-tmpfs-probe-v1' if probe else 'enroot-import-v2')
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
        if available < 256*1024**3 or shutil.disk_usage(run.root).free < 300*1024**3:
            phase.finish('fail', failure_summary='Import requires 256 GiB available memory and 300 GiB shared free space.', refresh=False)
            return 1
        root.mkdir(parents=True, exist_ok=False)
        command = ['unshare', '--mount', '--propagation', 'private', sys.executable, str(Path(__file__).resolve()),
                   '--run-dir', str(run.root), '--namespace-child'] + (['--probe'] if probe else [])
        rc, _, _ = phase.command(command, timeout=60 if probe else 1860)
        if not probe and not rc:
            partial = root / 'miles-amd64.sqsh.partial'
            final = root / 'miles-amd64.sqsh'
            checksum = sha256(partial)
            os.link(partial, final)
            partial.unlink()
            atomic(root / 'image-manifest.json', {'image': IMAGE, 'sqsh_sha256': checksum,
                    'size_bytes': final.stat().st_size, 'output': str(final),
                    'temporary_filesystem': '128 GiB tmpfs in child-private mount namespace',
                    'source_code_sha256': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('import_pinned_image.py')]}})
        phase.finish('fail' if rc else 'ok', exit_code=rc,
            failure_summary='Private-tmpfs probe/import failed; dependent runtime stages remain stopped.' if rc else None,
            metadata={'image': IMAGE, 'artifacts': [str(root.relative_to(run.root))],
                      'scope': 'Pinned image preparation; no GPU runtime or policy execution.'}, refresh=False)
        if rc:
            return rc
    print(json.dumps({'image_ready': str(final), 'sha256': checksum}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
