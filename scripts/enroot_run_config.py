"""Copy and minimally patch site Enroot settings inside one immutable run scope."""
import difflib
import os
from pathlib import Path
import shutil

from evidence import atomic, sha256


def prepare(root, source=Path('/etc/enroot')):
    target = root / 'site-config'
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Unexpected symlink in Enroot site configuration.')
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.name != 'enroot.conf' and path.suffix not in ('.sh', '.fstab', '.env'):
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = sha256(path)
    hook = target / 'hooks.d/10-devices.sh'
    before = hook.read_text()
    lines = before.splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if line.split()[:2] == ['/dev/log', '/dev/log']]
    if len(indices) != 1 or 'nofail' in lines[indices[0]]:
        raise ValueError('Device hook differs from the inspected required /dev/log mount.')
    i = indices[0]
    lines[i] = lines[i].rstrip('\n') + ',nofail,silent\n'
    after = ''.join(lines)
    atomic(hook, after)
    hook.chmod(0o755)
    atomic(root / 'enroot-dev-log.patch', ''.join(difflib.unified_diff(before.splitlines(keepends=True),
           after.splitlines(keepends=True), fromfile='site/hooks.d/10-devices.sh', tofile='run/hooks.d/10-devices.sh')))
    atomic(root / 'enroot-site-config.json', {'source_sha256': hashes, 'patched_hook_sha256': sha256(hook),
           'patch_reason': 'Optional /dev/log does not exist in the worker container; retain restricted devices and IPC isolation.',
           'system_configuration_modified': False})
    env = os.environ.copy()
    for key, name in {'ENROOT_SYSCONF_PATH': 'site-config', 'ENROOT_CONFIG_PATH': 'user-config',
        'ENROOT_RUNTIME_PATH': 'runtime', 'ENROOT_DATA_PATH': 'data', 'ENROOT_CACHE_PATH': 'cache',
        'ENROOT_TEMP_PATH': 'tmp'}.items():
        (root / name).mkdir(exist_ok=True)
        env[key] = str(root / name)
    env['ENROOT_MOUNT_HOME'] = 'n'
    return env
