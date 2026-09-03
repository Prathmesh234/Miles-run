"""Restore pinned processor metadata in a NEW package; do not requantize weights."""
import argparse
import json
import os
from pathlib import Path

from evidence import atomic, sha256


def repackage(run, source, parent, destination, parent_checksum, frozen_files):
    run, source, parent, destination = [Path(p).resolve() for p in (run, source, parent, destination)]
    if not (run / 'run.json').is_file() or not all(p.is_relative_to(run / 'models') for p in (source, parent, destination)):
        raise ValueError('All packages must remain in the current run models directory.')
    if any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in (source, parent)):
        raise ValueError('Input/output package trees must be disjoint.')
    partial = destination.with_name(destination.name + '.partial')
    if destination.exists() or partial.exists():
        raise ValueError('Destination exists; no implicit repair or overwrite.')
    if sha256(parent / 'checksums.sha256') != parent_checksum:
        raise ValueError('Parent checksum pin differs.')
    if json.loads((parent / 'CONVERSION_COMPLETE.json').read_text())['checksums_sha256'] != parent_checksum:
        raise ValueError('Parent conversion was not complete.')
    extras = ('preprocessor_config.json', 'video_preprocessor_config.json')
    for name in extras:
        if (source / name).is_symlink() or sha256(source / name) != frozen_files[name]:
            raise ValueError('Original processor configuration differs from frozen model.')
    partial.mkdir()
    manifest, linked = {}, 0
    for line in (parent / 'checksums.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or (parent / name).is_symlink():
            raise ValueError('Nonlocal or linked parent artifact rejected.')
        if name.endswith('.safetensors'):
            os.link(parent / name, partial / name)
            linked += 1
        else:
            atomic(partial / name, (parent / name).read_text())
            if sha256(partial / name) != digest:
                raise ValueError('Parent metadata checksum differs: ' + name)
        manifest[name] = digest
    for name in extras:
        if name in manifest:
            raise ValueError('Metadata already exists in parent; this repair is not applicable.')
        atomic(partial / name, (source / name).read_text())
        manifest[name] = sha256(partial / name)
        if manifest[name] != frozen_files[name]:
            raise ValueError('Copied processor metadata changed bytes.')
    record = {'schema_version': 1, 'parent': str(parent.relative_to(run)),
              'parent_checksums_sha256': parent_checksum, 'processor_sha256': {k: manifest[k] for k in extras},
              'hardlinked_weight_shards': linked, 'weights_changed': False, 'optimizer_steps_enabled': False,
              'scope': 'New immutable package; shared weight inodes must never be modified in either package. Serialized audit must pass before serving.'}
    atomic(partial / 'packaging.json', record)
    manifest['packaging.json'] = sha256(partial / 'packaging.json')
    atomic(partial / 'checksums.sha256', ''.join(f'{checksum}  {name}\n' for name, checksum in sorted(manifest.items())))
    atomic(partial / 'CONVERSION_COMPLETE.json', {'status': 'repackaged_not_runtime_qualified',
           'checksums_sha256': sha256(partial / 'checksums.sha256'), 'optimizer_steps_enabled': False})
    os.rename(partial, destination)
    return dict(record, destination=str(destination.relative_to(run)), checksums_sha256=sha256(destination / 'checksums.sha256'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'source', 'parent', 'destination', 'source-manifest', 'parent-checksums-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    frozen = {row['path']: row['sha256'] for row in json.loads(Path(args.source_manifest).read_text())['files']['files']}
    print(json.dumps(repackage(args.run_dir, args.source, args.parent, args.destination, args.parent_checksums_sha256, frozen), indent=2))
