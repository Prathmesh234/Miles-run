"""Complete selected, pinned task assets without changing failed source attempts.

Only the controller can read this directory. No task image, reference solution,
test, policy, or optimizer is executed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading

from evidence import Run, atomic, sha256
from materialize_terminal_lego import configure_git_environment, inventory_tasks
from pull_pinned_model import download_file


def tracked_files(text, task_ids):
    records = []
    seen = set()
    for record in text.rstrip('\0').split('\0'):
        header, name = record.split('\t', 1)
        mode, kind, checksum = header.split()
        path = PurePosixPath(name)
        if (mode not in ('100644', '100755') or kind != 'blob'
                or not re.fullmatch('[0-9a-f]{40}', checksum)
                or path.is_absolute() or '..' in path.parts
                or len(path.parts) < 2 or path.parts[0] not in task_ids
                or name in seen):
            raise ValueError('Unexpected tracked task asset.')
        seen.add(name)
        records.append({'path': name, 'git_blob_sha1': checksum, 'mode': int(mode, 8) & 0o777})
    return records


def lfs_item(payload, name):
    if not payload.startswith(b'version https://git-lfs.github.com/spec/v1'):
        return None
    match = re.fullmatch(
        rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n', payload)
    if not match or int(match[2]) > 64 * 1024**2:
        raise ValueError('Nonstandard or over-budget LFS task asset.')
    return {'path': name, 'size': int(match[2]), 'lfs': {'sha256': match[1].decode()}}


def copy_verified_sources(source, destination, records):
    pending = []
    total = 0
    for record in records:
        origin = source / record['path']
        if origin.is_symlink() or not origin.is_file() or origin.stat().st_nlink != 1:
            raise ValueError('Linked or missing source task asset.')
        if any(p.is_symlink() for p in origin.parents):
            raise ValueError('Symlink in source parents.')
        total += origin.stat().st_size
        if total > 2 * 1024**3 or origin.stat().st_size > 64 * 1024**2:
            raise ValueError('Selected sources exceed the preparation budget.')
        payload = origin.read_bytes()
        blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        if blob != record['git_blob_sha1']:
            raise ValueError('Source differs from the pinned Git tree: ' + record['path'])
        item = lfs_item(payload, record['path'])
        if item:
            pending.append(item)
            continue
        target = destination / record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(str(target) + '.partial')
        with partial.open('xb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(partial, record['mode'])
        os.link(partial, target)
        partial.unlink()
    if total + sum(x['size'] for x in pending) > 2 * 1024**3:
        raise ValueError('Materialized sources exceed the preparation budget.')
    return pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('00-terminal-lego-controller-task-materialization-v3')
    split_path = Path(__file__).parent / 'terminal-lego-subset.json'
    split = json.loads(split_path.read_text())
    task_ids = split['runtime_validation_task_ids'] + split['training_task_ids'] + split['development_task_ids']
    source = run.root / 'environments/terminal-lego-controller-v2/source'
    parent = run.root / 'environments/terminal-lego-controller-v3'
    destination = parent / 'source'
    findings, data, downloads = [], {}, []
    try:
        if (split['repository'] != 'Lego-X/Terminal-Lego-15k'
                or split['revision'] != '9c197f1c2e87b64cc316b1a5bfcef57b584929f0'
                or len(set(task_ids)) != len(task_ids)
                or any(not re.fullmatch('task_[0-9]{5}', name) for name in task_ids)):
            raise ValueError('Unexpected taskset or split.')
        if shutil.disk_usage(run.root).free < 512 * 1024**3:
            raise ValueError('Require a 512 GiB shared-filesystem reserve.')
        configure_git_environment()
        git = ['git', '-c', 'credential.helper=', '-c', 'core.hooksPath=/dev/null', '-C', str(source)]
        rc, revision, _ = phase.command(git + ['rev-parse', 'HEAD'])
        if rc or revision.strip() != split['revision']:
            raise ValueError('Source revision mismatch.')
        rc, tree, _ = phase.command(git + ['ls-tree', '-r', '-z', '--full-tree', 'HEAD', '--', *task_ids])
        if rc:
            raise ValueError('Cannot inspect the pinned task tree.')
        records = tracked_files(tree, set(task_ids))
        parent.mkdir(mode=0o700, exist_ok=False)
        destination.mkdir()
        pending = copy_verified_sources(source, destination, records)
        atomic(phase.path / 'lfs-pointers.json', pending)
        base = f"https://huggingface.co/datasets/{split['repository']}/resolve/{split['revision']}"
        for item in pending:
            if shutil.disk_usage(run.root).free < 256 * 1024**3:
                raise ValueError('Shared-filesystem reserve fell below 256 GiB.')
            try:
                result = download_file(item, base, destination, threading.Event())
            except Exception as exc:
                # Never serialize CDN exception URLs: they can carry signed tokens.
                failure = {'path': item['path'], 'error_type': type(exc).__name__,
                           'http_status': getattr(exc, 'code', None)}
                atomic(phase.path / 'download-error.json', failure)
                raise ValueError('LFS fetch failed; see sanitized download-error.json.') from None
            downloads.append(result)
            atomic(phase.path / 'downloads.json', downloads)
        data = inventory_tasks(destination, task_ids)
        data.update(schema_version=1, repository=split['repository'], revision=split['revision'],
                    split_sha256=sha256(split_path), task_count=len(task_ids),
                    source_relpath=str(destination.relative_to(run.root)),
                    prior_source_relpath=str(source.relative_to(run.root)),
                    git_tree_verification='Every selected source file matches its pinned Git blob before LFS materialization.',
                    scope='Controller-only task sources; no task execution or runtime eligibility claim.')
        atomic(parent / 'files.manifest.json', data)
        atomic(parent / 'checksums.sha256', ''.join(f"{f['sha256']}  source/{f['path']}\n" for f in data['files']))
    except Exception as exc:
        findings.append(str(exc))
    metadata = {'findings': findings, 'task_count': data.get('task_count'),
                'payload_bytes': data.get('payload_bytes'), 'categories': data.get('categories'),
                'lfs_files_verified': len(downloads), 'prior_attempts_unchanged': True,
                'manifest': str((parent / 'files.manifest.json').relative_to(run.root)),
                'artifacts': [str(parent.relative_to(run.root))],
                'scope': 'Task source acquisition only; no images, references, trajectories or optimizer steps.'}
    phase.finish('fail' if findings else 'ok', failure_summary='; '.join(findings) or None,
                 metadata=metadata, refresh=False)
    print(json.dumps(metadata), flush=True)
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
