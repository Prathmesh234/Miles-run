"""Fetch the selected clean tasks into a controller-only, run-owned checkout.

This does not build images, run reference solutions, expose grader assets to a
policy, or declare runtime eligibility. Every selected file is hashed.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import stat
import tomllib
import traceback

from evidence import Run, atomic, sha256


def configure_git_environment():
    # GIT_CURL_VERBOSE is enabled by presence, even when set to the string "0".
    # Remove diagnostic flags instead of risking verbose HTTP header logs.
    for name in ('GIT_TRACE', 'GIT_TRACE_PACKET', 'GIT_TRACE_CURL', 'GIT_CURL_VERBOSE'):
        os.environ.pop(name, None)
    os.environ.update({'GIT_TERMINAL_PROMPT': '0', 'GIT_LFS_SKIP_SMUDGE': '1'})


def inventory_tasks(root, task_ids):
    tasks, files, categories = [], [], collections.Counter()
    total = 0
    for task_id in task_ids:
        task = root / task_id
        required = ('instruction.md', 'task.toml', 'environment/Dockerfile', 'tests/test.sh', 'solution/solve.sh')
        for name in required:
            if not (task / name).is_file() or (task / name).is_symlink():
                raise ValueError(f'Missing regular task component: {task_id}/{name}')
        config = tomllib.loads((task / 'task.toml').read_text())
        metadata = config.get('metadata', {})
        category = metadata.get('category', 'unspecified')
        categories[category] += 1
        task_files = []
        for path in sorted(task.rglob('*')):
            info = path.lstat()
            if path.is_symlink() or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError('Linked or special task asset needs explicit review: ' + str(path.relative_to(root)))
            if path.is_dir():
                continue
            if info.st_nlink != 1:
                raise ValueError('Hard-linked task asset needs review.')
            with path.open('rb') as handle:
                if handle.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise ValueError('Selected LFS asset is not materialized: ' + str(path.relative_to(root)))
            total += info.st_size
            if total > 2 * 1024**3:
                raise ValueError('Selected task payload exceeds its 2 GiB preparation budget.')
            record = {'path': str(path.relative_to(root)), 'bytes': info.st_size, 'sha256': sha256(path),
                      'access': 'grader_only' if path.relative_to(task).parts[0] in ('tests', 'solution') else 'task_source'}
            files.append(record)
            task_files.append(record['path'])
        tasks.append({'task_id': task_id, 'category': category, 'difficulty': metadata.get('difficulty'),
                      'tags': metadata.get('tags', []), 'files': task_files,
                      'agent_timeout_s': config.get('agent', {}).get('timeout_sec'),
                      'verifier_timeout_s': config.get('verifier', {}).get('timeout_sec'),
                      'environment_requirements': config.get('environment', {}),
                      'runtime_eligibility': 'not_yet_validated'})
    return {'files': files, 'tasks': tasks, 'categories': dict(categories), 'payload_bytes': total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, choices=range(2, 10), default=2)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'00-terminal-lego-controller-task-materialization-v{args.attempt}')
    split = json.loads((Path(__file__).parent / 'terminal-lego-subset.json').read_text())
    task_ids = split['runtime_validation_task_ids'] + split['training_task_ids'] + split['development_task_ids']
    parent = run.root / f'environments/terminal-lego-controller-v{args.attempt}'
    root = parent / 'source'
    errors, data = [], {}
    try:
        if len(set(task_ids)) != len(task_ids):
            raise ValueError('Task roles overlap.')
        if shutil.disk_usage(run.root).free < 512 * 1024**3:
            raise ValueError('Require a 512 GiB shared-filesystem reserve before task materialization.')
        parent.mkdir(parents=True, mode=0o700, exist_ok=False)
        configure_git_environment()
        git = ['git', '-c', 'credential.helper=', '-c', 'core.hooksPath=/dev/null']
        commands = [git + ['init', str(root)],
                    git + ['-C', str(root), 'remote', 'add', 'origin', 'https://huggingface.co/datasets/' + split['repository']],
                    git + ['-C', str(root), 'sparse-checkout', 'init', '--no-cone'],
                    git + ['-C', str(root), 'sparse-checkout', 'set', '--no-cone', '--stdin'],
                    # Avoid the failed bulk promisor-object request. Keep the same
                    # exact revision and sparse working tree, with a complete shallow pack.
                    ['timeout', '--kill-after=10s', '600s', *git, '-C', str(root), 'fetch', '--depth=1', 'origin', split['revision']],
                    ['timeout', '--kill-after=10s', '600s', *git, '-C', str(root), 'checkout', '--detach', 'FETCH_HEAD']]
        for command in commands:
            if shutil.disk_usage(run.root).free < 256 * 1024**3:
                raise ValueError('Shared filesystem reserve fell below 256 GiB.')
            stdin = ''.join('/' + task_id + '/\n' for task_id in task_ids) if command[-1] == '--stdin' else None
            rc, _, _ = phase.command(command, stdin=stdin, timeout=630)
            if rc:
                raise RuntimeError('Pinned task checkout command failed; preserve partial source and stop.')
        rc, revision, _ = phase.command(git + ['-C', str(root), 'rev-parse', 'HEAD'])
        if rc or revision.strip() != split['revision']:
            raise ValueError('Materialized taskset revision differs from the pinned split.')
        data = inventory_tasks(root, task_ids)
        data.update(schema_version=1, repository=split['repository'], revision=split['revision'],
                    split_sha256=sha256(Path(__file__).parent / 'terminal-lego-subset.json'),
                    source_relpath=str(root.relative_to(run.root)), task_count=len(task_ids),
                    scope='Controller-only clean task sources. No images, references, policy trajectories or optimizer steps executed.')
        atomic(parent / 'files.manifest.json', data)
        atomic(parent / 'checksums.sha256', ''.join(f"{f['sha256']}  source/{f['path']}\n" for f in data['files']))
    except Exception as exc:
        errors.append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None,
        metadata={'findings': errors, 'task_count': data.get('task_count'), 'payload_bytes': data.get('payload_bytes'),
                  'categories': data.get('categories'), 'manifest': str((parent / 'files.manifest.json').relative_to(run.root)),
                  'scope': 'Task source preparation only; not environment correctness or training.'}, refresh=False)
    print(json.dumps({'findings': errors, 'task_count': data.get('task_count'),
                      'payload_bytes': data.get('payload_bytes'), 'categories': data.get('categories')}), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
