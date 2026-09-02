"""Read-only structural audit of the complete prospective clean task split.

No task is replaced or selected using rewards. This is not runtime eligibility:
the known-good solution and isolated grading must still pass for every task.
"""
import argparse
import inspect
import json
from pathlib import Path

from evidence import Run, atomic
from prepare_local_task_images import BASE_IMAGES, MANIFEST_SHA, offline_harness


def inspect_corpus(source, split, expected_manifest, bases):
    import collections
    import hashlib
    import json
    from pathlib import Path
    import re

    source = Path(source)
    manifest_path = source / 'files.manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest:
        raise ValueError('Pinned materialization manifest changed.')
    manifest = json.loads(manifest_path.read_text())
    roles = {'training': split['training_task_ids'], 'development': split['development_task_ids'],
             'runtime_validation': split['runtime_validation_task_ids']}
    ids = [task for tasks in roles.values() for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('Clean split roles overlap.')
    files_by_task = collections.defaultdict(list)
    for row in manifest['files']:
        task = Path(row['path']).parts[0]
        if task not in ids:
            raise ValueError('Unexpected task in frozen materialization.')
        path = source / 'source' / row['path']
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']):
            raise ValueError('Task artifact changed: ' + row['path'])
        files_by_task[task].append(row)
    tasks, requirements = [], collections.Counter()
    for role, task_ids in roles.items():
        for task in task_ids:
            root = source / 'source' / task
            dockerfile = root / 'environment/Dockerfile'
            harness = root / 'tests/test.sh'
            needs = []
            if not dockerfile.is_file() or not harness.is_file():
                needs.append('missing_dockerfile_or_harness')
            text = dockerfile.read_text() if dockerfile.is_file() else ''
            base_tags = re.findall(r'^FROM ([^\s]+)\s*$', text, re.M)
            if len(base_tags) != 1 or base_tags[0] not in bases:
                needs.append('additional_base_pin_or_multistage_review')
            if not (root / 'environment/task_file').is_dir():
                needs.append('missing_public_task_directory_requires_review')
            try:
                derived = offline_harness(harness.read_text())
                derived_hash = hashlib.sha256(derived.encode()).hexdigest()
            except (ValueError, OSError):
                needs.append('offline_harness_setup_requires_review')
                derived_hash = None
            environment_paths = [x['path'] for x in files_by_task[task]
                                 if Path(x['path']).parts[1] == 'environment']
            if any(part in ('tests', 'solution') for p in environment_paths for part in Path(p).parts[2:]):
                needs.append('forbidden_asset_name_in_policy_context')
            requirements.update(needs)
            tasks.append({'task_id': task, 'role': role, 'source_files': len(files_by_task[task]),
                          'source_bytes': sum(x['bytes'] for x in files_by_task[task]),
                          'base_tags': base_tags, 'environment_file_count': len(environment_paths),
                          'workdir_lines': re.findall(r'^WORKDIR\s+(.+)$', text, re.M),
                          'copy_lines': re.findall(r'^COPY\s+(.+)$', text, re.M),
                          'offline_harness_candidate_sha256': derived_hash,
                          'structural_review_requirements': needs,
                          'live_runtime_qualified_by_this_audit': False})
    return {'schema_version': 1, 'source_manifest_sha256': expected_manifest,
            'taskset_revision': split['revision'], 'ordered_training_ids': roles['training'],
            'task_count_by_role': {k: len(v) for k, v in roles.items()},
            'review_requirement_counts': dict(requirements), 'tasks': tasks, 'findings': [],
            'scope': 'All pinned source hashes and structural requirements, not image/runtime qualification. No selection, source mutation, solution execution, or policy outcome inspection.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-clean-corpus-structural-audit-v{args.attempt}')
    split = json.loads((Path(__file__).resolve().parents[1] / 'locks/terminal-lego-subset.json').read_text())
    program = 'import re\n' + inspect.getsource(offline_harness) + '\n' + inspect.getsource(inspect_corpus)
    program += '\nimport json,sys\nprint(json.dumps(inspect_corpus(sys.argv[1],json.loads(sys.argv[2]),sys.argv[3],json.loads(sys.argv[4]))))\n'
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program,
        remote + '/environments/terminal-lego-controller-v3', json.dumps(split), MANIFEST_SHA,
        json.dumps(BASE_IMAGES)], timeout=120)
    result = json.loads(out) if not rc else {'findings': ['Read-only source audit failed; inspect retained stderr.']}
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if result['findings'] else 'ok', metadata=result,
                 failure_summary='; '.join(result['findings']) or None, refresh=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('tasks', 'ordered_training_ids')}))
    return int(bool(result['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
