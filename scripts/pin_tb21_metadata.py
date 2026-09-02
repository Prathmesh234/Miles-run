"""Freeze evaluation-only TB2.1 task IDs without fetching tests or solutions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from evidence import Run, atomic


REPOSITORY = 'harbor-framework/terminal-bench-2-1'


def api(phase, endpoint):
    rc, out, _ = phase.command(['gh', 'api', 'repos/' + REPOSITORY + '/' + endpoint], timeout=45)
    if rc:
        raise RuntimeError('GitHub metadata read failed: ' + endpoint)
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('00-terminal-bench-21-evaluation-metadata')
    try:
        commit = api(phase, 'commits/main')
        tree = api(phase, 'git/trees/' + commit['commit']['tree']['sha'])
        tasks = [x for x in tree['tree'] if x['path'] == 'tasks' and x['type'] == 'tree']
        if len(tasks) != 1 or tree.get('truncated'):
            raise ValueError('Unexpected benchmark repository layout.')
        subtree = api(phase, 'git/trees/' + tasks[0]['sha'])
        entries = sorted((x for x in subtree['tree'] if x['type'] == 'tree'), key=lambda x: x['path'])
        if len(entries) != 89 or subtree.get('truncated'):
            raise ValueError('TB2.1 metadata does not contain the expected 89 complete task directories.')
        ids = [x['path'] for x in entries]
        data = {'schema_version': 1, 'benchmark': 'Terminal-Bench 2.1', 'role': 'evaluation_only',
                'repository': 'https://github.com/' + REPOSITORY, 'git_sha': commit['sha'],
                'git_tree_sha': commit['commit']['tree']['sha'], 'tasks_tree_sha': tasks[0]['sha'],
                'ordered_task_ids': ids,
                'ordered_task_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                'task_trees': {x['path']: x['sha'] for x in entries}, 'task_count': len(ids),
                'harness_version': 'harbor==0.21.0', 'harness_compatibility_tested': False,
                'policy_data_exposure': 'No task instructions, tests, solutions, or oracle files fetched by this metadata stage.',
                'training_forbidden': True,
                'version_selection': 'Explicit 2.1 repository; do not follow the generic documentation redirect to Terminal-Bench 4.0.'}
        atomic(phase.path / 'taskset.metadata.json', data)
        lock = Path(__file__).resolve().parents[1] / 'locks/terminal-bench-2.1.metadata.json'
        if lock.exists():
            raise ValueError('Evaluation taskset lock already exists; refuse silent repinning.')
        atomic(lock, data)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        phase.finish('fail', failure_summary=str(exc))
        return 1
    phase.finish('ok', metadata={'task_count': len(ids), 'git_sha': commit['sha'],
        'scope': 'Evaluation task IDs and git trees only; no baseline outcomes examined and no environment executed.',
        'artifacts': [str((phase.path / 'taskset.metadata.json').relative_to(run.root))]})
    print(json.dumps({'task_count': len(ids), 'git_sha': commit['sha'], 'ordered_ids_sha256': data['ordered_task_ids_sha256']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
