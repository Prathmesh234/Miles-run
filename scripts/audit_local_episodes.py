"""Audit local task events in a terminal job window, without inventing sample joins."""
import argparse
import inspect
import json

from evidence import Run, atomic, metric


def join_episode_ids(episodes, samples, trained_indices):
    """Join native metadata, never task names or coincident timestamps."""
    by_episode = {e['episode_id']: e for e in episodes}
    if len(by_episode) != len(episodes):
        raise ValueError('Duplicate controller episode identity.')
    by_sample, owners, rows, findings = {}, {}, [], []
    trained = set(trained_indices)
    if len(trained) != len(trained_indices):
        raise ValueError('A sample appears in more than one trainer input.')
    for sample in samples:
        index = sample['index']
        if index in by_sample:
            raise ValueError('Duplicate native qualification sample identity.')
        by_sample[index] = sample
        attempts = sample['metadata'].get('posttrainingx_environment_attempts', [])
        if not attempts:
            findings.append(f'Sample {index}: no durable controller episode identity.')
        for position, attempt in enumerate(attempts):
            eid = attempt['episode_id']
            if eid in owners:
                raise ValueError('Controller episode assigned to multiple samples or attempts.')
            owners[eid] = index
            episode = by_episode.get(eid)
            if episode is None:
                findings.append(f'Sample {index}: controller episode {eid} is missing.')
                continue
            if episode['task_id'] != attempt['task_id'] or attempt['task_id'] != sample['metadata']['task_id']:
                findings.append(f'Sample {index}: controller task identity differs.')
            final = position == len(attempts) - 1
            if final and episode['category'] == 'graded' and episode['reward'] != sample['reward']:
                findings.append(f'Sample {index}: canonical reward differs from training reward.')
            if final and episode['category'] != 'graded':
                timeout = sample['metadata'].get('exit_status') == 'timeout' and sample['reward'] == 0.0
                if not timeout:
                    findings.append(f'Sample {index}: final environment attempt has no canonical verdict or explicit timeout.')
            rows.append(dict(sample_index=index, group_index=sample['group_index'], episode_id=eid,
                task_id=episode['task_id'], attempt=position, final_attempt=final,
                disposition='trained_input' if index in trained else 'qualified_not_trained',
                category=episode['category'], reward=episode['reward'], duration_s=episode['event_span_s']))
    missing = sorted(trained - by_sample.keys())
    unjoined = sorted(by_episode.keys() - owners.keys())
    if missing:
        findings.append('Trainer sample IDs absent from native qualification evidence: ' + str(missing))
    if unjoined:
        findings.append(f'{len(unjoined)} controller episodes have no native sample join; disposition must not be guessed.')
    return dict(findings=findings, rows=rows, unjoined_episode_ids=unjoined,
        missing_trainer_sample_ids=missing, counts=dict(controller_episodes=len(episodes),
        native_samples=len(samples), trained_inputs=len(trained), qualified_not_trained=len(by_sample.keys() - trained),
        joined_episodes=len(rows), unjoined_episodes=len(unjoined),
        retry_attempts=sum(not row['final_attempt'] for row in rows)),
        scope='Exact native sample-to-controller IDs and trainer-input membership. Unjoined episodes are unresolved, not assumed cancellations or stale drops.')


def summarize_episode(rows):
    counts = {}
    for row in rows:
        counts[row['event']] = counts.get(row['event'], 0) + 1
    verdicts = [r for r in rows if r['event'] == 'graded']
    findings = []
    if len(verdicts) > 1 or any(v['reward'] not in (0.0, 1.0) for v in verdicts):
        findings.append('Invalid or repeated canonical verdict.')
    if verdicts:
        stopped = [i for i, r in enumerate(rows) if r['event'] == 'policy_stopped']
        staged = [i for i, r in enumerate(rows) if r['event'] == 'grader_assets_staged']
        if len(stopped) != 1 or len(staged) != 1 or stopped[0] >= staged[0]:
            findings.append('No unique policy-stop-before-grader-stage boundary.')
        elif any(r['event'] == 'command_finished' and not r.get('grading') for r in rows[stopped[0]+1:]):
            findings.append('Policy command completed after policy was sealed.')
    if counts.get('workspace_volume_created', 0) != counts.get('workspace_volume_removed', 0):
        findings.append('Workspace volume lifecycle does not balance.')
    category = ('graded' if verdicts else 'grading_error' if counts.get('grading_error') else 'no_verdict')
    return {'episode_id': rows[0]['episode_id'], 'task_id': rows[0]['task_id'], 'category': category,
            'reward': verdicts[0]['reward'] if len(verdicts) == 1 else None,
            'started_at': rows[0]['time'], 'last_event_at': rows[-1]['time'],
            'event_span_s': rows[-1]['monotonic_s'] - rows[0]['monotonic_s'],
            'policy_commands_completed': sum(r['event'] == 'command_finished' and not r.get('grading') for r in rows),
            'events': counts, 'findings': findings}


def collect(root, attempt, trained_indices=None):
    import hashlib
    import json
    from pathlib import Path

    root = Path(root)
    train = root / f'training/sync-grpo-v{attempt}'
    started = json.loads((train / 'training-command.json').read_text())['time']
    ended = json.loads((train / 'driver.finished.json').read_text())['time']
    episodes, source_hashes = [], {}
    for path in sorted((root / 'environments/local-file-runtime-v3/episodes').glob('*/events.jsonl')):
        if path.is_symlink():
            raise ValueError('Linked episode evidence refused.')
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if not rows or rows[0]['purpose'] != 'policy' or not started <= rows[0]['time'] <= ended:
            continue
        row = summarize_episode(rows)
        row['events_path'] = str(path.relative_to(root))
        source_hashes[row['events_path']] = hashlib.sha256(raw).hexdigest()
        episodes.append(row)
    counts = {'created': len(episodes), 'graded': sum(e['category'] == 'graded' for e in episodes),
              'passed': sum(e['reward'] == 1.0 for e in episodes),
              'failed_task': sum(e['reward'] == 0.0 for e in episodes),
              'grading_error': sum(e['category'] == 'grading_error' for e in episodes),
              'no_verdict': sum(e['category'] == 'no_verdict' for e in episodes)}
    result = {'scope': 'Controller-side policy episode events selected by driver time window; not an exact Miles sample-ID join or trained-trajectory count.',
            'selection_window': {'start': started, 'end': ended}, 'counts': counts, 'episodes': episodes,
            'source_sha256': source_hashes, 'findings': [f"{e['episode_id']}: {f}" for e in episodes for f in e['findings']],
            'notes': ['No-verdict episodes are not counted as reward zero.',
                      'Task rewards here are clean training-task outcomes, not held-out TB2.1 quality.',
                      'Event-span duration includes environment cleanup, not just policy inference.',
                      'Persisted controller events can outnumber fire-and-forget dashboard completion events after a crash.']}
    if trained_indices is not None:
        samples = []
        for path in sorted((train / 'dump_details/qualification-groups').glob('*.json')):
            raw = path.read_bytes()
            source_hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            samples.extend(json.loads(raw)['samples'])
        result['identity_join'] = join_episode_ids(episodes, samples, trained_indices)
        result['findings'].extend(result['identity_join']['findings'])
        result['scope'] = 'Controller isolation/lifecycle audit plus explicit native sample-ID join; full asynchronous trajectory accounting is not implied.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--tensor-audit', help='Run-relative successful tensor audit result for exact trainer input membership.')
    parser.add_argument('--join-attempt', type=int, default=1)
    args = parser.parse_args()
    run = Run(args.run_dir)
    suffix = f'-join-v{args.join_attempt}' if args.tensor_audit else ''
    phase = run.phase(f'02-sync-grpo-environment-audit-v{args.attempt}' + suffix)
    trained_indices = None
    if args.tensor_audit:
        from pathlib import Path
        path = Path(args.tensor_audit)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Tensor audit must be relative to the current run.')
        proof = json.loads((run.root / path).read_text())
        if proof['findings']:
            raise ValueError('A failed tensor audit cannot establish trainer membership.')
        trained_indices = [sample['sample_index'] for batch in proof['batches'] for sample in batch['samples']]
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    program = inspect.getsource(summarize_episode) + '\n' + inspect.getsource(join_episode_ids) + '\n' + inspect.getsource(collect)
    program += '\nimport json,sys\nprint(json.dumps(collect(sys.argv[1],int(sys.argv[2]),json.loads(sys.argv[3]))))\n'
    atomic(phase.path / 'audit-remote.py', program)
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program, remote, str(args.attempt), json.dumps(trained_indices)], timeout=45)
    data = json.loads(out) if not rc else {'findings': ['Controller event audit failed; see raw evidence.']}
    atomic(phase.path / 'episodes.json', data)
    values = [metric('environment_' + key, value, 'count') for key, value in data.get('counts', {}).items()]
    phase.finish('fail' if data['findings'] else 'ok', results=values, metadata=data,
                 failure_summary='; '.join(data['findings']) or None, refresh=False)
    print(json.dumps({'counts': data.get('counts'), 'findings': data['findings']}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
