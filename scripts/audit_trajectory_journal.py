"""Reconcile immutable sync rollout receipts without guessing missing work."""


def read_journal(root, train):
    import hashlib
    import json

    directory = train / 'trajectory-journal'
    events, hashes = [], {}
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError('Missing or linked trajectory journal.')
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != '.json':
            raise ValueError('Unexpected or unfinished journal artifact.')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if not path.name.endswith('-' + digest + '.json'):
            raise ValueError('Journal event checksum mismatch.')
        event = json.loads(raw)
        if event.get('schema_version') != 1:
            raise ValueError('Unsupported journal schema.')
        events.append(event)
        hashes[str(path.relative_to(root))] = digest
    if not events:
        raise ValueError('Empty trajectory journal.')
    return events, hashes


def audit_journal(events, episodes, native_samples, trained_indices):
    """Qualify sync accounting only; selected inputs are not optimizer receipts."""
    terminal = {'selected_for_training', 'sync_unused_discarded', 'sync_excess_discarded',
                'dynamic_filter_drop', 'group_cancelled', 'group_failed', 'sync_unused_recycled'}
    by_episode = {row['episode_id']: row for row in episodes}
    native = {row['index']: row for row in native_samples}
    if len(by_episode) != len(episodes) or len(native) != len(native_samples):
        raise ValueError('Duplicate controller or native sample identity.')
    if len(set(trained_indices)) != len(trained_indices):
        raise ValueError('Repeated trainer input identity.')
    attempts, owners, findings, event_counts = {}, {}, [], {}
    for event in events:
        kind = event['event']
        event_counts[kind] = event_counts.get(kind, 0) + 1
        if kind.startswith('async_'):
            findings.append('Async accounting is not qualified by this sync auditor.')
        if kind == 'environment_attempt':
            eid = event['episode_id']
            if eid in owners:
                raise ValueError('Repeated durable environment identity.')
            owners[eid] = event
            continue
        for sample in event.get('samples', []):
            key = (sample['attempt_id'], sample['sample_index'])
            if key[0] is None or key[1] is None:
                raise ValueError('Journal lacks native attempt/sample identity.')
            attempt = attempts.setdefault(key, dict(sample=sample, events={}, episodes=[]))
            if (sample['group_index'], sample['task_id']) != (attempt['sample']['group_index'], attempt['sample']['task_id']):
                raise ValueError('Task or group identity changed within an attempt.')
            attempt['events'].setdefault(kind, []).append(sample)
    for eid, owner in owners.items():
        key = (owner['attempt_id'], owner['sample_index'])
        attempt, episode = attempts.get(key), by_episode.get(eid)
        if attempt is None or episode is None:
            findings.append('Missing sample/controller for environment ' + eid)
            continue
        if owner['task_id'] != episode['task_id'] or owner['task_id'] != attempt['sample']['task_id']:
            findings.append('Task identity mismatch for environment ' + eid)
        if owner['group_index'] != attempt['sample']['group_index']:
            findings.append('Group identity mismatch for environment ' + eid)
        attempt['episodes'].append(eid)
    rows, selected = [], []
    native_fields = ('tokens', 'rollout_log_probs', 'loss_mask', 'response_length', 'reward')
    for (aid, index), attempt in sorted(attempts.items()):
        kinds = attempt['events']
        outcomes = [kind for kind in terminal for _ in kinds.get(kind, [])]
        if len(kinds.get('group_submitted', [])) != 1 or len(outcomes) != 1:
            findings.append(f'Sample {index}/{aid}: missing or ambiguous submission/disposition.')
        disposition = outcomes[0] if len(outcomes) == 1 else 'unresolved'
        returned = kinds.get('group_returned', [])
        if len(returned) > 1:
            findings.append(f'Sample {index}/{aid}: repeated return.')
        if disposition == 'selected_for_training':
            selected.append(index)
            if len(returned) != 1 or index not in native:
                findings.append(f'Sample {index}/{aid}: selected without native return/qualification.')
            else:
                for field in native_fields:
                    if returned[0].get(field) != native[index].get(field):
                        findings.append(f'Sample {index}/{aid}: native {field} changed.')
                if native[index]['metadata'].get('_miles_journal_attempt_id') != aid:
                    findings.append(f'Sample {index}/{aid}: trainer candidate attempt mismatch.')
            if not attempt['episodes']:
                findings.append(f'Sample {index}/{aid}: selected without environment identity.')
        if returned:
            for env in returned[0]['environment_attempts']:
                owner = owners.get(env['episode_id'])
                if owner is None or owner['sample_index'] != index or owner['task_id'] != env['task_id']:
                    findings.append(f'Sample {index}/{aid}: returned environment has no matching durable owner.')
        rows.append(dict(attempt_id=aid, sample_index=index, group_index=attempt['sample']['group_index'],
            task_id=attempt['sample']['task_id'], disposition=disposition,
            environment_ids=sorted(attempt['episodes']), weight_versions=returned[0]['weight_versions'] if returned else None))
    if sorted(selected) != sorted(trained_indices):
        findings.append('Selected sample identities do not exactly match audited trainer inputs.')
    unjoined = sorted(set(by_episode) - set(owners))
    if unjoined:
        findings.append(f'{len(unjoined)} controller episodes lack durable sample identity.')
    dispositions = {kind: sum(row['disposition'] == kind for row in rows) for kind in sorted(terminal | {'unresolved'})}
    return dict(findings=sorted(set(findings)), samples=rows, event_counts=event_counts,
        counts=dict(sample_attempts=len(rows), environment_attempts=len(owners), controller_episodes=len(episodes),
            selected_inputs=len(selected), unjoined_episodes=len(unjoined), dispositions=dispositions),
        unjoined_episode_ids=unjoined,
        scope='Synchronous native attempt/controller/disposition reconciliation. Trainer membership requires the separate tensor audit; optimizer completion and asynchronous accounting are not inferred.')
