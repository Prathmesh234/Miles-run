"""Operator-authorized, run-specific checkpoint pruning; no recursive deletion."""
import argparse
import inspect
import json

from evidence import Run, atomic, metric


def prune(root_text, apply=False, attempt=1, profile='original'):
    import datetime
    import hashlib
    import json
    import os
    from pathlib import Path
    import shutil
    import stat
    import subprocess
    import tempfile

    root = Path(root_text)
    expected = Path('/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210')
    if root != expected or not (root / 'run.json').is_file():
        raise ValueError('Authorization is restricted to the current dummy run.')
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink():
            raise ValueError('Run path contains a symlink.')
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError('Invalid evidence attempt.')
    evidence = root / f'tests/02-checkpoint-retention-v{attempt}'
    keep = 'training/sync-grpo-v9/checkpoints/iter_0000002'
    expected_hashes = {
        'training/sync-grpo-v6/checkpoints/iter_0000000': '936f3e46762bfd3c264d4019f65cc4b2276b8cd7cb621690dc27ee243aacc0bf',
        'training/sync-grpo-v6/checkpoints/iter_0000001': 'f6493b9a4a1728b9aee180bab4a83efe0c6e1b6c3834854f95e1ce91b022540e',
        'training/sync-grpo-v9/checkpoints/iter_0000000': '77ba87b7319a71aa2a70fd2cc3b768e041d85f2786596716b6a4f476a7156139',
        'training/sync-grpo-v9/checkpoints/iter_0000001': 'b49c6d59804631c3ab7f6c5dcc79604592fd3941f9259add6d6df36bb5048662',
        keep: 'd2129901a8ad1d1fbc4e9e6f606df8971d5ea0afdc440cef13c5a5d7dbfc0651',
    }
    retained = [keep]
    pointer_values = {'training/sync-grpo-v9/checkpoints/latest_checkpointed_iteration.txt': '2',
                      'training/sync-grpo-v6/checkpoints/latest_checkpointed_iteration.txt': '1'}
    obsolete_pointers = ['training/sync-grpo-v6/checkpoints/latest_checkpointed_iteration.txt']
    source_jobs = {'139': ['FAILED', '1:0'], '143': ['COMPLETED', '0:0']}
    if profile == 'post-job161':
        # Read-only inventory: 02-checkpoint-retention-followup-inventory-v1.
        # Keep job154's A/B resume pair and both job161 checkpoints.
        keep = 'training/sync-grpo-v12/checkpoints/iter_0000001'
        expected_hashes = {
            'training/sync-grpo-v9/checkpoints/iter_0000002': 'd2129901a8ad1d1fbc4e9e6f606df8971d5ea0afdc440cef13c5a5d7dbfc0651',
            'training/sync-grpo-v10/checkpoints/iter_0000000': '4a2d5e15a30a71c8e1f2014e688f9e9905ff6749d478c4652849ec9391bd6e29',
            'training/sync-grpo-v10/checkpoints/iter_0000001': '615a4beb9ce67cc390fddc5ae852212707fdae2b03a851ef448fe38fe697bd54',
            'training/sync-grpo-v10/checkpoints/iter_0000002': '42d217369d2ca3e2dd33517a690af2fe575f4fba1db38ea06eae5b970f7cab75',
            'training/sync-grpo-v12/checkpoints/iter_0000000': '626e5b8936a75847a41ca7cc6a7b0b8e245dda039f59237e8266b0eb3973e3b7',
            keep: 'ae80d8caba7f048ed20b52f42723788aacbb4b2712221677afbba34b3934d268',
        }
        retained = list(expected_hashes)[2:]
        pointer_values = {'training/sync-grpo-v9/checkpoints/latest_checkpointed_iteration.txt': '2',
                          'training/sync-grpo-v10/checkpoints/latest_checkpointed_iteration.txt': '2',
                          'training/sync-grpo-v12/checkpoints/latest_checkpointed_iteration.txt': '1'}
        obsolete_pointers = ['training/sync-grpo-v9/checkpoints/latest_checkpointed_iteration.txt']
        source_jobs = {'143': ['COMPLETED', '0:0'], '154': ['FAILED', '1:0'], '161': ['FAILED', '1:0']}
    elif profile != 'original':
        raise ValueError('Unknown explicit retention profile.')
    payload_names = {f'__{rank}_0.distcp' for rank in range(16)}
    small_names = {'.metadata', 'metadata.json', 'debug_events/actor_cell0_rank0.jsonl',
                   'debug_events/rollout_manager.jsonl'}

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
        with os.fdopen(fd, 'w') as handle:
            if isinstance(value, str):
                handle.write(value)
            else:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def inventory(relative):
        directory = root / relative
        for component in (directory, *directory.parents):
            if component.is_symlink():
                raise ValueError('Checkpoint path contains a symlink.')
            if component == root:
                break
        rows = []
        for path in sorted(directory.rglob('*')):
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise ValueError('Checkpoint contains a symlink.')
            if stat.S_ISDIR(st.st_mode):
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise ValueError('Checkpoint contains a special file or hard link.')
            name = str(path.relative_to(directory))
            if st.st_size <= 0 or (name in small_names and st.st_size > 64 * 1024**2):
                raise ValueError('Unexpected checkpoint file size.')
            rows.append({'path': name, 'bytes': st.st_size, 'inode': st.st_ino,
                         'mtime_ns': st.st_mtime_ns,
                         'sha256': sha(path) if name in small_names else None})
        if {row['path'] for row in rows} != payload_names | small_names:
            raise ValueError('Checkpoint file set differs from inspected 20-file layout.')
        if sha(directory / '.metadata') != expected_hashes[relative]:
            raise ValueError('Checkpoint metadata changed since inspection.')
        return rows

    def scheduler_gate():
        queue = subprocess.check_output(['squeue', '-h', '-o', '%i|%T|%N'], text=True)
        if queue.strip():
            raise RuntimeError('Refusing to prune while scheduler jobs are active.')
        jobs = subprocess.check_output(['sacct', '-j', ','.join(source_jobs), '-X', '-n', '-P',
                                        '-o', 'JobID,State,ExitCode'], text=True)
        states = {line.split('|')[0]: line.split('|')[1:3] for line in jobs.splitlines() if line.strip()}
        if any(states.get(job) != value for job, value in source_jobs.items()):
            raise RuntimeError('Expected completed/failed source jobs were not found.')
        return {'squeue': queue, 'sacct': jobs}

    scheduler = scheduler_gate()
    for relative, value in pointer_values.items():
        path = root / relative
        if path.is_symlink() or path.read_text().strip() != value:
            raise ValueError('Latest checkpoint marker differs from expected completed iteration.')
    training_directory = (root / keep).parents[1]
    iteration = int(Path(keep).name.removeprefix('iter_'))
    log = training_directory / 'logs/gpu-nodes-0/miles.out'
    success = [line.strip() for line in log.open(errors='replace')
               if f'successfully saved checkpoint from iteration {iteration:7d}' in line]
    if not success:
        raise RuntimeError('Latest checkpoint has no completed-save log receipt.')
    manifests = {relative: inventory(relative) for relative in expected_hashes}
    remove = [relative for relative in expected_hashes if relative not in retained]
    usage = shutil.disk_usage(root)
    result = {
        'schema_version': 1, 'status': 'planned', 'root': str(root), 'retained_checkpoint': keep,
        'retained_checkpoints': retained, 'profile': profile,
        'pruned_checkpoints': remove, 'inventory': manifests, 'scheduler': scheduler,
        'latest_save_receipt': success, 'disk_before': dict(zip(('total', 'used', 'free'), usage)),
        'planned_payload_bytes': sum(row['bytes'] for relative in remove for row in manifests[relative]
                                     if row['path'] in payload_names),
        'authorization': 'User explicitly authorized removal of older checkpoints from this dummy run.',
        'resume_verified': False,
        'payload_hashes_computed': False,
        'scope': 'Preserve retained saves, all metadata/debug logs and rollout state; delete only explicitly enumerated old tensor payloads. No archive upload or resume claim.',
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if not apply:
        return result
    if usage.free < 128 * 1024**2:
        raise RuntimeError('Insufficient space for durable cleanup evidence.')
    if evidence.exists():
        raise RuntimeError('Retention evidence already exists; inspect it, never retry blindly.')
    evidence.mkdir()
    write(evidence / 'plan.json', result)
    # Recheck all source identities and scheduler immediately before the first mutation.
    scheduler_gate()
    if {relative: inventory(relative) for relative in expected_hashes} != manifests:
        raise RuntimeError('A checkpoint changed while preparing the cleanup.')
    completed = []
    for relative in remove:
        destination = evidence / 'preserved' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError('Preservation destination already exists.')
        os.rename(root / relative, destination)
        for row in manifests[relative]:
            if row['path'] not in payload_names:
                continue
            path = destination / row['path']
            st = path.lstat()
            if (st.st_ino, st.st_size, st.st_mtime_ns, st.st_nlink) != (
                    row['inode'], row['bytes'], row['mtime_ns'], 1) or not stat.S_ISREG(st.st_mode):
                raise RuntimeError('Payload identity changed; remaining cleanup stopped.')
            path.unlink()
        for row in manifests[relative]:
            if row['path'] in small_names and sha(destination / row['path']) != row['sha256']:
                raise RuntimeError('Preserved metadata checksum mismatch.')
        completed.append(relative)
        write(evidence / 'progress.json', {'completed': completed})
    # Remove the failed attempt's stale resume pointer, preserving its exact bytes.
    for relative in obsolete_pointers:
        archived_pointer = evidence / 'preserved' / relative
        archived_pointer.parent.mkdir(parents=True, exist_ok=True)
        os.rename(root / relative, archived_pointer)
    if any(inventory(relative) != manifests[relative] for relative in retained):
        raise RuntimeError('Retained checkpoint changed during cleanup.')
    for relative, value in pointer_values.items():
        path = evidence / 'preserved' / relative if relative in obsolete_pointers else root / relative
        if path.read_text().strip() != value:
            raise RuntimeError('Checkpoint pointer changed during cleanup.')
    if any((root / relative).exists() for relative in remove):
        raise RuntimeError('An old checkpoint directory remains in the original location.')
    result.update(status='ok', completed=completed,
                  disk_after=dict(zip(('total', 'used', 'free'), shutil.disk_usage(root))),
                  ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    result['observed_free_space_delta_bytes'] = result['disk_after']['free'] - result['disk_before']['free']
    write(evidence / 'result.json', result)
    checksums = []
    for path in sorted(evidence.rglob('*')):
        if path.is_file():
            checksums.append(f'{sha(path)}  {path.relative_to(evidence)}')
    write(evidence / 'checksums.sha256', '\n'.join(checksums) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--profile', choices=['original', 'post-job161'], default='original')
    args = parser.parse_args()
    run = Run(args.run_dir)
    if run.root.name != '20260902-172037-a3b210':
        raise ValueError('Only the operator-authorized dummy run is permitted.')
    label = f'02-checkpoint-retention-v{args.attempt}' + ('' if args.apply else '-plan')
    phase = run.phase(label)
    source = inspect.getsource(prune) + '\nimport sys,json\nprint(json.dumps(prune(sys.argv[1], sys.argv[2] == "apply", int(sys.argv[3]), sys.argv[4])))\n'
    atomic(phase.path / 'remote-prune.py', source)
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', source, remote,
        'apply' if args.apply else 'plan', str(args.attempt), args.profile], timeout=120)
    result = json.loads(out.splitlines()[-1]) if not rc else {'status': 'fail', 'error': 'Cleanup stopped; inspect command logs and remote progress before any retry.'}
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if rc else 'ok',
                 results=[] if rc else [metric('checkpoint_payload_bytes_to_prune' if not args.apply else 'checkpoint_payload_bytes_pruned', result['planned_payload_bytes'], 'bytes')],
                 metadata={'result': str((phase.path / 'result.json').relative_to(run.root)),
                           'retained_checkpoint': result.get('retained_checkpoint'),
                           'resume_verified': False, 'applied': args.apply},
                 failure_summary=result.get('error'), refresh=False)
    print(json.dumps({key: result.get(key) for key in (
        'status', 'retained_checkpoint', 'planned_payload_bytes', 'observed_free_space_delta_bytes', 'disk_after', 'error')}))
    return int(bool(rc))


if __name__ == '__main__':
    raise SystemExit(main())
