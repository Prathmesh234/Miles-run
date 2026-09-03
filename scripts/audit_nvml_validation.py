"""Fetch a compact qualification result; keep raw GPU evidence on shared storage."""
import argparse
import inspect
import json

from evidence import Run, atomic, markdown


def audit_teardown_log(text, hostname, profile):
    """Join bounded rank events without treating NCCL stdout as JSON lines."""
    import json
    import math

    decoder, cursor, events = json.JSONDecoder(), 0, []
    while True:
        index = text.find('{"event":', cursor)
        if index < 0:
            break
        event, length = decoder.raw_decode(text[index:])
        events.append(event)
        cursor = index + length
    pinned = profile in ('pinned-host-nccl-teardown', 'pinned-host-clean-teardown')
    clean = profile == 'pinned-host-clean-teardown'
    expected = (['host_capacity_guard'] if pinned else []) + ['before_allocate', 'allocated']
    expected += ['before_pinned_release', 'pinned_released'] if clean else []
    expected += ['exit_with_live_context']
    if {e['rank'] for e in events} != set(range(8)) or len(events) != 8 * len(expected):
        raise ValueError('Missing or repeated rank teardown evidence.')
    result = []
    for rank in range(8):
        rows = [e for e in events if e['rank'] == rank]
        if [e['event'] for e in rows] != expected or any(e['hostname'] != hostname for e in rows):
            raise ValueError('Wrong teardown rank order or hostname.')
        times = [e['monotonic_s'] for e in rows]
        if any(not math.isfinite(t) for t in times) or times != sorted(times):
            raise ValueError('Invalid teardown monotonic timestamps.')
        allocation = next(e for e in rows if e['event'] == 'allocated')
        host_bytes = 24 * 1024**3 if pinned else 0
        if allocation['allocation_count'] * allocation['chunk_mib'] * 1024**2 != 64 * 1024**3:
            raise ValueError('GPU allocation does not match the frozen 64 GiB control.')
        if allocation['pinned_bytes'] != host_bytes or allocation['pinned_allocation_count'] != (3072 if pinned else 0):
            raise ValueError('Pinned host allocation does not match the control.')
        if rows[-1]['pinned_bytes'] != (0 if clean else host_bytes):
            raise ValueError('Exit marker disagrees with pinned-buffer lifetime.')
        summary = dict(rank=rank, allocation_time=allocation['time'], exit_time=rows[-1]['time'],
                       pinned_bytes=host_bytes, exit_monotonic_s=rows[-1]['monotonic_s'])
        if clean:
            release = next(e for e in rows if e['event'] == 'pinned_released')
            if release['expected_released_bytes'] != host_bytes:
                raise ValueError('Pinned release size mismatch.')
            for key in ('active_bytes.current', 'allocated_bytes.current'):
                if release['before'][key] - release['after'][key] < host_bytes or release['after'][key] < 0:
                    raise ValueError('Pinned allocator did not release the full control payload.')
            if not math.isfinite(release['duration_s']) or release['duration_s'] < 0:
                raise ValueError('Invalid host release duration.')
            summary['release_duration_s'] = release['duration_s']
        result.append(summary)
    return result


def audit_remote(root_text, attempt, job):
    import hashlib
    import json
    from pathlib import Path
    import subprocess

    root = Path(root_text)
    label = f'nvml-qualification-v{attempt}'
    accounting = subprocess.check_output(['sacct', '-j', str(job), '--noheader', '--parsable2',
        '--format=JobID,State,ExitCode,Elapsed,NodeList'], text=True)
    jobs = [line.split('|') for line in accounting.splitlines() if line.split('|')[0] == str(job)]
    if len(jobs) != 1 or jobs[0][1].split()[0] not in ('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL'):
        raise ValueError('Job is not unambiguously terminal; no final result emitted.')
    findings, nodes = [], []
    if jobs[0][1:3] != ['COMPLETED', '0:0']:
        findings.append('Slurm qualification did not complete successfully.')
    for index in range(4):
        host = f'gpu-nodes-{index}'
        phase = root / 'tests' / ('01-' + label + '-' + host)
        finals = list(phase.glob('*.values.json')) + list(phase.glob('*.failed.json'))
        if len(finals) != 1:
            findings.append(host + ': missing or ambiguous phase result.')
            continue
        data = json.loads(finals[0].read_text())
        node = {'hostname': host, 'phase': data, 'path': str(finals[0].relative_to(root)),
                'sha256': hashlib.sha256(finals[0].read_bytes()).hexdigest()}
        profile = data.get('metadata', {}).get('load_profile')
        # Old controls used a smaller event schema. Require the complete new
        # allocation/release contract only for the new pinned-host profiles.
        if profile in ('pinned-host-nccl-teardown', 'pinned-host-clean-teardown'):
            log = phase / 'logs/load.out'
            try:
                raw = log.read_bytes()
                node['load_ranks'] = audit_teardown_log(raw.decode(), host, profile)
                node['load_log_sha256'] = hashlib.sha256(raw).hexdigest()
            except Exception as exc:
                findings.append(host + ': load event audit: ' + str(exc))
        for name in ('failure.json', 'nvml-validation.json', 'heartbeat.json'):
            path = root / 'telemetry' / label / host / name
            if path.is_file():
                node[name] = json.loads(path.read_text())
        if data['status'] != 'ok':
            findings.append(host + ': ' + data['failure_summary'])
        nodes.append(node)
    return dict(slurm_job_id=str(job), attempt=attempt, findings=findings, nodes=nodes,
        slurm_accounting=accounting, scope='GPU/NVLink collector qualification, not full training telemetry or quality.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--job-id', type=int, required=True)
    a = ap.parse_args()
    run = Run(a.run_dir)
    phase = run.phase(f'01-nvml-result-audit-v{a.attempt}')
    program = inspect.getsource(audit_teardown_log) + '\n' + inspect.getsource(audit_remote) + '\nimport sys\nprint(json.dumps(audit_remote(sys.argv[1], int(sys.argv[2]), sys.argv[3])))\n'
    # json is imported inside the function too, but needed by the final serializer.
    program = 'import json\n' + program
    atomic(phase.path / 'audit-remote.py', program)
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', a.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program,
        '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name, str(a.attempt), str(a.job_id)], timeout=45)
    data = json.loads(out) if not rc else {'findings': ['Read-only qualification audit failed; inspect logs.']}
    atomic(phase.path / 'result.json', data)
    phase.finish('fail' if data['findings'] else 'ok', metadata=data,
        failure_summary='; '.join(data['findings']) or None, refresh=False)
    print(json.dumps({'job_id': a.job_id, 'findings': data['findings']}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
