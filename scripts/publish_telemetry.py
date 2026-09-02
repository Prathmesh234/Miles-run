"""Publish dense metric summaries per job; full-resolution telemetry stays outside Git.

Use --watch to commit and optionally push every 300 seconds until a Slurm job is
terminal. Preserve full JSONL privately; publish statistics, trends and failures.
The publication path is independent of the private run evidence directory.
"""
import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from evidence import atomic, utcnow

FILES = ('nvidia-smi', 'nvlink', 'infiniband', 'cpu-memory-numa', 'lustre')
SOURCES = {'nvidia-smi', 'nvlink', 'perfquery', 'ib-sysfs', 'proc-meminfo', 'proc-stat',
           'proc-vmstat', 'proc-sysfs', 'collector', 'statvfs', 'lustre-host-debugfs', 'persistent-nvml'}
FIELDS = {'time', 'monotonic_s', 'hostname', 'source', 'metric', 'value', 'unit', 'gpu_uuid',
          'hca', 'hca_port', 'role', 'rank', 'policy_version', 'slurm_job_id', 'link', 'kind',
          'raw_unit', 'raw_value', 'counter_group', 'lustre_client', 'error', 'requested_metric',
          'path', 'nvml_sample_timestamp_us', 'nvml_query_latency_us', 'nvml_field_id'}
SECRET = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                    rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
                    rb'hf_[A-Za-z0-9]{25,}|xox[baprs]-[A-Za-z0-9-]{20,}|'
                    rb'(?:AKIA|ASIA)[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{32,})\b')


def export_chunk(root_text, relative, offset, expected_inode, anchor):
    import base64
    import gzip
    import hashlib
    from pathlib import Path
    import re

    if not re.fullmatch(r'/shared/posttrainingx/runs/vultr-b200-slurm/\d{8}-\d{6}-[a-z0-9]+', root_text):
        raise ValueError('Invalid run root.')
    root = Path(root_text)
    if not re.fullmatch(r'telemetry/(?:lustre-)?[a-z0-9][a-z0-9-]*/gpu-nodes-[0-3]/(?:nvidia-smi|nvlink|infiniband|cpu-memory-numa|lustre)\.jsonl', relative):
        raise ValueError('Only explicitly allowed metric streams can be exported.')
    final = root / relative
    path = final if final.exists() else final.with_suffix('.jsonl.partial')
    if not path.exists():
        return {'status': 'missing', 'path': relative}
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError('Telemetry path contains a symlink.')
    before = path.stat()
    if offset < 0 or offset > before.st_size or (expected_inode is not None and expected_inode != before.st_ino):
        raise ValueError('Telemetry source was truncated or replaced.')
    with path.open('rb') as handle:
        if offset:
            handle.seek(max(0, offset - 4096))
            prior = handle.read(min(offset, 4096))
            if not prior.endswith(b'\n') or hashlib.sha256(prior).hexdigest() != anchor:
                raise ValueError('Previously published stream boundary changed.')
        handle.seek(offset)
        block = handle.read(16 * 1024**2)
        cut = block.rfind(b'\n') + 1
        raw = block[:cut]
        end = offset + len(raw)
        handle.seek(max(0, end - 4096))
        new_anchor = hashlib.sha256(handle.read(min(end, 4096))).hexdigest()
        import os
        after = os.fstat(handle.fileno())
    if before.st_ino != after.st_ino or after.st_size < before.st_size:
        raise ValueError('Telemetry source changed identity during export.')
    finalized = final.exists() and final.stat().st_ino == before.st_ino
    complete = finalized and end == after.st_size
    if finalized and not raw and offset < after.st_size:
        raise ValueError('Finalized JSONL has an incomplete or oversized line.')
    digest = None
    if complete:
        h = hashlib.sha256()
        with final.open('rb') as handle:
            for part in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(part)
        digest = h.hexdigest()
    payload = gzip.compress(raw, mtime=0)
    return {'status': 'ok', 'path': relative, 'inode': before.st_ino, 'offset': offset, 'end': end,
            'anchor': new_anchor, 'complete': complete, 'source_bytes_observed': after.st_size,
            'source_sha256': digest, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'gzip_sha256': hashlib.sha256(payload).hexdigest(),
            'gzip_base64': base64.b64encode(payload).decode()}


def validate_rows(raw, hostname, job_id):
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Chunk ends inside a JSONL record.')
    if SECRET.search(raw):
        raise ValueError('Credential-like content detected; refusing publication without printing it.')
    count, errors, first, last = 0, 0, None, None
    for line in raw.splitlines():
        row = json.loads(line)
        if set(row) - FIELDS or row.get('source') not in SOURCES:
            raise ValueError('Unreviewed telemetry schema/source; no publication.')
        if any(isinstance(value, (dict, list)) for value in row.values()):
            raise ValueError('Nested data is not allowed in public metric records.')
        if row.get('hostname') != hostname or str(row.get('slurm_job_id', job_id)) != str(job_id):
            raise ValueError('Telemetry node/job identity mismatch.')
        if not {'time', 'monotonic_s', 'metric', 'value', 'unit'} <= set(row):
            raise ValueError('Incomplete normalized telemetry record.')
        if not isinstance(row['monotonic_s'], (int, float)) or not math.isfinite(row['monotonic_s']):
            raise ValueError('Invalid monotonic timestamp.')
        if row['metric'] == 'collector_error':
            if row['value'] is not None:
                raise ValueError('Collector errors must not masquerade as numeric measurements.')
            errors += 1
        elif not isinstance(row['value'], (int, float)) or not math.isfinite(row['value']):
            raise ValueError('Non-numeric or nonfinite metric.')
        count += 1
        first = min(first, row['time']) if first else row['time']
        last = max(last, row['time']) if last else row['time']
    return {'records': count, 'collector_errors': errors, 'first_time': first, 'last_time': last}


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fetch_source(base_command, program, remote, relative, previous, destination, job_id):
    import base64

    state = dict(previous or {'path': relative, 'chunks': [], 'end': 0, 'inode': None, 'anchor': None})
    state['chunks'] = list(state['chunks'])
    if state.get('complete'):
        return state
    hostname = Path(relative).parts[-2]
    while True:
        result = subprocess.run(base_command + ['python3', '-c', program, remote, relative,
            str(state['end']), json.dumps(state['inode']), json.dumps(state['anchor'])],
            capture_output=True, text=True, timeout=120, check=True)
        packet = json.loads(result.stdout)
        if packet['status'] == 'missing':
            state.update(status='missing', complete=False)
            return state
        payload = base64.b64decode(packet.pop('gzip_base64'), validate=True)
        if hashlib.sha256(payload).hexdigest() != packet['gzip_sha256']:
            raise ValueError('Compressed transfer checksum mismatch.')
        raw = gzip.decompress(payload)
        if len(raw) != packet['end'] - packet['offset'] or hashlib.sha256(raw).hexdigest() != packet['raw_sha256']:
            raise ValueError('Raw transfer checksum/length mismatch.')
        stats = validate_rows(raw, hostname, job_id)
        if raw:
            name = Path('chunks') / Path(relative).relative_to('telemetry').parent / (
                Path(relative).stem + f'-{packet["offset"]:012d}-{packet["end"]:012d}.jsonl.gz')
            target = destination / name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != packet['gzip_sha256']:
                raise ValueError('Existing immutable chunk differs; stop and inspect.')
            if not target.exists():
                atomic_bytes(target, payload)
            state['chunks'].append(dict(path=str(name), offset=packet['offset'], end=packet['end'],
                raw_sha256=packet['raw_sha256'], gzip_sha256=packet['gzip_sha256'], gzip_bytes=len(payload), **stats))
        state.update({key: packet[key] for key in ('end', 'inode', 'anchor', 'complete', 'source_bytes_observed', 'source_sha256')})
        state['status'] = 'complete' if packet['complete'] else 'partial'
        if packet['complete']:
            digest = hashlib.sha256()
            for chunk in state['chunks']:
                digest.update(gzip.decompress((destination / chunk['path']).read_bytes()))
            if digest.hexdigest() != packet['source_sha256']:
                raise ValueError('Reconstructed stream does not match the finalized source.')
            return state
        if not raw or packet['end'] >= packet['source_bytes_observed']:
            return state


def publish(args):
    repo = Path(__file__).resolve().parents[1]
    run = Path(args.run_dir).resolve()
    if not (run / 'run.json').is_file() or not re.fullmatch(r'\d{8}-\d{6}-[a-z0-9]+', run.name):
        raise ValueError('A valid run manifest is required.')
    if not re.fullmatch('[a-z0-9][a-z0-9-]*', args.stream_label):
        raise ValueError('Invalid stream label.')
    if subprocess.check_output(['git', '-C', str(repo), 'diff', '--cached', '--name-only'], text=True).strip():
        raise ValueError('Refusing to mix telemetry with already staged changes.')
    destination = repo / 'telemetry' / 'vultr-b200-slurm' / run.name / f'job-{args.job_id}'
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError('Publication destination contains a symlink.')
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < 1024**3:
        raise RuntimeError('Publication requires a 1 GiB free-space reserve.')
    result_names = ('telemetry.values.json', 'telemetry.failed.json', 'telemetry.partial.json', 'telemetry.skipped.json')
    present = [destination / name for name in result_names if (destination / name).exists()]
    if len(present) > 1:
        raise ValueError('More than one current summary result exists.')
    manifest_path = present[0] if present else destination / 'telemetry.values.json'
    cache = run / 'provenance' / 'telemetry-publisher-cache' / f'job-{args.job_id}'
    if any(path.is_symlink() for path in (cache, *cache.parents)):
        raise ValueError('Local telemetry cache contains a symlink.')
    cache.mkdir(parents=True, exist_ok=True)
    state_path = cache / 'state.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    previous = {row['path']: row for row in old.get('streams', [])}
    paths = [f'telemetry/{args.stream_label}/gpu-nodes-{i}/{name}.jsonl'
             for i in range(4) for name in FILES]
    paths += [f'telemetry/lustre-{args.stream_label}/gpu-nodes-{i}/lustre.jsonl' for i in range(4)]
    if previous and set(previous) != set(paths):
        raise ValueError('Publication source set changed for an existing job.')
    base_command = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec', 'slurm-worker-gpu-nodes-0', '--']
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.name
    program = inspect.getsource(export_chunk) + '\nimport json,sys\nprint(json.dumps(export_chunk(sys.argv[1],sys.argv[2],int(sys.argv[3]),json.loads(sys.argv[4]),json.loads(sys.argv[5]))))\n'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_source, base_command, program, remote, path,
                               previous.get(path), cache, args.job_id) for path in paths]
        streams = [future.result() for future in futures]
    accounting = subprocess.check_output(base_command + ['sacct', '-X', '-j', str(args.job_id), '-n', '-P',
        '-o', 'JobID,State,ExitCode'], text=True, timeout=30)
    jobs = [line.split('|') for line in accounting.splitlines() if line.split('|')[0] == str(args.job_id)]
    if len(jobs) != 1:
        raise ValueError('Slurm job state is ambiguous; retain chunks without a completion claim.')
    terminal = jobs[0][1].split()[0] in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'}
    if terminal and not all(row.get('complete') for row in streams):
        # Capture final flushes that may have occurred after the concurrent read.
        streams = [fetch_source(base_command, program, remote, row['path'], row, cache, args.job_id)
                   for row in streams]
    if (manifest_path.exists()
            and old.get('published_manifest_sha256') == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            and old.get('streams') == streams
            and old.get('slurm_state') == jobs[0][1] and old.get('slurm_exit_code') == jobs[0][2]):
        if args.push:
            push(repo)
        print(json.dumps({'slurm_job_id': args.job_id, 'status': 'unchanged', 'terminal': terminal}), flush=True)
        return terminal
    from summarize_telemetry import build_summary, render
    manifest, timeline = build_summary(run, cache, streams, args.job_id, jobs[0][1], jobs[0][2])
    suffix = {'ok': 'values', 'fail': 'failed', 'partial': 'partial', 'skip': 'skipped'}[manifest['status']]
    manifest_path = destination / f'telemetry.{suffix}.json'
    expected_files = set(result_names) | {'README.md', 'timeline.csv', 'checksums.sha256'}
    if any(path.name not in expected_files or not path.is_file() for path in destination.iterdir()):
        raise ValueError('Publication contains raw archives or unexpected files; inspect before publishing.')
    atomic(manifest_path, manifest)
    # Keep only the current status file in Git. Earlier versions remain in history;
    # private snapshots retain failure evidence independently of publication.
    snapshot = cache / ('summary-' + str(time.time_ns()) + '.json')
    atomic(snapshot, manifest)
    for previous_result in present:
        if previous_result != manifest_path:
            previous_result.unlink()
    frozen = json.loads(manifest_path.read_text())
    atomic(destination / 'README.md', render(frozen))
    atomic(destination / 'timeline.csv', timeline)
    hashes = [f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}'
              for path in sorted(destination.iterdir()) if path.name != 'checksums.sha256']
    atomic(destination / 'checksums.sha256', '\n'.join(hashes) + '\n')
    relative = str(destination.relative_to(repo))
    subprocess.run(['git', '-C', str(repo), 'add', '--', relative], check=True)
    changed = subprocess.run(['git', '-C', str(repo), 'diff', '--cached', '--quiet']).returncode
    if changed not in (0, 1):
        raise RuntimeError('Unable to inspect staged telemetry changes.')
    staged = subprocess.check_output(['git', '-C', str(repo), 'diff', '--cached', '--name-only'], text=True).splitlines()
    if any(not name.startswith(relative + '/') for name in staged):
        raise ValueError('Unrelated files were staged during publication; refusing to commit.')
    if changed == 1:
        subprocess.run(['git', '-C', str(repo), 'commit', '-m', f'Summarize telemetry for Slurm job {args.job_id}'], check=True)
    atomic(state_path, {'streams': streams, 'slurm_state': jobs[0][1], 'slurm_exit_code': jobs[0][2],
                        'published_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
    if args.push:
        push(repo)
    print(json.dumps({'slurm_job_id': args.job_id, 'status': manifest['status'],
        'source_records': manifest['metadata']['raw_records'], 'collector_errors': manifest['metadata']['collector_error_count'],
        'summary_metrics': len(manifest['results']), 'published_files': len(list(destination.iterdir())),
        'published_bytes': sum(path.stat().st_size for path in destination.iterdir())}), flush=True)
    return terminal


def push(repo):
    remote_url = subprocess.check_output(['git', '-C', str(repo), 'remote', 'get-url', 'origin'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
    if remote_url != 'https://github.com/Prathmesh234/Miles-run.git' or branch != 'main':
        raise ValueError('Publication remote/branch differs from the operator-authorized repository.')
    # The initial upload hit an HTTP rewind error. Keep the larger buffer scoped
    # to this command; never change the operator's global Git configuration.
    subprocess.run(['git', '-C', str(repo), '-c', 'http.postBuffer=134217728',
                    'push', 'origin', 'HEAD:main'], check=True, timeout=300)


def start_watcher(run_dir, kubeconfig, stream_label, job_id):
    """Start a bounded, run-owned local publisher after an unambiguous submission."""
    run = Path(run_dir).resolve()
    folder = run / 'provenance' / f'telemetry-publisher-job-{int(job_id)}'
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(run),
               '--kubeconfig', str(Path(kubeconfig).resolve()), '--stream-label', stream_label,
               '--job-id', str(int(job_id)), '--watch', '--push',
               '--interval-seconds', '300', '--max-seconds', '6000']
    with (folder / 'stdout.log').open('xb') as stdout, (folder / 'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
    receipt = {'started_at': utcnow(), 'pid': process.pid, 'command': command,
               'interval_seconds': 300, 'max_seconds': 6000,
               'log_directory': str(folder.relative_to(run)),
               'status': 'spawned', 'scope': 'Local watcher spawned; successful publication is not yet verified.'}
    atomic(folder / 'started.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--stream-label', required=True)
    parser.add_argument('--job-id', required=True, type=int)
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval-seconds', type=int, default=300)
    parser.add_argument('--max-seconds', type=int, default=6000)
    args = parser.parse_args()
    if args.interval_seconds < 60 or args.max_seconds < 1:
        raise ValueError('Invalid bounded publication cadence.')
    repo = Path(__file__).resolve().parents[1]
    with (repo / '.git/posttrainingx-telemetry-publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + args.max_seconds
        while True:
            terminal = publish(args)
            if terminal or not args.watch:
                break
            wake = min(deadline, time.monotonic() + args.interval_seconds)
            while time.monotonic() < wake:
                time.sleep(min(30, wake - time.monotonic()))
            if time.monotonic() >= deadline:
                raise TimeoutError('Bounded publisher expired; job state was not terminal.')


if __name__ == '__main__':
    main()
