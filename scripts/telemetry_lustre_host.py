"""Bounded per-node Lustre sampling from a read-only host debugfs mount."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import time


def duration_seconds(value):
    seconds = int(value)
    if not 1 <= seconds <= 86400:
        raise argparse.ArgumentTypeError('Collector duration must be between 1 and 86400 seconds.')
    return seconds


def stats_records(text):
    units = {'bytes': 'B', 'usecs': 'us', 'reqs': 'count', 'pages': 'pages'}
    rows = []
    for line in text.splitlines():
        found = re.fullmatch(r'(\w+)\s+(\d+) samples \[(\w+)\](?:\s+(.*))?', line)
        if not found:
            continue
        operation, count, raw_unit, tail = found.groups()
        if raw_unit not in units:
            raise ValueError('Unknown Lustre statistic unit: ' + raw_unit)
        rows.append({'metric': operation + '.samples', 'value': int(count), 'unit': 'count',
                     'kind': 'counter'})
        values = list(map(int, (tail or '').split()))
        if len(values) not in (0, 3, 4):
            raise ValueError('Unexpected aggregate fields for ' + operation)
        # Sum-of-squares often saturates at INT64_MAX. Preserve it in raw text;
        # never use it to infer variances or percentiles from cumulative stats.
        for key, value in zip(('min', 'max', 'sum'), values[:3]):
            rows.append({'metric': operation + '.' + key, 'value': value, 'unit': units[raw_unit],
                         'kind': 'counter' if key == 'sum' else 'lifetime_aggregate'})
    if not rows:
        raise ValueError('No Lustre counter rows in statistics file.')
    return rows


def collect(root, host, source, duration, label='lustre-host-validation-v1',
            marker_name='control/lustre-validation-job.json', stop_name=None,
            role='infrastructure-collector-validation'):
    if root.is_symlink() or not (root / 'run.json').is_file():
        raise ValueError('A mounted run directory is required.')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', label):
        raise ValueError('Invalid collector stream label.')
    for name in (marker_name, stop_name):
        if name and (Path(name).is_absolute() or '..' in Path(name).parts):
            raise ValueError('Collector marker must be a relative run path.')
    output = root / 'telemetry' / label / host
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    paths = {'normalized': output / 'lustre.jsonl.partial', 'raw': output / 'raw-lustre.jsonl.partial'}
    handles = {k: p.open('x') for k, p in paths.items()}
    def write(key, value):
        handles[key].write(json.dumps(value, allow_nan=False) + '\n')
    try:
        while time.monotonic() - start < duration and not (stop_name and (root / stop_name).exists()):
            tick = time.monotonic()
            common = {'time': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
                      'monotonic_s': tick, 'hostname': host, 'source': 'lustre-host-debugfs',
                      'role': role}
            marker = root / marker_name
            if marker.exists():
                allocation = json.loads(marker.read_text())
                if allocation.get('active'):
                    common['slurm_job_id'] = allocation['slurm_job_id']
            try:
                if shutil.disk_usage(root).free < 10*1024**3:
                    raise ValueError('Lustre collector requires 10 GiB free space.')
                files = sorted(source.glob('*/stats'))
                if not files:
                    raise ValueError('No host Lustre client statistics files.')
                for path in files:
                    text = path.read_text()
                    attrs = dict(common, lustre_client=path.parent.name)
                    write('raw', dict(attrs, path=str(path), text=text))
                    try:
                        for row in stats_records(text):
                            write('normalized', dict(attrs, **row))
                    except ValueError as exc:
                        # A never-used mount may contain only time headers.
                        if ' samples [' in text:
                            raise
                        write('normalized', dict(attrs, metric='client_without_operation_samples', value=1,
                                                 unit='state', explanation=str(exc)))
            except (OSError, ValueError) as exc:
                write('normalized', dict(common, metric='collector_error', value=None, unit='event', error=str(exc)))
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
            time.sleep(max(0, 1 - (time.monotonic() - tick)))
    finally:
        for key, handle in handles.items():
            handle.close()
            os.rename(paths[key], paths[key].with_suffix(''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--hostname', required=True, choices=[f'gpu-nodes-{i}' for i in range(4)])
    ap.add_argument('--source', type=Path, default=Path('/host-lustre'))
    ap.add_argument('--duration-s', type=duration_seconds, default=180)
    ap.add_argument('--stream-label', default='lustre-host-validation-v1')
    ap.add_argument('--job-marker', default='control/lustre-validation-job.json')
    ap.add_argument('--stop-marker')
    ap.add_argument('--role', default='infrastructure-collector-validation')
    args = ap.parse_args()
    collect(args.run_dir, args.hostname, args.source, args.duration_s, args.stream_label,
            args.job_marker, args.stop_marker, args.role)
