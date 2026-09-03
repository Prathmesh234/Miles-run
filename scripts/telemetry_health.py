"""Fail-closed collector health: durable errors and host-local monotonic heartbeat."""
import json
import math

from evidence import atomic, utcnow


def heartbeat(directory, hostname, job_id, ticks, errors, now):
    atomic(directory / 'heartbeat.json', dict(time=utcnow(), monotonic_s=now,
        hostname=hostname, slurm_job_id=str(job_id), ticks=ticks, errors=errors))


def assert_healthy(directory, hostname, job_id, now, max_age_s=12):
    """Only call on the collector's host; monotonic epochs are node-local."""
    marker = directory / 'failure.json'
    if marker.exists():
        raise RuntimeError('Telemetry failure marker: ' + str(marker))
    try:
        row = json.loads((directory / 'heartbeat.json').read_text())
    except FileNotFoundError:
        return False  # Startup only; the caller enforces its startup deadline.
    if row.get('hostname') != hostname or str(row.get('slurm_job_id')) != str(job_id):
        raise RuntimeError('Telemetry heartbeat identity mismatch.')
    timestamp = row.get('monotonic_s')
    if (not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp)
            or not 0 <= now - timestamp <= max_age_s):
        raise RuntimeError('Telemetry heartbeat stalled or has an invalid clock.')
    if row.get('errors') != 0 or not isinstance(row.get('ticks'), int) or row['ticks'] < 1:
        raise RuntimeError('Telemetry heartbeat contains errors or no completed samples.')
    return True


def require_healthy(directory, hostname, job_id, now):
    if not assert_healthy(directory, hostname, job_id, now):
        raise RuntimeError('Required telemetry heartbeat is missing.')
