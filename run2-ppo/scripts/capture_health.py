"""Supplement two-second host sampling with read-only NVLink/health/storage.

Unsupported telemetry emits an error record, never a fabricated zero. Commands
never reset counters. These counters are node-wide, not process-attributed.
"""
import datetime
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
assert root.is_relative_to(Path('/shared/clustermax-campaigns')) and root.name.startswith('job-')
stop = False


def finish(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, finish)
signal.signal(signal.SIGINT, finish)
commands = {'nvlink': ['nvidia-smi', 'nvlink', '-gt', 'd'],
            'health': ['nvidia-smi', '--query-gpu=uuid,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,clocks_event_reasons.active', '--format=csv,noheader,nounits'],
            'lustre': ['lctl', 'get_param', 'llite.*.stats']}
with (root/'infra'/f'{socket.gethostname()}-health.jsonl').open('x', buffering=1) as out:
    while not stop and not (root/'exit-code.txt').exists():
        started = time.monotonic()
        row = {'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'monotonic':started, 'hostname':socket.gethostname(),
               'shared_free_bytes':shutil.disk_usage(root).free, 'sources':{}}
        for source, command in commands.items():
            try:
                p = subprocess.run(command, capture_output=True, text=True, timeout=3)
                row['sources'][source] = {'argv':command, 'exit_code':p.returncode,
                    'stdout':p.stdout, 'stderr':p.stderr,
                    'status':'ok' if p.returncode == 0 else 'collector_error'}
            except (OSError, subprocess.TimeoutExpired) as error:
                row['sources'][source] = {'argv':command, 'status':'collector_error', 'error':str(error)}
        out.write(json.dumps(row)+'\n')
        while time.monotonic()-started < 5 and not stop:
            time.sleep(.5)
