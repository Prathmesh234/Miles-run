"""Read the effective limits of this experiment's container, not its environment."""
import datetime
import json
import os
from pathlib import Path
import socket
import sys

out = Path(sys.argv[1])
assert str(out).startswith('/campaign/runs/job-')
files = {}
for path in ['/proc/self/status', '/proc/self/limits', '/proc/self/cgroup',
             '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu.stat',
             '/sys/fs/cgroup/cpuset.cpus.effective', '/sys/fs/cgroup/cpuset.mems.effective',
             '/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.high',
             '/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.events',
             '/sys/fs/cgroup/pids.max', '/sys/fs/cgroup/pids.current']:
    try:
        files[path] = Path(path).read_text()
    except OSError as error:
        files[path] = {'error':str(error)}
processes = []
for proc in Path('/proc').glob('[0-9]*'):
    try:
        status = dict(line.split(':', 1) for line in (proc/'status').read_text().splitlines())
        processes.append({key:status.get(key, '').strip() for key in
                          ['Name','Pid','PPid','Threads','VmRSS','Cpus_allowed_list','Mems_allowed_list']})
    except (OSError, ValueError):
        pass
result = {'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'hostname':socket.gethostname(), 'affinity':sorted(os.sched_getaffinity(0)),
          'files':files, 'processes':processes}
phase = sys.argv[2] if len(sys.argv) > 2 else 'initial'
assert phase.replace('-', '').isalnum()
target = out/'infra'/f'{socket.gethostname()}-runtime-constraints-{phase}.json'
assert not target.exists(), target
target.write_text(json.dumps(result, indent=2))
print(target)
