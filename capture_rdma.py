"""Read only counters of this worker's local RDMA ports; never reset or scan."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

root = Path(sys.argv[1])
assert root.resolve().is_relative_to(Path('/shared/clustermax-campaigns')) and root.name.startswith('job-')
ports = [(p.parents[1].name,p.name) for p in Path('/sys/class/infiniband').glob('*/ports/*')
         if (p/'link_layer').read_text().strip() == 'InfiniBand']
stop = False
def finish(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM,finish)
signal.signal(signal.SIGINT,finish)
def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=4)
        return {'argv':argv,'status':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except Exception as e:
        return {'argv':argv,'error':repr(e)}
deadline=time.monotonic()+9000
out=root/'infra'/f'{socket.gethostname()}-rdma-counters.jsonl'
with out.open('x',buffering=1) as f, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    while not stop and time.monotonic()<deadline and not (root/'exit-code.txt').exists():
        started=time.monotonic()
        row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'monotonic':started, 'hostname':socket.gethostname(),
             'scope':'local worker ports; PMA extended counters, no reset flags or remote destination',
             'rdma':command(['rdma','-j','statistic','show']),
             'ports':list(pool.map(command, [['perfquery','-C',d,'-P',p,'-x','-t','1000'] for d,p in ports]))}
        f.write(json.dumps(row)+'\n')
        while time.monotonic()-started<10 and not stop and not (root/'exit-code.txt').exists():
            time.sleep(.5)
