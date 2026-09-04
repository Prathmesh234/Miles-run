"""Capture the experiment's SGLang Prometheus endpoint until its job finishes."""
import datetime
import json
from pathlib import Path
import signal
import sys
import time
import urllib.request

root = Path(sys.argv[1])
url = sys.argv[2]
assert str(root).startswith('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030/runs/job-')
assert url == 'http://10.244.180.242:15000/metrics'
stop = False

def finish(*_):
    global stop
    stop = True

signal.signal(signal.SIGTERM, finish)
signal.signal(signal.SIGINT, finish)
deadline = time.monotonic() + 9000
with (root/'infra/sglang-prometheus.jsonl').open('x', buffering=1) as out:
    while not stop and time.monotonic() < deadline and not (root/'exit-code.txt').exists():
        entry = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'url':url}
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                entry.update(status=response.status, text=response.read().decode())
        except Exception as error:
            entry['error'] = repr(error)
        out.write(json.dumps(entry)+'\n')
        for _ in range(10):
            if stop or (root/'exit-code.txt').exists():
                break
            time.sleep(1)
