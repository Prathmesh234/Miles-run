"""Read-only local 400-Gb/s IB and NVLink counters during an existing allocation.

Never reset counters or address remote ports. No training process manipulation.
Stop with allocation end, exit-code.txt, a STOP_FABRIC_1S file, or one hour.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


def command(argv):
    before=time.time();start=time.monotonic()
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=2)
        return dict(start=before,end=time.time(),monotonic=time.monotonic(),seconds=time.monotonic()-start,
                    status=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except (OSError,subprocess.TimeoutExpired) as error:
        return dict(start=before,end=time.time(),monotonic=time.monotonic(),seconds=time.monotonic()-start,
                    status=-1,error=str(error))


def sample_port(device):
    result=command(['perfquery','-C',device,'-P','1','-x','-t','1000'])
    if result['status']==0:
        result['counters']={k:int(v) for k,v in re.findall(r'^([A-Za-z0-9]+):\.+([0-9]+)',result.pop('stdout'),re.M)}
    return {'device':device,**result}


def sample_nvlink():
    result=command(['nvidia-smi','nvlink','-gt','d'])
    if result['status']==0:
        raw=result.pop('stdout');gpu=None;data={}
        for line in raw.splitlines():
            m=re.search(r'UUID: (GPU-[^)]+)',line)
            if m: gpu=m[1]
            m=re.search(r'Link (\d+): Data (Tx|Rx): (\d+) KiB',line)
            if m and gpu:
                link,direction,value=m.groups()
                data.setdefault(gpu+'/'+link,{})[direction]=int(value)
        result.update(counters=data,raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
    return result


def main():
    root=Path(sys.argv[1]).resolve()
    assert root.is_relative_to(Path('/shared/clustermax-campaigns')) and root.name=='job-197'
    host=socket.gethostname();inventory=json.loads((root/'infra'/f'{host}-before.json').read_text())
    ib=next(c['stdout'] for c in inventory['commands'] if c['argv']==['ibstat'] and c['exit_code']==0)
    devices=[d for d,body in re.findall(r"CA '([^']+)'\n(.*?)(?=\nCA '|\Z)",ib,re.S)
             if 'Rate: 400' in body and 'Link layer: InfiniBand' in body and 'State: Active' in body]
    assert len(devices)==8, devices
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    deadline=time.monotonic()+3600
    output=root/'infra'/f'{host}-fabric-1s.jsonl'
    with output.open('x',buffering=1) as f,ThreadPoolExecutor(max_workers=8) as pool:
        while not stopped and time.monotonic()<deadline and not (root/'exit-code.txt').exists() and not (root/'STOP_FABRIC_1S').exists():
            start=time.monotonic()
            pending=pool.submit(sample_nvlink)
            ports=list(pool.map(sample_port,devices))
            row=dict(schema_version=1,time=datetime.now(timezone.utc).isoformat(),host=host,
                     ib=ports,nvlink=pending.result(),collector_wall_seconds=time.monotonic()-start,
                     requested_interval_seconds=1,ib_unit='4-byte words',nvlink_unit='KiB')
            f.write(json.dumps(row,separators=(',',':'))+'\n')
            while not stopped and time.monotonic()-start<1: time.sleep(.05)
    print(json.dumps({'host':host,'output':str(output),'finished':True}),flush=True)


if __name__=='__main__':main()
