"""Summarize captured evidence; counters are node-wide, not process attribution."""
import datetime
import json
from pathlib import Path
import re
import sys

root=Path(sys.argv[1])
start=float(sys.argv[2])
end=float(sys.argv[3])
def epoch(value):
    return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def stats(values):
    return {'count':len(values),'mean':sum(values)/len(values),'max':max(values),'min':min(values)} if values else {}
result={'window_epoch':[start,end],'nodes':{},'counter_scope':'node-wide local ports; not exclusively training; TX and RX reported separately',
        'counter_units_source':'https://man7.org/linux/man-pages/man8/perfquery.8.html',
        'rdma_window':'all successfully captured PMA samples; timestamps per port, which may begin after process startup'}
training_windows=[]
pending=None
for line in (root/'training.log').read_text().splitlines():
    match=re.search(r'\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) actor_cell0_rank0\].*Timer train (start|end)',line)
    if match:
        ts=epoch(match[1]+'+00:00')
        if match[2]=='start':pending=ts
        elif pending is not None:
            training_windows.append((pending,ts));pending=None
result['training_windows_epoch']=training_windows
for node in range(4):
    name=f'gpu-nodes-{node}'
    data={'gpu':{},'rdma_ports':{},'rdma_unavailable_ports':{},'rdma_hw_error_changes':{}}
    values={k:[] for k in ['util_percent','memory_mib','power_w','temp_c','clock_mhz']}
    stages=[{k:[] for k in values} for _ in training_windows]
    for line in (root/'infra'/f'{name}-timeseries.jsonl').read_text().splitlines():
        row=json.loads(line)
        ts=epoch(row['timestamp'])
        if not start<=ts<=end:
            continue
        for gpu in row['gpu'].get('stdout','').splitlines():
            fields=gpu.split(',')
            for key,index in [('util_percent',3),('memory_mib',5),('power_w',7),('temp_c',9),('clock_mhz',10)]:
                try:
                    value=float(fields[index]);values[key].append(value)
                    for stage,(t0,t1) in zip(stages,training_windows):
                        if t0<=ts<=t1:stage[key].append(value)
                except (ValueError,IndexError):pass
    data['gpu']={k:stats(v) for k,v in values.items()}
    data['training_windows']=[{k:stats(v) for k,v in stage.items()} for stage in stages]
    ports={}
    hardware=[]
    path=root/'infra'/f'{name}-rdma-counters.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            for port in row['ports']:
                device=port['argv'][2]
                if port.get('status')!=0:
                    data['rdma_unavailable_ports'][device]=port.get('stderr',port.get('error',''))
                    continue
                counters={k:int(v) for k,v in re.findall(r'^([A-Za-z0-9]+):\.+([0-9]+)',port['stdout'],re.M)}
                ports.setdefault(device,[]).append((epoch(row['utc']),counters))
            if row['rdma'].get('status')==0:
                hardware.append(json.loads(row['rdma']['stdout']))
        for device,rows in ports.items():
            ts0,c0=rows[0];ts1,c1=rows[-1]
            decreases=[k for k in c0 if k in c1 and c1[k]<c0[k]]
            delta={k:c1[k]-v for k,v in c0.items() if k in c1 and c1[k]>=v}
            data['rdma_ports'][device]={'start_epoch':ts0,'end_epoch':ts1,'samples':len(rows),'decreased_counters':decreases,
                'tx_bytes':delta.get('PortXmitData',0)*4,'rx_bytes':delta.get('PortRcvData',0)*4,
                'counter_deltas':delta}
        if hardware:
            first={p['ifname']:p for p in hardware[0]}
            for port in hardware[-1]:
                old=first.get(port['ifname'],{})
                changes={k:v-old[k] for k,v in port.items() if isinstance(v,int) and k in old and isinstance(old[k],int) and v!=old[k] and any(x in k for x in ['err','retry','retrans','nak','buffer','sequence'])}
                if changes:data['rdma_hw_error_changes'][port['ifname']]=changes
    result['nodes'][name]=data
out=root/'infrastructure-runtime-summary.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2));print(out)
