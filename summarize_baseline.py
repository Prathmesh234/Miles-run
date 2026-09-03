"""Summarize original GPU telemetry and transport evidence without modifying it."""
import csv
import json
from pathlib import Path

ROOT = Path('/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00/runs/20260903-150011')
OUT = Path('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030/preflight/baseline-infra-summary.json')
WINDOWS = {'setup':('2026-09-03T15:00:11','2026-09-03T15:24:22'),
           'rollout_and_training':('2026-09-03T15:24:22','2026-09-03T15:30:03'),
           'stranded_allocation':('2026-09-03T15:30:20','2026-09-03T19:00:11')}
FIELDS = ['utilization.gpu','utilization.memory','memory.used','power.draw','temperature.gpu','clocks.sm']
result = {'windows_utc':WINDOWS,'nodes':{},'nccl_evidence':{}}
for path in (ROOT/'telemetry/gpu').glob('gpu-nodes-*.csv'):
    stats = {w:{k:[] for k in FIELDS} for w in WINDOWS}
    with path.open() as file:
        for row in csv.DictReader(file):
            timestamp = row['collector_time_utc'][:19]
            for window,(start,end) in WINDOWS.items():
                if start <= timestamp < end:
                    for key in FIELDS:
                        try: stats[window][key].append(float(row[key]))
                        except (ValueError,KeyError): pass
    result['nodes'][path.stem] = {w:{k:{'count':len(v),'mean':sum(v)/len(v),'max':max(v)} for k,v in fields.items() if v} for w,fields in stats.items()}
for path in (ROOT/'logs').glob('nccl-*.out'):
    evidence = []
    with path.open(errors='replace') as file:
        for i,line in enumerate(file):
            if any(x in line for x in ['Using network','NET/IB : Using','NET/Socket : Using','NCCL version','Bootstrap : Using','GDRDMA']):
                evidence.append(line.strip())
            if len(evidence)>=12 or i>3000: break
    result['nccl_evidence'][path.name] = evidence
OUT.write_text(json.dumps(result,indent=2))
print(OUT)
for node,stats in result['nodes'].items():
    print(node,stats['rollout_and_training'].get('utilization.gpu'),stats['rollout_and_training'].get('memory.used'))
for key,value in list(result['nccl_evidence'].items())[:2]:print(key,value[:4])
