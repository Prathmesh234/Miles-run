"""Summarize original GPU telemetry read-only, excluding its idle allocation tail."""
import csv
import datetime
import json
from pathlib import Path

C=Path('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030')
B=Path('/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00/runs/20260903-150011')
windows={'ready_to_checkpoint':('15:24:22','15:30:03'),
         'ready_to_step1':('15:24:22','15:27:43'),
         'step1_to_step2':('15:27:43','15:29:46')}
fields=['utilization.gpu','utilization.memory','memory.used','power.draw','temperature.gpu','clocks.sm','ecc.errors.uncorrected.volatile.total']
result={'windows_utc':windows,'nodes':{},'note':'Observed GPU samples per window; mean utilization is not training-only utilization. Original four-hour idle tail excluded.'}
for node in range(4):
    values={w:{k:[] for k in fields} for w in windows}
    with (B/f'telemetry/gpu/gpu-nodes-{node}.csv').open() as f:
        for row in csv.DictReader(f):
            tod=row['collector_time_utc'][11:19]
            for w,(start,end) in windows.items():
                if start <= tod <= end:
                    for k in fields:
                        try: values[w][k].append(float(row[k]))
                        except ValueError:pass
    result['nodes'][f'gpu-nodes-{node}']={w:{k:{'samples':len(v),'mean':sum(v)/len(v),'max':max(v),'min':min(v)} for k,v in data.items() if v} for w,data in values.items()}
out=C/'baseline-gpu-profile.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2));print(out)
