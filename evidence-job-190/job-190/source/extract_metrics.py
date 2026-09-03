"""Extract full TensorBoard scalars and compact telemetry summaries for a finished run."""
import datetime
import json
from pathlib import Path
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(sys.argv[1])
result = {'run':str(root),'extracted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scalars':{},'rollouts':[],'nodes':{},'checkpoints':[]}
for file in root.rglob('events.out.tfevents.*'):
    acc = EventAccumulator(str(file),size_guidance={'scalars':0}).Reload()
    for tag in acc.Tags().get('scalars',[]):
        result['scalars'].setdefault(tag,[]).extend({'step':x.step,'wall_time':x.wall_time,'value':x.value,'file':str(file.relative_to(root))} for x in acc.Scalars(tag))
for file in sorted((root/'rollouts').glob('step*-samples.json')):
    data=json.loads(file.read_text());samples=[s for group in data['groups'] for s in group]
    result['rollouts'].append({'file':file.name,'metrics':data['metrics'],'samples':len(samples),
        'tasks':[group[0]['metadata']['task'] for group in data['groups']],
        'input_tokens':sum(len(s['tokens']) for s in samples),
        'active_training_tokens':sum(sum(s['loss_mask']) for s in samples),
        'truncated':sum(s['truncated'] for s in samples),
        'trace_truncated':sum(s['metadata'].get('trace_truncated',s['truncated']) for s in samples)})
for file in (root/'infra').glob('*-timeseries.jsonl'):
    values={k:[] for k in ['gpu_util_percent','memory_mib','power_w','temperature_c','sm_clock_mhz']}
    times=[]
    with file.open() as stream:
        for line in stream:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            times.append(row['timestamp'])
            for gpu in row['gpu'].get('stdout','').splitlines():
                fields=gpu.split(',')
                for k,index in [('gpu_util_percent',3),('memory_mib',5),('power_w',7),('temperature_c',9),('sm_clock_mhz',10)]:
                    try:values[k].append(float(fields[index]))
                    except (IndexError,ValueError):pass
    result['nodes'][file.stem]={'start':times[0] if times else None,'end':times[-1] if times else None,
        'whole_attempt_stats':{k:{'count':len(v),'mean':sum(v)/len(v),'max':max(v)} for k,v in values.items() if v}}
for file in (root/'checkpoints').rglob('*'):
    if file.is_file():result['checkpoints'].append({'path':str(file.relative_to(root)),'size':file.stat().st_size})
path=root/'metrics-extracted.json';path.write_text(json.dumps(result,indent=2));print(path)
print(json.dumps({'rollouts':result['rollouts'],'scalar_tags':list(result['scalars']),'checkpoint_files':len(result['checkpoints'])},indent=2))
