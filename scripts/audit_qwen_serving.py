"""Audit finalized serving correctness and native telemetry without launching work."""
import argparse
import inspect
import json
from pathlib import Path

from evidence import Run, atomic, metric
from summarize_native import percentile, summary


AUDIT = r'''
import collections,json,math,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]);attempt=int(sys.argv[2])
result={'cases':[],'telemetry':[],'gpu_statistics':[],'findings':[]}
for mode in ['off','on']:
 p=root/f'tests/02-qwen-serving-mtp-{mode}-attempt-{attempt}'
 d=json.loads((p/'probe-result.json').read_text())
 expected=['4','infrastructure-ready']
 texts=[g['response']['text'].strip() for g in d.get('generations',[])]
 if texts!=expected:result['findings'].append(f'MTP {mode}: deterministic responses mismatch')
 if d['errors']:result['findings'].extend(d['errors'])
 outputs=[]
 for g in d.get('generations',[]):
  meta=g['response']['meta_info'];logprobs=meta.get('output_token_logprobs',[])
  valid=(len(logprobs)==meta['completion_tokens'] and
   all(isinstance(x[0],(float,int)) and math.isfinite(x[0]) and type(x[1]) is int for x in logprobs))
  if not valid:result['findings'].append(f'MTP {mode}: output token/logprob accounting failed')
  outputs.append({'text':g['response']['text'],'completion_tokens':meta['completion_tokens'],
   'output_token_ids':[x[1] for x in logprobs],'logprobs_valid':valid,
   'client_first_text_s':g['client_first_text_s'],'duration_s':g['duration_s']})
 result['cases'].append({'mtp':mode,'precision':d.get('precision','bf16'),'startup_s':d['startup_s'],'outputs':outputs,
  'acceptance':d['mtp_acceptance_metrics'],'cleanup':d['cleanup'],
  'metrics_coverage':d['metrics_coverage'],'raw_path':str(p.relative_to(root))})
 if mode=='on' and not any(m['metric']=='sglang:spec_accept_rate' and m['value']>0 for m in d['mtp_acceptance_metrics']):
  result['findings'].append('MTP acceptance was not positive on either smoke request')
paths=[root/f'telemetry/native-qwen-serving-v{attempt}/gpu-nodes-0/{name}.jsonl'
 for name in ['nvidia-smi','nvlink','infiniband','cpu-memory-numa','lustre']]
paths.append(root/f'telemetry/lustre-qwen-serving-v{attempt}/gpu-nodes-0/lustre.jsonl')
for p in paths:
 if not p.is_file():
  result['findings'].append('Missing finalized telemetry: '+str(p.relative_to(root)));continue
 count=errors=0;times=set();groups=collections.defaultdict(list);ids=set()
 with p.open() as f:
  for line in f:
   row=json.loads(line);count+=1;times.add(row['monotonic_s'])
   if row['metric']=='collector_error':errors+=1;continue
   if p.name=='nvidia-smi.jsonl':
    ids.add(row['gpu_uuid']);groups[(row['gpu_uuid'],row['metric'],row['unit'])].append(row['value'])
 if errors or not count:result['findings'].append('Missing/error samples in '+str(p.relative_to(root)))
 if p.name=='nvidia-smi.jsonl' and len(ids)!=8:result['findings'].append('GPU telemetry does not cover eight UUIDs')
 ts=sorted(times);gaps=[b-a for a,b in zip(ts,ts[1:])]
 result['telemetry'].append({'path':str(p.relative_to(root)),'records':count,'sample_times':len(ts),
  'collector_errors':errors,'interval_s':summary(gaps) if gaps else None})
 for (gpu,name,unit),values in sorted(groups.items()):
  result['gpu_statistics'].append({'gpu_uuid':gpu,'metric':name,'unit':unit,'statistics':summary(values)})
print(json.dumps(result,allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('02-qwen-serving-result-audit-attempt-' + str(args.attempt))
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    code = inspect.getsource(percentile) + '\n' + inspect.getsource(summary) + '\n' + AUDIT
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', code, remote, str(args.attempt)], timeout=60)
    data = json.loads(out) if not rc else {'findings': ['Read-only serving audit failed.']}
    data['scope'] = ('Single-node deterministic serving smoke, not RL, throughput/latency characterization, '
                     'a full telemetry qualification or quality evaluation. MTP timings are not a controlled comparison.')
    data['limitations'] = [
        'Native coverage here does not establish all required DCGM, XID, throttle, Ray, Miles or weight-transfer metrics.',
        'Short deterministic prompts do not establish task quality or trainer/rollout logprob equivalence.']
    if data.get('cases') and not any(case['metrics_coverage'].get('finite_forward_timer_samples', 0) for case in data['cases']):
        data['limitations'].append('No finite forward-timer sample during these very short requests.')
    atomic(phase.path / 'audit.json', data)
    values = []
    for case in data.get('cases', []):
        values.append(metric('server_startup', case['startup_s'], 's', 'gpu-nodes-0', mtp=case['mtp']))
    for row in data.get('gpu_statistics', []):
        for name, value in row['statistics'].items():
            if value is not None:
                values.append(dict(metric(row['metric'] + '.' + name, value,
                    'count' if name == 'n' else 'ratio' if name == 'coefficient_of_variation' else row['unit'],
                    'gpu-nodes-0'), gpu_uuid=row['gpu_uuid']))
    phase.finish('fail' if data['findings'] else 'ok', results=values, metadata=data,
                 failure_summary='; '.join(data['findings']) or None)
    print(json.dumps({'findings': data['findings'], 'telemetry_streams': len(data.get('telemetry', [])),
                      'gpu_statistics': len(data.get('gpu_statistics', []))}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
