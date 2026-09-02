"""Audit terminal Slurm state, exact comparison rows and finalized telemetry."""
import argparse
import json

from evidence import Run, atomic, metric


AUDIT = r'''
import collections,hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);attempt=int(sys.argv[2]);job=sys.argv[3]
phase=root/f'tests/02-checkpoint-parity-child-v{attempt}'
out={'findings':[],'coverage':[],'comparison_counts':{},'slurm_job_id':job}
accounting=subprocess.check_output(['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed'],text=True)
out['slurm_accounting']=accounting
parent=[line.split('|') for line in accounting.splitlines() if line.split('|')[0]==job]
if len(parent)!=1 or parent[0][1] not in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL']:
 raise ValueError('Job is not unambiguously terminal; do not finalize this audit.')
if parent[0][1:3]!=['COMPLETED','0:0']:out['findings'].append('Slurm job did not complete with exit zero.')
result_path=phase/'parity-result.json'
if result_path.is_file():
 result=json.loads(result_path.read_text());out['parity']=result
 out['findings'].extend(result['findings'])
else:out['findings'].append('No finalized parity child result.')
p=phase/'tensor-comparisons.jsonl'
if p.is_file():
 counts=collections.Counter();digest=hashlib.sha256();names=set();bad=[]
 contract=out.get('parity',{}).get('comparison_contract',{});version=contract.get('version',1)
 allowed={f'model.language_model.layers.{i}.linear_attn.A_log' for i in range(40) if i%4!=3}
 if version==2 and (set(contract['allowlisted_widenings'])!=allowed or contract['upstream_source_sha256']!='2e64b4703a26b786a1c7026be67d3c090f35981947f198a31c80145db472d009'):
  bad.append('Comparison contract differs from the pinned narrow widening rule.')
 with p.open('rb') as f:
  for line in f:
   digest.update(line);row=json.loads(line);name=row['converted_name']
   if name in names:bad.append('Duplicate converted name: '+name)
   names.add(name);counts['compared']+=1;counts['mtp_compared']+=int(name.startswith('mtp.'))
   counts['equal' if row['equal'] else 'mismatched']+=1
   if row['equal'] and any(row[a]!=row[b] for a,b in [('shape','reference_shape'),('dtype','reference_dtype'),('actual_sha256','reference_sha256')]):
    bad.append('Invalid equality claim: '+name)
   if version==2:
    counts['qualified' if row['qualified'] else 'rejected']+=1
    widened=row['qualification']=='lossless_bf16_to_fp32_alog';counts['lossless_widened']+=int(widened)
    w=row.get('widening',{})
    valid_widening=(widened and name in allowed and row['shape']==row['reference_shape']==[32]
     and row['dtype']=='torch.float32' and row['reference_dtype']=='torch.bfloat16'
     and w.get('lift_exact') is True and w.get('inverse_exact') is True
     and w.get('reference_lifted_fp32_sha256')==row['actual_sha256']
     and w.get('inverse_bf16_sha256')==row['reference_sha256']
     and w.get('max_absolute_difference_in_fp32')==0)
    qualified=row['finite'] and ((row['equal'] and row['qualification']=='bitwise_equal') or valid_widening)
    if row['qualified']!=qualified:bad.append('Invalid qualified parity claim: '+name)
 out['comparison_counts']=dict(counts);out['comparison_sha256']=digest.hexdigest()
 if counts!=collections.Counter(out.get('parity',{}).get('counts',{})):bad.append('Child counts differ from raw comparison rows.')
 if not counts['compared'] or not counts['mtp_compared']:bad.append('Incomplete text/MTP weight comparisons.')
 if version==1 and counts['mismatched']:bad.append('Mismatched strict dtype/byte comparisons.')
 if version==2 and (counts['rejected'] or counts['lossless_widened']!=30 or counts['qualified']!=counts['compared']):bad.append('Incomplete or rejected qualified weight comparisons.')
 if version not in [1,2]:bad.append('Unknown comparison contract version.')
 out['findings'].extend(bad)
else:out['findings'].append('No finalized full comparison JSONL.')
paths=[root/f'telemetry/native-checkpoint-parity-v{attempt}/gpu-nodes-0/{name}.jsonl'
 for name in ['nvidia-smi','nvlink','infiniband','cpu-memory-numa','lustre']]
paths.append(root/f'telemetry/lustre-checkpoint-parity-v{attempt}/gpu-nodes-0/lustre.jsonl')
for p in paths:
 count=errors=0;times=set();gpu=set();before={};after={}
 if not p.is_file():out['findings'].append('Missing finalized telemetry: '+str(p.relative_to(root)));continue
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for line in f:
   digest.update(line);row=json.loads(line);count+=1;times.add(row['monotonic_s'])
   if row['metric']=='collector_error':errors+=1;continue
   if p.name=='nvidia-smi.jsonl':gpu.add(row['gpu_uuid'])
   if p.name=='lustre.jsonl' and row['metric'] in ['read_bytes.sum','write_bytes.sum']:
    before.setdefault(row['metric'],row['value']);after[row['metric']]=row['value']
 if errors or count==0:out['findings'].append('Telemetry errors or empty stream: '+str(p.relative_to(root)))
 if p.name=='nvidia-smi.jsonl' and len(gpu)!=8:out['findings'].append('GPU UUID coverage is not eight.')
 ordered=sorted(times);gaps=[b-a for a,b in zip(ordered,ordered[1:])]
 out['coverage'].append({'path':str(p.relative_to(root)),'sha256':digest.hexdigest(),'records':count,
  'sample_times':len(times),'collector_errors':errors,'max_interval_s':max(gaps) if gaps else None,
  'host_client_counter_deltas':{k:after[k]-v for k,v in before.items()}})
out['scope']='Audit of exact text/MTP weight parity and this stage telemetry only; no optimizer or held-out result.'
out['notes']=['Payload rehash precedes checkpoint loading; read timing is cache-warm, not a cold restart benchmark.',
 'Host Lustre deltas include other host clients, not exclusively this process.',
 'Raw earlier MTP metadata-key count includes extra_state objects; weight tensor count is reported separately.']
print(json.dumps(out,allow_nan=False))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--job-id', type=int, required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-checkpoint-parity-result-audit-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', AUDIT, remote, str(args.attempt), str(args.job_id)], timeout=90)
    data = json.loads(out) if not rc else {'findings': ['Read-only parity audit failed; inspect captured stderr.']}
    atomic(phase.path / 'audit.json', data)
    values = [metric(k, v, 'count', 'gpu-nodes-0') for k, v in data.get('comparison_counts', {}).items()]
    phase.finish('fail' if data['findings'] else 'ok', results=values, metadata=data,
                 failure_summary='; '.join(data['findings']) or None)
    print(json.dumps({'findings': data['findings'], 'comparison_counts': data.get('comparison_counts'),
                      'coverage_streams': len(data.get('coverage', []))}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
