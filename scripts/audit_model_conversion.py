"""Collect finalized conversion provenance and audit its telemetry coverage."""
import argparse
import json
from pathlib import Path

from evidence import Run, atomic, metric


AUDIT = r'''
import collections,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);attempt=int(sys.argv[2]);phase=root/f'tests/02-model-conversion-child-v{attempt}'
result=json.loads((phase/'conversion-result.json').read_text())
if result['findings'] or result['conversion_exit_code']!=0:raise ValueError('Conversion did not finish successfully')
checkpoint=root/result['checkpoint_relpath'];manifest_bytes=(checkpoint/'conversion.manifest.json').read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest()!=result['manifest_sha256']:raise ValueError('Conversion manifest hash mismatch')
manifest=json.loads(manifest_bytes)
for file in manifest['files']:
 p=checkpoint/file['path']
 if p.is_symlink() or p.stat().st_size!=file['bytes']:raise ValueError('Converted file changed size/type')
out={'conversion':result,'manifest':manifest,'coverage':[],'gpu_peak_hbm_mib':{},'warnings':{},'findings':[]}
paths=[root/f'telemetry/native-model-conversion-v{attempt}/gpu-nodes-0/{name}.jsonl'
 for name in ['nvidia-smi','nvlink','infiniband','cpu-memory-numa','lustre']]
paths.append(root/f'telemetry/lustre-model-conversion-v{attempt}/gpu-nodes-0/lustre.jsonl')
for p in paths:
 count=errors=0;times=set();gpu=set();before={};after={}
 if not p.is_file():out['findings'].append('Missing finalized telemetry: '+str(p.relative_to(root)));continue
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for line in f:
   digest.update(line);row=json.loads(line);count+=1;times.add(row['monotonic_s'])
   if row['metric']=='collector_error':errors+=1;continue
   if p.name=='nvidia-smi.jsonl':
    gpu.add(row['gpu_uuid'])
    if row['metric']=='memory.used':
     out['gpu_peak_hbm_mib'][row['gpu_uuid']]=max(out['gpu_peak_hbm_mib'].get(row['gpu_uuid'],0),row['value'])
   if p.name=='lustre.jsonl' and row['metric'] in ['read_bytes.sum','write_bytes.sum']:
    before.setdefault(row['metric'],row['value']);after[row['metric']]=row['value']
 if errors or count==0:out['findings'].append('Telemetry errors/empty stream: '+str(p.relative_to(root)))
 if p.name=='nvidia-smi.jsonl' and len(gpu)!=8:out['findings'].append('GPU UUID coverage is not eight')
 ordered=sorted(times);gaps=[b-a for a,b in zip(ordered,ordered[1:])]
 out['coverage'].append({'path':str(p.relative_to(root)),'sha256':digest.hexdigest(),'records':count,
  'sample_times':len(times),'collector_errors':errors,'max_interval_s':max(gaps) if gaps else None,
  'host_client_counter_deltas':{k:after[k]-v for k,v in before.items()}})
text=(phase/'logs/conversion.out').read_text()+(phase/'logs/conversion.err').read_text()
for key,needle in {'qwen_asr_docstring':'[ERROR] `cache_position`','rank_common_args':'common state dict differs',
 'deprecated_checkpoint_backend':'MCore\'s async save is deprecated','empty_python_stack':'<no Python frame>',
 'router_dtype':'without fp32 routing'}.items():out['warnings'][key]=text.count(needle)
out['scope']='Candidate weights and native coverage only. No tensor parity, trainer step, optimizer resume or quality result is established.'
out['notes']=['Host Lustre deltas include other clients on this host and are not isolated checkpoint throughput.',
 'The conversion itself hashed every payload; this audit rechecks manifest identity and file sizes, not all payload bytes.',
 'All warning categories remain review items; zero exit status is not a general correctness certificate.']
print(json.dumps(out,allow_nan=False))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-model-conversion-result-audit-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', AUDIT, remote, str(args.attempt)], timeout=90)
    data = json.loads(out) if not rc else {'findings': ['Read-only conversion audit failed.']}
    atomic(phase.path / 'audit.json', data)
    values = []
    if not rc:
        atomic(run.root / 'provenance/converted-checkpoint-manifest.json', data['manifest'])
        for key, unit in [('checkpoint_bytes', 'B'), ('checkpoint_files', 'count'),
                          ('conversion_duration_s', 's'), ('hash_duration_s', 's'), ('mtp_tensor_keys', 'count')]:
            values.append(metric(key, data['conversion'][key], unit, 'gpu-nodes-0'))
        for gpu, value in data['gpu_peak_hbm_mib'].items():
            values.append(dict(metric('peak_used_hbm', value*1024**2, 'B', 'gpu-nodes-0'), gpu_uuid=gpu))
    phase.finish('fail' if data['findings'] else 'ok', results=values, metadata=data,
                 failure_summary='; '.join(data['findings']) or None)
    print(json.dumps({'findings': data['findings'], 'coverage_streams': len(data.get('coverage', [])),
                      'warnings': data.get('warnings')}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
