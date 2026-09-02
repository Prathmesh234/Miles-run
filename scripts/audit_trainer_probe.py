"""Read-only audit of terminal trainer-probe state, rank evidence and telemetry."""
import argparse
import inspect
import json
import math

from evidence import Run, atomic, metric


def validate_rank_evidence(before, after, grads, forward, inputs, result):
    """Cross-check finalized records, without claiming to recompute GPU tensors."""
    findings = []
    key = lambda row: (row['chunk'], row['name'])
    if not before or before != after or len({key(row) for row in before}) != len(before):
        findings.append('Parameter hashes changed, missing or duplicated.')
    if {key(row) for row in before} != {key(row) for row in grads} or len(grads) != len(before):
        findings.append('Gradient inventory differs from parameters.')
    for row in before:
        digest = row.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            findings.append('Invalid parameter digest.')
    for grad in grads:
        if grad['mtp'] != ('.mtp.' in grad['name']):
            findings.append('MTP gradient classification differs from parameter name.')
        if grad.get('present'):
            stats = [grad.get(k, float('nan')) for k in ['max_abs', 'local_buffer_l2']]
            if grad.get('finite') is not True or not all(math.isfinite(x) and x >= 0 for x in stats):
                findings.append('Gradient statistics are nonfinite or negative.')
            elif bool(grad.get('nonzero')) != all(x > 0 for x in stats):
                findings.append('Gradient nonzero flag differs from statistics.')
        elif grad.get('nonzero'):
            findings.append('Absent gradient is claimed nonzero.')
    counts = {'gradient_tensors_present': sum(g.get('present', False) for g in grads),
              'gradient_tensors_nonzero': sum(g.get('nonzero', False) for g in grads),
              'mtp_gradient_tensors_nonzero': sum(g['mtp'] and g.get('nonzero', False) for g in grads)}
    if any(result.get(k) != value for k, value in counts.items()):
        findings.append('Reported gradient counts differ from raw gradient rows.')
    for mtp in (False, True):
        if not any(g['mtp'] == mtp and g.get('present') and g.get('nonzero') for g in grads):
            findings.append('Missing nonzero ' + ('MTP' if mtp else 'main') + ' gradients.')
    ids = inputs.get('token_ids', [])
    if (len(ids) != 128 or inputs.get('cu_seqlens') != [0, 128]
            or any(type(x) is not int or not 0 <= x < 248320 for x in ids)):
        findings.append('Packed input mismatch.')
    lp = forward.get('teacher_forced_log_probs', [[]])
    valid_lp = (len(lp) == 1 and len(lp[0]) == 127 and all(math.isfinite(x) and x <= 0 for x in lp[0]))
    if forward.get('logit_shape') != [1, 128, 248320] or not valid_lp:
        findings.append('Logits/logprob shape or finiteness mismatch.')
    loss = forward.get('main_cross_entropy', float('nan'))
    if not math.isfinite(loss) or loss < 0:
        findings.append('Main diagnostic loss is nonfinite or negative.')
    elif valid_lp and not math.isclose(loss, -sum(lp[0]) / 127, rel_tol=1e-5, abs_tol=1e-5):
        findings.append('Diagnostic loss differs from recorded token logprobs.')
    if result.get('parameters_unchanged') is not True:
        findings.append('Rank did not attest unchanged parameters.')
    return findings, counts


AUDIT = r'''
import collections,hashlib,json,math,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);attempt=int(sys.argv[2]);job=sys.argv[3]
phase=root/f'tests/02-trainer-probe-child-v{attempt}'
out={'findings':[],'ranks':[],'coverage':[],'slurm_job_id':job}
accounting=subprocess.check_output(['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed'],text=True)
out['slurm_accounting']=accounting
parents=[x.split('|') for x in accounting.splitlines() if x.split('|')[0]==job]
if len(parents)!=1 or parents[0][1] not in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL']:
 raise ValueError('Job is not unambiguously terminal; do not finalize this audit.')
if parents[0][1:3]!=['COMPLETED','0:0']:out['findings'].append('Slurm job did not complete with exit zero.')
result_path=phase/'result.json'
if result_path.is_file():
 result=json.loads(result_path.read_text());out['child_findings']=result['findings'];out['findings'].extend(result['findings'])
 out['imports']=result.get('imports');out['input_rehash_duration_s']=result.get('input_rehash_duration_s')
else:out['findings'].append('No finalized child result.')
expected={'tp':1,'pp':1,'cp':1,'ep':8,'etp':1,'dense_dp':8,'expert_dp':1}
for rank in range(8):
 p=phase/f'ranks/rank-{rank:02d}';result=p/'result.json'
 if not result.is_file():out['findings'].append(f'Missing finalized rank {rank}.');continue
 r=json.loads(result.read_text());item={k:v for k,v in r.items() if k!='forward'}
 item['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
 if r.get('status')!='ok':out['findings'].append(f'Rank {rank} failed: '+r.get('failure_summary','unknown'))
 if r.get('topology')!=expected:out['findings'].append(f'Rank {rank} topology mismatch.')
 if r.get('optimizer_constructed') is not False or r.get('optimizer_steps')!=0:
  out['findings'].append(f'Rank {rank} no-optimizer contract mismatch.')
 names=['parameters-before.json','parameters-after.json','gradients.json','forward.json','input.json']
 if all((p/name).is_file() for name in names):
  before,after,grads,forward,inputs=[json.loads((p/name).read_text()) for name in names]
  findings,counts=validate_rank_evidence(before,after,grads,forward,inputs,r)
  out['findings'].extend(f'Rank {rank}: '+finding for finding in findings)
  item.update(counts)
  item.update(parameter_tensor_count=len(before),parameter_bytes=sum(x['bytes'] for x in before),
   main_cross_entropy=forward['main_cross_entropy'],forward_backward_duration_s=forward['forward_backward_duration_s'],
   artifact_hashes={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in names})
 else:out['findings'].append(f'Rank {rank} has incomplete finalized audit artifacts.')
 out['ranks'].append(item)
if len(out['ranks'])==8 and len({r['gpu_uuid'] for r in out['ranks']})!=8:
 out['findings'].append('CUDA rank UUIDs are not eight unique devices.')
paths=[root/f'telemetry/native-trainer-probe-v{attempt}/gpu-nodes-0/{name}.jsonl'
 for name in ['nvidia-smi','nvlink','infiniband','cpu-memory-numa','lustre']]
paths.append(root/f'telemetry/lustre-trainer-probe-v{attempt}/gpu-nodes-0/lustre.jsonl')
for p in paths:
 if not p.is_file():out['findings'].append('Missing finalized telemetry: '+str(p.relative_to(root)));continue
 count=errors=0;times=set();gpu=set();digest=hashlib.sha256()
 with p.open('rb') as f:
  for line in f:
   digest.update(line);row=json.loads(line);count+=1;times.add(row['monotonic_s'])
   if row['metric']=='collector_error':errors+=1;continue
   if p.name=='nvidia-smi.jsonl':gpu.add(row['gpu_uuid'])
 if not count or errors:out['findings'].append('Empty telemetry or collector errors: '+str(p.relative_to(root)))
 if p.name=='nvidia-smi.jsonl' and len(gpu)!=8:out['findings'].append('GPU telemetry UUID coverage is not eight.')
 ordered=sorted(times);gaps=[b-a for a,b in zip(ordered,ordered[1:])]
 out['coverage'].append({'path':str(p.relative_to(root)),'sha256':digest.hexdigest(),'records':count,
  'sample_times':len(times),'collector_errors':errors,'max_interval_s':max(gaps) if gaps else None})
out['scope']='Native EP8 checkpoint load, packed forward/backward and finite-gradient/unchanged-parameter audit only. Not GRPO, optimizer, resume, independent EP8 shard-value parity, serving-logprob equivalence or held-out quality.'
out['notes']=['Input rehash precedes checkpoint load, so load timing is cache-warm.',
 'Per-rank gradient-buffer norms are not deduplicated global gradient norms.',
 'The 128-token diagnostic main loss is cross entropy; native MTP auxiliary loss remains enabled.']
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
    phase = run.phase(f'02-trainer-probe-result-audit-v{args.attempt}')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c',
        'import math\n'+inspect.getsource(validate_rank_evidence)+'\n'+AUDIT,
        remote, str(args.attempt), str(args.job_id)], timeout=120)
    data = json.loads(out) if not rc else {'findings': ['Read-only trainer audit failed; inspect stderr.']}
    atomic(phase.path / 'audit.json', data)
    values = []
    for rank in data.get('ranks', []):
        for key, unit in [('checkpoint_load_duration_s', 's'), ('forward_backward_duration_s', 's'),
                          ('cuda_peak_allocated_bytes', 'B'), ('gradient_tensors_nonzero', 'count'),
                          ('mtp_gradient_tensors_nonzero', 'count')]:
            if key in rank:
                values.append(metric(key, rank[key], unit, rank['hostname'], rank=rank['rank'], gpu_uuid=rank['gpu_uuid']))
    phase.finish('fail' if data['findings'] else 'ok', results=values, metadata=data,
                 failure_summary='; '.join(data['findings']) or None)
    print(json.dumps({'findings': data['findings'], 'rank_records': len(data.get('ranks', [])),
                      'coverage_streams': len(data.get('coverage', []))}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
