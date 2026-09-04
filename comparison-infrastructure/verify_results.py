"""Verify preserved evidence and explicit completion claims before publication."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent

def main():
    data=json.loads((ROOT/'results.json').read_text())
    scheduler=json.loads((ROOT/'evidence/final-scheduler.json').read_text())
    states={}
    for line in scheduler['stdout'].splitlines()[1:]:
        if line.strip():
            fields=line.split('|');states[int(fields[0])]=dict(zip(['job','state','exit','start','end','elapsed'],fields))
    verification={'verified_utc':datetime.now(timezone.utc).isoformat(),'runs':{},'limitations':['Weights-only; no full distributed optimizer/RNG resume test','No scaling sweep or isolated fabric capacity benchmark']}
    for run in data['runs']:
        job=run['job_id'];root=Path(run['source_root']);state=states[job]
        assert state['state']=='COMPLETED' and state['exit']=='0:0',(job,state)
        assert run['exit_code']=='0' and len(run['batches'])==2
        assert all(b['samples']==16 and b['active_tokens']>0 for b in run['batches'])
        for file,digest in run['source_sha256'].items():
            assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest,(job,file)
        snapshot=root/'snapshot-manifest.json';verified_count=0
        if snapshot.exists():
            for item in json.loads(snapshot.read_text())['included']:
                assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
                verified_count+=1
        checks={}
        for role,name in [('actor','checkpoint-verification.json'),('critic','critic-checkpoint-verification.json')]:
            if role=='critic' and job==190:continue
            check=json.loads((root/name).read_text())
            assert check['completed_optimizer_updates']==2
            assert len(check['shard_files'])==16
            assert len(check['sampled_tensor_reads'])>=3
            assert all(x['finite'] for x in check['sampled_tensor_reads'].values())
            if job in [196,197]:
                assert any(x['changed_elements']>0 for x in check['selected_tensors_vs_base'].values())
            checks[role]={'source':name,'shards':len(check['shard_files']),'storage_entries_checked':check['storage_entries_checked'],
                          'finite_cpu_tensor_reads':len(check['sampled_tensor_reads']),
                          'sampled_changed_elements_vs_base':sum(x['changed_elements'] for x in check.get('selected_tensors_vs_base',{}).values())}
        gates=[]
        for b in run['batches']:
            assert math.isfinite(b['scalar_actor']['train/grad_norm']) and b['scalar_actor']['train/grad_norm']>0
            if job in [196,197]:
                grad=b['scalar_critic']['train/critic-grad_norm'];assert math.isfinite(grad) and grad>0
                gates.append(json.loads((root/f"policy-validity-{b['step']}.json").read_text()))
        warning_counts=Counter()
        for line in (root/'training.log').read_text(errors='replace').splitlines():
            for label,needle in [('startup_http_retry','Connection refused'),('gloo_peer_reset','Connection reset by peer'),('traceback','Traceback (most recent call last)')]:
                if needle in line:warning_counts[label]+=1
        if job==197:
            assert [b['recorded_policy_lag'] for b in run['batches']]==[0,1]
            assert run['rollout_compute_overlap_seconds']>0
            args=run['validated_arguments'];assert args['use_tis'] and not args['use_rollout_logprobs'] and args['calculate_per_token_loss']
            assert args['optimizer_cpu_offload'] and args['offload_train'] and args['tis_clip']==2 and args['tis_clip_low']==0
            assert len(run['optimizer_steps'])==64 and all(p['ok'] for p in run['optimizer_steps'])
            groups=Counter((p['resolved_role'],p['rollout_id']) for p in run['optimizer_steps'])
            assert groups==Counter({('actor',0):16,('actor',1):16,('critic',0):16,('critic',1):16}),groups
            # Rollout hook emits an end record but no `ok` field; validate its
            # produced batches/gates separately. Timed awaits must report ok.
            assert all(p.get('ok') is True for p in run['phases'] if p['role']=='driver' and p['name']!='rollout')
            assert len(run['weight_transfers'])==3
            for p in run['weight_transfers'][1:]:assert all(v['overlapping_bins']>0 for v in p['ib_TX_highrate_bins'].values())
            for b in run['batches']:
                for key in ['train/tis','train/tis_weight','train/tis_weight_squared','train/tis_upper_clipfrac']:
                    assert math.isfinite(b['scalar_actor'][key])
                assert 0<=b['scalar_actor']['train/tis_weight']<=2
        verification['runs'][str(job)]={'slurm_state':state['state'],'slurm_exit':state['exit'],
            'start_utc':state['start'],'end_utc':state['end'],'allocated_elapsed':state['elapsed'],
            'training_exit':run['exit_code'],'input_hashes_checked':len(run['source_sha256']),
            'snapshot_files_checked':verified_count,'checkpoint_checks':checks,'policy_validity':gates,
            'warning_line_counts_not_unique_events':dict(warning_counts),
            'optimizer_role_assignment_basis':dict(Counter(p['role_assignment_basis'] for p in run['optimizer_steps']))}
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'verified_jobs':list(verification['runs']),'output':str(ROOT/'verification.json')}))

if __name__=='__main__':main()
