"""Analyze recorded workload telemetry, not synthetic scaling benchmarks.

Only standard-library dependencies. Rates retain their sampling intervals;
missing data is never zero-filled. Run `python3 analyze.py --help`.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def epoch(value):
    if isinstance(value, (int, float)): return float(value)
    value = re.sub(r'(\.\d{6})\d+', r'\1', value).replace('Z', '+00:00')
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def jsonlines(path):
    if not path.exists(): return
    with path.open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''): result.update(chunk)
    return result.hexdigest()


def stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values: return None
    def q(p):
        index = (len(values)-1)*p; low = int(index); high = min(low+1, len(values)-1)
        return values[low]+(values[high]-values[low])*(index-low)
    return dict(count=len(values), mean=statistics.mean(values), min=values[0],
                p50=q(.5), p95=q(.95), max=values[-1])


def overlap(a, b, c, d): return max(0., min(b, d)-max(a, c))


def union_seconds(intervals):
    merged = []
    for a,b in sorted(intervals):
        if merged and a <= merged[-1][1]: merged[-1][1] = max(b, merged[-1][1])
        else: merged.append([a,b])
    return sum(b-a for a,b in merged)


def hardware(root):
    result = {}
    for node in range(4):
        host = f'gpu-nodes-{node}'
        inventory = read_json(root/'infra'/f'{host}-before.json', {})
        ports, links, gpu = {}, {}, None
        for command in inventory.get('commands', []):
            if command.get('exit_code') != 0: continue
            if command['argv'] == ['ibstat']:
                for device, body in re.findall(r"CA '([^']+)'\n(.*?)(?=\nCA '|\Z)", command['stdout'], re.S):
                    speed = re.search(r'Rate: (\d+)', body)
                    if speed and 'Link layer: InfiniBand' in body and 'State: Active' in body:
                        ports[device] = float(speed[1])
            if command['argv'] == ['nvidia-smi', 'nvlink', '-s']:
                for line in command['stdout'].splitlines():
                    m = re.search(r'UUID: (GPU-[^)]+)', line)
                    if m: gpu = m[1]
                    m = re.search(r'Link (\d+): ([\d.]+) GB/s', line)
                    if m and gpu: links[gpu+'/'+m[1]] = float(m[2])
        result[host] = {'ib_port_gbps': ports, 'nvlink_reported_line_GBps': links,
                        'nvlink_nominal_payload_GBps_per_link': 50.,
                        'nvlink_basis': 'B200 nominal 1.8 TB/s bidirectional / 18 / 2; not measured capacity'}
    return result


def timers_and_scalars(root):
    timers, starts, scalars = [], {}, defaultdict(dict)
    path = root/'training.log'
    if not path.exists(): return timers, scalars
    for line in path.read_text(errors='replace').splitlines():
        m = re.search(r'\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) (actor|critic)_cell0_rank0\]', line)
        if not m: continue
        stamp, role = m.groups(); t = epoch(stamp)
        timer = re.search(r'Timer (\w+) (start|end)', line)
        if timer:
            name, phase = timer.groups(); key = role, name
            if phase == 'start': starts[key] = t
            elif key in starts:
                a = starts.pop(key)
                timers.append(dict(role=role, name=name, start=a, end=t, seconds=t-a, source='rank0 native Timer'))
        metric = re.search(r'(?:step|perf) (\d+): (\{.*\})', line)
        if metric:
            try: value = ast.literal_eval(metric[2])
            except (ValueError, SyntaxError): continue
            for k,v in value.items():
                if isinstance(v, (int,float)) and math.isfinite(v): scalars[(role,int(metric[1]))][k] = v
    extracted = read_json(root/'metrics-extracted.json', {})
    for k, rows in extracted.get('scalars', {}).items():
        if k.startswith(('train/', 'perf/', 'rollout/')):
            for row in rows: scalars[('actor',row['step'])].setdefault(k, row['value'])
    return timers, scalars


def fabric(root, hw):
    rows, errors = [], Counter()
    for node in range(4):
        host = f'gpu-nodes-{node}'; previous = {}
        for sample in jsonlines(root/'infra'/f'{host}-rdma-counters.jsonl'):
            t = epoch(sample['utc']); clock = sample.get('monotonic', t)
            for port in sample['ports']:
                device = port['argv'][2]
                if port.get('status') != 0:
                    errors[f'{host}:IB:{device}:collector_error'] += 1; continue
                counters = {k:int(v) for k,v in re.findall(r'^([A-Za-z0-9]+):\.+([0-9]+)', port['stdout'], re.M)}
                capacity = hw[host]['ib_port_gbps'].get(device)
                for key, direction in [('PortXmitData','TX'), ('PortRcvData','RX')]:
                    if key not in counters: continue
                    ident = device,direction
                    if ident in previous:
                        t0,clock0,count0 = previous[ident]; delta = counters[key]-count0; elapsed=clock-clock0
                        if elapsed > 0 and t > t0 and delta >= 0:
                            gbps = delta*4*8/elapsed/1e9
                            rows.append(dict(fabric='IB',host=host,device=device,direction=direction,
                                start=t0,end=t,interval_s=elapsed,bytes=delta*4,gbps=gbps,capacity_gbps=capacity,
                                utilization_pct=100*gbps/capacity if capacity else None))
                        else: errors[f'{host}:IB:{device}:counter_reset_or_clock'] += 1
                    previous[ident] = t,clock,counters[key]
        previous = {}
        for sample in jsonlines(root/'infra'/f'{host}-health.jsonl'):
            t = epoch(sample['time']); clock=sample.get('monotonic',t)
            for key,value in sample['sources'].items():
                if value.get('status') != 'ok': errors[f'{host}:{key}:collector_error'] += 1
            value = sample['sources'].get('nvlink',{})
            if value.get('status') != 'ok': continue
            gpu = None
            for line in value['stdout'].splitlines():
                match=re.search(r'UUID: (GPU-[^)]+)',line)
                if match: gpu=match[1]
                match=re.search(r'Link (\d+): Data (Tx|Rx): (\d+) KiB',line)
                if not match or not gpu: continue
                link,direction,count=match.groups();count=int(count);ident=gpu,link,direction
                if ident in previous:
                    t0,clock0,count0=previous[ident];delta=count-count0;elapsed=clock-clock0
                    if elapsed>0 and t>t0 and delta>=0:
                        gbps=delta*1024*8/elapsed/1e9
                        rows.append(dict(fabric='NVLink',host=host,device=gpu+'/'+link,direction=direction.upper(),
                            start=t0,end=t,interval_s=elapsed,bytes=delta*1024,gbps=gbps,
                            capacity_gbps=400.,utilization_pct=gbps/4))
                    else: errors[f'{host}:NVLink:counter_reset_or_clock'] += 1
                previous[ident]=t,clock,count
    return rows, dict(errors)


def aggregate_fabric(rows, hw):
    groups=defaultdict(list)
    for row in rows:
        # The 100-Gb/s storage ports are not GPU scale-out links.
        if row['fabric']=='IB' and row['capacity_gbps'] != 400: continue
        groups[(row['fabric'],row['host'],row.get('group_end',row['end']),row['direction'])].append(row)
    result=[]
    for (kind,host,end,direction),items in sorted(groups.items()):
        expected = sum(v==400 for v in hw[host]['ib_port_gbps'].values()) if kind=='IB' else len(hw[host]['nvlink_reported_line_GBps'])
        actual=len({r['device'] for r in items}); gbps=sum(r['gbps'] for r in items)
        complete = expected>0 and actual==expected
        result.append(dict(fabric=kind,host=host,end=end,start=min(r['start'] for r in items),direction=direction,
            gbps=gbps,capacity_gbps=expected*400.,utilization_pct=100*gbps/(expected*400.) if complete else None,
            hottest_link_utilization_pct=max((r['utilization_pct'] for r in items if r['utilization_pct'] is not None),default=None),
            collected_links=actual,expected_links=expected,complete=complete))
    return result


def highrate_fabric(root):
    rows=[];errors=Counter();costs=[]
    for node in range(4):
        host=f'gpu-nodes-{node}';previous={}
        for sample in jsonlines(root/'infra'/f'{host}-fabric-1s.jsonl'):
            end=epoch(sample['time']);costs.append(sample['collector_wall_seconds'])
            counters=[]
            for item in sample['ib']:
                if item['status']!=0:
                    errors[f'{host}:IB:{item["device"]}']+=1;continue
                for key,direction in [('PortXmitData','TX'),('PortRcvData','RX')]:
                    if key in item.get('counters',{}):
                        counters.append(('IB',item['device'],direction,item['counters'][key]*4,item))
            item=sample['nvlink']
            if item['status']!=0: errors[f'{host}:NVLink']+=1
            else:
                for device,values in item['counters'].items():
                    for direction,count in values.items():
                        counters.append(('NVLink',device,direction.upper(),count*1024,item))
            for kind,device,direction,count,item in counters:
                key=kind,device,direction;t=item['end'];clock=item['monotonic']
                if key in previous:
                    t0,clock0,count0=previous[key];elapsed=clock-clock0;delta=count-count0
                    if elapsed>0 and t>t0 and delta>=0:
                        gbps=delta*8/elapsed/1e9
                        rows.append(dict(fabric=kind,host=host,device=device,direction=direction,
                            start=t0,end=t,group_end=end,interval_s=elapsed,bytes=delta,gbps=gbps,
                            capacity_gbps=400.,utilization_pct=gbps/4))
                    else: errors[f'{host}:{kind}:reset_or_clock']+=1
                previous[key]=t,clock,count
    return rows,dict(errors),stats(costs)


def link_statistics(rows, start, end):
    grouped=defaultdict(list)
    for row in rows:
        if row['direction']=='TX' and row['capacity_gbps']==400 and overlap(row['start'],row['end'],start,end)>0:
            grouped[(row['fabric'],row['host'],row['device'])].append(row['utilization_pct'])
    return [dict(fabric=k[0],host=k[1],device=k[2],utilization_pct=stats(v)) for k,v in sorted(grouped.items())]


def gpu_samples(root):
    result=[]
    for node in range(4):
        host=f'gpu-nodes-{node}'
        for row in jsonlines(root/'infra'/f'{host}-timeseries.jsonl'):
            if row.get('gpu',{}).get('exit_code') != 0: continue
            devices=[]
            for line in row['gpu']['stdout'].splitlines():
                values=line.split(',')
                try: devices.append({'index':int(values[0]),'util_pct':float(values[3]),'memory_GiB':float(values[5])/1024,
                                     'memory_total_GiB':float(values[6])/1024,'power_W':float(values[7])})
                except (ValueError,IndexError): continue
            if len(devices)==8:
                result.append(dict(time=epoch(row['timestamp']),host=host,devices=devices,
                    util_pct=statistics.mean(d['util_pct'] for d in devices),
                    memory_GiB=statistics.mean(d['memory_GiB'] for d in devices),
                    power_W=sum(d['power_W'] for d in devices)))
    return result


def optimizer_role(row, pid_role, phases):
    explicit=row.get('role')
    if explicit in ['actor','critic']: return explicit,'recorded optimizer role'
    explicit=pid_role.get((row['host'],row['pid']))
    if explicit: return explicit,'same-process lifecycle/native-log role'
    # These two train calls are serialized in the pinned disjoint async driver.
    # Infer only when the entire optimizer span belongs to exactly one call.
    start=row['time']-row['elapsed_seconds'];end=row['time']
    candidates={p['name'].removesuffix('_train') for p in phases
                if p['name'] in ['actor_train','critic_train'] and p['start']<=start<=end<=p['end']}
    return (next(iter(candidates)),'inferred from unique enclosing serialized driver train span') if len(candidates)==1 else ('unknown','ambiguous or missing role')


def native_worker_roles(root, events):
    ips={};pids=defaultdict(set);result={}
    for row in events:pids[row['pid']].add(row['host'])
    for path in (root/'infra').glob('*-before.json'):
        inv=read_json(path)
        for cmd in inv.get('commands',[]):
            if cmd['argv']==['ip','-j','address','show'] and cmd.get('exit_code')==0:
                for device in json.loads(cmd['stdout']):
                    for address in device.get('addr_info',[]):
                        if address.get('scope')=='global':ips[address['local']]=inv['hostname']
    for line in (root/'training.log').read_text(errors='replace').splitlines():
        worker=re.search(r'MegatronTrainRayActor pid=(\d+)(?:, ip=([\d.]+))?\)',line)
        label=re.search(r'\d{2}:\d{2}:\d{2}\.\d+ (actor|critic)_cell\d+_rank\d+\]',line)
        if not worker or not label:continue
        pid=int(worker[1]);host=ips.get(worker[2]) if worker[2] else None
        if not worker[2] and len(pids[pid])==1:host=next(iter(pids[pid]))
        if not host:continue
        key=host,pid
        assert key not in result or result[key]==label[1],f'Conflicting native roles: {key}'
        result[key]=label[1]
    return result


def native_lifecycle(root):
    rows=[]
    for line in (root/'training.log').read_text(errors='replace').splitlines():
        stamp=re.search(r'\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) (actor|critic)_cell(\d+)_rank(\d+)\]',line)
        event=re.search(r'ft cls=MegatronTrainRayActor fn=(sleep|wake_up|update_weights) phase=end ok=(true|false) elapsed_s=([\d.]+)',line)
        if stamp and event:
            rows.append(dict(time=epoch(stamp[1]),role=stamp[2],cell=int(stamp[3]),rank=int(stamp[4]),
                operation=event[1],ok=event[2]=='true',elapsed_seconds=float(event[3]),
                source='native structured log; printed precision and Ray deduplication limit coverage'))
    return rows


def analyze_run(root, job, label):
    hw=hardware(root); timers,scalars=timers_and_scalars(root)
    phases=[]; events=list(jsonlines(root/'timeline.jsonl'))
    stamps=[epoch(r['time']) for r in events]
    if not stamps: raise ValueError(f'{root}: missing coordinator timeline')
    allocation_start,allocation_end=min(stamps),max(stamps)
    async_rows=[r for path in sorted((root/'infra').glob('async-events-*.jsonl')) for r in jsonlines(path)]
    phase_starts={}
    for row in sorted(async_rows,key=lambda r:r['time']):
        key=row['host'],row['pid'],row['operation'],row.get('rollout_id',-1)
        if row['phase']=='start': phase_starts[key]=row
        elif row['phase']=='end' and key in phase_starts:
            beginning=phase_starts.pop(key)
            phases.append(dict(name=row['operation'],role='driver',start=beginning['time'],end=row['time'],
                seconds=row['time']-beginning['time'],step=row.get('rollout_id'),
                behavior_version=beginning.get('behavior_version'),policy_lag=beginning.get('policy_lag'),
                ok=row.get('ok'),
                source='async driver/rollout events'))
    batches=[]
    generation=list(jsonlines(root/'generation-timing.jsonl'))
    for path in sorted((root/'rollouts').glob('step*-samples.json')):
        if not re.fullmatch(r'step\d+-samples.json',path.name): continue
        data=read_json(path); samples=[r for group in data['groups'] for r in group]
        idx=int(re.search(r'step(\d+)',path.name)[1])-1
        versions=sorted({str(r['metadata']['policy_version']) for r in samples if 'policy_version' in r['metadata']})
        cohort_start=[];cohort_end=[]
        for ep in sorted((root/'rollouts').glob(f'step{idx+1}-group*.json')):
            for trace in read_json(ep).get('traces',[]):
                timing=trace.get('timing',{})
                if 'start' in timing: cohort_start.append(timing['start'])
                finish=timing.get('scoring',{}).get('end') or timing.get('finalize',{}).get('end')
                if finish is not None: cohort_end.append(finish)
        a,b=(min(cohort_start),max(cohort_end)) if cohort_start and cohort_end else (None,None)
        actual_phase=next((p for p in phases if p['name']=='rollout' and p['step']==idx),None)
        if actual_phase: a,b=actual_phase['start'],actual_phase['end']
        elif a is not None:
            phases.append(dict(name='rollout',role='harness',start=a,end=b,seconds=b-a,step=idx,
                               source='first episode start to last episode scoring/finalize; excludes bridge serialization'))
        responses=[r for r in generation if a is not None and a<=r['time']<=b]
        served_versions=sorted({str(r['meta']['weight_version']) for r in responses if r.get('meta',{}).get('weight_version') is not None})
        batches.append(dict(step=idx,samples=len(samples),raw_reward_mean=data['metrics']['reward_mean'],
            rollout_seconds=data['metrics']['episode_seconds'],active_tokens=sum(sum(r['loss_mask']) for r in samples),
            attempted_groups=data['metrics']['attempted_groups'],start=a,end=b,
            metadata_policy_versions=versions,served_weight_versions=served_versions,
            recorded_policy_lag=actual_phase.get('policy_lag') if actual_phase else None,
            scalar_actor=scalars.get(('actor',idx),{}),scalar_critic=scalars.get(('critic',idx),{})))
    for timer in timers:
        if timer['name'] in ['actor_train','critic_train','update_weights','save_model']:
            if job==197 and timer['name'] in ['actor_train','critic_train','update_weights']: continue
            phases.append({**timer,'name':'weight_transfer' if timer['name']=='update_weights' else timer['name']})
    useful=[p['start'] for p in phases if p['name']=='rollout']
    useful_end=[p['end'] for p in phases if p['name'] in ['actor_train','critic_train','save_model']]
    active_start=min(useful) if useful else allocation_start
    active_end=max(useful_end) if useful_end else allocation_end
    links,errors=fabric(root,hw); totals=aggregate_fabric(links,hw)
    fast_links,fast_errors,collector_cost=highrate_fabric(root)
    fast_totals=aggregate_fabric(fast_links,hw)
    fabric_summary={}
    for kind in ['IB','NVLink']:
        for direction in ['TX','RX']:
            for host in hw:
                relevant=[r for r in totals if r['fabric']==kind and r['direction']==direction and r['host']==host
                          and r['complete'] and overlap(r['start'],r['end'],active_start,active_end)>0]
                elapsed=sum(overlap(r['start'],r['end'],active_start,active_end) for r in relevant)
                fabric_summary[f'{kind}:{host}:{direction}']={'utilization_pct':stats([r['utilization_pct'] for r in relevant]),
                    'sampled_seconds':elapsed,'seconds_ge_90pct':sum(overlap(r['start'],r['end'],active_start,active_end)
                        for r in relevant if r['utilization_pct']>=90),
                    'mean_pct_time_weighted':sum(r['utilization_pct']*overlap(r['start'],r['end'],active_start,active_end)
                        for r in relevant)/elapsed if elapsed else None}
    transfer_windows=[]
    for index,p in enumerate(sorted((p for p in phases if p['name']=='weight_transfer'),key=lambda p:p['start'])):
        by_host={}
        for host in hw:
            relevant=[r for r in totals if r['fabric']=='IB' and r['direction']=='TX' and r['host']==host and r['complete']
                      and overlap(r['start'],r['end'],p['start'],p['end'])>0]
            by_host[host]={'overlapping_bins':len(relevant),'node_TX_utilization_pct':stats([r['utilization_pct'] for r in relevant]),
                           'sampling_intervals_s':stats([r['end']-r['start'] for r in relevant])}
        fine={}
        for host in hw:
            relevant=[r for r in fast_totals if r['fabric']=='IB' and r['direction']=='TX' and r['host']==host and r['complete']
                      and overlap(r['start'],r['end'],p['start'],p['end'])>0]
            hotlinks=[r for r in fast_links if r['fabric']=='IB' and r['direction']=='TX' and r['host']==host
                      and overlap(r['start'],r['end'],p['start'],p['end'])>0]
            fine[host]={'overlapping_bins':len(relevant),'node_TX_utilization_pct':stats([r['utilization_pct'] for r in relevant]),
                        'individual_port_TX_utilization_pct':stats([r['utilization_pct'] for r in hotlinks]),
                        'sampling_intervals_s':stats([r['end']-r['start'] for r in relevant])}
        transfer_windows.append({**p,'publication_index':index,'ib_TX_overlapping_bins':by_host,'ib_TX_highrate_bins':fine})
    compute=[(p['start'],p['end']) for p in phases if p['name'] in ['actor_train','critic_train']]
    rollout=[(p['start'],p['end']) for p in phases if p['name']=='rollout']
    intersections=[(max(a,c),min(b,d)) for a,b in compute for c,d in rollout if overlap(a,b,c,d)>0]
    pid_role={(r['host'],r['pid']):r['role'] for r in async_rows if r.get('role') in ['actor','critic']}
    pid_role.update(native_worker_roles(root,async_rows))
    optimizers=[]
    for row in async_rows:
        if row['operation']=='optimizer_step':
            role,basis=optimizer_role(row,pid_role,phases)
            optimizers.append({**row,'resolved_role':role,'role_assignment_basis':basis})
    exit_code=(root/'exit-code.txt').read_text().strip() if (root/'exit-code.txt').exists() else None
    snapshot=read_json(root/'snapshot-manifest.json',{})
    source_paths=[root/'training.log',root/'timeline.jsonl',root/'generation-timing.jsonl']
    source_paths += list((root/'infra').glob('*timeseries.jsonl'))+list((root/'infra').glob('*rdma-counters.jsonl'))
    source_paths += list((root/'infra').glob('*health.jsonl'))+list((root/'infra').glob('async-events-*.jsonl'))
    source_paths += list((root/'infra').glob('*fabric-1s.jsonl'))
    source_paths += list((root/'infra').glob('*before.json'))+list((root/'rollouts').glob('step*-samples.json'))
    source_paths += list((root/'rollouts').glob('step*-group*.json'))
    source_paths += [root/'metrics-extracted.json',root/'validated-arguments.json',root/'exit-code.txt']
    return dict(job_id=job,label=label,source_root=str(root),remote_campaign=snapshot.get('campaign'),
        exit_code=exit_code,complete=bool(exit_code is not None),allocation_window=[allocation_start,allocation_end],
        active_window=[active_start,active_end],active_window_definition='first rollout start through last recorded model save/train end',
        hardware=hw,batches=batches,phases=sorted(phases,key=lambda p:p['start']),timers=timers,
        fabric_intervals=totals,fabric_summary=fabric_summary,fabric_collector_errors=errors,
        fabric_link_statistics=link_statistics(links,active_start,active_end),
        fabric_highrate_intervals=fast_totals,fabric_highrate_errors=fast_errors,collector_wall_seconds=collector_cost,
        fabric_highrate_link_statistics=link_statistics(fast_links,active_start,active_end),
        weight_transfers=transfer_windows,gpu_samples=gpu_samples(root),
        rollout_compute_overlap_seconds=union_seconds(intersections),
        rollout_seconds_union=union_seconds(rollout),compute_seconds_union=union_seconds(compute),
        optimizer_steps=optimizers,rank_events=[r for r in async_rows if r['operation'].startswith('rank_')],
        native_lifecycle_events=native_lifecycle(root),
        source_sha256={str(p.relative_to(root)):sha(p) for p in source_paths if p.exists()},
        validated_arguments=read_json(root/'validated-arguments.json',{}))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='append',nargs=3,metavar=('JOB','LABEL','DIRECTORY'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    runs=[analyze_run(Path(path),int(job),label) for job,label,path in args.run]
    output={'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'runs':runs,
        'scope':'Two-update matched-workload runs; different algorithms, generated tokens and compilation costs. Not a controlled scaling sweep.',
        'semantics':{'bandwidth':'PMA delta x4 bytes; NVLink data delta x1024 bytes. TX and RX are separate. Node-wide counters.',
        'weight_windows':'Statistics of complete node sampling bins that overlap publication. No attribution of all bytes to model weights; bins include boundary traffic.',
        'saturation':'90% is a descriptive utilization threshold, not proof of a workload bandwidth ceiling. Coarse intervals hide bursts.',
        'NVLink_denominator':'50 GB/s nominal payload per direction/link, versus reported raw line rate 53.125 GB/s. nvidia-smi -dr unsupported on installed driver.',
        'missing':'Null/absent, never filled with zero. Job190 has no continuous NVLink data counters.',
        'off_policy':'Async recorded lag comes from engine version passed to the rollout hook; synchronous legacy metadata policy_version was rollout ID, not engine version.',
        'optimizer':'Referenced storages by device, not transferred bytes; CPU Adam states are separate from TMS parameter/gradient offload.'}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'runs':[{'job':r['job_id'],'batches':len(r['batches']),
                      'complete':r['complete'],'fabric_intervals':len(r['fabric_intervals'])} for r in runs]}))


if __name__=='__main__': main()
