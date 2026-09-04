"""Consolidate immutable raw archives into dense, source-linked comparison data.

Uses only the standard library. Run render_comparison.py on the resulting JSON.
Missing measurements stay null/absent; node-wide counters are not attributed to
one process. Raw logs, tensors and transcripts stay outside the Git worktree.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys


def epoch(t):
    # Python 3.9 accepts microseconds but the original collector emits ns.
    t=re.sub(r'(\.\d{6})\d+',r'\1',t)
    return dt.datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    def percentile(q):
        pos = (len(values)-1)*q
        low = int(pos); high = min(low+1, len(values)-1)
        return values[low]*(high-pos)+values[high]*(pos-low) if low != high else values[low]
    mean = statistics.mean(values)
    return {'count':len(values), 'min':values[0], 'mean':mean,
            'median':statistics.median(values), 'p90':percentile(.90),
            'p95':percentile(.95), 'p99':percentile(.99), 'max':values[-1],
            'cv':statistics.pstdev(values)/abs(mean) if mean else None}


def lines(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def phase_results(root):
    starts = {}
    phases = []
    for row in lines(root/'timeline.jsonl'):
        event = row['event']
        if event.endswith('_start'):
            starts[event[:-6]] = row
        elif event.endswith('_end') and event[:-4] in starts:
            name = event[:-4]; start = starts.pop(name)
            log = root/(name+'.log')
            phases.append({'schema_version':1, 'runner':name,
                'status':'ok' if row.get('exit_code', 0) == 0 else 'fail',
                'started_at':start['time'], 'ended_at':row['time'],
                'duration_s':epoch(row['time'])-epoch(start['time']),
                'exit_code':row.get('exit_code', 0), 'timeout':False,
                'metadata':{'argv':start.get('argv')},
                'results':[{'metric':'duration','value':epoch(row['time'])-epoch(start['time']),
                            'unit':'s','node':'coordinator','labels':{}}],
                'failure_summary':None if row.get('exit_code',0)==0 else f'{name} returned nonzero; see retained log',
                'log_relpath':log.name if log.exists() else None,
                'log_sha256':sha(log) if log.exists() else None})
    events=list(lines(root/'timeline.jsonl'))
    for name,start in starts.items():
        log=root/(name+'.log')
        phases.append({'schema_version':1,'runner':name,'status':'fail',
            'started_at':start['time'],'ended_at':events[-1]['time'],
            'duration_s':epoch(events[-1]['time'])-epoch(start['time']),
            'exit_code':None,'timeout':False,'metadata':{'argv':start.get('argv')},'results':[],
            'failure_summary':'No completion event; run interrupted or command raised',
            'log_relpath':log.name if log.exists() else None,'log_sha256':sha(log) if log.exists() else None})
    return phases


def episode_results(root):
    episodes = []
    shipped = set()
    for path in sorted((root/'rollouts').glob('step*-samples.json')):
        data = load(path)
        if isinstance(data, dict):
            shipped.update(row['metadata']['trace_id'] for group in data['groups'] for row in group)
    for path in sorted((root/'rollouts').glob('*.json')):
        match = re.fullmatch(r'step(\d+)-group(\d+)-(task_\d+)-(\d+)\.json', path.name)
        if not match:
            continue
        episode = load(path)
        traces = episode['traces']
        trace = traces[0] if traces else {}
        timing = trace.get('timing', {})
        scores = trace.get('rewards', {}).values()
        reward = sum(v['score']*v.get('weight', 1) for v in scores)
        calls = trace.get('calls', [])
        end = timing.get('scoring', {}).get('end') or timing.get('finalize', {}).get('end')
        episodes.append({'step':int(match[1]), 'group':int(match[2]), 'task':match[3],
            'episode_id':episode['id'], 'trace_id':trace.get('id'), 'ok':episode['ok'],
            'errors':episode.get('errors', []), 'reward':reward,
            'accepted':trace.get('id') in shipped, 'turns':len(calls),
            'tool_calls':sum(len(n.get('message', {}).get('tool_calls') or []) for n in trace.get('nodes', [])),
            'output_tokens':sum(c.get('usage', {}).get('completion_tokens', 0) or 0 for c in calls),
            'duration_s':end-timing['start'] if end is not None and 'start' in timing else None,
            'boot_s':timing.get('boot', {}).get('end', 0)-timing.get('boot', {}).get('start', 0),
            'model_s':timing.get('agent', {}).get('model', {}).get('duration'),
            'stop_condition':trace.get('stop_condition'), 'source':str(path.relative_to(root)),
            'source_sha256':sha(path)})
    return episodes


GPU_FIELDS = {'gpu_util_percent':3, 'hbm_util_percent':4, 'hbm_used_mib':5,
              'power_w':7, 'temperature_c':9, 'sm_clock_mhz':10}


def telemetry(root, start, end):
    series = []; by_gpu = collections.defaultdict(lambda: collections.defaultdict(list))
    errors = collections.Counter(); sources = {}; nvlink_statistics = collections.defaultdict(list)
    sample_monotonic = None
    health_states = {}
    def add(t, host, metric, value, unit, device=None):
        if value is not None:
            source = 'sglang' if metric.startswith('sglang:') else 'infiniband' if metric.startswith('ib_') else 'nvlink' if metric.startswith('nvlink_') else 'host' if metric.startswith(('cpu_','host_','shared_')) else 'nvidia-smi'
            series.append({'time':t, 'time_utc':dt.datetime.fromtimestamp(t,dt.timezone.utc).isoformat(),
                           'monotonic':sample_monotonic, 'source':source,
                           'elapsed_s':round(t-start, 3), 'hostname':host,
                           'role':'trainer' if host in ['gpu-nodes-0','gpu-nodes-1'] else 'rollout',
                           'metric':metric, 'value':value, 'unit':unit, 'device':device})
    for node in range(4):
        host = f'gpu-nodes-{node}'; previous = None
        path = root/'infra'/f'{host}-timeseries.jsonl'
        if path.exists(): sources[str(path.relative_to(root))] = sha(path)
        for row in lines(path):
            sample_monotonic = row.get('monotonic')
            t = epoch(row['timestamp'])
            if not start <= t <= end: continue
            gpus = row['gpu']; values = collections.defaultdict(list)
            if gpus.get('exit_code') != 0: errors[host+':gpu'] += 1
            for gpu in gpus.get('stdout', '').splitlines():
                fields = [x.strip() for x in gpu.split(',')]
                for metric, index in GPU_FIELDS.items():
                    value = number(fields[index]) if len(fields) > index else None
                    if value is not None:
                        values[metric].append(value)
                        by_gpu[host+'/'+fields[1]][metric].append(value)
            for metric, vals in values.items():
                unit = 'MiB' if 'mib' in metric else 'W' if metric == 'power_w' else 'C' if metric == 'temperature_c' else 'MHz' if 'mhz' in metric else '%'
                add(t, host, metric+'_mean', statistics.mean(vals), unit)
                if metric in ['gpu_util_percent','hbm_used_mib']: add(t, host, metric+'_max', max(vals), unit)
            system = row.get('system', {})
            mem = system.get('/proc/meminfo', '')
            if isinstance(mem, str):
                match = re.search(r'^MemAvailable:\s+(\d+)', mem, re.M)
                if match: add(t, host, 'host_available_gib', int(match[1])/1024**2, 'GiB')
            cpu = system.get('/proc/stat', '')
            if isinstance(cpu, str) and cpu.startswith('cpu '):
                counters = [int(v) for v in cpu.splitlines()[0].split()[1:9]]
                if previous is not None:
                    delta = [a-b for a,b in zip(counters, previous)]
                    if sum(delta)>0:
                        add(t, host, 'cpu_busy_percent', 100*(sum(delta)-delta[3]-delta[4])/sum(delta), '%')
                        add(t, host, 'cpu_iowait_percent', 100*delta[4]/sum(delta), '%')
                previous = counters
        path = root/'infra'/f'{host}-rdma-counters.jsonl'; last = {}
        if path.exists(): sources[str(path.relative_to(root))] = sha(path)
        for row in lines(path):
            sample_monotonic = row.get('monotonic')
            t = epoch(row['utc'])
            for port in row['ports']:
                device = port['argv'][2]
                if port.get('status') != 0:
                    errors[host+':rdma:'+device] += 1; continue
                counters = {k:int(v) for k,v in re.findall(r'^([A-Za-z0-9]+):\.+([0-9]+)', port['stdout'], re.M)}
                if device in last and start <= t <= end:
                    t0, old = last[device]
                    for key, metric in [('PortXmitData','ib_tx_gbps'),('PortRcvData','ib_rx_gbps')]:
                        if key in counters and key in old and t > t0:
                            delta = counters[key]-old[key]
                            if delta >= 0: add(t, host, metric, delta*4*8/(t-t0)/1e9, 'Gb/s', device)
                            else: errors[host+':counter_reset:'+device] += 1
                last[device] = (t, counters)
        path = root/'infra'/f'{host}-health.jsonl'; nvlast = {}
        if path.exists(): sources[str(path.relative_to(root))] = sha(path)
        for row in lines(path):
            sample_monotonic = row.get('monotonic')
            t = epoch(row['time'])
            if not start <= t <= end: continue
            add(t, host, 'shared_free_gib', row['shared_free_bytes']/1024**3, 'GiB')
            for source, result in row['sources'].items():
                if result['status'] != 'ok': errors[host+':'+source] += 1
            health = row['sources'].get('health', {})
            if health.get('status') == 'ok':
                health_states.setdefault(host, {'first':health['stdout']})['last'] = health['stdout']
            result = row['sources'].get('nvlink', {})
            if result.get('status') == 'ok':
                device = None; sums = collections.defaultdict(float)
                for line in result['stdout'].splitlines():
                    match = re.search(r'UUID: (GPU-[^)]+)', line)
                    if match: device = match[1]
                    match = re.search(r'Link (\d+): Data (Tx|Rx): (\d+) KiB', line)
                    if match and device:
                        link, direction, count = match.groups(); count = int(count)
                        key = host+'/'+device+'/'+link+'/'+direction
                        if key in nvlast:
                            t0, c0 = nvlast[key]
                            if t > t0 and count >= c0:
                                rate = (count-c0)*1024*8/(t-t0)/1e9
                                nvlink_statistics[key].append(rate); sums[direction] += rate
                            elif count < c0: errors[key+':counter_reset'] += 1
                        nvlast[key] = (t, count)
                for direction, rate in sums.items():
                    add(t, host, 'nvlink_'+direction.lower()+'_gbps', rate, 'Gb/s')
    path = root/'infra/sglang-prometheus.jsonl'
    if path.exists(): sources[str(path.relative_to(root))] = sha(path)
    prometheus_final = {}
    for row in lines(path):
        sample_monotonic = row.get('monotonic')
        t = epoch(row['utc'])
        if not start <= t <= end: continue
        if row.get('status') != 200: errors['sglang_scrape'] += 1; continue
        for line in row['text'].splitlines():
            match = re.fullmatch(r'(sglang:[a-zA-Z_0-9]+)(\{.*\})?\s+(\S+)(?:\s+\S+)?', line)
            if not match: continue
            name, labels, raw = match.groups(); value = number(raw)
            if value is None: continue
            # The pinned model identity is stored once in provenance. Do not
            # repeat its 160-character checkpoint path in every metric key.
            if labels:
                pairs = re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)="((?:\\.|[^"\\])*)"', labels)
                labels = '{'+','.join(f'{k}="{v}"' for k,v in pairs if k != 'model_name')+'}'
            prometheus_final[name+(labels or '')] = value
            if name in ['sglang:num_queue_reqs','sglang:num_running_reqs','sglang:token_usage',
                        'sglang:gen_throughput','sglang:num_retracted_reqs','sglang:cache_hit_rate']:
                add(t, 'gpu-nodes-3', name, value, 'tokens/s' if name.endswith('gen_throughput') else 'ratio' if name.endswith(('token_usage','cache_hit_rate')) else 'count', labels)
    grouped = collections.defaultdict(list)
    for row in series: grouped[row['hostname']+'/'+row['metric']].append(row['value'])
    return {'series':series, 'node_statistics':{k:stats(v) for k,v in grouped.items()},
            'gpu_statistics':{g:{k:stats(v) for k,v in metrics.items()} for g,metrics in by_gpu.items()},
            'collector_errors':dict(errors), 'source_sha256':sources,
            'health_first_last':health_states, 'nvlink_per_link_gbps':{k:stats(v) for k,v in nvlink_statistics.items()},
            'prometheus_last_sample':prometheus_final}


def miles_run(root, job, algorithm):
    extracted = load(root/'metrics-extracted.json', {'scalars':{},'rollouts':[]})
    events = list(lines(root/'timeline.jsonl'))
    text = (root/'training.log').read_text() if (root/'training.log').exists() else ''
    # Timing boundaries are explicit and always retained alongside the series.
    allocation_window = [epoch(events[0]['time']), epoch(events[-1]['time'])]
    ready = re.findall(r'\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) rollout_manager\].*mark_alive end', text)
    saved = re.findall(r'\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) [^\]]*rank0\].*Timer save_model end', text)
    start = min(epoch(t+'+00:00') for t in ready) if ready else allocation_window[0]
    end = max(epoch(t+'+00:00') for t in saved) if saved else allocation_window[1]
    window_definition = 'rollout engine mark_alive to last actor/critic save_model end' if ready and saved else 'incomplete run: available coordinator window'
    phases = phase_results(root)
    scalars = extracted['scalars']
    timers=[]; pending={}
    for line in text.splitlines():
        match=re.search(r'\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) (actor|critic)_cell0_rank0\].*Timer (\w+) (start|end)',line)
        if match:
            stamp,role,name,state=match.groups(); key=(role,name); t=epoch(stamp+'+00:00')
            if state=='start':pending[key]=t
            elif key in pending:
                t0=pending.pop(key)
                timers.append({'role':role,'name':name,'start_epoch':t0,'end_epoch':t,'seconds':t-t0})
    critic_timers=[t['seconds'] for t in timers if t['role']=='critic' and t['name']=='critic_train']
    steps = []
    for i, rollout in enumerate(extracted['rollouts']):
        scalar = {k:next((x['value'] for x in v if x['step']==i), None) for k,v in scalars.items()}
        steps.append({'step':i+1, 'accepted':rollout['samples'],
                      'reward':rollout['metrics']['reward_mean'],
                      'rollout_s':rollout['metrics']['episode_seconds'],
                      'actor_train_s':scalar.get('perf/actor_train_time'),
                      'active_tokens_per_actor_second':rollout['active_training_tokens']/scalar['perf/actor_train_time'] if scalar.get('perf/actor_train_time') else None,
                      'critic_train_s':critic_timers[i] if i<len(critic_timers) else scalar.get('perf/critic_train_time'),
                      'active_tokens':rollout['active_training_tokens'],
                      'input_tokens':rollout['input_tokens'], 'truncated':rollout['truncated'],
                      'tasks':rollout['tasks'], 'scalars':scalar})
    episodes = episode_results(root)
    cohorts=collections.defaultdict(list)
    for episode in episodes:
        cohorts[(episode['step'],episode['group'],episode['task'])].append(episode)
    groups=[{'step':key[0],'group':key[1],'task':key[2],
             'samples':len(rows),'accepted':sum(e['accepted'] for e in rows),
             'reward_mean':statistics.mean(e['reward'] for e in rows),
             'zero_variance':len({e['reward'] for e in rows})==1} for key,rows in cohorts.items()]
    # Use first real environment episode through final save if log evidence is
    # present. Allocation-wide telemetry is retained separately from this window.
    telem = telemetry(root, start, end)
    exit_path = root/'exit-code.txt'
    exit_code = int(exit_path.read_text()) if exit_path.exists() else None
    generation=[]; excluded_validation_calls=[]
    training_start=min((epoch(e['time']) for e in events if e['event']=='training_start'),default=allocation_window[0])
    for row in lines(root/'generation-timing.jsonl'):
        meta=row['meta']
        if row['time'] < training_start and row['prompt_tokens']==3 and meta.get('output_token_logprobs')==[[-.7,34,None],[-.9,99,None]]:
            excluded_validation_calls.append({'time':row['time'],'reason':'exact pre-training test_transport mock fixture; not a model trajectory'})
            continue
        generation.append({'time':row['time'],'seconds':row['seconds'],
            'prompt_tokens':row['prompt_tokens'],'completion_tokens':row['completion_tokens'],
            'finish':row['finish'],'dp_rank':meta.get('dp_rank'),'weight_version':meta.get('weight_version'),
            'weight_versions':meta.get('weight_versions'), 'retractions':meta.get('num_retractions'),
            'queue_s':meta.get('queue_time'),'e2e_s':meta.get('e2e_latency'),
            'reported_decode_tokens_per_s':meta.get('decode_throughput'),'cached_tokens':meta.get('cached_tokens'),
            'request_to_prefill_finish_s':meta['prefill_finished_time']-meta['request_received_ts'] if 'prefill_finished_time' in meta and 'request_received_ts' in meta else None})
    archive=load(root/'archive-manifest.json',{})
    return {'job_id':job, 'algorithm':algorithm, 'label':f'Miles {algorithm.upper()} ({job})',
            'raw_archive':archive.get('remote_run',str(root)), 'local_archive':str(root.resolve()),
            'window_epoch':[start,end], 'window_definition':window_definition,
            'coordinator_window_epoch':allocation_window, 'ready_to_checkpoint_s':end-start if ready and saved else None,
            'status':'ok' if exit_code == 0 else 'fail' if exit_code is not None else 'running',
            'exit_code':exit_code, 'phases':phases, 'phase_counts':dict(collections.Counter(p['status'] for p in phases)),
            'completed_actor_updates':len(scalars.get('train/loss',[])),
            'completed_critic_updates':len(scalars.get('train/critic-value_loss',[])),
            'positive_actor_gradient_steps':sum(x['value']>0 and math.isfinite(x['value']) for x in scalars.get('train/grad_norm',[])),
            'steps':steps, 'all_scalars':scalars, 'episodes':episodes, 'groups':groups,'timers':timers,
            'generation_calls':generation,
            'excluded_validation_calls':excluded_validation_calls,
            'generation_statistics':{k:stats([g[k] for g in generation]) for k in ['seconds','completion_tokens','queue_s','e2e_s','reported_decode_tokens_per_s','request_to_prefill_finish_s']},
            'accounting':{'generated':len(episodes), 'accepted':sum(s['accepted'] for s in steps),
                'consumed_by_logged_actor_step':sum(s['accepted'] for s in steps if any(x['step']==s['step']-1 for x in scalars.get('train/loss',[]))),
                'errors':sum(not e['ok'] for e in episodes),
                'not_accepted':sum(not e['accepted'] for e in episodes),
                'zero_variance_generated_groups':sum(g['zero_variance'] for g in groups),
                'accepted_zero_variance_groups':sum(g['zero_variance'] and g['accepted']>0 for g in groups),
                'output_tokens':sum(e['output_tokens'] for e in episodes)},
            'episode_statistics':{k:stats([e[k] for e in episodes]) for k in ['reward','turns','tool_calls','output_tokens','duration_s','model_s']},
            'checkpoints':load(root/'checkpoint-verification.json'),
            'critic_checkpoint':load(root/'critic-checkpoint-verification.json'),
            'telemetry':telem}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ipo', type=Path, required=True)
    parser.add_argument('--ppo', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-gpu-csv', type=Path)
    parser.add_argument('--failed',type=Path,action='append',default=[])
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    runs = [miles_run(args.ipo, 190, 'ipo'), miles_run(args.ppo, int(args.ppo.name.split('-')[-1]), 'ppo')]
    original = load(args.baseline/'baseline-step-summary.json')
    historical=load(args.baseline/'comparison-results.json')['baseline']
    for step in original.values():
        step['approx_actor_s']=step['time/step']-step['time/wait_for_batch']-step['time/broadcast_weights']-step['time/load_data']
    baseline = {'job_id':181,'label':'Prime-RL IPO (181)', 'algorithm':'ipo',
                'status':'two_updates_completed_allocation_timed_out',
                'ready_to_checkpoint_s':historical['ready_to_checkpoint_seconds'],
                'accounting':{'generated':historical['episodes'],'accepted':historical['accepted_traces']},
                'historical_comparison_sha256':sha(args.baseline/'comparison-results.json'),
                'steps':list(original.values()), 'gpu_statistics':load(args.baseline/'baseline-gpu-profile.json'),
                'source_sha256':sha(args.baseline/'baseline-step-summary.json')}
    baseline_series = []
    if args.baseline_gpu_csv:
        cohorts = collections.defaultdict(lambda: collections.defaultdict(list))
        columns = {'utilization.gpu':('gpu_util_percent_mean','%'),
                   'utilization.memory':('hbm_util_percent_mean','%'),
                   'memory.used':('hbm_used_mib_mean','MiB'), 'power.draw':('power_w_mean','W'),
                   'temperature.gpu':('temperature_c_mean','C'), 'clocks.sm':('sm_clock_mhz_mean','MHz')}
        for row in csv.DictReader(args.baseline_gpu_csv.open()):
            t=epoch(row['collector_time_utc'])
            if epoch('2026-09-03T15:24:22Z') <= t <= epoch('2026-09-03T15:30:03Z'):
                for field, (metric, unit) in columns.items():
                    value=number(row[field])
                    if value is not None:cohorts[(t,row['node'])][(metric,unit)].append(value)
        for (t,host), metrics in cohorts.items():
            for (metric,unit),values in metrics.items():
                baseline_series.append({'job_id':181,'time':t,'time_utc':dt.datetime.fromtimestamp(t,dt.timezone.utc).isoformat(),
                    'monotonic':None,'source':'nvidia-smi','elapsed_s':round(t-epoch('2026-09-03T15:24:22Z'),3),
                    'hostname':host,'role':'trainer' if host in ['gpu-nodes-0','gpu-nodes-1'] else 'rollout',
                    'metric':metric,'value':statistics.mean(values),'unit':unit,'device':None})
        baseline['gpu_csv_sha256']=sha(args.baseline_gpu_csv)
    result = {'schema_version':1, 'scope':'two-update matched-workload smoke tests, not a quality or speedup claim',
        'baseline':baseline, 'runs':runs,
        'metric_semantics':{
            'reward':'Raw accepted training reward, task mix changes after algorithm-dependent admission; not held-out evaluation.',
            'ppo':'Learned critic, native mask-aware GAE; raw rewards; gamma=lambda=1; whitened advantages; clipped policy/value loss.',
            'ipo':'Centered group rewards with zero-advantage filtering. Its custom entropy metric is a zero placeholder, not measured entropy.',
            'memory':'nvidia-smi device allocated HBM, distinct from trainer peak allocator memory.',
            'ib':'PMA counters use four-byte units. Rates are counter differences / elapsed time. Each port is separate; TX/RX are not added together.',
            'time':'Different framework timer boundaries, compilation/cache state, and generated work prevent an isolated loss-function speedup claim.',
            'missing':'Omitted/null, never converted to zero. Unsupported collectors and missing historical measurements are listed.',
            'checkpoints':'Weights-only. No optimizer/RNG/scheduler resume fidelity claim.',
            'weight_transfer':'update_weights timers include the native broadcast path but exclude the separately recorded actor wake_up/sleep transfers.',
            'sglang_scope':'The node-3 serving endpoint exposes DP0 scheduler gauges and API-level token counters. Do not interpret the queue gauge as a complete DP0+DP1 queue.',
            'staleness':'Synchronous Miles driver, one actor update per batch. Fully asynchronous queue/staleness benchmark not performed.',
            'summary_statistics':'Linearly interpolated sample quantiles; population CV. Samples are correlated, not independent benchmark repetitions.'},
        'coverage_gaps':['No held-out TB2.1 evaluation or quality confidence interval.',
            'No full ClusterMAX destructive/burn-in or collectives sweep was requested for this matched repeat.',
            'Historical job190 lacks continuous NVLink payload and Lustre client counters.',
            'PMA failures, Lustre permissions, and unsupported GPU fields remain collector errors.',
            'SGLang histograms are retained; no fabricated per-request TTFT/ITL from aggregate end-to-end times.',
            'DP1 scheduler gauges are not exposed in the retained single-endpoint scrape; GPU/fabric collectors cover both rollout nodes.',
            'No MTP in matched job190 recipe; MTP acceptance is not applicable.']}
    result['failed_attempts']=[]
    for path in args.failed:
        phases=phase_results(path)
        failed_log=(path/'training.log').read_text() if (path/'training.log').exists() else ''
        actor_calls=len(set(re.findall(r'log_utils.py:\d+ - step (\d+):',failed_log)))
        critic_calls=len(set(re.findall(r'log_utils.py:\d+ - critic-step (\d+):',failed_log)))
        result['failed_attempts'].append({'job_id':int(path.name.split('-')[-1]),
            'exit_code':int((path/'exit-code.txt').read_text()),'phases':phases,
            'actor_step_calls':actor_calls,'critic_step_calls':critic_calls,
            'valid_actor_updates':0,'raw_archive':load(path/'archive-manifest.json',{}).get('remote_run',str(path)),
            'failure_events':[e for e in lines(path/'timeline.jsonl') if e['event']=='failure']})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    manifests=[]
    for path in [args.ppo,*args.failed]:
        manifest_path=path/'archive-manifest.json';archive_path=path.with_suffix('.tar.gz')
        if manifest_path.exists():
            archive=load(manifest_path)
            manifests.append({'job_id':int(path.name.split('-')[-1]),'remote_run':archive['remote_run'],
                'local_archive':str(path.resolve()),'files':len(archive['included']),
                'uncompressed_bytes':sum(x['bytes'] for x in archive['included']),
                'manifest_sha256':sha(manifest_path),
                'compressed_archive_sha256':sha(archive_path) if archive_path.exists() else None,
                'compressed_archive_bytes':archive_path.stat().st_size if archive_path.exists() else None,
                'checkpoint_shards_retained_remote':archive['checkpoint_files_retained_remote']})
    packages={str(path.relative_to(args.ppo)):load(path) for path in (args.ppo/'infra').rglob('python-packages.json')}
    common=next(iter(packages.values()),{})
    hosts=load(args.ppo/'hosts.json',{})
    reverse={ip:host for host,ip in hosts.items()}
    log=(args.ppo/'training.log').read_text()
    placement=[{'bundle':int(b),'actual_bundle':int(a),'host':reverse.get(ip,ip),'gpu_index':int(g),
                'role':'actor_and_critic' if int(b)<16 else 'rollout'} for b,a,ip,g in re.findall(
                    r'bundle\s+(\d+), actual_bundle_index:\s+(\d+), node: ([0-9.]+), gpu: (\d+)',log)]
    validated=load(args.ppo/'validated-arguments.json',{})
    key_arguments=['advantage_estimator','use_critic','actor_num_nodes','actor_num_gpus_per_node',
        'critic_num_nodes','critic_num_gpus_per_node','rollout_num_gpus','rollout_num_gpus_per_engine',
        'tensor_model_parallel_size','expert_model_parallel_size','context_parallel_size','pipeline_model_parallel_size',
        'expert_tensor_parallel_size','lr','critic_lr','gamma','lambd','eps_clip','eps_clip_high','value_clip',
        'num_rollout','global_batch_size','rollout_batch_size','n_samples_per_prompt','seed',
        'num_critic_only_steps','normalize_advantages','rewards_normalization','disable_rewards_normalization','loss_type','offload_train',
        'no_save_optim','no_save_rng','save','critic_save','load','critic_load','update_weight_transfer_mode']
    provenance={'schema_version':1,'launch_manifest':load(args.ppo/'campaign-preflight/run.json'),
        'resolved_parameters':{k:validated[k] for k in key_arguments if k in validated},
        'resolved_arguments_sha256':sha(args.ppo/'validated-arguments.json') if validated else None,
        'driver_patch':load(args.ppo/'ppo-driver-patch.json'),'ray_placement':placement,
        'source_at_launch':load(args.ppo/'source-sha256.json'),
        'package_lock':common,
        'package_inventory_sha256':{name:sha(args.ppo/name) for name in packages},
        'package_differences':{name:{k:v for k,v in value.items() if common.get(k)!=v} for name,value in packages.items()},
        'archives':manifests,
        'model_hashes':load(args.ppo/'model-checksums.json'),
        'checkpoint_hashes':load(args.ppo/'checkpoint-checksums.json'),
        'native_tests':load(args.ppo/'ppo-native-tests.json'),'transport_tests':load(args.ppo/'ppo-transport-tests.json'),
        'policy_validity':[load(p) for p in sorted(args.ppo.glob('policy-validity-*.json'))],
        'final_runtime':load(args.ppo/'final-runtime.json'),
        'analysis_versions':{'python':sys.version,'package_lock':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}}
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    with (args.out/'timeseries.csv').open('w') as stream:
        fields = ['job_id','time','time_utc','monotonic','source','elapsed_s','hostname','role','metric','value','unit','device']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows(baseline_series)
        for run in runs:
            writer.writerows({'job_id':run['job_id'], **r} for r in run['telemetry'].pop('series'))
    # The CSV owns time series; avoid duplicating them in the readable JSON.
    for run in runs: run['telemetry']['series_file'] = 'timeseries.csv'
    (args.out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'out':str(args.out), 'runs':[{k:r[k] for k in ['job_id','status','accounting']} for r in runs]}))


if __name__ == '__main__':
    main()
