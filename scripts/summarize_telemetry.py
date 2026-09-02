"""Dense ClusterMAX-style publication; full-resolution evidence stays outside Git."""
from collections import Counter, defaultdict
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path

from summarize_native import counter_rate, summary

CMAX_SHA = 'fed871df5321d42706c98701522cc3ccd55898d5'
GPU_GAUGES = {
    'utilization.gpu': ('gpu_utilization', '%', 1),
    'memory.used': ('gpu_hbm_used', 'GiB', 1 / 1024),
    'power.draw': ('gpu_power', 'W', 1),
    'temperature.gpu': ('gpu_temperature', 'degC', 1),
    'clocks.current.sm': ('gpu_sm_clock', 'MHz', 1),
}
GAUGES = {
    ('proc-meminfo', 'MemAvailable'): ('host_memory_available', 'GiB', 1 / 1024**3),
    ('proc-meminfo', 'Dirty'): ('host_dirty_memory', 'GiB', 1 / 1024**3),
    ('statvfs', 'available'): ('shared_storage_available', 'TiB', 1 / 1024**4),
    ('collector', 'collection_duration'): ('collection_duration', 's', 1),
}
RATES = {
    ('perfquery', 'PortXmitData'): ('ib_rail_tx', 'GB/s', 1e-9),
    ('perfquery', 'PortRcvData'): ('ib_rail_rx', 'GB/s', 1e-9),
    ('perfquery', 'PortXmitWait'): ('ib_rail_xmit_wait', 'pma_ticks/s', 1),
    ('lustre-host-debugfs', 'read_bytes.sum'): ('lustre_client_read', 'GB/s', 1e-9),
    ('lustre-host-debugfs', 'write_bytes.sum'): ('lustre_client_write', 'GB/s', 1e-9),
    ('proc-vmstat', 'pgfault'): ('host_page_faults', 'count/s', 1),
    ('proc-vmstat', 'pgmajfault'): ('host_major_page_faults', 'count/s', 1),
}
IB_ERRORS = {'ExcessiveBufferOverrunErrors', 'LinkDownedCounter', 'LinkErrorRecoveryCounter',
    'LocalLinkIntegrityErrors', 'PortRcvConstraintErrors', 'PortRcvErrors',
    'PortRcvRemotePhysicalErrors', 'PortRcvSwitchRelayErrors', 'PortXmitConstraintErrors',
    'PortXmitDiscards', 'QP1Dropped', 'SymbolErrorCounter', 'VL15Dropped'}
TIMELINE_METRICS = {'gpu_utilization', 'gpu_hbm_used', 'gpu_power', 'nvlink_gpu_tx',
    'ib_rail_tx', 'ib_rail_rx', 'lustre_client_read', 'lustre_client_write',
    'host_memory_available', 'host_cpu_busy', 'shared_storage_available'}
CPU_FIELDS = {'cpu_user', 'cpu_nice', 'cpu_system', 'cpu_idle', 'cpu_iowait',
              'cpu_irq', 'cpu_softirq', 'cpu_steal'}  # guest is already included in user/nice.


def rounded(value):
    return float(f'{value:.6g}') if isinstance(value, float) else value


def compact_stats(values):
    return {key: rounded(value) for key, value in summary(values).items()}


def verified_rows(cache, stream):
    digest, offset, records = hashlib.sha256(), 0, 0
    for chunk in stream['chunks']:
        path = cache / chunk['path']
        if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError('Unsafe telemetry cache path.')
        payload = path.read_bytes()
        raw = gzip.decompress(payload)
        if (hashlib.sha256(payload).hexdigest() != chunk['gzip_sha256']
                or hashlib.sha256(raw).hexdigest() != chunk['raw_sha256']
                or chunk['offset'] != offset or len(raw) != chunk['end'] - offset):
            raise ValueError('Telemetry cache checksum/offset mismatch.')
        digest.update(raw)
        offset = chunk['end']
        for line in raw.splitlines():
            records += 1
            yield json.loads(line)
    if offset != stream['end'] or records != sum(c['records'] for c in stream['chunks']):
        raise ValueError('Telemetry cache accounting mismatch.')
    if stream.get('complete') and digest.hexdigest() != stream['source_sha256']:
        raise ValueError('Telemetry source checksum mismatch.')


def summarize_sources(cache, streams):
    pooled, entities, minutes = defaultdict(list), defaultdict(list), defaultdict(list)
    errors, gaps, counters, constants = {}, Counter(), {}, {}
    coverage, roles, gpu_ids, rails = [], {}, defaultdict(set), defaultdict(set)
    excluded = Counter()

    def observe(row, metric, unit, value, entity):
        host = row['hostname']
        pooled[(host, metric, unit)].append(value)
        entities[(host, entity, metric, unit)].append(value)
        if metric in TIMELINE_METRICS:
            minute = row['time'][:16] + ':00Z'
            minutes[(minute, host, metric, unit)].append(value)

    for stream in streams:
        previous, links, cpu_ticks = {}, {}, {}
        times, count, error_count, first, last = set(), 0, 0, None, None
        host = Path(stream['path']).parts[-2]
        for row in verified_rows(cache, stream):
            count += 1
            times.add(row['monotonic_s'])
            first = min(first, row['time']) if first else row['time']
            last = max(last, row['time']) if last else row['time']
            roles[host] = row.get('role', roles.get(host, 'unknown'))
            source, name = row['source'], row['metric']
            if name == 'collector_error':
                error_count += 1
                reason = 'timeout' if 'timed out' in row.get('error', '') else 'collection_failed'
                key = (host, source, reason, row.get('requested_metric'))
                event = errors.setdefault(key, {'node': host, 'source': source, 'reason': reason,
                    'requested_metric': row.get('requested_metric'), 'count': 0, 'times': []})
                event['count'] += 1
                event['times'].append(row['time'])
                continue
            entity = row.get('gpu_uuid') or (row['hca'] + '/' + str(row['hca_port'])
                if 'hca' in row else row.get('lustre_client', 'host'))
            if row.get('gpu_uuid'):
                gpu_ids[host].add(row['gpu_uuid'])
            if row.get('hca'):
                rails[host].add(entity)
            value = row['value']
            if source in ('nvidia-smi', 'persistent-nvml') and name in GPU_GAUGES:
                metric, unit, scale = GPU_GAUGES[name]
                observe(row, metric, unit, value * scale, entity)
            elif (source, name) in GAUGES:
                metric, unit, scale = GAUGES[(source, name)]
                observe(row, metric, unit, value * scale, entity)
            elif (source in ('nvidia-smi', 'persistent-nvml') and name in ('memory.total', 'clocks.current.memory')) or (source, name) in {
                    ('proc-meminfo', 'MemTotal'), ('statvfs', 'capacity')}:
                key = (host, name, row['unit'])
                item = constants.setdefault(key, {'values': set(), 'entities': set(), 'samples': 0})
                item['values'].add(value)
                item['entities'].add(entity)
                item['samples'] += 1
            elif (source == 'perfquery' and name in IB_ERRORS) or name.startswith('ecc.errors.'):
                key = (host, entity, name)
                item = counters.setdefault(key, {'first': value, 'last': value, 'min': value, 'max': value,
                    'resets': 0, 'samples': 0, 'unit': row['unit']})
                item['resets'] += int(value < item['last'])
                item.update(last=value, min=min(value, item['min']), max=max(value, item['max']), samples=item['samples'] + 1)
            elif (source, name) in RATES or name in ('nvlink_data_tx_bytes_total', 'nvlink_data_rx_bytes_total'):
                key = (entity, row.get('link'), name)
                old = previous.get(key)
                previous[key] = row
                if old is None:
                    continue
                rate = counter_rate(old, row)
                if rate is None:
                    gaps[(host, name, 'reset_nonpositive_time_or_gap_over_5s')] += 1
                    continue
                if name.startswith('nvlink_data_'):
                    # Keep Tx only: Rx often describes the same endpoint transfer.
                    # Raw Tx and Rx both remain in the source evidence.
                    if name.endswith('rx_bytes_total'):
                        continue
                    entities[(host, entity + '/link-' + str(row['link']), 'nvlink_link_tx', 'GB/s')].append(rate * 1e-9)
                    tick = (entity, row['time'])
                    link_values = links.setdefault(tick, {})
                    if row['link'] in link_values:
                        raise ValueError('Duplicate NVLink within one sample tick.')
                    link_values[row['link']] = rate
                    if len(link_values) == 18:
                        observe(row, 'nvlink_gpu_tx', 'GB/s', sum(link_values.values()) * 1e-9, entity)
                        del links[tick]
                else:
                    metric, unit, scale = RATES[(source, name)]
                    observe(row, metric, unit, rate * scale, entity)
            elif source == 'proc-stat' and name in CPU_FIELDS:
                tick = cpu_ticks.setdefault(row['time'], {'row': row, 'values': {}})
                tick['values'][name] = value
                if len(tick['values']) == len(CPU_FIELDS):
                    current = cpu_ticks.pop(row['time'])
                    old = previous.get('cpu')
                    previous['cpu'] = current
                    if old:
                        delta = {key: value - old['values'][key] for key, value in current['values'].items()}
                        elapsed = row['monotonic_s'] - old['row']['monotonic_s']
                        total = sum(delta.values())
                        if 0 < elapsed <= 5 and total > 0 and min(delta.values()) >= 0:
                            observe(row, 'host_cpu_busy', '%', 100 * (total - delta['cpu_idle'] - delta['cpu_iowait']) / total, 'host')
                            observe(row, 'host_cpu_iowait', '%', 100 * delta['cpu_iowait'] / total, 'host')
                        else:
                            gaps[(host, 'cpu', 'reset_nonpositive_time_or_gap_over_5s')] += 1
            else:
                excluded[(source, 'not_in_headline_set')] += 1
        if error_count != sum(c['collector_errors'] for c in stream['chunks']):
            raise ValueError('Collector-error count differs from publication cache.')
        for unused in links.values():
            gaps[(host, 'nvlink_gpu_tx', 'incomplete_18_link_tick')] += 1
        if cpu_ticks:
            gaps[(host, 'cpu', 'incomplete_cpu_tick')] += len(cpu_ticks)
        ordered = sorted(times)
        intervals = [b - a for a, b in zip(ordered, ordered[1:])]
        coverage.append({'path': stream['path'], 'state': stream['status'], 'records': count,
            'collector_errors': error_count, 'first_time': first, 'last_time': last,
            'max_observed_interval_s': rounded(max(intervals)) if intervals else None,
            'captured_bytes': stream['end'], 'sha256': stream.get('source_sha256'),
            'hash_scope': 'complete_file' if stream.get('complete') else 'not_finalized'})

    results = []
    for (host, name, unit), values in sorted(pooled.items()):
        stats = compact_stats(values)
        results.append({'metric': name, 'value': stats.pop('mean'), 'unit': unit, 'node': host,
            'labels': {'statistic': 'sample_mean'}, 'statistics': stats})
    health = []
    for host in sorted(roles):
        rows = [(entity, metric, item) for (node, entity, metric), item in counters.items() if node == host]
        health.append({'node': host, 'observed_counter_series': len(rows),
            'all_zero_series': sum(item['max'] == item['min'] == 0 for _, _, item in rows),
            'unchanged_nonzero_series': sum(item['max'] == item['min'] != 0 for _, _, item in rows),
            'changed_or_reset_series': sum(item['max'] != item['min'] or item['resets'] > 0 for _, _, item in rows),
            'metrics_observed': sorted({metric for _, metric, _ in rows}),
            'exceptions': [dict(entity=entity, metric=metric, **item) for entity, metric, item in rows
                           if item['min'] != 0 or item['max'] != 0 or item['resets']]})
    gpu_summary, outliers = [], []
    for host in sorted(roles):
        for gpu in sorted(gpu_ids[host]):
            item = {'node': host, 'gpu_uuid': gpu}
            for name, unit, field in (('gpu_utilization', '%', 'mean'), ('gpu_hbm_used', 'GiB', 'max'),
                                      ('gpu_power', 'W', 'max'), ('gpu_temperature', 'degC', 'max')):
                values = entities.get((host, gpu, name, unit))
                item[name + '_' + field] = compact_stats(values)[field] if values else None
            gpu_summary.append(item)
        for name in ('gpu_utilization', 'ib_rail_tx', 'nvlink_link_tx'):
            means = [(sum(values) / len(values), entity, unit) for (node, entity, metric, unit), values
                     in entities.items() if node == host and metric == name]
            if not means:
                continue
            stats = compact_stats([value for value, _, _ in means])
            low, high = min(means), max(means)
            outliers.append({'node': host, 'metric': name, 'unit': low[2], 'entity_count': len(means),
                'lowest_mean': {'entity': low[1], 'value': rounded(low[0])},
                'highest_mean': {'entity': high[1], 'value': rounded(high[0])},
                'between_entity_mean_cv': stats['coefficient_of_variation']})
    timeline = []
    for (minute, host, name, unit), values in sorted(minutes.items()):
        stats = compact_stats(values)
        timeline.append(dict(time=minute, node=host, metric=name, unit=unit,
                             **{k: stats[k] for k in ('n', 'min', 'mean', 'p95', 'max')}))
    return {'results': results, 'coverage': coverage, 'collector_errors': list(errors.values()),
        'counter_gaps': [dict(node=h, metric=m, reason=r, count=n) for (h, m, r), n in sorted(gaps.items())],
        'health': health, 'gpu_summary': gpu_summary, 'entity_extremes': outliers,
        'topology': [dict(node=h, role=roles[h], observed_gpus=len(gpu_ids[h]), observed_rails=len(rails[h])) for h in sorted(roles)],
        'inventory_values': [dict(node=h, metric=m, unit=u, observed_values=sorted(item['values']),
             entities=len(item['entities']), samples_collapsed=item['samples']) for (h,m,u),item in sorted(constants.items())],
        'omitted_samples': [dict(source=s, reason=r, count=n) for (s,r),n in sorted(excluded.items())],
        'timeline': timeline}


def timeline_csv(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['time', 'node', 'metric', 'unit', 'n', 'min', 'mean', 'p95', 'max'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def render(data):
    meta = data['metadata']
    lines = [f"# Job {meta['slurm_job_id']} — telemetry summary", '',
        f"**Telemetry gate: {data['status'].upper()}** · Slurm: {meta['slurm_state']} ({meta['slurm_exit_code']}) · "
        f"{meta['raw_records']:,} source records · {meta['collector_error_count']} collector errors.", '',
        f"Observed window: {data['started_at']} to {data['ended_at']} ({data['duration_s']:.1f} s).", '',
        'Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.', '',
        '## Nodes and headline measurements', '',
        'GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.', '',
        '| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    index = {(row['node'], row['metric']): row for row in data['results']}
    def val(host, metric, stat='mean'):
        row = index.get((host, metric))
        value = None if row is None else row['value'] if stat == 'mean' else row['statistics'].get(stat)
        return '—' if value is None else f'{value:.3g}'
    for node in meta['topology']:
        h = node['node']
        lines.append(f"| {h} | {node['role']} | {node['observed_gpus']}/{node['observed_rails']} | "
            f"{val(h,'gpu_utilization')} / {val(h,'gpu_utilization','p95')} | {val(h,'gpu_hbm_used','max')} | "
            f"{val(h,'gpu_power','max')} | {val(h,'nvlink_gpu_tx')} | {val(h,'ib_rail_tx')} | {val(h,'lustre_client_write','max')} |")
    lines += ['', '## Failures and coverage', '']
    for event in meta['collector_errors']:
        lines.append(f"- **{event['node']} / {event['source']}**: {event['count']} × {event['reason']}; UTC " + ', '.join(t[11:].rstrip('Z') for t in event['times']) + '.')
    if not meta['collector_errors']:
        lines.append('No collector-error records observed; this alone does not establish complete coverage.')
    for row in meta['coverage']:
        if row['state'] != 'complete':
            lines.append(f"- {row['path']}: **{row['state']}**.")
    lines += ['', f"Invalid intervals among summarized counters: **{sum(row['count'] for row in meta['counter_gaps'])}** (per-series intervals, not independent outages). Missing/reset/>5 s intervals are excluded, never zero-filled.", '',
        '| Node | Observed health-counter series | Always zero | Unchanged nonzero | Changed/reset |', '|---|---:|---:|---:|---:|']
    for row in meta['health']:
        lines.append(f"| {row['node']} | {row['observed_counter_series']} | {row['all_zero_series']} | {row['unchanged_nonzero_series']} | {row['changed_or_reset_series']} |")
    lines += ['', 'Unchanged nonzero values predate the observation window; they are not new errors during this run. These are only the ECC/IB counters actually collected. They do not establish XID, throttle, row-remap, PCIe or DCGM coverage.', '',
        '## Largest entity differences', '',
        'Lowest/highest time-mean within each node; descriptive differences, not hardware-fault diagnoses.', '',
        '| Node | Metric | Lowest entity : mean | Highest entity : mean | Across-entity CV |', '|---|---|---|---|---:|']
    for row in meta['entity_extremes']:
        def entity_label(item):
            label = item['entity'].replace('GPU-', '')
            if '/link-' in label:
                gpu, link = label.split('/link-'); label = gpu[:8] + '/link-' + link
            elif len(label) > 16:
                label = label[:8]
            return f"{label}: {item['value']:.3g}"
        cv = row['between_entity_mean_cv']
        lines.append(f"| {row['node']} | {row['metric']} ({row['unit']}) | {entity_label(row['lowest_mean'])} | {entity_label(row['highest_mean'])} | {'—' if cv is None else f'{cv:.3g}'} |")
    lines += ['', '## What is retained', '',
        '- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.',
        '- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.',
        '- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.', '',
        '## Evidence and limits', '',
        f"Raw evidence root: `{meta['raw_evidence_root']}`. All {len(meta['coverage'])} source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.", '',
        f"Formatting reference: ClusterMAX `{meta['clustermax_reference_sha']}`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.", '',
        'Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.', '']
    if meta.get('training'):
        train = meta['training']
        lines += ['## Training context', '',
            f"Observed optimizer updates: **{len(train['steps'])}**. These are log receipts, not proof of complete resume fidelity or held-out quality.", '',
            '| Step | UTC | Train reward | Grad norm | Trainer time (s) | Weight update (s) |', '|---:|---|---:|---:|---:|---:|']
        for row in train['steps']:
            values = [str(row['step']), row['time']] + ['—' if row.get(k) is None else f"{row[k]:.4g}" for k in ('train_reward','grad_norm','train_time_s','weight_update_s')]
            lines.append('| ' + ' | '.join(values) + ' |')
        lines += ['', 'Training reward is not held-out Terminal-Bench accuracy. MTP acceptance rate is omitted because this summary does not independently validate that field.', '']
    return '\n'.join(lines)


def training_context(run, job_id):
    paths = sorted((run / 'tests').glob('02-sync-grpo-*-optimizer-observation*/result.json'))
    candidates = []
    for path in paths:
        item = json.loads(path.read_text())
        if str(item.get('slurm_job_id')) == str(job_id) and item.get('optimizer_steps_observed') == len(item.get('steps', [])):
            candidates.append((path, item))
    if not candidates:
        return None
    path, data = max(candidates, key=lambda pair: (len(pair[1]['steps']), pair[0].name))
    if hashlib.sha256((run / data['source_log']).read_bytes()).hexdigest() != data['log_sha256']:
        raise ValueError('Training observation source hash changed.')
    performance = {(row['role'], row['rollout_id']): row['metrics'] for row in data['performance']}
    steps = []
    for row in data['steps']:
        rollout = performance.get(('rollout', row['step']), {})
        trainer = performance.get(('trainer', row['step']), {})
        steps.append(dict(step=row['step'], time=row['time'],
            train_reward=rollout.get('rollout/episode_raw_reward'), grad_norm=row['metrics'].get('train/grad_norm'),
            train_time_s=trainer.get('perf/train_time'), weight_update_s=trainer.get('perf/update_weights_time')))
    return {'steps': steps, 'observation_path': str(path.relative_to(run)),
        'observation_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_log': data['source_log'], 'log_sha256': data['log_sha256']}


def build_summary(run, cache, streams, job_id, state, exit_code):
    data = summarize_sources(cache, streams)
    timeline = data.pop('timeline')
    results = data.pop('results')
    times = [row[key] for row in data['coverage'] for key in ('first_time','last_time') if row[key]]
    start, end = (min(times), max(times)) if times else (None, None)
    duration = (dt.datetime.fromisoformat(end.replace('Z','+00:00')) - dt.datetime.fromisoformat(start.replace('Z','+00:00'))).total_seconds() if times else 0
    errors = sum(row['count'] for row in data['collector_errors'])
    complete = all(row['state'] == 'complete' for row in data['coverage'])
    status = 'skip' if not times else 'fail' if errors else 'ok' if complete else 'partial'
    metadata = dict(data, run_id=run.name, slurm_job_id=job_id, slurm_state=state,
        slurm_exit_code=exit_code, raw_records=sum(row['records'] for row in data['coverage']),
        collector_error_count=errors, all_streams_finalized=complete, clustermax_reference_sha=CMAX_SHA,
        summary_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistics_helper_sha256=hashlib.sha256(Path(__file__).with_name('summarize_native.py').read_bytes()).hexdigest(),
        raw_evidence_root='/shared/posttrainingx/runs/vultr-b200-slurm/' + run.name,
        aggregation='Sample-weighted distributions; six significant digits. Counter rates require positive monotonic intervals <=5s and nondecreasing counters. GPU NVLink Tx requires all 18 links. IB remains per rail; no asynchronous rail sum.',
        timeline_window_s=60, training=training_context(run, job_id))
    source = next((row for row in data['coverage'] if row['records']), None)
    result = {'schema_version': 1, 'runner': 'telemetry-summary', 'status': status,
        'started_at': start, 'ended_at': end, 'duration_s': rounded(duration),
        'metadata': metadata, 'results': results,
        'log_relpath': source['path'] if source else None, 'log_sha256': source['sha256'] if source else None}
    if errors:
        result.update(exit_code=1, timeout=any(row['reason'] == 'timeout' for row in data['collector_errors']),
                      failure_summary=f'{errors} collector errors; completed Slurm job is not telemetry-qualified.')
    if status == 'skip':
        result['reason'] = 'no_normalized_telemetry_observed'
    return result, timeline_csv(timeline)
