"""Generate a bounded-preflight report and plots from retained JSON evidence."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import json
import math
from pathlib import Path
import statistics

from evidence import Run, atomic, metric, sha256


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Statistics require finite observed values.')
    mean = statistics.mean(values)
    return {'n': len(values), 'min': min(values), 'mean': mean,
            'median': statistics.median(values), 'p90': percentile(values, .90),
            'p95': percentile(values, .95), 'p99': percentile(values, .99),
            'max': max(values), 'coefficient_of_variation': statistics.pstdev(values) / abs(mean) if mean else None}


def counter_rate(previous, current):
    delta_t = current['monotonic_s'] - previous['monotonic_s']
    delta_v = current['value'] - previous['value']
    if delta_t <= 0 or delta_t > 5 or delta_v < 0:
        return None
    return delta_v / delta_t


def read_stream(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def analyze(run):
    grouped, by_node, series = defaultdict(list), defaultdict(list), defaultdict(list)
    coverage, errors, resets = [], [], []
    sources, job_ids = {}, set()
    for name in ['nvidia-smi', 'nvlink', 'infiniband', 'cpu-memory-numa', 'lustre']:
        path = run.root / 'telemetry' / (name + '.jsonl')
        sources[str(path.relative_to(run.root))] = sha256(path)
        counts, failures, times = Counter(), Counter(), defaultdict(set)
        for row in read_stream(path):
            host = row['hostname']
            counts[host] += 1
            times[host].add(row['monotonic_s'])
            if row['metric'] == 'collector_error':
                failures[host] += 1
                errors.append({'stream': name, **row})
                continue
            if name == 'nvidia-smi':
                job_ids.add(row['slurm_job_id'])
                key = (host, row['gpu_uuid'], row['metric'], row['unit'])
                grouped[key].append(row['value'])
                by_node[(host, row['metric'], row['unit'])].append(row['value'])
                series[(host, row['metric'], row['time'])].append(row['value'])
        for host, count in counts.items():
            ts = sorted(times[host])
            gaps = [b - a for a, b in zip(ts, ts[1:])]
            coverage.append({'stream': name, 'hostname': host, 'records': count,
                'collector_errors': failures[host], 'sample_times': len(ts),
                'sample_interval_s': summary(gaps) if gaps else None})
    gpu_stats = [dict(hostname=h, gpu_uuid=g, metric=m, unit=u, statistics=summary(v))
                 for (h, g, m, u), v in sorted(grouped.items())]
    node_stats = [dict(hostname=h, metric=m, unit=u, statistics=summary(v))
                  for (h, m, u), v in sorted(by_node.items())]
    timeline = [dict(hostname=h, metric=m, time=t, value=statistics.mean(v), contributing_gpus=len(v))
                for (h, m, t), v in sorted(series.items())]
    previous, node_link_rates = {}, defaultdict(list)
    for row in read_stream(run.root / 'telemetry/nvlink.jsonl'):
        if row['metric'] == 'collector_error':
            continue
        key = (row['hostname'], row['gpu_uuid'], row['link'], row['metric'])
        if key in previous:
            rate = counter_rate(previous[key], row)
            if rate is None:
                resets.append({'key': key, 'time': row['time'], 'reason': 'reset_nonpositive_time_or_gap_over_5s'})
            else:
                node_link_rates[(row['hostname'], row['metric'], row['time'])].append(rate)
        previous[key] = row
    for (host, name, timestamp), values in sorted(node_link_rates.items()):
        if len(values) == 8 * 18:
            timeline.append({'hostname': host, 'metric': name.replace('_bytes_total', '_bytes_per_s'),
                             'time': timestamp, 'value': sum(values), 'contributing_links': len(values)})
    collectives, storage = [], []
    for path in sorted((run.root / 'tests').glob('01-native-allreduce-*/*.values.json')):
        data = json.loads(path.read_text())
        collectives.append({'phase': data['runner'], 'hosts': data['metadata']['hosts'],
                            'results': data['results'], 'evidence': str(path.relative_to(run.root))})
    for path in sorted((run.root / 'tests').glob('01-storage-smoke-*/*.values.json')):
        data = json.loads(path.read_text())
        job = data['metadata']['fio']['jobs'][0]
        storage.append({'phase': data['runner'], 'read_Bps': job['read']['bw_bytes'],
                        'write_Bps': job['write']['bw_bytes'], 'io_bytes': job['write']['io_bytes'],
                        'evidence': str(path.relative_to(run.root))})
    outliers = []
    for name in sorted({x['metric'] for x in gpu_stats}):
        rows = [x for x in gpu_stats if x['metric'] == name]
        means = [x['statistics']['mean'] for x in rows]
        q1, q3 = percentile(means, .25), percentile(means, .75)
        low, high = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        outliers.extend(dict(row, criterion='per-GPU time mean outside 1.5 IQR across 32 GPUs')
                        for row in rows if not low <= row['statistics']['mean'] <= high)
    if len(job_ids) != 1:
        raise ValueError('Native report requires exactly one Slurm allocation.')
    return {'schema_version': 1, 'run_id': run.root.name, 'slurm_job_id': next(iter(job_ids)),
        'scope': 'Exploratory native all-reduce and 256 MiB fio preflight; no model or RL workload.',
        'full_telemetry_gate': 'fail', 'source_sha256': sources, 'coverage': coverage,
        'collector_errors': errors, 'counter_rate_discontinuities': resets,
        'gpu_statistics': gpu_stats, 'node_statistics': node_stats, 'gpu_outliers': outliers,
        'outlier_caveat': 'Descriptive flags, not hardware-fault diagnoses; all samples include idle/startup periods.',
        'collectives': collectives, 'storage_smoke': storage, 'timeline': timeline,
        'unavailable': ['IB traffic/errors during load', 'Lustre client performance counters',
            'Complete DCGM coverage', 'CollectiveX', 'Model serving/MTP', 'RL reward and quality',
            'Async queues/staleness/weight updates', 'Model checkpoint/restart performance'],
        'notes': ['Missing samples are not converted to zero.',
            'GPU statistics include idle, startup, collective and storage periods. They are not training utilization.',
            'NVLink rates sum 144 transmit or receive links per node; endpoints may count the same fabric transfer.',
            'The prior ClusterMAX envelope used 16 GiB all-reduce, versus 512 MiB maximum here, with a different harness. No regression percentage is inferred.']}


def render(data):
    lines = ['# Native infrastructure preflight', '', data['scope'], '',
             '**Full telemetry gate: failed. No training or quality result exists.**', '',
             '## All-reduce at 512 MiB', '', '| Nodes | Out-of-place bus GB/s | In-place bus GB/s | Evidence |',
             '|---|---:|---:|---|']
    for item in data['collectives']:
        rows = {r['labels']['mode']: r['value'] for r in item['results']
                if r['metric'] == 'nccl_bus_bandwidth' and r['labels']['message_bytes'] == 536870912}
        lines.append(f"| {', '.join(item['hosts'])} | {rows.get('out_of_place')} | {rows.get('in_place')} | [JSON](../{item['evidence']}) |")
    lines += ['', '## Checkpoint-path smoke', '',
              '256 MiB direct writes with verification per node. This is not a full model checkpoint or saturated storage benchmark.', '',
              '| Phase | Write MB/s | Verify-read MB/s | Evidence |', '|---|---:|---:|---|']
    for item in data['storage_smoke']:
        lines.append(f"| {item['phase']} | {item['write_Bps']/1e6:.2f} | {item['read_Bps']/1e6:.2f} | [JSON](../{item['evidence']}) |")
    lines += ['', '## Time-series coverage', '', '| Stream | Host | Sample times | Collector errors |', '|---|---|---:|---:|']
    for row in data['coverage']:
        lines.append(f"| {row['stream']} | {row['hostname']} | {row['sample_times']} | {row['collector_errors']} |")
    lines += ['', '## Interpretation', ''] + ['- ' + s for s in data['notes']]
    lines += ['', '## Unavailable measurements', ''] + ['- ' + s for s in data['unavailable']]
    lines += ['', '## Per-node statistics', '',
        'Pooled GPU samples over the complete preflight. CV is undefined when the mean is zero.', '',
        '| Host | Metric | Unit | Min | Mean | Median | p90 | p95 | p99 | Max | CV |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['node_statistics']:
        s = row['statistics']
        vals = [row['hostname'], row['metric'], row['unit']] + [s[k] for k in
            ('min', 'mean', 'median', 'p90', 'p95', 'p99', 'max', 'coefficient_of_variation')]
        lines.append('| ' + ' | '.join('undefined' if v is None else f'{v:.4g}' if isinstance(v, float) else str(v) for v in vals) + ' |')
    lines += ['', '## Per-GPU statistics and outliers', '', data['outlier_caveat'], '',
        f"{len(data['gpu_outliers'])} per-GPU metric flags. Complete per-GPU distributions and flags are in `infrastructure.json`.", '',
        'Plot: [native-preflight.png](native-preflight.png). All plotted values derive from `infrastructure.json`.', '',
        'Raw finalized time series: `../telemetry/*.jsonl`. Individual phase JSON above points to the retained shared raw logs.', '']
    return '\n'.join(lines)


def plot(data, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin = min(dt.datetime.fromisoformat(x['time'].replace('Z', '+00:00')) for x in data['timeline'])
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    specs = [('utilization.gpu', 'GPU utilization (%)', 1), ('power.draw', 'Power per GPU (W)', 1),
             ('temperature.gpu', 'GPU temperature (°C)', 1), ('memory.used', 'Used HBM per GPU (GiB)', 1/1024),
             ('nvlink_data_tx_bytes_per_s', 'Node aggregate NVLink Tx (GB/s)', 1/1e9)]
    for axis, (name, title, factor) in zip(axes.flat, specs):
        for host in sorted({x['hostname'] for x in data['timeline']}):
            rows = sorted((x for x in data['timeline'] if x['hostname'] == host and x['metric'] == name), key=lambda r:r['time'])
            x = [(dt.datetime.fromisoformat(r['time'].replace('Z', '+00:00')) - origin).total_seconds() for r in rows]
            axis.plot(x, [r['value']*factor for r in rows], label=host, linewidth=1.1)
        axis.set_title(title)
        axis.grid(alpha=.2)
        axis.set_xlabel('Seconds since first sample (UTC alignment)')
    axes.flat[0].legend(fontsize=8)
    axes.flat[5].axis('off')
    axes.flat[5].text(.02, .85, f"Native preflight only, Slurm job {data['slurm_job_id']}\n\nNo model training or quality measurement\n\nIB counters during load: missing\nLustre performance counters: missing\nDCGM: incomplete\n\nMissing data is not plotted as zero.", va='top', transform=axes.flat[5].transAxes)
    fig.suptitle('Vultr B200: measured native preflight, telemetry gate failed')
    fig.tight_layout()
    temporary = target.with_suffix('.partial.png')
    fig.savefig(temporary, dpi=160)
    temporary.replace(target)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('01-native-infrastructure-report')
    try:
        data = analyze(run)
        atomic(run.root / 'reports/infrastructure.json', data)
        atomic(run.root / 'reports/infrastructure.md', render(data))
        plot(data, run.root / 'reports/native-preflight.png')
    except Exception as exc:
        phase.finish('fail', failure_summary='Native report generation failed: ' + str(exc))
        raise
    phase.finish('ok', metadata={'scope': 'Report generation only; the complete telemetry gate remains failed.',
        'artifacts': ['reports/infrastructure.json', 'reports/infrastructure.md', 'reports/native-preflight.png']},
        results=[metric('observed_gpu_uuid_count', len({r['gpu_uuid'] for r in data['gpu_statistics']}), 'count'),
                 metric('collector_error_count', len(data['collector_errors']), 'count')])
    print(json.dumps({'report': str(run.root / 'reports/infrastructure.md'), 'collector_errors': len(data['collector_errors'])}))


if __name__ == '__main__':
    main()
