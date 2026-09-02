"""Derive a diagnostic trainer report and plots from immutable remote telemetry."""
import argparse
from collections import defaultdict
import datetime as dt
import inspect
import json
import math
from pathlib import Path
import statistics

from evidence import Run, atomic, metric, sha256
from summarize_native import counter_rate, percentile, summary


def analyze_streams(root, coverage):
    import hashlib
    groups, series = defaultdict(list), []
    gaps, sources, findings = [], {}, []
    for source in coverage:
        path = root / source['path']
        if path.is_symlink() or not path.is_file():
            raise ValueError('Telemetry is missing or linked: ' + source['path'])
        digest, previous = hashlib.sha256(), {}
        with path.open('rb') as handle:
            for line in handle:
                digest.update(line)
                row = json.loads(line)
                if row['metric'] == 'collector_error':
                    findings.append('Collector error in ' + source['path'])
                    continue
                name, unit, value = row['metric'], row['unit'], row['value']
                if path.name == 'nvidia-smi.jsonl':
                    entity = row['gpu_uuid']
                elif path.name == 'nvlink.jsonl':
                    entity = row['gpu_uuid'] + '/link-' + str(row['link'])
                elif path.name == 'infiniband.jsonl':
                    entity = row['hca'] + '/' + row['hca_port']
                elif row['source'] == 'lustre-host-debugfs' and name in ['read_bytes.sum', 'write_bytes.sum']:
                    entity = row['lustre_client']
                else:
                    continue
                key = (row['hostname'], entity, name)
                if path.name != 'nvidia-smi.jsonl':
                    old = previous.get(key)
                    previous[key] = row
                    if old is None:
                        continue
                    value = counter_rate(old, row)
                    if value is None:
                        gaps.append({'path': source['path'], 'entity': entity, 'metric': name,
                                     'time': row['time'], 'reason': 'reset_nonpositive_time_or_gap_over_5s'})
                        continue
                    name, unit = name + '.rate', unit + '/s'
                if not math.isfinite(value):
                    findings.append('Nonfinite metric in ' + source['path'])
                    continue
                groups[(row['hostname'], entity, name, unit)].append(value)
                plotted = (name in ['utilization.gpu', 'power.draw', 'memory.used',
                                    'nvlink_data_tx_bytes_total.rate', 'PortXmitData.rate',
                                    'read_bytes.sum.rate', 'write_bytes.sum.rate'])
                if plotted:
                    series.append({'time': row['time'], 'monotonic_s': row['monotonic_s'],
                                   'hostname': row['hostname'], 'entity': entity,
                                   'metric': name, 'value': value, 'unit': unit})
        sources[source['path']] = digest.hexdigest()
        if digest.hexdigest() != source['sha256']:
            findings.append('Telemetry changed since result audit: ' + source['path'])
    distributions = [dict(hostname=h, entity=e, metric=m, unit=u, statistics=summary(values))
                     for (h, e, m, u), values in sorted(groups.items())]
    outliers = []
    for name in sorted({row['metric'] for row in distributions}):
        rows = [row for row in distributions if row['metric'] == name]
        if len(rows) < 4:
            continue
        means = [row['statistics']['mean'] for row in rows]
        q1, q3 = percentile(means, .25), percentile(means, .75)
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outliers.extend(row for row in rows if not low <= row['statistics']['mean'] <= high)
    # Keep the plot source compact without smoothing or aligning independent HCA clocks.
    # NVLink rows share a native sampling tick and can be summed exactly per GPU.
    grouped_links, compact = defaultdict(list), []
    for row in series:
        if row['metric'] == 'nvlink_data_tx_bytes_total.rate':
            grouped_links[(row['hostname'], row['entity'].split('/')[0], row['time'], row['monotonic_s'])].append(row['value'])
        else:
            compact.append(row)
    for (host, gpu, timestamp, monotonic), values in sorted(grouped_links.items()):
        if len(values) != 18:
            findings.append('Incomplete per-GPU NVLink tick; not plotted as zero.')
            continue
        compact.append({'hostname': host, 'entity': gpu, 'time': timestamp, 'monotonic_s': monotonic,
                        'metric': 'nvlink_gpu_tx_bytes_per_s', 'value': sum(values), 'unit': 'B/s'})
    return {'findings': findings, 'source_sha256': sources, 'distributions': distributions,
            'descriptive_outliers': outliers, 'counter_discontinuities': gaps, 'timeline': compact}


def render(data):
    lines = ['# EP8 trainer diagnostic', '', data['scope'], '',
             f"Slurm job **{data['slurm_job_id']}**. Status: **{data['status']}**. Optimizer steps: **0**.", '',
             '## Per-rank checks', '',
             '| Rank | Parameter tensors unchanged | Nonzero gradients | Nonzero MTP gradients | Load (s) | Forward/backward (s) | Peak allocated GiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in data['ranks']:
        lines.append(f"| {row['rank']} | {row['parameter_tensor_count']} | {row['gradient_tensors_nonzero']} | "
                     f"{row['mtp_gradient_tensors_nonzero']} | {row['checkpoint_load_duration_s']:.3f} | "
                     f"{row['forward_backward_duration_s']:.3f} | {row['cuda_peak_allocated_bytes'] / 1024**3:.3f} |")
    lines += ['', '## Interpretation and limits', ''] + ['- ' + text for text in data['notes']]
    lines += ['', '## Telemetry', '', '[Diagnostic time series](trainer-probe-v1.png). Missing values are not zero-filled.', '',
              '| Stream | Records | Collector errors | Maximum gap (s) |', '|---|---:|---:|---:|']
    for row in data['coverage']:
        lines.append(f"| [{row['path']}](../{row['path']}) | {row['records']} | {row['collector_errors']} | {row['max_interval_s']:.3f} |")
    lines += ['', '## Per-GPU statistics', '',
              'All samples, including startup, checkpoint reads, hashing and idle periods. These are not steady-state GRPO measurements.', '',
              '| GPU / entity | Metric | Unit | Min | Mean | Median | p90 | p95 | p99 | Max | CV |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields = ('min', 'mean', 'median', 'p90', 'p95', 'p99', 'max', 'coefficient_of_variation')
    for row in data['distributions']:
        # Full link/rail distributions remain in JSON; GPU gauges are concise enough for Markdown.
        if '/' in row['entity'] or row['metric'].endswith('.rate'):
            continue
        values = [row['entity'], row['metric'], row['unit']] + [row['statistics'][key] for key in fields]
        lines.append('| ' + ' | '.join('undefined' if v is None else f'{v:.4g}' if isinstance(v, float) else str(v) for v in values) + ' |')
    lines += ['', '## Unvalidated measurements', ''] + ['- ' + text for text in data['unvalidated']]
    lines += ['', '## Evidence', '', f"Result audit: [{data['audit_path']}](../{data['audit_path']}), SHA256 `{data['audit_sha256']}`.", '',
              'The JSON source includes all per-link/rail distributions, descriptive outliers, source hashes and the exact plotted samples.', '',
              'Raw rank files and torchrun output: `../tests/02-trainer-probe-child-v1/`.', '']
    return '\n'.join(lines)


def plot(data, path, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs = [('utilization.gpu', 'GPU utilization (%)', 1), ('memory.used', 'HBM used (GiB)', 1 / 1024),
             ('power.draw', 'GPU power (W)', 1), ('nvlink_gpu_tx_bytes_per_s', 'NVLink Tx per GPU (GB/s)', 1e-9),
             ('PortXmitData.rate', 'IB Tx per rail (GB/s)', 1e-9), ('lustre', 'Host Lustre client I/O (GB/s)', 1e-9)]
    origin = min(dt.datetime.fromisoformat(row['time'].replace('Z', '+00:00')) for row in data['timeline'])
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, (name, axis_title, factor) in zip(axes.flat, specs):
        groups = defaultdict(list)
        for row in data['timeline']:
            if row['metric'] == name or (name == 'lustre' and row['metric'] in ['read_bytes.sum.rate', 'write_bytes.sum.rate']):
                groups[(row['hostname'], row['entity'], row['metric'])].append(row)
        for (host, entity, metric_name), rows in sorted(groups.items()):
            rows.sort(key=lambda row: row['time'])
            times = [(dt.datetime.fromisoformat(row['time'].replace('Z', '+00:00')) - origin).total_seconds() for row in rows]
            label = host + '/' + (entity[:8] if name != 'lustre' else entity[:8] + ' ' + metric_name.split('_')[0])
            axis.plot(times, [row['value'] * factor for row in rows], label=label, linewidth=.8)
        axis.set_title(axis_title)
        axis.grid(alpha=.2)
        axis.set_xlabel('Seconds since first plotted sample (UTC alignment)')
        axis.legend(fontsize=6, ncol=2)
    fig.suptitle(title or 'Job 120: EP8 load and diagnostic forward/backward; no optimizer or GRPO\nIncludes startup/JIT/hashing. Host fabric/storage counters are not process-exclusive.', fontsize=11)
    fig.tight_layout()
    temporary = path.with_suffix('.partial.png')
    fig.savefig(temporary, dpi=150)
    temporary.replace(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('02-trainer-probe-report-v1')
    audit_path = run.root / 'tests/02-trainer-probe-result-audit-v1/audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['findings']:
        phase.finish('fail', failure_summary='Trainer result audit did not pass.')
        return 1
    code = ('import json,math,statistics,sys\nfrom pathlib import Path\nfrom collections import defaultdict\n'
            + '\n'.join(inspect.getsource(fn) for fn in [percentile, summary, counter_rate, analyze_streams])
            + '\nprint(json.dumps(analyze_streams(Path(sys.argv[1]),json.loads(sys.argv[2])),allow_nan=False))')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    try:
        rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
                                   'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', code, remote,
                                   json.dumps(audit['coverage'])], timeout=120)
        if rc:
            raise ValueError('Read-only telemetry analysis failed; raw stderr retained.')
        data = json.loads(out)
        data.update(schema_version=1, status='fail' if data['findings'] else 'ok', scope=audit['scope'],
                    slurm_job_id=audit['slurm_job_id'], ranks=audit['ranks'], coverage=audit['coverage'],
                    audit_path=str(audit_path.relative_to(run.root)), audit_sha256=sha256(audit_path),
                    training_started=False, heldout_quality_measured=False,
                    notes=audit['notes'] + [
                        'First forward/backward includes compilation/warmup. Do not use its duration as steady-state training throughput.',
                        'Megatron resolves sequence_parallel=false at TP1; TP1/EP8/PP1/CP1/ETP1 and MTP settings were checked.',
                        'Successful loading and unchanged hashes do not replace independent EP8 shard-to-checkpoint value parity.',
                        'Per-link/rail counter rates are node-wide and may include other clients; NVLink endpoints can count the same transfer.',
                        'Outliers use per-entity time means outside 1.5 IQR. They are descriptive, not a diagnosis of faulty hardware.',
                        'This one-node diagnostic cannot attribute an end-to-end RL bottleneck or compare role splits.'],
                    unvalidated=['Independent EP8 shard-value and serving-logprob equivalence.',
                                 'Real GRPO grouping, rewards, gradients, optimizer update and complete checkpoint/resume.',
                                 'Complete DCGM, throttle/XID, Ray, SGLang and Miles telemetry during the combined RL workload.',
                                 'Local policy/grader isolation and task runtime validation.',
                                 'Async overlap, queues, staleness, broadcast activation, role-split comparisons and held-out quality.'])
        # Reports are rendered only from the serialized JSON source.
        target = run.root / 'reports/trainer-probe-v1.json'
        if target.exists():
            raise FileExistsError('Refusing to replace an existing diagnostic report.')
        atomic(target, data)
        frozen = json.loads(target.read_text())
        atomic(target.with_suffix('.md'), render(frozen))
        plot(frozen, target.with_suffix('.png'))
        phase.finish(data['status'], failure_summary='; '.join(data['findings']) or None,
                     metadata={'artifacts': [str(target.relative_to(run.root)), 'reports/trainer-probe-v1.md', 'reports/trainer-probe-v1.png'],
                               'scope': audit['scope'], 'counter_discontinuities': len(data['counter_discontinuities'])},
                     results=[metric('telemetry_distribution_count', len(data['distributions']), 'count', 'gpu-nodes-0')])
        print(json.dumps({'findings': data['findings'], 'report': str(target),
                          'distributions': len(data['distributions']), 'timeline_records': len(data['timeline'])}))
        return int(bool(data['findings']))
    except Exception as exc:
        phase.finish('fail', failure_summary=str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
