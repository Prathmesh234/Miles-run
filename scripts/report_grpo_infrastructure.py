"""Report finalized GRPO evidence, including failed runs and missing telemetry."""
import argparse
import inspect
import json

from evidence import Run, atomic, metric, sha256
from report_trainer_probe import analyze_streams, plot
from summarize_native import counter_rate, percentile, summary


def render(data):
    lines = [f"# Job{data['slurm_job_id']}: GRPO infrastructure observation", '',
             data['scope'], '', f"Qualification: **{data['qualification_status']}**. "
             f"Observed optimizer updates: **{data['optimizer_steps_observed']}**.", '',
             '## Limitations', '']
    lines += ['- ' + text for text in data['limitations']]
    lines += ['', '## Collection failures', '']
    lines += ['- ' + text for text in sorted(set(data['findings']))] or ['None.']
    plot_name = data.get('plot_basename', f"grpo-v{data['attempt']}-infrastructure.png")
    lines += ['', '## Time series', '', f"![GPU, fabric and storage]({plot_name})", '',
              'UTC alignment; clocks are not proven synchronized below a sample interval. Missing rates are not zero-filled.', '',
              '## Per-node and per-entity distributions', '',
              '| Node | GPU/link/rail/client | Metric | Unit | Min | Mean | Median | p90 | p95 | p99 | Max | CV |',
              '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['distributions']:
        fields = ('min', 'mean', 'median', 'p90', 'p95', 'p99', 'max', 'coefficient_of_variation')
        values = [row['hostname'], row['entity'], row['metric'], row['unit']] + [row['statistics'][f] for f in fields]
        lines.append('| ' + ' | '.join('undefined' if v is None else f'{v:.5g}' if isinstance(v, float) else str(v) for v in values) + ' |')
    lines += ['', '## Evidence', '', f"[Terminal audit](../{data['audit_path']}) SHA256 `{data['audit_sha256']}`.", '',
              'JSON retains exact plotted samples, every source hash, descriptive outliers and counter discontinuities.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--observation-path', help='Run-relative immutable optimizer observation JSON.')
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-grpo-infrastructure-report-v{args.attempt}')
    audit_path = run.root / f'tests/02-sync-grpo-result-audit-v{args.attempt}/audit.json'
    audit = json.loads(audit_path.read_text())
    observation_path = args.observation_path or f'tests/02-sync-grpo-v{args.attempt}-optimizer-observation/result.json'
    observed = json.loads((run.root / observation_path).read_text())
    program = ('import json,math,statistics,sys\nfrom pathlib import Path\nfrom collections import defaultdict\n'
        + '\n'.join(inspect.getsource(fn) for fn in (percentile, summary, counter_rate, analyze_streams))
        + '\nprint(json.dumps(analyze_streams(Path(sys.argv[1]),json.loads(sys.argv[2])),allow_nan=False))')
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', program, remote, json.dumps(audit['coverage'])], timeout=180)
    if rc:
        phase.finish('fail', failure_summary='Read-only telemetry analysis failed.', refresh=False)
        return 1
    data = json.loads(out)
    data.update(schema_version=1, attempt=args.attempt, slurm_job_id=audit['slurm_job_id'],
        scope='Exploratory qualification, not a controlled role comparison or steady-state performance claim.',
        qualification_status=observed['qualification_status'], optimizer_steps_observed=observed['optimizer_steps_observed'],
        audit_path=str(audit_path.relative_to(run.root)), audit_sha256=sha256(audit_path),
        limitations=observed.get('limitations', [observed.get('failure', 'Qualification incomplete.')]) + [
            'Includes model startup, cold compilation, checkpoints and shutdown.',
            'Host IB and Lustre counters are not process-exclusive. Node/GPU roles differ; outliers do not diagnose hardware faults.',
            'Native collector errors remain in raw evidence and missing samples are not zero-filled. See the per-run error diagnosis for timing.',
            'Complete DCGM/XID/throttle, async queue/staleness and all required RL telemetry are not qualified.',
            'No held-out quality improvement or full checkpoint/resume equivalence demonstrated.'])
    target = run.root / f'reports/grpo-v{args.attempt}-infrastructure.json'
    if target.exists():
        raise FileExistsError(target)
    atomic(target, data)
    frozen = json.loads(target.read_text())
    atomic(target.with_suffix('.md'), render(frozen))
    plot(frozen, target.with_suffix('.png'), title=f"Job {audit['slurm_job_id']}: 2T/2R GRPO qualification; {observed['optimizer_steps_observed']} observed updates\nIncludes startup/JIT/checkpoints/shutdown. Host fabric/storage counters are not process-exclusive.")
    phase.finish('fail' if data['findings'] else 'ok',
        failure_summary='; '.join(sorted(set(data['findings']))) or None,
        results=[metric('telemetry_distribution_count', len(data['distributions']), 'count')],
        metadata={'scope':data['scope'], 'artifacts':[str(target.relative_to(run.root)), str(target.with_suffix('.md').relative_to(run.root)),
            str(target.with_suffix('.png').relative_to(run.root))], 'qualification_status':data['qualification_status']}, refresh=False)
    print(json.dumps({'report':str(target), 'distributions':len(data['distributions']), 'timeline_records':len(data['timeline']),
        'collector_findings':sorted(set(data['findings']))}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
