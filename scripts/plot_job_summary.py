"""Plot compact telemetry envelopes without disguising failed coverage gates."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys

from evidence import Run, atomic, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--summary-dir', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    folder = Path(args.summary_dir)
    candidates = list(folder.glob('telemetry.*.json'))
    if len(candidates) != 1:
        raise ValueError('Require exactly one current telemetry summary.')
    data = json.loads(candidates[0].read_text())
    rows = list(csv.DictReader((folder / 'timeline.csv').open()))
    if not rows:
        raise ValueError('No observations to plot; missing data is not zero.')
    sources = {str(path): sha256(path) for path in [candidates[0], folder / 'timeline.csv']}
    run = Run(args.run_dir)
    job = data['metadata']['slurm_job_id']
    phase = run.phase(f'02-summary-plot-job{job}-v{args.attempt}')
    if shutil.disk_usage(phase.path).free < 128 * 1024**2:
        raise RuntimeError('Plot requires 128 MiB free space.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def timestamp(value):
        return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))

    origin = timestamp(min(row['time'] for row in rows))
    specs = [('gpu_utilization', 'GPU utilization (%)'), ('gpu_hbm_used', 'HBM per GPU (GiB)'),
             ('nvlink_gpu_tx', 'NVLink Tx per GPU (GB/s)'), ('ib_rail_tx', 'IB Tx per rail (GB/s)'),
             ('lustre_client_write', 'Lustre writes per client (GB/s)'),
             ('shared_storage_available', 'Shared storage available (TiB)')]
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for axis, (metric, title) in zip(axes.flat, specs):
        for index, host in enumerate(sorted({row['node'] for row in rows})):
            selected = sorted((row for row in rows if row['node'] == host and row['metric'] == metric),
                              key=lambda row: row['time'])
            if not selected:
                continue
            x = [(timestamp(row['time']) - origin).total_seconds() / 60 for row in selected]
            mean, high = [[float(row[key]) for row in selected] for key in ('mean', 'max')]
            # Unobserved minutes remain gaps, never connecting a missing interval.
            for position in reversed(range(1, len(x))):
                if x[position] - x[position-1] > 1.01:
                    x.insert(position, float('nan')); mean.insert(position, float('nan')); high.insert(position, float('nan'))
            axis.plot(x, mean, color=f'C{index}', label=host)
            axis.plot(x, high, color=f'C{index}', linestyle=':', alpha=.6)
        axis.set_title(title)
        axis.grid(alpha=.2)
        axis.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Minutes from ' + origin.isoformat())
    figure.suptitle(f"Job {job}: {data['status'].upper()} telemetry gate; Slurm {data['metadata']['slurm_state']}\n"
        'One-minute sample-weighted means (solid) and maxima (dotted); startup/checkpoints/teardown included.\n'
        'Raw sampling gaps remain failures. These envelopes are not evidence of uninterrupted coverage.', fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .91))
    destination = phase.path / 'infrastructure.png'
    temporary = phase.path / '.infrastructure.png'
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    if any(sha256(path) != digest for path, digest in sources.items()):
        phase.finish('fail', failure_summary='Summary changed during rendering.', refresh=False)
        return 1
    result = dict(source_sha256=sources, plot_sha256=sha256(destination),
        command=[sys.executable, *sys.argv], script_sha256=sha256(__file__),
        matplotlib_version=matplotlib.__version__, python_version=sys.version,
        source_gate_status=data['status'], slurm_job_id=job,
        scope='Rendering verified; source gate status is unchanged. No async or held-out quality claim.')
    atomic(phase.path / 'result.json', result)
    phase.finish('ok', metadata=result, refresh=False)
    print(json.dumps({'plot': str(destination), 'source_gate_status': data['status']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
