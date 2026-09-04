"""Render measured GPU telemetry and async stage durations as PNG and SVG.

Usage: python3 plot_run.py SNAPSHOT_DIR CHART_DIR --title 'job 197'
Requires matplotlib. Missing data is reported, never replaced with fake curves.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path


def rows(path):
    for line in path.read_text().splitlines():
        try: yield json.loads(line)
        except json.JSONDecodeError: continue


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('snapshot', type=Path); ap.add_argument('output', type=Path)
    ap.add_argument('--title', required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    sources, charts = [], []
    series = {}
    for path in sorted(args.snapshot.glob('infra/*timeseries.jsonl')):
        sources.append(path)
        samples = []
        for row in rows(path):
            gpu = row.get('gpu', {})
            if gpu.get('exit_code') != 0: continue
            query = next(x for x in gpu['argv'] if x.startswith('--query-gpu='))
            keys = query.split('=',1)[1].split(',')
            values = [dict(zip(keys, x)) for x in csv.reader(gpu['stdout'].splitlines())]
            try:
                samples.append((datetime.fromisoformat(row['timestamp']).timestamp(),
                    *[sum(float(x[key]) for x in values)/len(values) for key in
                      ['utilization.gpu','memory.used','power.draw']]))
            except (KeyError, ValueError, ZeroDivisionError): continue
        if samples: series[path.name.split('-timeseries')[0]] = samples
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    if series:
        start = min(values[0][0] for values in series.values())
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        for node, samples in series.items():
            for index, ax in enumerate(axes):
                scale = 1024 if index == 1 else 1
                ax.plot([(x[0]-start)/60 for x in samples], [x[index+1]/scale for x in samples], label=node)
        for ax, ylabel in zip(axes, ['GPU utilization (%)','GPU memory (GiB)','GPU power (W)']):
            ax.set_ylabel(ylabel); ax.grid(alpha=.2)
        axes[0].legend(ncol=2); axes[-1].set_xlabel('Minutes from first captured sample')
        fig.suptitle(args.title+'\nMean per GPU on each node · observed snapshot only')
        for ext in ('png','svg'): fig.savefig(args.output/('gpu-telemetry.'+ext), dpi=160)
        plt.close(fig); charts.append('gpu-telemetry')
    durations = {}
    for path in sorted(args.snapshot.glob('infra/async-events-*.jsonl')):
        sources.append(path)
        for row in rows(path):
            if row.get('phase') == 'end' and row.get('ok') and row.get('operation') in (
                'weight_transfer','broadcast_actor_onload','broadcast_actor_offload','actor_train','critic_train'):
                durations.setdefault(row['operation'], []).append((row.get('rollout_id',-1),row['elapsed_seconds']))
    if durations:
        fig, ax = plt.subplots(figsize=(11,5), constrained_layout=True)
        for op, values in durations.items():
            ax.plot([x[0] for x in values], [x[1] for x in values], 'o-', label=op)
        ax.set(xlabel='Rollout index (-1 = initial publication)', ylabel='Host wall time (seconds)', title=args.title)
        ax.legend(); ax.grid(alpha=.2)
        for ext in ('png','svg'): fig.savefig(args.output/('stage-durations.'+ext), dpi=160)
        plt.close(fig); charts.append('stage-durations')
    manifest = {'charts':charts, 'title':args.title, 'sources':[
        {'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        'no_data':not charts, 'note':'No interpolation of missing runs; host timing is not wire bandwidth.'}
    (args.output/'plot-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'charts': charts, 'output': str(args.output)}))


if __name__ == '__main__': main()
