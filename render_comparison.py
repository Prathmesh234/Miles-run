"""Generate shareable plots and Markdown from consolidated JSON/CSV, never stdout.

Plot environment: matplotlib==3.9.4, numpy==2.0.2. The complete environment is
recorded in results/provenance.json. Missing measurements are omitted.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--preview',action='store_true',help='Also write local PNG previews for visual QA')
    args = parser.parse_args(); out = args.directory
    data = json.loads((out/'comparison.json').read_text())
    runs = data['runs']; baseline = data['baseline']
    labels = [baseline['label']]+[r['label'] for r in runs]
    colors = ['#777777', '#305b8c', '#a55730']
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none', 'svg.hashsalt':'miles-ppo-comparison',
                         'axes.grid':True, 'grid.alpha':.16})
    figure, axes = plt.subplots(3, 3, figsize=(15,11), layout='constrained')
    axes = axes.flatten()
    figure.suptitle('Qwen3.6-35B-A3B on 32 B200 GPUs: two-update workload comparison\n'
                    'Training rewards are not held-out quality; task mix and timer boundaries differ.', fontsize=14)
    bsteps = baseline['steps']
    def plot_steps(ax, title, ylabel, bfield, field, scalar=None):
        if bfield:
            pairs = [(s['step'],s.get(bfield)) for s in bsteps if s.get(bfield) is not None]
            if pairs: ax.plot(*zip(*pairs), 'o-', color=colors[0], label=labels[0])
        for index, run in enumerate(runs):
            pairs = [(s['step'],s['scalars'].get(scalar) if scalar else s.get(field)) for s in run['steps']]
            pairs = [(x,y) for x,y in pairs if y is not None]
            if pairs: ax.plot(*zip(*pairs), 'o-', color=colors[index+1], label=labels[index+1])
        ax.set(title=title, ylabel=ylabel, xlabel='Optimizer update', xticks=[1,2])
        if ax.lines: ax.legend(fontsize=7)
    plot_steps(axes[0], 'Raw reward of accepted training traces', 'Mean reward', 'mean_reward', 'reward')
    axes[0].set_ylim(0,1.05)
    plot_steps(axes[1], 'Active training tokens per actor update', 'Unmasked tokens', 'active_training_tokens', 'active_tokens')
    # Prime timer removes wait/broadcast/load; Miles train timer includes backend
    # overhead. Explicitly label this approximate comparison.
    plot_steps(axes[2], 'Approximate actor training path', 'Seconds (different timer boundaries)', 'approx_actor_s', None, 'perf/train_time')
    plot_steps(axes[3], 'Critic compute (PPO only)', 'Seconds', None, 'critic_train_s')
    counts = [(r['accounting']['generated'],r['accounting']['accepted']) for r in [baseline]+runs]
    x = np.arange(len(labels))
    axes[4].bar(x-.18,[p[0] for p in counts],.36,color='#777777',label='Generated')
    axes[4].bar(x+.18,[p[1] for p in counts],.36,color='#305b8c',label='Accepted')
    axes[4].set(title='Trajectory accounting',ylabel='Traces',xlabel='Run',xticks=x,xticklabels=labels)
    axes[4].tick_params(axis='x',labelsize=7); axes[4].legend(fontsize=7)
    times = [r['ready_to_checkpoint_s'] for r in [baseline]+runs]
    for i, value in enumerate(times):
        if value is not None:
            axes[5].bar(i,value,color=colors[i]); axes[5].text(i,value,f'{value:.1f}',ha='center',va='bottom')
    axes[5].set(title='Inference ready to final checkpoint',ylabel='Seconds',xlabel='Run',xticks=x,xticklabels=labels)
    axes[5].tick_params(axis='x',labelsize=7)
    plot_steps(axes[6], 'Actor gradient norm', 'Pre-clipping L2 norm', 'optim/grad_norm', None, 'train/grad_norm')
    plot_steps(axes[7], 'Accepted batch reward acquisition', 'Seconds (prefetch wait vs full rollout)', 'time/wait_for_batch', 'rollout_s')
    for i, run in enumerate(runs):
        vals=[]
        for node in range(4):
            item=run['telemetry']['node_statistics'].get(f'gpu-nodes-{node}/hbm_used_mib_max')
            vals.append(item['max']/1024 if item else np.nan)
        axes[8].bar(np.arange(4)+(i-.5)*.35,vals,.35,color=colors[i+1],label=run['label'])
    axes[8].set(title='Peak observed device HBM',ylabel='GiB per GPU (maximum on node)',xlabel='Node role',
                xticks=range(4),xticklabels=['0 train','1 train','2 rollout','3 rollout']); axes[8].legend(fontsize=7)
    figure.savefig(out/'training-comparison.svg', metadata={'Date':None})
    if args.preview: figure.savefig(out/'training-comparison.png',dpi=120)
    plt.close(figure)

    ppo = next(r for r in runs if r['algorithm'] == 'ppo')
    figure, axes = plt.subplots(3,3,figsize=(14,10),layout='constrained')
    figure.suptitle('PPO correctness and weight-transfer diagnostics\n'
                    'Native Miles scalars; near-zero first-epoch KL/clip can occur before the policy update.',fontsize=14)
    fields=[('train/pg_loss','Policy surrogate loss','Loss'),
            ('train/critic-value_loss','Clipped value loss','Loss'),
            ('train/entropy_loss','Measured actor entropy','Nats / token'),
            ('train/grad_norm','Actor gradient norm','Pre-clipping L2 norm'),
            ('train/critic-grad_norm','Critic gradient norm','Pre-clipping L2 norm'),
            ('train/pg_clipfrac','Policy clipped fraction','Fraction'),
            ('train/critic-value_clipfrac','Value clipped fraction','Fraction'),
            ('train/ppo_kl','Behavior/current sampled logprob difference','Native PPO KL metric')]
    for ax,(field,title,unit) in zip(axes.flatten(),fields):
        rows=ppo['all_scalars'].get(field,[])
        if rows: ax.plot([r['step']+1 for r in rows],[r['value'] for r in rows],'o-',color=colors[2],label=ppo['label'])
        ax.set(title=title,xlabel='Optimizer update',ylabel=unit,xticks=[1,2])
        if rows: ax.legend(fontsize=7)
    ax=axes.flatten()[-1]
    for i,run in enumerate(runs):
        transfers=[t for t in run['timers'] if t['role']=='actor' and t['name']=='update_weights']
        if transfers: ax.plot(range(len(transfers)),[t['seconds'] for t in transfers],'o-',color=colors[i+1],label=run['label'])
    ax.set(title='Broadcast path (excludes actor offload)',xlabel='Broadcast index (0 = initial)',ylabel='Seconds',xticks=[0,1,2])
    if ax.lines: ax.legend(fontsize=7)
    figure.savefig(out/'ppo-diagnostics.svg',metadata={'Date':None})
    if args.preview: figure.savefig(out/'ppo-diagnostics.png',dpi=120)
    plt.close(figure)

    series = list(csv.DictReader((out/'timeseries.csv').open()))
    figure, axes = plt.subplots(4,2,figsize=(15,14),layout='constrained'); axes=axes.flatten()
    figure.suptitle('Measured infrastructure and inference timelines\n'
                   'Time zero: rollout engines marked alive. Curves use sampled counters; unsupported metrics are omitted.',fontsize=14)
    panels=[('gpu_util_percent_mean','GPU utilization','% (mean of GPUs in each role)'),
            ('hbm_used_mib_mean','Allocated device HBM','GiB per GPU (role mean)'),
            ('power_w_mean','GPU board power','W per GPU (role mean)'),
            ('cpu_busy_percent','Host CPU busy','% (role mean across logical CPUs)'),
            ('ib_tx_gbps','Fabric transmit payload','Gb/s (sum of local ports in role)'),
            ('nvlink_tx_gbps','NVLink transmit payload','Gb/s (sum of local links in role)'),
            ('sglang:num_queue_reqs','SGLang queue (reported DP0 scheduler)','Requests (reported gauge, not DP0+DP1)'),
            ('shared_free_gib','Shared filesystem free space','GiB')]
    for ax,(metric,title,unit) in zip(axes,panels):
        for i,run in enumerate([baseline]+runs):
            for role,style in [('trainer','-'),('rollout','--')]:
                if metric=='shared_free_gib' and role=='rollout': continue
                exact=collections.defaultdict(list)
                for row in series:
                    if int(row['job_id'])==run['job_id'] and row['metric']==metric and row['role']==role:
                        exact[(float(row['elapsed_s']),row['hostname'])].append(float(row['value']))
                bins=collections.defaultdict(list)
                sum_metric=metric in ['ib_tx_gbps','nvlink_tx_gbps','sglang:num_queue_reqs']
                for (t,host),values in exact.items():
                    value=sum(values) if sum_metric else float(np.mean(values))
                    if metric=='hbm_used_mib_mean':value/=1024
                    bins[(int(t//5)*5,host)].append(value)
                by_time=collections.defaultdict(list)
                for (t,host),values in bins.items():
                    by_time[t].append(float(np.mean(values)))
                if by_time:
                    ts=sorted(by_time)
                    values=[sum(by_time[t]) if sum_metric else float(np.mean(by_time[t])) for t in ts]
                    ax.plot(ts,values,style,color=colors[i],linewidth=1.25,label=run['label']+' '+role)
        ax.set(title=title,xlabel='Seconds since inference ready (5-second bins)',ylabel=unit)
        if ax.lines: ax.legend(fontsize=7)
        else: ax.text(.5,.5,'Not observed in these archives',ha='center',transform=ax.transAxes)
    figure.savefig(out/'infrastructure-timeline.svg',metadata={'Date':None})
    if args.preview: figure.savefig(out/'infrastructure-timeline.png',dpi=120)
    plt.close(figure)

    ipo=next(r for r in runs if r['algorithm']=='ipo')
    complete_timing=ppo['ready_to_checkpoint_s'] is not None and ipo['ready_to_checkpoint_s'] is not None
    change=(ppo['ready_to_checkpoint_s']/ipo['ready_to_checkpoint_s']-1)*100 if complete_timing else None
    critic_seconds=sum(t['seconds'] for t in ppo['timers'] if t['role']=='critic' and t['name']=='critic_train')
    residency_seconds=sum(t['seconds'] for t in ppo['timers'] if t['name'] in ['sleep','wake_up'])
    headline=(f"The PPO repeat logged **{ppo['completed_actor_updates']} actor and {ppo['completed_critic_updates']} critic updates**. Inference-ready to final checkpoint was **{ppo['ready_to_checkpoint_s']:.2f} s**, "
            f"versus **{ipo['ready_to_checkpoint_s']:.2f} s** for Miles IPO ({change:+.1f}%). This is an observed run difference, not an isolated PPO-versus-IPO speed penalty.",'',
            f"PPO generated {ppo['accounting']['generated']} episodes / {ppo['accounting']['output_tokens']:,} output tokens; IPO generated "
            f"{ipo['accounting']['generated']} / {ipo['accounting']['output_tokens']:,}. Both consumed 32 accepted traces. "
            f"PPO added {critic_seconds:.2f} s of logged critic compute and {residency_seconds:.2f} s of model sleep/wake transitions across the full driver lifecycle. "
            'These timers have different boundaries and are not an additive critical-path decomposition.') if complete_timing else (
            f"PPO attempt status: **{ppo['status']}**. No complete checkpoint timing comparison is available.",)
    report=['# PPO comparison with the previous Miles run','',data['scope']+'.','',*headline,'',
            '## Results','', 'The plots and this report are generated from [comparison.json](comparison.json) and [timeseries.csv](timeseries.csv).', '',
            '![Training comparison](training-comparison.svg)','',
            '![Infrastructure timelines](infrastructure-timeline.svg)','',
            '![PPO diagnostics](ppo-diagnostics.svg)','']
    for run in runs:
        report += [f"### {run['label']}",'',
            f"Status: **{run['status']}**. The archive contains {len(run['steps'])} accepted rollout batches, "
            f"{run['accounting']['generated']} generated episodes and {run['accounting']['accepted']} accepted traces. "
            f"Logged optimizer steps: **{run['completed_actor_updates']} actor / {run['completed_critic_updates']} critic**. "
            f"Episode errors: {run['accounting']['errors']}.",'',
            f"Raw evidence: `{run['raw_archive']}`. Plot window: {run['window_definition']}.",'']
        for step in run['steps']:
            report += [f"- Update {step['step']}: raw reward {step['reward']:.4f}; "
                       f"{step['active_tokens']:,} active tokens; {step['rollout_s']:.2f} s of rollout; tasks {', '.join(step['tasks'])}."]
        report += ['',f"Zero-variance reward groups: {run['accounting']['zero_variance_generated_groups']} generated; "
            f"{run['accounting']['accepted_zero_variance_groups']} accepted. "
            f"Traces consumed by logged actor steps: {run['accounting']['consumed_by_logged_actor_step']}."]
        report += ['', 'Per-task generated-episode rewards (not held-out evaluations):','']
        for task in sorted({e['task'] for e in run['episodes']}):
            rows=[e for e in run['episodes'] if e['task']==task]
            report += [f"- `{task}`: {np.mean([e['reward'] for e in rows]):.4f} mean reward across {len(rows)} episodes; {sum(e['accepted'] for e in rows)} accepted traces."]
        if run['ready_to_checkpoint_s'] is not None: report += ['',f"Inference ready to final save: **{run['ready_to_checkpoint_s']:.2f} seconds**."]
        duration=run['episode_statistics']['duration_s']
        if duration: report += ['',f"Episode duration: median {duration['median']:.2f} s; p95 {duration['p95']:.2f} s; p99 {duration['p99']:.2f} s; maximum {duration['max']:.2f} s."]
        for role,nodes in [('trainer',[0,1]),('rollout',[2,3])]:
            means=[run['telemetry']['node_statistics'].get(f'gpu-nodes-{n}/gpu_util_percent_mean',{}).get('mean') for n in nodes]
            means=[x for x in means if x is not None]
            if means: report += ['',f"{role.capitalize()} GPU utilization over the plotted window: **{np.mean(means):.2f}%** (mean of node means, not achieved FLOP utilization)."]
        gpu_values=[(name,v['gpu_util_percent']['mean']) for name,v in run['telemetry']['gpu_statistics'].items() if 'gpu_util_percent' in v]
        if gpu_values:
            low=min(gpu_values,key=lambda x:x[1]);high=max(gpu_values,key=lambda x:x[1])
            report += ['',f"Observed GPU utilization extremes: `{low[0]}` at {low[1]:.2f}% mean; `{high[0]}` at {high[1]:.2f}% mean. "
                'Role differences matter; these are descriptive outliers, not hardware fault labels.']
        for key in ['checkpoints','critic_checkpoint']:
            checkpoint=run.get(key)
            if checkpoint:
                comparison=checkpoint.get('selected_tensors_vs_base')
                delta=f"{sum(v['changed_elements'] for v in comparison.values())} changed elements in sampled tensors versus base." if comparison else 'Base-weight deltas were not measured for this historical checkpoint.'
                report += ['',f"{checkpoint.get('role','actor' if key=='checkpoints' else 'critic')} checkpoint: {sum(checkpoint['shard_files'].values())/1024**3:.2f} GiB, "
                    f"{len(checkpoint['sampled_tensor_reads'])} finite small tensors loaded on CPU; {delta} "
                    'This is structural/sample validation, not full-state resume validation.']
        report += ['',f"Collector-error counts inside the plotted window: `{json.dumps(run['telemetry']['collector_errors'],sort_keys=True)}`.",'']
    report += ['## Failures and attribution','',
        'The successful comparison excludes failed attempts; their cost and failure phases remain in the JSON and [intervention log](interventions.json). '
        'Job 195 reached a critic step but had a zero-gradient, uniform-output actor and was deliberately stopped. '
        'It is not evidence of PPO learning. Configuration/staging failures and the actor offload lifecycle defects are not attributed to Vultr hardware.','']
    for attempt in data['failed_attempts']:
        report += [f"- Job {attempt['job_id']}: exit {attempt['exit_code']}; {attempt['actor_step_calls']} actor / {attempt['critic_step_calls']} critic step calls; zero accepted valid actor updates."]
    report += ['', 'Native PPO adds critic forward/backward work and actor/critic offload transitions. '
        'The synchronous recipe serializes environment rollouts and optimizer work; trainer waiting and rollout idle periods are expected. '
        'Low whole-window GPU utilization does not by itself diagnose weak GPUs or fabric. '
        'Use the preserved timers, per-node variability and link counters to distinguish time spent waiting from compute. '
        'These short runs do not establish a storage/fabric bandwidth ceiling or a statistically reliable framework speedup.','']
    provenance=json.loads((out/'provenance.json').read_text())
    runtime=provenance.get('final_runtime') or {}
    for command in runtime.get('commands',[]):
        if command['argv'][0]=='squeue' and command['stdout'].strip():
            report += ['### Other allocation after this run','',
                f"The post-run queue snapshot was `{command['stdout'].strip()}`. This is not the completed PPO allocation. "
                'Our job containers were stopped and GPU UUIDs reconciled. The cluster was subsequently allocated to another job; '
                'do not claim it remained globally idle. Post-run hashing is outside this run\'s measured window but can overlap another allocation\'s startup.','']
    for command in runtime.get('commands',[]):
        if command['argv'][0]=='sacct' and command['exit_code']==0:
            report += ['### Allocation accounting','',
                'Slurm allocation durations include startup and cleanup. Failed attempts are not included in the successful-run charts.','']
            records=list(csv.DictReader(command['stdout'].splitlines(),delimiter='|'))
            for row in records:
                report += [f"- Job {row['JobID']}: {row['State']}, elapsed {row['Elapsed']}, exit {row['ExitCode']}."]
    report += ['', '## Interpretation and limits','']
    report += [f'- **{name}:** {description}' for name,description in data['metric_semantics'].items()]
    report += ['', '## Missing or unperformed measurements','']+['- '+gap for gap in data['coverage_gaps']]
    report += ['', '## Evidence retention','',
        'Each phase records command arguments, exit status, duration, and raw-log SHA-256. All TensorBoard scalar series, '
        'compact task-level episode records, GPU/node/link sample statistics, and collector errors are consolidated in the JSON. '
        'The CSV retains sampled node/engine timelines. Full token transcripts, model/checkpoint shards, repeated process logs, '
        'and raw collector output remain in checksummed local and cluster archives, outside Git. '
        'No held-out improvement is inferred from these two training rewards.','']
    (out/'REPORT.md').write_text('\n'.join(report))
    # Matplotlib emits trailing spaces inside multiline SVG path attributes.
    # Newline separators preserve those paths while keeping generated diffs clean.
    for path in out.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(out/'REPORT.md')


if __name__ == '__main__':
    main()
