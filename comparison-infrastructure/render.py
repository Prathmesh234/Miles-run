"""Render exportable evidence figures. Matplotlib is the only extra dependency."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np

COLORS=['#4263a0','#138b87','#c57925']
NODE_COLORS=['#345995','#168b84','#ae6a1f','#995c83']
INK='#203044'; MUTED='#64748b'; GRID='#e6ebef'
LABELS={190:'GRPO-style / IPO',196:'Synchronous PPO',197:'Async PPO + TIS'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
 'axes.labelsize':10,'axes.titleweight':'semibold','text.color':INK,'axes.labelcolor':INK,
 'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':GRID,'axes.spines.top':False,
 'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white',
 'savefig.facecolor':'white','svg.fonttype':'none'})


def clean(ax,which='y'):
    ax.grid(axis=which,color=GRID,lw=.7);ax.set_axisbelow(True)
    ax.tick_params(length=0,pad=6)


def finish(fig,out,name,title,subtitle,foot):
    fig.suptitle(title,x=.075,y=.98,ha='left',fontsize=20,fontweight='bold',color=INK)
    fig.text(.075,.935,subtitle,ha='left',va='top',fontsize=10.5,color=MUTED)
    fig.text(.075,.035,foot,ha='left',va='bottom',fontsize=8.5,color=MUTED,linespacing=1.5)
    fig.subplots_adjust(left=.13 if name in ['02-execution-timeline','10-broadcast-offload'] else .085,right=.97,top=.84,bottom=.18,hspace=.55,wspace=.35)
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']: fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)


def phase_values(run,name):
    return sorted([p for p in run['phases'] if p['name']==name],key=lambda p:p['start'])


def comparison(runs,out):
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    specs=[('Rollout generation','Seconds',lambda r:[b['rollout_seconds'] for b in r['batches']]),
           ('Actor training','Seconds',lambda r:[p['seconds'] for p in r['timers'] if p['name']=='actor_train']),
           ('Critic training','Seconds',lambda r:[p['seconds'] for p in r['timers'] if p['name']=='critic_train']),
           ('Generated training work','Active action tokens (thousands)',lambda r:[b['active_tokens']/1000 for b in r['batches']])]
    for ax,(title,ylabel,values) in zip(axes.flat,specs):
        for i,r in enumerate(runs):
            v=values(r);x=np.arange(len(v))+(i-(len(runs)-1)/2)*.24
            bars=ax.bar(x,v,width=.22,color=COLORS[i],label=LABELS[r['job_id']])
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set_xticks([0,1],['Update 0','Update 1']);ax.set_title(title,loc='left',pad=12)
        ax.set_ylabel(ylabel);clean(ax);ax.margins(y=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(.075,.90))
    finish(fig,out,'01-work-and-time','The same task does not mean the same amount of work',
           'Two updates per run · 32 B200 GPUs · generated token counts and learning algorithms differ',
           'Rollout: bridge episode_seconds. Actor/critic: native rank-0 train timers (host wall time). No critic in job 190.\nFirst-update compilation and initialization inflate training time; this is not a steady-state throughput benchmark.')


def timelines(runs,out):
    fig,axes=plt.subplots(len(runs),1,figsize=(14,3+2.5*len(runs)),squeeze=False)
    lanes={'rollout':4,'critic_train':3,'actor_train':2,'weight_transfer':1,'save_model':0}
    palette={'rollout':'#138b87','critic_train':'#8c70a0','actor_train':'#4263a0','weight_transfer':'#c57925','save_model':'#8c9bad'}
    last=max(p['end']-r['active_window'][0] for r in runs for p in r['phases'])
    for ax,r in zip(axes.flat,runs):
        zero=r['active_window'][0]
        for p in r['phases']:
            if p['name'] not in lanes: continue
            ax.broken_barh([(p['start']-zero,p['seconds'])],(lanes[p['name']]-.32,.64),facecolors=palette[p['name']],edgecolors='white',linewidth=.6)
            if p['name']=='rollout':ax.text(p['start']-zero+p['seconds']/2,lanes[p['name']],f"Batch {p['step']}",ha='center',va='center',color='white',fontsize=8)
        ax.axvline(0,color=MUTED,lw=.8,ls=':')
        ax.set_yticks(list(lanes.values()),['Rollout','Critic','Actor','Publish weights','Save weights'])
        ax.set_ylim(-.6,4.7);ax.set_xlim(-30,last+15);ax.set_title(f"{LABELS[r['job_id']]}  ·  job {r['job_id']}   |   measured rollout/train overlap {r['rollout_compute_overlap_seconds']:.1f}s",loc='left',fontsize=11)
        clean(ax,'x')
    axes[-1,0].set_xlabel('Seconds from first rollout start (each run has its own origin)')
    finish(fig,out,'02-execution-timeline','Asynchrony, measured rather than inferred',
           'Rollout and training lanes show recorded wall-clock spans; simultaneous work is not added twice',
           'Jobs 190/196: native rank-0 training timers and first-to-last episode span. Job 197: instrumented driver spans.\nNative one-batch-ahead async waits for prefetched rollout completion before weight publication; it is not fully-async continuous serving.')


def transfers(runs,out):
    fig,axes=plt.subplots(1,2,figsize=(13,6.4))
    ax=axes[0]
    for i,r in enumerate(runs):
        vals=[p['seconds'] for p in r['weight_transfers']]
        bars=ax.bar(np.arange(len(vals))+(i-1)*.24,vals,width=.22,color=COLORS[i],label=LABELS[r['job_id']])
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks([0,1,2],['Initial','After update 0','After update 1']);ax.set_ylabel('Publication span (seconds)');clean(ax)
    ax.legend(frameon=False,fontsize=9);ax.margins(y=.23);ax.set_title('Wall time to publish model weights',loc='left')
    ax=axes[1]
    async_run=next((r for r in runs if r['job_id']==197),None)
    if async_run:
        for i,p in enumerate(async_run['weight_transfers']):
            vals=[x['individual_port_TX_utilization_pct']['max'] for x in p['ib_TX_highrate_bins'].values() if x['individual_port_TX_utilization_pct']]
            if vals:
                ax.bar(i,max(vals),color=COLORS[2],width=.5)
                ax.text(i,max(vals),f'{max(vals):.1f}%',ha='center',va='bottom',fontsize=10)
            else: ax.text(i,.2,'No 1s\ncoverage',ha='center',va='bottom',color=MUTED,fontsize=9)
        ax.set_xticks([0,1,2],['Initial','After update 0','After update 1'])
    else: ax.text(.5,.5,'Async runtime evidence pending',transform=ax.transAxes,ha='center')
    ax.set_ylabel('Hottest 400-Gb/s port TX (% of line rate)');clean(ax);ax.margins(y=.3)
    ax.set_xlim(-.6,2.6)
    ax.set_title('Async transfer windows · finer IB samples',loc='left')
    finish(fig,out,'03-weight-publication','Weight publication: latency and observed link pressure',
           'Publication wall time and sampled fabric utilization are different measurements',
           'Whole-node counters include other traffic. Bars use bins overlapping the publication window, not isolated model bytes.\n1s sampler began 01:21:22 UTC, after initial async publication; coarse baseline sampling is approximately 10s. No wire-throughput claim.')


def fabric_comparison(runs,out,kind,name):
    fig,axes=plt.subplots(1,2,figsize=(13,6.6))
    for i,r in enumerate(runs):
        nodes=[];hot=[];x=[]
        for n in range(4):
            host=f'gpu-nodes-{n}';s=r['fabric_summary'][f'{kind}:{host}:TX']['utilization_pct']
            links=[v['utilization_pct']['max'] for v in r['fabric_link_statistics'] if v['fabric']==kind and v['host']==host]
            if s and links: x.append(n+(i-1)*.24);nodes.append(s['max']);hot.append(max(links))
        if x:
            axes[0].bar(x,nodes,width=.22,color=COLORS[i],label=LABELS[r['job_id']])
            axes[1].bar(x,hot,width=.22,color=COLORS[i],label=LABELS[r['job_id']])
    for ax in axes:
        clean(ax);ax.set_xticks(range(4),['Node 0\ntrain','Node 1\ntrain','Node 2\nrollout','Node 3\nrollout'])
        ax.set_xlim(-.5,3.5)
        ax.set_ylabel('Maximum sampled TX utilization (%)');ax.margins(y=.2)
    axes[0].set_title('Aggregate node links',loc='left');axes[1].set_title('Hottest individual link / port',loc='left')
    axes[0].legend(frameon=False,fontsize=9)
    if kind=='IB':
        title='Scale-out fabric: how much InfiniBand was used?'
        sub='Eight active 400-Gb/s GPU-fabric ports per node · TX only · common ≈10s sampling'
        foot='Node denominator: 8 × 400 Gb/s = 3.2 Tb/s one-way. Hot-port denominator: 400 Gb/s. Storage ports excluded.\nFirst rollout through last model save/train end. Sampled utilization is not a 1→2→4-node scaling curve; coarse bins hide bursts.'
    else:
        title='Scale-up fabric: observed NVLink payload utilization'
        sub='144 NVLink links per node · TX only · common ≈5s sampling · job 190 has no continuous payload data'
        foot='Denominator: nominal 50 GB/s per link/direction (1.8 TB/s bidirectional per B200), not the reported 53.125 GB/s raw line rate.\nFirst rollout through last save/train end. No isolated all-reduce or 1→8-GPU scaling test was performed; missing data is not zero.'
    finish(fig,out,name,title,sub,foot)


def fabric_detail(runs,out):
    r=next((r for r in runs if r['job_id']==197),None)
    if not r or not r['fabric_highrate_intervals']:return
    transfers=[p for p in r['weight_transfers'] if any(v['overlapping_bins'] for v in p['ib_TX_highrate_bins'].values())]
    if not transfers:return
    fig,axes=plt.subplots(len(transfers),2,figsize=(13,3.5+3*len(transfers)),squeeze=False)
    for axs,p in zip(axes,transfers):
        for n,color in enumerate(NODE_COLORS):
            rows=[v for v in r['fabric_highrate_intervals'] if v['fabric']=='IB' and v['direction']=='TX' and v['host']==f'gpu-nodes-{n}' and v['complete'] and p['start']-5<=v['end']<=p['end']+5]
            for ax,key in zip(axs,['utilization_pct','hottest_link_utilization_pct']):
                ax.plot([v['end']-p['start'] for v in rows],[v[key] for v in rows],marker='.',ms=3,lw=1.5,color=color,label=f'Node {n}')
        for ax in axs:
            ax.axvspan(0,p['seconds'],color=COLORS[2],alpha=.12)
            ax.set_xlabel('Seconds from publication start');ax.set_ylabel('TX utilization (%)');clean(ax)
        axs[0].set_title(f"Publication {p['publication_index']} · node aggregate",loc='left')
        axs[1].set_title(f"Publication {p['publication_index']} · hottest port on each node",loc='left')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(.075,.90))
    finish(fig,out,'06-transfer-bursts','Resolving the short weight-transfer bursts',
           'One-second requested cadence · shaded area is driver publication span · counters are node-wide',
           'Each point is a delta over its actual counter interval and is plotted at interval end. Boundary points mix neighboring phases.\n8 × 400 Gb/s node denominator; 400 Gb/s per-port denominator. Passive sampler command time/coverage are retained in results.json.')


def offpolicy(runs,out):
    r=next((r for r in runs if r['job_id']==197),None)
    if not r or not r['batches']:return
    fig,axes=plt.subplots(1,3,figsize=(14,6.5))
    x=[b['step'] for b in r['batches']]
    behavior=[];trainer=[]
    for b in r['batches']:
        p=next((p for p in r['phases'] if p['name']=='rollout' and p.get('step')==b['step']),{})
        v=p.get('behavior_version');lag=p.get('policy_lag')
        behavior.append(v if v is not None else np.nan);trainer.append(v+lag if v is not None and lag is not None else np.nan)
    axes[0].plot(x,behavior,'o-',color=COLORS[1],label='Behavior version')
    axes[0].plot(x,trainer,'s--',color=COLORS[2],label='Expected trainer version')
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True));axes[0].set_ylabel('Recorded version');axes[0].legend(frameon=False,fontsize=9)
    axes[0].set_title('Behavior policy lags behind training',loc='left',fontsize=11)
    for key,label,color in [('train/tis','Raw IS ratio',COLORS[0]),('train/tis_weight','Clamped weight',COLORS[1])]:
        axes[1].plot(x,[b['scalar_actor'].get(key,np.nan) for b in r['batches']],'o-',color=color,label=label)
    axes[1].axhline(1,color=MUTED,ls=':',lw=1);axes[1].set_ylabel('Valid-token mean');axes[1].legend(frameon=False,fontsize=9)
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
    axes[1].set_title('Importance sampling is measured',loc='left',fontsize=11)
    vals=[100*b['scalar_actor'].get('train/tis_upper_clipfrac',np.nan) for b in r['batches']]
    axes[2].bar(x,vals,color=COLORS[2],width=.48);axes[2].set_ylabel('Valid action tokens clipped (%)')
    axes[2].set_title('TIS upper threshold = 2',loc='left',fontsize=11)
    for ax in axes:ax.set_xticks(x,[f'Update {v}' for v in x]);clean(ax);ax.margins(y=.2)
    finish(fig,out,'07-off-policy-tis','One-batch-ahead PPO with explicit behavior correction',
           'TIS weight = clamp(exp(trainer-before-update logprob − behavior logprob), 0, 2)',
           'Native reducer over valid action tokens; PPO clipping is separate. Means can hide tails, so upper-clipped fraction is shown too.\nRecorded rollout lag is an instrumentation field; generation response versions and TIS mismatch provide independent runtime evidence.')


def optimizer(runs,out):
    r=next((r for r in runs if r['job_id']==197),None)
    if not r or not r['optimizer_steps']:return
    rows=r['optimizer_steps'];groups=defaultdict(list)
    for x in rows:groups[(x['resolved_role'],x['rollout_id'])].append(x)
    fig,axes=plt.subplots(1,3,figsize=(14,6.6))
    labels=[]
    for i,((role,step),items) in enumerate(sorted(groups.items())):
        labels.append(f'{role}\nu{step}')
        cpu=[sum(v for k,v in p['after']['unique_storage_bytes'].items() if k.startswith('state:cpu'))/1024**3 for p in items]
        gpu=[sum(v for k,v in p['after']['unique_storage_bytes'].items() if k.startswith('state:cuda'))/1024**3 for p in items]
        axes[0].bar(i-.16,statistics.mean(cpu),width=.3,color=COLORS[1],label='CPU' if i==0 else None)
        axes[0].bar(i+.16,statistics.mean(gpu),width=.3,color=COLORS[2],label='CUDA' if i==0 else None)
        durations=[p['elapsed_seconds'] for p in items]
        axes[1].bar(i,statistics.median(durations),color=COLORS[0],width=.5)
        axes[1].scatter(i,max(durations),color=COLORS[2],marker='_',s=150,label='Slowest rank' if i==0 else None)
        overhead=[p['inventory_overhead_seconds']*1000 for p in items]
        axes[2].bar(i,statistics.median(overhead),color=COLORS[1],width=.5)
    for ax in axes:ax.set_xticks(range(len(labels)),labels);clean(ax);ax.margins(y=.2)
    axes[0].set_title('Referenced optimizer states',loc='left');axes[0].set_ylabel('GiB / rank · mean');axes[0].legend(frameon=False)
    axes[1].set_title('Optimizer step host wall time',loc='left');axes[1].set_ylabel('Seconds · median rank');axes[1].legend(frameon=False,fontsize=9)
    axes[2].set_title('Inventory instrumentation cost',loc='left');axes[2].set_ylabel('Milliseconds · median rank')
    finish(fig,out,'08-optimizer-offload','Optimizer offloading: placement, time, and measurement cost',
           'CPU optimizer-state offload enabled · parameter/gradient offload is a separate mechanism',
           'Unique referenced tensor storages per process, not physical resident pages or transferred bytes. Rank replicas are not deduplicated across processes.\nHost wall time adds no CUDA barrier and is not a direct PCIe copy measurement. Unknown role is retained where rank identity cannot be proven.')


def gpu_context(runs,out):
    fig,axes=plt.subplots(len(runs),2,figsize=(13,3+2.6*len(runs)),squeeze=False)
    for axs,r in zip(axes,runs):
        a,b=r['active_window']
        for node,color in enumerate(NODE_COLORS):
            rows=[v for v in r['gpu_samples'] if v['host']==f'gpu-nodes-{node}' and a<=v['time']<=b]
            for ax,key in zip(axs,['util_pct','memory_GiB']):
                ax.plot([v['time']-a for v in rows],[v[key] for v in rows],color=color,lw=1.1,label=f'Node {node}')
        axs[0].set_ylabel('GPU busy (%)');axs[0].set_ylim(0,105)
        axs[1].set_ylabel('Used HBM (GiB / GPU)');axs[1].set_ylim(bottom=0)
        for ax in axs:
            clean(ax);ax.set_title(LABELS[r['job_id']],loc='left',fontsize=11)
            ax.set_xlim(0,max(v['active_window'][1]-v['active_window'][0] for v in runs))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(.075,.90))
    for ax in axes[-1]:ax.set_xlabel('Seconds from first rollout start')
    finish(fig,out,'09-gpu-context','Device activity and memory through each run',
           'Each line is the mean of eight GPUs on one node · roughly two-second nvidia-smi samples',
           'Nodes 0–1 host training; nodes 2–3 host rollout. These are GPU busy-time percentages, not MFU or achieved FLOP/s.\nReported HBM use includes allocator/cache effects. Memory drops do not directly measure optimizer-transfer volume.')


def broadcast_lifecycle(runs,out):
    r=next((r for r in runs if r['job_id']==197),None)
    if not r:return
    fig,ax=plt.subplots(figsize=(12,6.4));bottom=np.zeros(3)
    for name,label,color in [('broadcast_actor_onload','Actor onload',COLORS[0]),
                            ('weight_transfer','Publish weights',COLORS[2]),
                            ('broadcast_actor_offload','Actor offload',COLORS[1])]:
        values=[p['seconds'] for p in phase_values(r,name)]
        if len(values)!=3:continue
        bars=ax.barh(np.arange(3),values,left=bottom,color=color,height=.55,label=label)
        ax.bar_label(bars,labels=[f'{v:.2f}s' for v in values],label_type='center',color='white',fontsize=10)
        bottom+=values
    ax.set_yticks(range(3),['Initial','After update 0','After update 1']);ax.invert_yaxis()
    ax.set_xlabel('Sequential driver spans (seconds)');clean(ax,'x')
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(.075,.90))
    ax.margins(x=.1)
    finish(fig,out,'10-broadcast-offload','Weight movement includes more than publication',
           'Async PPO · restore actor residency, publish to rollout engines, then offload actor',
           'Host wall time of driver await calls; not an isolated PCIe or CUDA-copy measurement. Components are sequential.\nOptimizer CPU-state placement is measured separately. Missing per-rank lifecycle instrumentation is disclosed in the report.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('results',type=Path);ap.add_argument('--out',type=Path,default=Path('comparison-infrastructure/charts'))
    args=ap.parse_args();data=json.loads(args.results.read_text());runs=data['runs']
    comparison(runs,args.out);timelines(runs,args.out);transfers(runs,args.out)
    fabric_comparison(runs,args.out,'IB','04-scale-out-ib');fabric_comparison(runs,args.out,'NVLink','05-scale-up-nvlink')
    fabric_detail(runs,args.out);offpolicy(runs,args.out);optimizer(runs,args.out);gpu_context(runs,args.out);broadcast_lifecycle(runs,args.out)
    (args.out/'manifest.json').write_text(json.dumps({'source':str(args.results),'generated_utc':data['generated_utc'],'figures':sorted(p.name for p in args.out.glob('*.png'))},indent=2)+'\n')
    print(json.dumps({'figures':len(list(args.out.glob('*.png'))),'output':str(args.out)}))

if __name__=='__main__':main()
