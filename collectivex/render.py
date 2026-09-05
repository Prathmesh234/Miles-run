#!/usr/bin/env python3
"""Import reviewed CollectiveX evidence and render a dependency-free HTML dashboard.

The compact JSON is the source of truth; HTML embeds it for offline file:// use.
Only allowlisted numeric measurements/configuration are imported, never Slack
transcripts, environment variables, private addresses, GPU UUIDs or credentials.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,math

HERE=Path(__file__).resolve().parent

def load(path):return json.loads(path.read_text())

def import_campaign(root):
    audit=load(root/'completion-audit.json'); fleet=load(root/'fleet-results/stats.json')
    series=load(root/'fleet-results/timeseries.json'); receipt=load(root/'fleet-publication.json')
    assert audit['status']=='verified' and fleet['status']=='ok'
    points=[]; nodes=[]; infrastructure={}; sources={}
    for node in audit['nodes']:
        name=node['node']; job=node['job_id']; folder=root/('evidence-job-'+job)
        nodes.append(dict(name=name,job=job,start=node['started_at'],end=node['ended_at'],
                          gpu_sample_gap_max_s=node['maximum_gpu_sample_gap_s'],phases=node['phase_counts']))
        for file in sorted((folder/'results').glob('measure-*.json')):
            raw=load(file); assert raw['outcome']['status']=='success'
            case=raw['identity']['case_factors']['case']
            sources[name+'/'+file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
            for row in raw['measurement']['rows']:
                assert row['correctness']['passed']
                components={}
                for component in ['dispatch','combine','roundtrip','pair_period']:
                    value=row['components'][component]
                    payload=row['byte_provenance']['roundtrip' if component=='pair_period' else component]
                    components[component]=dict(latency_us=value['percentiles_us'],samples=value['sample_count'],
                                               logical_bytes=payload['total_logical_bytes'],activation_bytes=payload['activation_data_bytes'])
                points.append(dict(node=name,job=job,phase=case['phase'],precision=case['precision'],tokens_per_gpu=row['tokens_per_rank'],
                                   global_tokens=row['global_tokens'],correct=row['correctness']['passed'],
                                   max_relative_error=row['correctness']['max_relative_error'],components=components))
        infra={}
        for field in ['utilization.gpu','power.draw','temperature.gpu','memory.used','clocks.sm']:
            divisor=1024 if field=='memory.used' else 1
            infra[field]=[[r['elapsed_s'],r.get(field)/divisor if r.get(field) is not None else None] for r in series['gpu'][name]]
        infra['nvlink_tx']=[[r['elapsed_s'],r['nvlink_tx_GB_s']] for r in series['nvlink'][name]]
        previous={}; monotonic_start=None
        for file in sorted((folder/'telemetry').glob('*.jsonl.gz')):
            for line in gzip.open(file,'rt'):
                r=json.loads(line); t=r['monotonic_s']
                if monotonic_start is None:monotonic_start=t
                if r['metric']=='collector_error':raise ValueError('Unexpected collector error')
                if r['source']=='proc' and r['metric'] in ['memory_available','load_1m']:
                    value=r['value']/(1024**3) if r['metric']=='memory_available' else r['value']
                    infra.setdefault(r['metric'],[]).append([t-monotonic_start,value])
                if r['source']=='filesystem' and r['metric']=='free':
                    infra.setdefault('shared_free',[]).append([t-monotonic_start,r['value']/1024**4])
                if r['source']=='infiniband' and r['metric']=='PortXmitData':
                    hca=r['hca']; old=previous.get(hca); previous[hca]=(t,r['value'])
                    if old:
                        elapsed=t-old[0]; delta=r['value']-old[1]
                        value=delta/elapsed*8/1e9 if 0<elapsed<=15 and delta>=0 else None
                        infra.setdefault('ib_'+hca,[]).append([t-monotonic_start,value])
        infrastructure[name]=infra
    data=dict(schema_version=1,scope='Four independent EP8 NVLink islands; not EP32 or inter-node RDMA',
              nodes=nodes,points=points,infrastructure=infrastructure,
              configuration=dict(gpus=32,ep=8,backend='DeepEP V2',mode='normal',shape='DeepSeek-V3',hidden=7168,experts=256,topk=8,
                                 routing='uniform',seed=67,repetitions_per_node=1,combine='BF16',
                                 isolated_samples_per_point=2048,pair_period_samples_per_point=448),
              pins={k:v for k,v in audit['pins'].items() if k not in ['image_record','native_runtime']},
              image=json.loads(audit['pins']['image_record']),runtime=json.loads(audit['pins']['native_runtime']),
              archive=dict(filename=Path(receipt['archive']).name,sha256=receipt['archive_sha256'],bytes=receipt['bytes'],verified_files=receipt['verified_files']),
              native_result_sha256=sources,
              reference_dashboard='https://inferencex.com/collectivex',
              warnings=['One repetition per node; no provider ranking or repeated-run confidence intervals.',
                        'Pair period is chained cross-rank-median timing; drained round trip is a different measurement.',
                        'Logical payload rate is not physical wire bandwidth; rates are evaluated at the selected latency percentile.',
                        'No missing samples are zero-filled. GPU lines break at gaps >2 s; NVLink at >3 s; IB at >15 s.',
                        'NVLink counter-rate spikes, including a node-total 7993.7 GB/s estimate, are retained but not validated as physical bandwidth.',
                        'These are synthetic communication tests, not Qwen inference or RL quality measurements.',
                        'KV-transfer and inter-node IB benchmarks were not run; background IB counters are observational only.'])
    # Public upstream source URL, not the local unpushed patch revision.
    data['metric_reference']='https://github.com/SemiAnalysisAI/InferenceX/tree/'+data['pins']['inferencex_base_sha']+'/experimental/CollectiveX'
    validate(data)
    return data

def validate(data):
    assert len(data['nodes'])==4 and len(data['points'])==112
    seen=set()
    for p in data['points']:
        identity=(p['node'],p['phase'],p['precision'],p['tokens_per_gpu'])
        assert identity not in seen;seen.add(identity)
        assert p['correct'] and p['global_tokens']==p['tokens_per_gpu']*8
        for c in p['components'].values():
            values=[c['latency_us'][q] for q in ['p50','p90','p95','p99']]
            assert all(math.isfinite(v) and v>0 for v in values) and sorted(values)==values
            assert c['logical_bytes']>=c['activation_bytes']>0 and c['samples']>0
    for node in data['nodes']:
        assert node['phases']=={'ok':16,'failed':0,'skipped':0}
        for phase,n in [('decode',10),('prefill',4)]:
            for precision in ['bf16','fp8']:
                assert sum(p['node']==node['name'] and p['phase']==phase and p['precision']==precision for p in data['points'])==n

TEMPLATE=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Measured CollectiveX DeepEP V2 EP8 results on four Vultr B200 nodes. Offline, self-contained dashboard.">
<title>CollectiveX · Vultr B200 | Miles experiments</title>
<style>
:root{color-scheme:light;--bg:#f6f8fb;--surface:#fff;--text:#172439;--muted:#637087;--line:#dce3ee;--accent:#225cb4;--good:#14765f;--warn:#8c5d12;--n0:#2563a7;--n1:#846036;--n2:#3d8069;--n3:#8069a7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent)}button,select,input{font:inherit}button,select{border:1px solid var(--line);background:var(--surface);color:var(--text);border-radius:5px;padding:7px 10px}button{cursor:pointer}button:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,circle:focus-visible{outline:2px solid var(--accent);outline-offset:3px}header{background:var(--surface);border-bottom:1px solid var(--line);padding:15px max(24px,calc((100vw - 1420px)/2));display:flex;gap:28px;align-items:center}header strong{font-size:18px;white-space:nowrap}header nav{display:flex;gap:20px;flex:1}header nav a{text-decoration:none;color:var(--muted)}main{max-width:1468px;margin:auto;padding:32px 24px 64px}.eyebrow{font-size:11px;font-weight:700;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase}h1{font-size:30px;letter-spacing:-.7px;margin:6px 0 9px}h2{font-size:19px;margin:0 0 10px}h3{font-size:15px;margin:0 0 10px}p{margin:8px 0}.muted,.caption{color:var(--muted)}.caption{font-size:12px}.hero{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:25px}.stamp{font-size:12px;text-align:right}.good{color:var(--good)}.stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--surface);margin-bottom:28px}.stat{padding:17px 20px;border-right:1px solid var(--line)}.stat:last-child{border:0}.stat strong{font-size:25px;display:block;font-weight:650}.stat span{font-size:12px;color:var(--muted)}.layout{display:grid;grid-template-columns:220px 1fr;gap:22px}.panel{background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:21px}.controls{align-self:start;position:sticky;top:15px}.controls label{display:block;font-size:12px;font-weight:600;margin-top:14px}.controls select{width:100%;margin-top:5px}.checks label{font-weight:400;margin-top:7px}.checks input{accent-color:var(--accent)}.bar{display:flex;align-items:center;justify-content:space-between;gap:15px;flex-wrap:wrap}.buttons{display:flex;gap:7px}.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;margin:5px 0 12px}.swatch{display:inline-block;width:21px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}.chart{width:100%;display:block;height:auto;overflow:visible}.chart text{font-size:11px;fill:var(--muted)}.chart .grid{stroke:var(--line);stroke-dasharray:2 4}.chart .axis{stroke:var(--line)}.chart circle{cursor:crosshair}.tip{border-left:3px solid var(--accent);padding:9px 14px;background:var(--bg);min-height:65px;margin:8px 0;font-size:12px}.table-wrap{overflow:auto;max-height:400px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 11px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}th{position:sticky;top:0;background:var(--surface);font-weight:600;color:var(--muted)}th:first-child,td:first-child{text-align:left}section{scroll-margin-top:20px}.section{margin-top:26px}.note{border-left:3px solid var(--warn);padding:8px 13px;color:var(--muted);font-size:12px;margin-top:15px}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.kv{display:grid;grid-template-columns:180px 1fr;gap:7px 14px;font-size:12px}.kv span:nth-child(odd){color:var(--muted)}code{font:11px/1.5 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:18px;margin-top:28px}.infra-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.infra-controls label{font-size:12px}details>summary{cursor:pointer;font-weight:600;margin:10px 0}.empty{padding:60px;text-align:center;color:var(--muted)}
.layout>div,.two>div,.kv span{min-width:0}.kv span{overflow-wrap:anywhere}.infra-controls select{max-width:100%}
@media(max-width:900px){.layout{grid-template-columns:1fr}.controls{position:static;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 16px}.controls h3,.controls .checks,.controls p{grid-column:1/-1}.two{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.stat:nth-child(2){border-right:0}.hero{display:block}.stamp{text-align:left;margin-top:10px}header nav,header>.caption{display:none}header{padding:15px 12px}main{padding:20px 12px}h1{font-size:25px}.kv{grid-template-columns:130px minmax(0,1fr)}.infra-controls,.infra-controls label{width:100%}.infra-controls select{display:block;width:100%}}
@media print{button,.controls{display:none}.layout{display:block}.panel{break-inside:avoid}body{background:white}.table-wrap{max-height:none}}
</style></head><body>
<header><strong>Miles experiments <span class="muted">/ CollectiveX</span></strong><nav><a href="#benchmarks">EP collectives</a><a href="#infrastructure">Infrastructure</a><a href="#evidence">Evidence & methods</a></nav><span class="caption">Offline · no external dependencies</span></header>
<main><div class="hero"><div><div class="eyebrow">Vultr · NVIDIA B200 · DeepEP V2</div><h1>CollectiveX: four independent EP8 islands</h1><p class="muted">Dispatch, combine and round-trip performance across 32 GPUs. Measured results—not a model-serving simulation.</p></div><div class="stamp"><strong class="good">All four jobs completed</strong><br>Jobs 221–224 · 5 September 2026 UTC<br>00:45–00:59 · one repetition per node</div></div>
<div class="stats"><div class="stat"><strong>112 / 112</strong><span>Measured points passed correctness</span></div><div class="stat"><strong id="reduction"></strong><span>Large-message FP8 pair-latency reduction</span></div><div class="stat"><strong>0</strong><span>Collector-error records</span></div><div class="stat"><strong>32 / 32</strong><span>GPUs released · no compute applications</span></div></div>
<section id="benchmarks" class="layout"><aside class="panel controls"><h3>Benchmark selection</h3>
<label>Phase<select id="phase"><option value="prefill">Prefill · 1,024–8,192</option><option value="decode">Decode · 1–512</option></select></label>
<label>Dispatch precision<select id="precision"><option value="both">BF16 + FP8</option><option value="bf16">BF16</option><option value="fp8">FP8</option></select></label>
<label>Operation<select id="operation"><option value="pair_period">Chained pair period</option><option value="roundtrip">Round trip (measured)</option><option value="dispatch">Dispatch</option><option value="combine">Combine</option></select></label>
<label>Latency percentile<select id="percentile"><option>p50</option><option>p90</option><option>p95</option><option>p99</option></select></label>
<label>Y-axis metric<select id="metric"><option value="latency">Latency · µs</option><option value="payload">Logical payload · GB/s/GPU</option><option value="activation">Activation data · GB/s/GPU</option><option value="tokens">Token rate · tokens/s (global)</option></select></label>
<label>Y scale<select id="scale"><option value="linear">Linear · zero baseline</option><option value="log">Logarithmic</option></select></label>
<div id="nodes" class="checks"></div><p class="caption">Normal mode · EP8 · NVLink<br>Combine is BF16 for both dispatch precisions. No low-latency or KV-transfer case was run.</p></aside>
<div><div class="panel"><div class="bar"><h2 id="chart-title">Collective latency</h2><div class="buttons"><button id="csv">Export CSV</button><button id="svg">Export SVG</button></div></div><div id="legend" class="legend"></div><div id="benchmark-chart"></div><div class="tip" id="tip">Hover or focus a measured point to inspect all four latency percentiles, logical bytes and correctness tolerance.</div><p id="metric-note" class="caption"></p><p class="caption">X: source tokens per GPU, logarithmic. Each node is a separate 8-GPU allocation. Source: pinned CollectiveX native JSON, jobs 221–224, 2026-09-05 00:45–00:59 UTC.</p></div>
<div class="panel section"><h2>Selected measurements</h2><div class="table-wrap"><table><thead><tr><th>Node / job</th><th>Dispatch</th><th>Tokens/GPU</th><th>p50 µs</th><th>p90 µs</th><th>p95 µs</th><th>p99 µs</th><th id="rate-head">GB/s/GPU at p50</th><th>Correct</th></tr></thead><tbody id="measurements"></tbody></table></div><p class="caption">Percentiles are measured kernel timings. Rate at p99 latency is not the p99 of a bandwidth distribution. Correctness passed within the harness tolerance, not necessarily zero numerical error.</p></div></div></section>
<section id="infrastructure" class="panel section"><div class="bar"><div><div class="eyebrow">Recorded during setup and benchmark execution</div><h2>Infrastructure time series</h2></div><div class="infra-controls"><label>Metric <select id="infra-metric"><option value="utilization.gpu">Mean GPU busy · %</option><option value="power.draw">Mean GPU power · W</option><option value="temperature.gpu">Mean GPU temperature · °C</option><option value="memory.used">Mean GPU HBM used · GiB</option><option value="clocks.sm">Mean GPU SM clock · MHz</option><option value="nvlink_tx">Sum of NVLink TX endpoints · GB/s/node</option><option value="memory_available">Host memory available · GiB</option><option value="load_1m">Host load · 1 minute</option><option value="shared_free">Shared filesystem free · TiB</option></select></label></div></div><div id="infra-legend" class="legend"></div><div id="infra-chart"></div><p class="caption" id="infra-caption"></p><div class="note">NVLink rates are counter differences, not kernel benchmarks. An isolated 7,993.7 GB/s node-total estimate is retained as an unvalidated counter-rate spike—not sustained physical bandwidth. IB port traffic is observational background traffic in these single-node EP8 tests.</div></section>
<section id="evidence" class="section two"><div class="panel"><h2>What was tested</h2><div class="kv" id="configuration"></div><div class="note">This is not a Qwen training run. FP8 is 22.75–22.84% lower pair latency at 8,192 tokens/GPU, but 6.64–16.18% slower at one token/GPU. Node variation is descriptive; one repetition cannot establish repeat-run stability or provider superiority.</div></div><div class="panel"><h2>Coverage and provenance</h2><div class="table-wrap"><table><thead><tr><th>Node</th><th>Job</th><th>Phases</th><th>Longest GPU interval</th></tr></thead><tbody id="coverage"></tbody></table></div><p class="caption">All 32 distinct GPU UUIDs were reconciled in the private raw evidence. No credentials, internal addresses or Slack transcripts are embedded here.</p><button id="json">Download source metrics JSON</button><details><summary>Evidence archive and source pins</summary><div id="provenance" class="kv"></div></details></div></section>
<section class="panel section"><h2>Reading these metrics</h2><p>The controls follow the <a href="https://inferencex.com/collectivex" target="_blank" rel="noreferrer">CollectiveX dashboard</a>: source tokens per rank, dispatch/combine/round trip, latency percentiles, and payload rate. This offline view adds chained pair-period timing and node telemetry. The Slack-linked dashboard was a presentation reference; all plotted values come from this Vultr campaign, not Slack anecdotes or other providers.</p><p>Payload rate = routed logical bytes ÷ selected latency ÷ 8 GPUs. Activation rate excludes scale bytes. Token rate is global source tokens ÷ drained-roundtrip latency and is available only for that operation. Chained pair period and drained round trip are distinct measurements; neither is the sum of separately reported percentiles.</p><ul id="limitations"></ul><p><strong>CMAX CLI:</strong> CollectiveX was not registered in the inspected ClusterMAX revision <code id="cmax-sha"></code>. Its DeepEP comm-lib-usability smoke checks are not this standalone InferenceX timing suite.</p></section>
<div class="footer">Synthetic DeepSeek-V3 communication shape · DeepEP V2 normal mode · FP8 dispatch / BF16 combine · No cross-node RDMA qualification. Raw evidence remains in a separately checksummed bundle; the repository contains this compact, reproducible view.</div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const colors=['#2563a7','#846036','#3d8069','#8069a7'],fmt=(x,n=2)=>Number(x).toLocaleString('en-US',{maximumFractionDigits:n,minimumFractionDigits:n}),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={pair_period:'Chained pair period',roundtrip:'Measured round trip',dispatch:'Dispatch',combine:'Combine'};
const units={latency:'Latency (µs)',payload:'Logical payload rate (GB/s/GPU)',activation:'Activation-data rate (GB/s/GPU)',tokens:'Global source-token rate (tokens/s)'};
const active=new Set(D.nodes.map(n=>n.name));
function selected(){return D.points.filter(p=>p.phase===$('phase').value&&($('precision').value==='both'||p.precision===$('precision').value)&&active.has(p.node));}
function value(p){const c=p.components[$('operation').value],lat=c.latency_us[$('percentile').value],m=$('metric').value;return m==='latency'?lat:m==='payload'?c.logical_bytes/lat/1000/8:m==='activation'?c.activation_bytes/lat/1000/8:p.global_tokens/lat*1e6;}
function el(tag,attrs,text){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs||{}))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function chart(target,series,{xLabel,yLabel,logX=false,logY=false,maxGap=Infinity,hover=null}={}){
 target.replaceChildren();const valid=series.flatMap(s=>s.points).filter(p=>p.y!==null&&Number.isFinite(p.y)&&(!logY||p.y>0));
 if(!valid.length){const p=document.createElement('p');p.className='empty';p.textContent='Select at least one node to show its recorded measurements.';target.append(p);return;}
 const W=1000,H=365,L=90,R=24,T=18,B=62,w=W-L-R,h=H-T-B;
 const tx=x=>logX?Math.log2(x):x,ty=y=>logY?Math.log10(y):y;
 let xmin=Math.min(...valid.map(p=>tx(p.x))),xmax=Math.max(...valid.map(p=>tx(p.x))),ymin=logY?Math.min(...valid.map(p=>ty(p.y))):0,ymax=Math.max(...valid.map(p=>ty(p.y)));
 if(xmin===xmax)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;if(logY)ymin-=.02*(ymax-ymin);ymax+=.08*(ymax-ymin);
 const X=x=>L+(tx(x)-xmin)/(xmax-xmin)*w,Y=y=>T+h-(ty(y)-ymin)/(ymax-ymin)*h;
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img',style:'font-family:Arial,sans-serif;font-size:11px','aria-label':`${yLabel} versus ${xLabel}`});
 svg.append(el('rect',{x:0,y:0,width:W,height:H,fill:'#fff'}));
 for(let i=0;i<=5;i++){const t=ymin+(ymax-ymin)*i/5,y=T+h-h*i/5,v=logY?10**t:t;svg.append(el('line',{x1:L,x2:W-R,y1:y,y2:y,class:'grid',stroke:'#dce3ee','stroke-dasharray':'2 4'}));svg.append(el('text',{x:L-12,y:y+4,'text-anchor':'end',fill:'#637087'},v>=10000?v.toExponential(1):fmt(v,v<10?2:0)));}
 const ticks=logX?[...new Set(valid.map(p=>p.x))].sort((a,b)=>a-b):Array.from({length:7},(_,i)=>xmin+(xmax-xmin)*i/6);
 for(const x of ticks){svg.append(el('line',{x1:X(x),x2:X(x),y1:T,y2:H-B,class:'grid',stroke:'#dce3ee','stroke-dasharray':'2 4'}));svg.append(el('text',{x:X(x),y:H-B+23,'text-anchor':'middle',fill:'#637087'},fmt(x,logX?0:1)));}
 svg.append(el('text',{x:L+w/2,y:H-10,'text-anchor':'middle',fill:'#637087'},xLabel));svg.append(el('text',{transform:`translate(17 ${T+h/2}) rotate(-90)`,'text-anchor':'middle',fill:'#637087'},yLabel));
 for(const s of series){let path='',last=null;for(const p of s.points){if(p.y===null||!Number.isFinite(p.y)||(logY&&p.y<=0)){last=null;continue;}path+=(last!==null&&p.x-last<=maxGap?'L':'M')+X(p.x).toFixed(2)+','+Y(p.y).toFixed(2);last=p.x;}svg.append(el('path',{d:path,fill:'none',stroke:s.color,'stroke-width':hover?2:1.3,'stroke-dasharray':s.dash||'none'}));
  if(hover)for(const p of s.points){if(p.y===null)continue;const dot=el('circle',{cx:X(p.x),cy:Y(p.y),r:4,fill:s.color,stroke:'#fff','stroke-width':1,tabindex:0,'aria-label':`${s.name}, ${p.x} tokens, ${p.y}`});dot.append(el('title',{},`${s.name}: ${fmt(p.y)} at ${p.x} tokens/GPU`));dot.addEventListener('mouseenter',()=>hover(p.data));dot.addEventListener('focus',()=>hover(p.data));svg.append(dot);}}
 target.append(svg);
}
function legend(target,series){target.innerHTML=series.map(s=>`<span><i class="swatch" style="border-color:${s.color};border-top-style:${s.dash?'dashed':'solid'}"></i>${esc(s.name)}</span>`).join('');}
function showPoint(p){const c=p.components[$('operation').value];$('tip').innerHTML=`<strong>${esc(p.node)} · job ${p.job} · ${p.precision.toUpperCase()} · ${p.tokens_per_gpu.toLocaleString()} tokens/GPU</strong><br>p50 / p90 / p95 / p99: ${['p50','p90','p95','p99'].map(q=>fmt(c.latency_us[q],3)).join(' / ')} µs · ${c.samples} samples<br>Logical payload: ${c.logical_bytes.toLocaleString()} bytes across EP8 · correct within tolerance: ${p.correct} · max relative error: ${fmt(p.max_relative_error,6)}`;}
function render(){
 const tokenOption=$('metric').querySelector('[value="tokens"]');tokenOption.disabled=$('operation').value!=='roundtrip';if(tokenOption.disabled&&$('metric').value==='tokens')$('metric').value='latency';
 const rows=selected().sort((a,b)=>a.node.localeCompare(b.node)||a.precision.localeCompare(b.precision)||a.tokens_per_gpu-b.tokens_per_gpu),series=[];
 D.nodes.forEach((n,i)=>{for(const precision of ['bf16','fp8']){const points=rows.filter(p=>p.node===n.name&&p.precision===precision);if(points.length)series.push({name:n.name+' · '+precision.toUpperCase(),color:colors[i],dash:precision==='fp8'?'7 4':null,points:points.map(p=>({x:p.tokens_per_gpu,y:value(p),data:p}))});}});
 $('chart-title').textContent=names[$('operation').value]+' · '+$('percentile').value;legend($('legend'),series);
 chart($('benchmark-chart'),series,{xLabel:'Source tokens per GPU (log₂)',yLabel:units[$('metric').value],logX:true,logY:$('scale').value==='log',hover:showPoint});
 $('metric-note').textContent=$('metric').value==='latency'?'Measured latency; chained and drained timings are distinct. FP8 affects dispatch; combine remains BF16.':'Rate is computed at the selected latency percentile, not a percentile of instantaneous bandwidth. Logical/activation rates are per GPU; token rate is global and uses drained round trip.';
 $('rate-head').textContent='GB/s/GPU at '+$('percentile').value;
 $('measurements').innerHTML=rows.map(p=>{const c=p.components[$('operation').value];return `<tr><td>${p.node} / ${p.job}</td><td>${p.precision.toUpperCase()}</td><td>${p.tokens_per_gpu.toLocaleString()}</td>${['p50','p90','p95','p99'].map(q=>'<td>'+fmt(c.latency_us[q],3)+'</td>').join('')}<td>${fmt(c.logical_bytes/c.latency_us[$('percentile').value]/1000/8,3)}</td><td class="good">Pass</td></tr>`;}).join('');
 $('tip').textContent='Hover or focus a measured point to inspect all percentiles and byte accounting.';renderInfra();
}
function renderInfra(){const key=$('infra-metric').value,label=$('infra-metric').selectedOptions[0].textContent,series=D.nodes.filter(n=>active.has(n.name)).map(n=>({name:n.name,color:colors[D.nodes.indexOf(n)],points:(D.infrastructure[n.name][key]||[]).map(r=>({x:r[0]/60,y:r[1]}))}));legend($('infra-legend'),series);chart($('infra-chart'),series,{xLabel:'Minutes since each node began telemetry (includes setup)',yLabel:label,maxGap:(key.startsWith('ib_')?15:key==='nvlink_tx'?3:2)/60});$('infra-caption').textContent='Source: recorded NVIDIA / IB PMA / proc / filesystem samples, 2026-09-05 00:45–00:59 UTC. GPU metrics are means of available observations; missing bins break lines. NVLink is summed over 144 TX endpoints per node. IB is per port, not node aggregate.';}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csv(){const header=['node','job','phase','precision','tokens_per_gpu','operation','p50_us','p90_us','p95_us','p99_us','logical_bytes','rate_latency_percentile','logical_GB_s_per_gpu','correct'];return [header.join(','),...selected().map(p=>{const c=p.components[$('operation').value];return [p.node,p.job,p.phase,p.precision,p.tokens_per_gpu,$('operation').value,...['p50','p90','p95','p99'].map(q=>c.latency_us[q]),c.logical_bytes,$('percentile').value,c.logical_bytes/c.latency_us[$('percentile').value]/1000/8,p.correct].join(',');})].join('\n')+'\n';}
function kv(target,rows){target.innerHTML=rows.map(([k,v])=>`<span>${esc(k)}</span><span>${esc(v)}</span>`).join('');}
for(const [i,n] of D.nodes.entries()){const l=document.createElement('label');l.innerHTML=`<input type="checkbox" checked> <span style="color:${colors[i]}">${n.name}</span> · job ${n.job}`;l.firstElementChild.addEventListener('change',e=>{e.target.checked?active.add(n.name):active.delete(n.name);render();});$('nodes').append(l);}
for(const hca of ['mlx5_0','mlx5_1','mlx5_2','mlx5_3','mlx5_4','mlx5_9','mlx5_12','mlx5_13']){const o=document.createElement('option');o.value='ib_'+hca;o.textContent='IB TX '+hca+':1 · Gb/s/port';$('infra-metric').append(o);}
const reduction=D.nodes.map(n=>{const rows=D.points.filter(p=>p.node===n.name&&p.tokens_per_gpu===8192);return 100*(1-rows.find(p=>p.precision==='fp8').components.pair_period.latency_us.p50/rows.find(p=>p.precision==='bf16').components.pair_period.latency_us.p50);});$('reduction').textContent=fmt(reduction.reduce((a,b)=>a+b,0)/4,1)+'%';
kv($('configuration'),[['Allocation','4 × 8 B200 GPUs; one EP8 group per node'],['Backend / mode','DeepEP V2 / normal'],['Synthetic shape','DeepSeek-V3: hidden 7168; 256 experts; top-k 8'],['Routing / seed','Uniform / 67'],['Timing budget','2,048 isolated / 448 chained samples per point'],['Precision','BF16 or FP8 dispatch; always BF16 combine'],['Runtime',D.runtime.framework+'; CUDA '+D.runtime.accelerator_runtime+'; NCCL '+D.runtime.collective_library.version]]);
kv($('provenance'),[['Archive',D.archive.filename],['Archive SHA-256',D.archive.sha256],['Verified bundle files',D.archive.verified_files],['InferenceX base',D.pins.inferencex_base_sha],['B200 profile patch',D.pins.inferencex_patched_sha],['DeepEP',D.pins.deepep_sha],['Image',D.image.image],['Image amd64 digest',D.image.amd64_digest]]);
 $('coverage').innerHTML=D.nodes.map(n=>`<tr><td>${n.name}</td><td>${n.job}</td><td>${n.phases.ok}/16</td><td>${fmt(n.gpu_sample_gap_max_s,3)} s</td></tr>`).join('');$('limitations').innerHTML=D.warnings.map(w=>'<li>'+esc(w)+'</li>').join('');$('cmax-sha').textContent=D.pins.clustermax_sha;
for(const id of ['phase','precision','operation','percentile','metric','scale'])$(id).addEventListener('change',render);$('infra-metric').addEventListener('change',renderInfra);
$('csv').addEventListener('click',()=>download('collectivex-selection.csv',csv(),'text/csv'));
$('json').addEventListener('click',()=>download('collectivex-metrics.json',JSON.stringify(D,null,2),'application/json'));
$('svg').addEventListener('click',()=>{const svg=$('benchmark-chart').querySelector('svg');if(svg)download('collectivex-chart.svg',new XMLSerializer().serializeToString(svg),'image/svg+xml');});
window.collectiveX={data:D,selected,value,csv,render};render();
</script></body></html>'''

def render(data):
    validate(data)
    payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\/')
    return TEMPLATE.replace('__DATA__',payload)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-campaign',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.import_campaign:
        data=import_campaign(args.import_campaign)
        (HERE/'metrics.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    else:data=load(HERE/'metrics.json')
    html=render(data); target=HERE/'index.html'
    if args.check:
        assert target.read_text()==html,'index.html differs from metrics.json / render.py; rerun renderer'
    else:target.write_text(html)
    print(json.dumps(dict(status='ok',measured_points=len(data['points']),dashboard=str(target),checked=args.check)))

if __name__=='__main__':main()
