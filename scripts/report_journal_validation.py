"""Build a dense failed-or-passed sync validation digest from audited evidence."""
import argparse
import collections
import json
from pathlib import Path

from evidence import atomic, sha256
from summarize_native import summary


def build(run, attempt, job):
    names = dict(allocation=f'tests/02-sync-grpo-result-audit-v{attempt}/audit.json',
        tensors=f'tests/02-grpo-tensor-audit-v{attempt}-a1/result.json',
        episodes=f'tests/02-sync-grpo-environment-audit-v{attempt}-join-v1/episodes.json',
        optimizer=f'tests/02-sync-grpo-v{attempt}-optimizer-observation-a1/result.json',
        nvml=f'tests/01-nvml-call-audit-job{job}-v1/result.json')
    sources = {key: json.loads((run / name).read_text()) for key, name in names.items()}
    allocation, tensors, episodes, optimizer, nvml = (sources[key] for key in names)
    placement_path = f'tests/02-ray-placement-observer-sync-grpo-v{attempt}/result.json'
    placement = None
    if (run / placement_path).exists():
        names['placement'] = placement_path
        source = json.loads((run / placement_path).read_text())
        counts = collections.Counter((r['class_name'], r['assigned_hostname']) for r in source['observed_alive_actors'])
        placement = dict(ticks=source['ticks'], findings=source['findings'], actors=[
            dict(class_name=k[0], hostname=k[1], count=v) for k, v in sorted(counts.items())])
    if any(str(sources[key]['slurm_job_id']) != str(job) for key in ('allocation', 'optimizer', 'nvml')):
        raise ValueError('Audits refer to different Slurm jobs.')
    journal = episodes['journal_accounting']
    by_id = {row['episode_id']: row for row in episodes['episodes']}
    populations = {}
    for disposition in ('selected_for_training', 'sync_unused_discarded'):
        samples = [sample for sample in journal['samples'] if sample['disposition'] == disposition]
        owned = [by_id[eid] for sample in samples for eid in sample['environment_ids']]
        populations[disposition] = dict(samples=len(samples), episodes=len(owned),
            passed=sum(row['reward'] == 1 for row in owned),
            environment_seconds=sum(row['event_span_s'] for row in owned))
    if journal['counts']['selected_inputs'] != tensors['trained_input_samples']:
        raise ValueError('Native journal and trainer counts disagree.')
    findings = allocation['findings'] + tensors['findings'] + episodes['findings'] + nvml['findings']
    if placement:
        findings += placement['findings']
    cleanup = allocation.get('host_cleanup')
    cleanup_summary = None
    if cleanup:
        completed = cleanup['completed']
        cleanup_summary = dict(findings=cleanup['findings'], completed_ranks=len(completed),
            released_tensor_bytes=sum(row['tensor_bytes'] for row in completed),
            duration_s=summary([row['duration_s'] for row in completed]),
            policy_versions=sorted({row['policy_version'] for row in completed}))
    return dict(schema_version=1, job_id=job, training_attempt=attempt,
        qualification='fail' if findings else 'component_validation_only',
        slurm_state=allocation['slurm_state'], slurm_exit_code=allocation['slurm_exit_code'],
        provenance=allocation['provenance'], findings=sorted(set(findings)),
        optimizer_steps=[dict(step=row['step'], time=row['time'], metrics=row['metrics']) for row in optimizer['steps']],
        checkpoint_saves=optimizer['checkpoint_saves'], accounting=journal['counts'], populations=populations,
        environment_outcomes=episodes['counts'], zero_variance_groups=sum(row['zero_variance_groups'] for row in tensors['batches']),
        native_tensor_audit_findings=tensors['findings'], episode_audit_findings=episodes['findings'],
        performance=optimizer['performance'], finalized_telemetry_streams=len(allocation['coverage']),
        host_cleanup=cleanup_summary, actor_placement=placement,
        native_telemetry=dict(records=sum(r['records'] for r in allocation['coverage']),
            collector_errors=sum(r['collector_errors'] for r in allocation['coverage']),
            maximum_gpu_sample_gap_s={r['hostname']: r['max_interval_s'] for r in allocation['coverage']
                                      if r['path'].endswith('/nvidia-smi.jsonl')}),
        slow_nvml_calls=[row for node in nvml['nodes'] for row in node['slowest_calls'][:3]],
        raw_evidence_root='/shared/posttrainingx/runs/vultr-b200-slurm/' + run.name,
        sources={key: dict(path=name, sha256=sha256(run / name)) for key, name in names.items()},
        limitations=['Synchronous small-batch validation, not an asynchronous placement comparison.',
            'Training-task rewards are not held-out TB2.1 quality; no quality delta is established.',
            'Full checkpoint resume, DCGM and all required telemetry remain unqualified; periodic actor placement is reported when captured.',
            'NVML call timing identifies the blocking API, not the underlying hardware or driver cause.',
            'Environment-seconds sum parallel episode lifetimes; they are not elapsed wall time.'])


def render(data):
    lines = [f"# Job {data['job_id']}: four-node GRPO validation", '',
        f"**{data['qualification'].upper()}** — Slurm {data['slurm_state']} ({data['slurm_exit_code']}). "
        f"{len(data['optimizer_steps'])} optimizer updates and {len(data['checkpoint_saves'])} save receipts verified.", '',
        '## Accounting', '',
        f"- {data['accounting']['selected_inputs']} selected native samples; tensor audit findings: {len(data['native_tensor_audit_findings'])}.",
        f"- {data['accounting']['controller_episodes']} controller episodes; "
        f"{data['accounting']['unjoined_episodes']} unjoined episodes and {data['accounting']['dispositions']['unresolved']} unresolved dispositions.",
        f"- Environment audit findings: {len(data['episode_audit_findings'])}.", '',
        '| Population | Samples | Passed episodes | Environment-seconds |', '|---|---:|---:|---:|']
    for name, row in data['populations'].items():
        lines.append(f"| {name} | {row['samples']} | {row['passed']} | {row['environment_seconds']:.2f} |")
    lines += ['', f"Zero-variance GRPO groups: {data['zero_variance_groups']}. These are clean training-task outcomes, not TB2.1 evaluation.", '',
        '## Audit findings', '']
    lines += ['- ' + finding for finding in data['findings']] or ['No findings in these component audits. Full benchmark gates remain open.']
    if data.get('host_cleanup'):
        cleanup = data['host_cleanup']
        lines += ['', f"Pinned-backup cleanup: {cleanup['completed_ranks']} completed rank receipts, "
            f"{cleanup['released_tensor_bytes']:,} tensor bytes released; {len(cleanup['findings'])} findings.", '']
    if data.get('native_telemetry'):
        telemetry = data['native_telemetry']
        lines += ['', f"Native telemetry: {telemetry['records']:,} records; {telemetry['collector_errors']} collector errors.", '',
            '| Node | Maximum GPU sampling gap (s) |', '|---|---:|']
        lines += [f'| {host} | {gap:.3f} |' for host, gap in telemetry['maximum_gpu_sample_gap_s'].items()]
    if data.get('actor_placement'):
        placement = data['actor_placement']
        lines += ['', f"Actor placement: {placement['ticks']} snapshots; {len(placement['findings'])} findings.", '',
                  '| Node | Actor class | Count |', '|---|---|---:|']
        lines += [f"| {r['hostname']} | {r['class_name']} | {r['count']} |" for r in placement['actors']
                  if r['class_name'] in ('MegatronTrainRayActor', 'SGLangEngine')]
    if data.get('infrastructure_plot'):
        lines += ['', '## Infrastructure time series', '', f"![Native infrastructure telemetry]({data['infrastructure_plot']})", '']
    if data.get('runtime_warnings'):
        lines += ['', '## Runtime warnings', '']
        lines += ['- ' + warning for warning in data['runtime_warnings']]
    lines += ['', '### Slow NVML calls', '', '| Node | API | Duration (s) | UTC start |', '|---|---|---:|---|']
    for row in data['slow_nvml_calls']:
        if row['value'] >= 1:
            lines.append(f"| {row['hostname']} | {row['api']} | {row['value']:.3f} | {row['time']} |")
    lines += ['', 'Slow API calls are observations, not a diagnosis. Allocation exit, sampling continuity and required metric coverage must be checked separately.', '',
        '## Timing observations', '', '| Rollout | Environment rollout (s) | Trainer call (s) | Trainer wait (s) |', '|---:|---:|---:|---:|']
    by_role = {(row['role'], row['rollout_id']): row['metrics'] for row in data['performance']}
    for step in data['optimizer_steps']:
        rid = step['step']; rollout = by_role[('rollout', rid)]; trainer = by_role[('trainer', rid)]
        lines.append(f"| {rid} | {rollout['perf/rollout_time']:.2f} | {trainer['perf/train_time']:.2f} | {trainer['perf/train_wait_time']:.2f} |")
    lines += ['', 'The first trainer call includes cold compilation. Waiting in this synchronous run includes checkpoint and rollout work; it does not classify an asynchronous role split.', '',
        '## Attribution and limits', '',
        '- **Infrastructure/driver:** use the recorded continuity findings and API timings; no hardware cause is inferred.',
        '- **Miles:** use the exact accounting above. Default resume scheduler mismatch is separately reproduced in the resume report.',
        '- **Model recipe:** native tensor findings are recorded above; held-out quality and quantized execution remain unqualified.',
        '- **Environment:** isolation/lifecycle findings apply only to this scoped task subset.',
        '- **Configuration:** tiny synchronous batches, cold compilation and checkpoint-every-step prevent a steady-state performance claim.', '']
    lines += ['- ' + text for text in data['limitations']]
    lines += ['', '## Reproducibility and raw evidence', '', f"Shared root: `{data['raw_evidence_root']}`.", '',
        f"Miles `{data['provenance']['miles_sha']}`; site code `{data['provenance']['root_sha']}`.", '']
    lines += [f"- {key}: `{row['path']}`; SHA256 `{row['sha256']}`." for key, row in data['sources'].items()]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--infrastructure-plot', type=Path)
    args = parser.parse_args()
    data = build(args.run_dir, args.attempt, args.job_id)
    if args.infrastructure_plot:
        if args.infrastructure_plot.parent.resolve() != args.output.parent.resolve():
            raise ValueError('The published plot must be alongside the report.')
        data['infrastructure_plot'] = args.infrastructure_plot.name
        data['infrastructure_plot_sha256'] = sha256(args.infrastructure_plot)
    log = args.run_dir / f'training/sync-grpo-v{args.attempt}/logs/gpu-nodes-0/miles.out'
    if log.exists():
        count = log.read_text().count('post-warmup freeze_gc failed')
        data['runtime_warnings'] = ([f'SGLang logged {count} post-warmup freeze_gc failures. Retained as runtime warnings; no configuration was changed to suppress them.'] if count else [])
    atomic(args.output, data)
    atomic(args.output.with_suffix('.md'), render(json.loads(args.output.read_text())))
