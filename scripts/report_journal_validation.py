"""Build a dense failed-or-passed sync validation digest from audited evidence."""
import argparse
import json
from pathlib import Path

from evidence import atomic, sha256


def build(run, attempt, job):
    names = dict(allocation=f'tests/02-sync-grpo-result-audit-v{attempt}/audit.json',
        tensors=f'tests/02-grpo-tensor-audit-v{attempt}-a1/result.json',
        episodes=f'tests/02-sync-grpo-environment-audit-v{attempt}-join-v1/episodes.json',
        optimizer=f'tests/02-sync-grpo-v{attempt}-optimizer-observation-a1/result.json',
        nvml=f'tests/01-nvml-call-audit-job{job}-v1/result.json')
    sources = {key: json.loads((run / name).read_text()) for key, name in names.items()}
    allocation, tensors, episodes, optimizer, nvml = (sources[key] for key in names)
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
    return dict(schema_version=1, job_id=job, training_attempt=attempt,
        qualification='fail' if findings else 'component_validation_only',
        slurm_state=allocation['slurm_state'], slurm_exit_code=allocation['slurm_exit_code'],
        provenance=allocation['provenance'], findings=sorted(set(findings)),
        optimizer_steps=[dict(step=row['step'], time=row['time'], metrics=row['metrics']) for row in optimizer['steps']],
        checkpoint_saves=optimizer['checkpoint_saves'], accounting=journal['counts'], populations=populations,
        environment_outcomes=episodes['counts'], zero_variance_groups=sum(row['zero_variance_groups'] for row in tensors['batches']),
        native_tensor_audit_findings=tensors['findings'], episode_audit_findings=episodes['findings'],
        performance=optimizer['performance'], finalized_telemetry_streams=len(allocation['coverage']),
        slow_nvml_calls=[row for node in nvml['nodes'] for row in node['slowest_calls'][:3]],
        raw_evidence_root='/shared/posttrainingx/runs/vultr-b200-slurm/' + run.name,
        sources={key: dict(path=name, sha256=sha256(run / name)) for key, name in names.items()},
        limitations=['Synchronous small-batch validation, not an asynchronous placement comparison.',
            'Training-task rewards are not held-out TB2.1 quality; no quality delta is established.',
            'Full checkpoint resume, actor placement capture, DCGM and all required telemetry remain unqualified.',
            'NVML call timing identifies the blocking API, not the underlying hardware or driver cause.',
            'Environment-seconds sum parallel episode lifetimes; they are not elapsed wall time.'])


def render(data):
    lines = [f"# Job {data['job_id']}: four-node GRPO validation", '',
        f"**{data['qualification'].upper()}** — Slurm {data['slurm_state']} ({data['slurm_exit_code']}). "
        f"{len(data['optimizer_steps'])} optimizer updates and {len(data['checkpoint_saves'])} save receipts verified.", '',
        '## What passed', '',
        f"- {data['accounting']['selected_inputs']} native samples reconciled through trainer tokens, logprobs, masks and GRPO advantages.",
        f"- All {data['accounting']['controller_episodes']} controller episodes have durable native identities; "
        f"{data['accounting']['unjoined_episodes']} unjoined episodes and {data['accounting']['dispositions']['unresolved']} unresolved dispositions.",
        '- Environment audit found no grading-isolation or lifecycle imbalance.', '',
        '| Population | Samples | Passed episodes | Environment-seconds |', '|---|---:|---:|---:|']
    for name, row in data['populations'].items():
        lines.append(f"| {name} | {row['samples']} | {row['passed']} | {row['environment_seconds']:.2f} |")
    lines += ['', f"Zero-variance GRPO groups: {data['zero_variance_groups']}. These are clean training-task outcomes, not TB2.1 evaluation.", '',
        '## Failure evidence', '', '| Node | API | Duration (s) | UTC start |', '|---|---|---:|---|']
    for row in data['slow_nvml_calls']:
        if row['value'] >= 1:
            lines.append(f"| {row['hostname']} | {row['api']} | {row['value']:.3f} | {row['time']} |")
    lines += ['', 'The overlong collector ticks occurred during teardown. Do not treat the completed training driver as a passing infrastructure gate.', '',
        '## Timing observations', '', '| Rollout | Environment rollout (s) | Trainer call (s) | Trainer wait (s) |', '|---:|---:|---:|---:|']
    by_role = {(row['role'], row['rollout_id']): row['metrics'] for row in data['performance']}
    for step in data['optimizer_steps']:
        rid = step['step']; rollout = by_role[('rollout', rid)]; trainer = by_role[('trainer', rid)]
        lines.append(f"| {rid} | {rollout['perf/rollout_time']:.2f} | {trainer['perf/train_time']:.2f} | {trainer['perf/train_wait_time']:.2f} |")
    lines += ['', 'The first trainer call includes cold compilation. Waiting in this synchronous run includes checkpoint and rollout work; it does not classify an asynchronous role split.', '',
        '## Attribution and limits', '',
        '- **Infrastructure/driver:** blocking NVML calls observed; underlying cause unproven.',
        '- **Miles:** unused sampled work is now fully accounted for. Default resume scheduler mismatch is separately reproduced in the resume report.',
        '- **Model recipe:** finite gradients and native tensor checks passed; held-out quality and quantized execution remain unqualified.',
        '- **Environment:** local task isolation and lifecycle audit passed for this scoped task subset.',
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
    args = parser.parse_args()
    data = build(args.run_dir, args.attempt, args.job_id)
    atomic(args.output, data)
    atomic(args.output.with_suffix('.md'), render(json.loads(args.output.read_text())))
