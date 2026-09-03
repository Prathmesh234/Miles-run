"""Render resume evidence as a compact component table, not a duplicated JSON dump."""
import json
from pathlib import Path

from evidence import atomic


def render(data):
    meta = data['metadata']
    lines = ['# Checkpoint and resume qualification', '',
        '**Full-state resume is not qualified.** Saving a checkpoint is not proof that restart preserves the next update.', '',
        '| Component | Current evidence |', '|---|---|']
    lines += ['| ' + row['component'] + ' | ' + row['state'].replace('|', '\\|') + ' |' for row in meta['components']]
    saved = meta.get('saved_checkpoint_inspection')
    if saved:
        lines += ['', '## Inspected checkpoint', '',
            f"`{saved['checkpoint']}` — iteration {saved['iteration']}, saved scheduler position {saved['scheduler_num_steps']}.", '',
            'Only DCP metadata and the small common/RNG byte ranges were read. Tensor payload checksums, actual model/optimizer loading and next-step equivalence remain untested.', '',
            'The Megatron consumed-sample counters are zero; Miles dataset cursor state is separate. Do not interpret those fields as a resumed data position.']
    defect = meta.get('scheduler_resume_reproducer')
    if defect:
        lines += ['', '## Reproduced scheduler mismatch', '',
            '| Configuration | Loaded position | After Miles initialization | LR |', '|---|---:|---:|---:|']
        for row in defect['rows']:
            lines.append(f"| use_checkpoint_opt_param_scheduler={row['use_checkpoint_opt_param_scheduler']} | "
                f"{row['loaded_num_steps']} | {row['after_miles_initialization_num_steps']} | {row['lr']} |")
        lines += ['', defect['smallest_fix'], '',
            'This reproduces the native scheduler plus the pinned Miles post-load branch, not a whole-model restart. The controlled GPU replay explicitly enables checkpoint-scheduler restoration.']
    replay = meta.get('checkpoint_replay_validation')
    if replay:
        lines += ['', '## Frozen-input replay', '',
            f"Status: **{replay['status']}**. Native CPU job **{replay['native_cpu_job']}**; "
            f"GPU job **{replay['gpu_job']}**, {replay['gpus']} GPUs, {replay['replicas']} independent trainer replicas.",
            '', replay['contract'], '', '| Limitation |', '|---|']
        lines += ['| ' + item.replace('|', '\\|') + ' |' for item in replay['limitations']]
        lines += ['', '| Preserved failure or source finding | Scoped change |', '|---|---|']
        for item in replay['interventions']:
            lines.append('| ' + item['failure'] + ' | ' + item['fix'] + ' |')
        lines += ['', f"CPU proof: `{replay['native_cpu_evidence']}`. Submission: `{replay['submission_evidence']}`."]
    lines += ['', '## Remaining requirements', '']
    lines += ['- ' + item for item in meta['required_before_resume_claim']]
    lines += ['', '## Evidence', '', f"- Dataset cursor probe: `{meta['cpu_probe']['path']}`."]
    lines += [f"- Cursor/buffer candidate: `{row['path']}`." for row in meta['opt_in_buffer_candidate']['evidence']]
    if saved:
        lines += [f"- Saved-state inspection: `{row['path']}`; SHA256 `{row['sha256']}`." for row in saved['evidence']]
    if defect:
        lines += [f"- Scheduler replay: `{defect['evidence']}`; SHA256 `{defect['sha256']}`."]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    source = Path('docs/resume-compatibility.json')
    atomic(source.with_suffix('.md'), render(json.loads(source.read_text())))
