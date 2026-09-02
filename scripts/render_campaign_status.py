"""Render campaign status from its machine-readable source."""
import argparse
import json
from pathlib import Path

from evidence import atomic


def render(data):
    lines = ['# Campaign status', '', 'Generated from `docs/current-status.json`.', '',
             f"Run: `{data['run_id']}`. Optimizer steps verified: **{data.get('optimizer_steps_verified', 'unverified')}**. "
             f"Held-out quality measured: **{data['heldout_quality_measured']}**."]
    if data.get('active_submission'):
        current = data['active_submission']
        lines += ['', f"Latest submission: Slurm **{current['slurm_job_id']}**, {current['gpus']} GPUs, "
                  f"{current['layout']}, {current['steps_requested']} requested steps. {current['scope']}",
                  f"Status snapshot: `{data.get('updated_at', 'unknown')}`; inspect Slurm for live state."]
    lines += ['', '## Historical milestones (later gates supersede earlier limits)', '']
    lines.extend('- ' + item for item in data['completed'])
    lines += ['', '## Remaining gates', '', '| Gate | State | Smallest next step |', '|---|---|---|']
    for gate in data['blocked_or_unvalidated']:
        lines.append('| ' + ' | '.join(gate[k].replace('|', '\\|') for k in ('gate', 'state', 'smallest_next_step')) + ' |')
    lines += ['', '## Quality budget', '', data['quality_note'], '']
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('docs/current-status.json'))
    args = ap.parse_args()
    atomic(args.source.with_suffix('.md'), render(json.loads(args.source.read_text())))
