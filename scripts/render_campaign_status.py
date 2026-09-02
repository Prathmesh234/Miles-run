"""Render campaign status from its machine-readable source."""
import argparse
import json
from pathlib import Path

from evidence import atomic


def render(data):
    lines = ['# Campaign status', '', 'Generated from `docs/current-status.json`.', '',
             f"Run: `{data['run_id']}`. Training started: **{data['training_started']}**. "
             f"Held-out quality measured: **{data['heldout_quality_measured']}**.", '', '## Completed work', '']
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
