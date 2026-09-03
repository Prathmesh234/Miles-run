"""Render the compact collector-control report from its JSON source."""
import json
from pathlib import Path

from evidence import atomic


def render(data):
    meta = data['metadata']
    lines = ['# GPU telemetry qualification', '',
        '**Collector controls passed; training telemetry remains unqualified.**', '',
        meta['scope'], '',
        '| Job | Workload | Maximum GPU sample gap (s) | Result |',
        '|---|---|---:|---|',
        f"| {meta['job_id']} | Node-local all-reduce | {max(n['max_gap_s'] for n in meta['nodes']):.3f} | passed |"]
    for control in meta['subsequent_controls']:
        lines.append(f"| {control['job_id']} | {control['scope']} | "
            f"{max(n['max_gpu_sample_gap_s'] for n in control['nodes']):.3f} | "
            f"{'failed' if control['findings'] else 'passed'} |")
    failed = meta['failed_training_validation']
    lines += ['', f"Job {failed['job_id']} had trainer-node gaps of " +
        ' / '.join(f'{gap:.3f} s' for gap in failed['gaps_s']) + '.', '',
        'None of these short controls reproduces that training-runtime stall. Passing them does not repair the failed training gate or prove a hardware cause.', '',
        '## Contract and next test', '', meta['runtime_contract'], '', failed['next_step'], '',
        '## Evidence', '', f"- Initial control: `{meta['evidence']}`."]
    lines += [f"- Job {row['job_id']}: `{row['evidence']}`; SHA256 `{row['sha256']}`." for row in meta['subsequent_controls']]
    lines += ['', 'Full per-node results, source pins and prior failures remain in `telemetry-qualification.json` and the linked raw bundles.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    source = Path('docs/telemetry-qualification.json')
    atomic(source.with_suffix('.md'), render(json.loads(source.read_text())))
