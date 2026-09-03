"""Render the compact collector-control report from its JSON source."""
import json
from pathlib import Path

from evidence import atomic


def render(data):
    meta = data['metadata']
    lines = ['# GPU telemetry qualification', '',
        '**Training telemetry remains unqualified; individual control results are below.**', '',
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
        'A control result does not repair the failed training gate or, by itself, prove a hardware cause.', '',
        '## Contract and next test', '', meta['runtime_contract'], '', failed['next_step'], '',
    ]
    if meta.get('next_control'):
        control = meta['next_control']
        lines += [f"**{control['status']} — {control['profile']}**", '', control['hypothesis'], '',
                  control['bounded_workload'], '', control['interpretation'], '']
    if meta.get('pinned_release_control'):
        contrast = meta['pinned_release_control']
        lines += ['## Pinned-host release control', '', contrast['interpretation'], '',
            '| Node | Ordinary-exit gap (s) | Explicit-release gap (s) | Host release min / max (s) | Verified ranks |',
            '|---|---:|---:|---:|---:|']
        for row in contrast['nodes']:
            stats = row['host_release_s']
            lines.append(f"| {row['hostname']} | {row['ordinary_exit_max_gap_s']:.3f} | "
                f"{row['explicit_release_max_gap_s']:.3f} | {stats['min']:.3f} / {stats['max']:.3f} | {stats['n']} |")
        lines += ['', contrast['next_step'], '']
    lines += ['## Evidence', '', f"- Initial control: `{meta['evidence']}`."]
    lines += [f"- Job {row['job_id']}: `{row['evidence']}`; SHA256 `{row['sha256']}`." for row in meta['subsequent_controls']]
    lines += ['', 'Full per-node results, source pins and prior failures remain in `telemetry-qualification.json` and the linked raw bundles.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    source = Path('docs/telemetry-qualification.json')
    atomic(source.with_suffix('.md'), render(json.loads(source.read_text())))
