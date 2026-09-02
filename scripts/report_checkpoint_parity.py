"""Render the retained failed gate and its independent numeric diagnosis."""
import argparse
import json

from evidence import Run, atomic, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    parity_path = run.root / 'tests/02-checkpoint-parity-result-audit-v1/audit.json'
    diagnosis_path = run.root / 'tests/02-checkpoint-dtype-result-audit-v1/audit.json'
    audit = json.loads(parity_path.read_text())
    diagnosis = json.loads(diagnosis_path.read_text())
    if diagnosis['findings'] or diagnosis['diagnosis']['lossless_widening_count'] != 30:
        raise ValueError('Do not report lossless widening without the completed diagnostic gate.')
    result = audit['parity']
    data = {
        'schema_version': 1, 'status': 'failed_gate_with_diagnosed_dtype_difference',
        'slurm_job_id': audit['slurm_job_id'], 'slurm_accounting': audit['slurm_accounting'],
        'training_started': False, 'heldout_quality_measured': False,
        'counts': audit['comparison_counts'], 'reference_weight_count': result['reference_weight_count'],
        'excluded_vision_weight_count': len(result['excluded_vision_weights']),
        'checkpoint_weight_tensor_count': result['tensor_metadata_count'],
        'mtp_weight_tensor_count': result['mtp_weight_tensor_count'],
        'non_tensor_metadata_count': result['non_tensor_metadata_count'],
        'saved_recipe': result['saved_recipe'],
        'timings': {k: v for k, v in result.items() if k.endswith('_duration_s')},
        'coverage': audit['coverage'], 'findings': audit['findings'],
        'comparison_sha256': audit['comparison_sha256'],
        'diagnosis': {
            'lossless_widening_tensors': diagnosis['diagnosis']['lossless_widening_count'],
            'lossless_widening_scalars': diagnosis['diagnosis']['scalar_count'],
            'negative_control_passed': diagnosis['diagnosis']['sub_bf16_perturbation_negative_control'],
            'sha256': diagnosis['diagnosis_sha256'], 'upstream_source_sha256': diagnosis['source_sha256'],
            'upstream_source_excerpt': diagnosis['intentional_widening_source'],
        },
        'attribution': 'Experimental validation criterion: exact dtype equality rejects intentional, lossless A_log widening. No observed numeric discrepancy in those 960 scalars; this is not evidence of a Vultr fault.',
        'smallest_proposed_fix': 'Version the validator to admit only the pinned A_log BF16-to-FP32 conversion after exact value, lifted-byte and inverse-byte tests; retain strict checks for all other tensors. Do not cast the checkpoint back to BF16 or edit prior failures.',
        'remaining_gates': ['Execute and audit the narrowly revised full parity gate.',
                            'EP8 trainer load/reshard and forward/logit parity.',
                            'Gradient, optimizer and full checkpoint/resume fidelity.',
                            'Local environment isolation and clean taskset freeze.',
                            'Full async telemetry, placement sweep and held-out quality improvement.'],
        'notes': audit['notes'],
        'artifacts': {
            str(parity_path.relative_to(run.root)): sha256(parity_path),
            str(diagnosis_path.relative_to(run.root)): sha256(diagnosis_path),
            'tests/02-checkpoint-parity-child-v1/tensor-comparisons.jsonl': audit['comparison_sha256'],
            'tests/02-checkpoint-dtype-diagnostic-v1/diagnosis.json': diagnosis['diagnosis_sha256'],
        },
    }
    target = run.root / 'reports/checkpoint-parity-v1.json'
    atomic(target, data)
    # Read the serialized source; Markdown is not a separately maintained claim.
    d = json.loads(target.read_text())
    lines = ['# Checkpoint parity and numeric diagnosis', '', f"Status: **{d['status']}**.", '',
             'No training or held-out quality evaluation has run.', '',
             '| Measurement | Value |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in d['counts'].items()]
    lines += [f"| Lossless BF16-to-FP32 A_log widenings | {d['diagnosis']['lossless_widening_tensors']} |",
              f"| Exactly preserved A_log scalars | {d['diagnosis']['lossless_widening_scalars']} |",
              f"| MTP weight tensors in checkpoint metadata | {d['mtp_weight_tensor_count']} |", '',
              '## Failure attribution', '', d['attribution'], '',
              '## Smallest proposed fix', '', d['smallest_proposed_fix'], '',
              '## Remaining gates', '']
    lines += ['- ' + text for text in d['remaining_gates']]
    lines += ['', '## Measurement caveats', ''] + ['- ' + text for text in d['notes']]
    lines += ['', '## Raw evidence', ''] + [f'- [{path}](../{path}), SHA256 `{digest}`'
                                             for path, digest in d['artifacts'].items()]
    lines += ['', 'Telemetry paths, hashes, gaps and counts are included in the JSON source.', '']
    atomic(target.with_suffix('.md'), '\n'.join(lines))
    run.refresh()
    print(json.dumps({'report': str(target), 'status': d['status']}))


if __name__ == '__main__':
    main()
