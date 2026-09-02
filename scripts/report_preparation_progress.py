"""Freeze preparation milestones without implying training or benchmark success."""
import argparse
import json
from pathlib import Path

from evidence import Run, atomic, markdown, metric, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    phase = run.phase('02-model-and-task-preparation-v2')
    sources = {
        'parity': run.root / 'tests/02-checkpoint-parity-result-audit-v2/audit.json',
        'tasks': run.root / 'tests/00-terminal-lego-materialized-source-audit-v1/audit.json',
        'linux_tests': run.root / 'tests/00-miles-linux-launch-regression-comparison-v2/comparison.json',
        'patch': repo / 'patches/manifest.json',
        'split': repo / 'locks/terminal-lego-subset.json',
        'collectives': repo / 'locks/collectivex.json',
    }
    data = {key: json.loads(path.read_text()) for key, path in sources.items()}
    parity, tasks, linux = (data[key] for key in ('parity', 'tasks', 'linux_tests'))
    counts = parity['comparison_counts']
    errors = []
    if parity['findings'] or counts['qualified'] != counts['compared'] or counts['lossless_widened'] != 30:
        errors.append('Qualified parity did not pass its frozen contract.')
    if tasks['findings'] or tasks['controller_directory_mode'] != '0o700':
        errors.append('Task source audit failed.')
    if linux['new_failing_ids'] or not data['patch']['replay_tree_matches']:
        errors.append('Launcher regression or patch replay failed.')
    metadata = {
        'scope': 'Preparation milestones only. The full benchmark is incomplete; no optimizer step or held-out result.',
        'findings': errors, 'training_started': False, 'heldout_quality_measured': False,
        'evidence': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in sources.items()},
        'parity': {
            'slurm_job_id': parity['slurm_job_id'], 'slurm_accounting': parity['slurm_accounting'],
            'counts': counts, 'contract_version': parity['parity']['comparison_contract']['version'],
            'comparison_sha256': parity['comparison_sha256'],
            'qualification': '30 named A_log tensors are exactly reversible BF16-to-FP32 widenings. All other comparisons require dtype/byte equality. Job118 remains failed; checkpoint bytes unchanged.',
            'telemetry_streams': len(parity['coverage']),
            'collector_errors': sum(stream['collector_errors'] for stream in parity['coverage']),
            'not_validated': ['EP8 trainer reshard', 'forward logits', 'gradients', 'optimizer', 'resume', 'quality'],
        },
        'clean_tasks': {
            'revision': data['split']['revision'], 'training_ids': len(data['split']['training_task_ids']),
            'development_ids': len(data['split']['development_task_ids']),
            'runtime_validation_ids': data['split']['runtime_validation_task_ids'],
            'task_count': tasks['task_count'], 'file_count': tasks['file_count'],
            'payload_bytes': tasks['payload_bytes'], 'manifest_sha256': tasks['manifest_sha256'],
            'runtime_eligibility': 'unvalidated; no images, references, tests or policy execution',
        },
        'launcher': {'revision': data['patch']['patched_revision'],
                     'patch_sha256': data['patch']['patch_sha256'],
                     'replay_tree': data['patch']['source_tree_sha1'],
                     'linux_suite': linux['current_counts'],
                     'new_failing_ids': linux['new_failing_ids'],
                     'note': 'Identical baseline failures are retained; full suite is not green.'},
        'quality_plan': {'optimizer_steps': 400, 'tb21_evaluation_steps': [0, 50, 100, 200, 400],
                         'stage4_global_batch': 64, 'proposed_common_sweep_global_batch': 96,
                         'prospective_eligible_trajectories_at_batch96': 38400,
                         'note': 'Requires remaining gates, budget and prospective freeze. No improvement guaranteed.'},
        'remaining_runtime_gates': [
            'Local policy/grader separation and dependency-compatible OpenEnv client are not validated.',
            'Docker VFS hard size quota probe failed; no daemon change made and no policy container started.',
            'CollectiveX normal DeepEP V2 needs ElasticBuffer, absent from Miles image. Qualify a separate pinned collective runtime; do not force-upgrade training NCCL.',
            'Full telemetry during RL, async accounting, broadcast activation and checkpoint/resume remain untested.',
        ],
    }
    result = phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None,
        results=[metric('checkpoint_components_qualified', counts['qualified'], 'count', 'gpu-nodes-0'),
                 metric('checkpoint_components_byte_and_dtype_equal', counts['equal'], 'count', 'gpu-nodes-0'),
                 metric('checkpoint_components_losslessly_widened', counts['lossless_widened'], 'count', 'gpu-nodes-0'),
                 metric('task_source_files_verified', tasks['file_count'], 'count'),
                 metric('task_source_bytes_verified', tasks['payload_bytes'], 'B')],
        metadata=metadata, refresh=False)
    atomic(run.root / 'reports/preparation-progress-v2.json', result)
    atomic(run.root / 'reports/preparation-progress-v2.md', markdown(result))
    run.refresh()
    print(json.dumps({'status': result['status'], 'report': str(run.root / 'reports/preparation-progress-v2.md'),
                      'training_started': False, 'heldout_quality_measured': False}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
