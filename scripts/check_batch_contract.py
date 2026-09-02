"""Exercise the exact pinned Megatron calculator, without GPU allocation or rounding."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from evidence import Run, atomic, metric


def render(data):
    lines = ['# Pinned Megatron batch contract', '',
             '| Trainer nodes | Dense DP | Global batch | Running batch | Microbatches | Status |',
             '|---:|---:|---:|---:|---:|---|']
    for case in data['cases']:
        lines.append('| ' + ' | '.join(str(case.get(k, 'unavailable')) for k in
            ('trainer_nodes', 'dense_dp', 'global_batch_size', 'running_global_batch_size', 'num_microbatches', 'status')) + ' |')
    lines += ['', data['recommendation'], '', 'Status: ' + data['status'] + '.', '',
              'CPU calculator evidence does not replace actual GRPO gradient and full-trainer validation.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--source-evidence', type=Path, required=True)
    ap.add_argument('--case', type=int, choices=(1, 2, 3))
    ap.add_argument('--batch', type=int, choices=(64, 96))
    args = ap.parse_args()
    data = json.loads(args.source_evidence.read_text())
    if hashlib.sha256(data['source'].encode()).hexdigest() != data['sha256']:
        raise ValueError('Pinned calculator source checksum mismatch.')
    if args.case:
        namespace = {'__name__': 'posttrainingx_pinned_batch_calculator'}
        exec(compile(data['source'], data['path'], 'exec'), namespace)
        calc = namespace['ConstantNumMicroBatchesCalculator'](
            global_batch_size=args.batch, micro_batch_size=1, data_parallel_size=args.case * 8,
            decrease_batch_size_if_needed=False, rank=0)
        print(json.dumps({'trainer_nodes': args.case, 'dense_dp': args.case * 8,
                          'expert_dp': args.case, 'global_batch_size': args.batch,
                          'running_global_batch_size': calc.get_current_running_global_batch_size(),
                          'num_microbatches': calc.get()}))
        return 0
    run = Run(args.run_dir)
    cases = []
    for batch in (64, 96):
        for nodes in (1, 2, 3):
            phase = run.phase(f'02-megatron-batch-contract-{nodes}t-b{batch}')
            rc, out, _ = phase.command([sys.executable, str(Path(__file__).resolve()), '--run-dir', str(run.root),
                '--source-evidence', str(args.source_evidence.resolve()), '--case', str(nodes), '--batch', str(batch)])
            result = json.loads(out) if not rc else {'trainer_nodes': nodes, 'dense_dp': 8*nodes, 'global_batch_size': batch}
            result.update(calculator_sha256=data['sha256'], rounding_enabled=False,
                          scope='Exact pinned calculator CPU contract only; full trainer/GRPO validation remains required.')
            phase.finish('fail' if rc else 'ok', exit_code=rc,
                failure_summary='Pinned calculator rejects the requested global batch and dense-DP combination.' if rc else None,
                results=[metric('global_batch_size', batch, 'trajectories'), metric('dense_dp', nodes*8, 'ranks')], metadata=result)
            cases.append(dict(result, status='fail' if rc else 'ok', phase=phase.name))
    report = {'cases': cases,
        'recommendation': 'Keep Stage 4 at batch64. Prospectively use batch96/rollout_batch12/group8 for every Stage5 layout; do not silently round 64 down.',
        'status': 'proposal; no measured role sweep or optimizer execution'}
    atomic(run.root / 'reports/batch-contract.json', report)
    atomic(run.root / 'reports/batch-contract.md', render(report))
    print(json.dumps({'cases': len(cases), 'failures': sum(c['status']=='fail' for c in cases)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
