"""Diagnose the 30 retained A_log dtype mismatches without changing the gate.

CPU-only, isolated and bounded. Selected tensor bytes must match the preceding
full parity run before checking exact BF16 -> FP32 promotion and its inverse.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from evidence import Run, atomic, metric, sha256


def probe():
    import hashlib
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors import safe_open

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    if torch.cuda.device_count():
        raise ValueError('This diagnostic must not expose GPU devices.')
    root = Path('/run-artifacts')
    old = root / 'tests/02-checkpoint-parity-child-v1'
    mismatches = []
    with (old / 'tensor-comparisons.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if not row['equal']:
                mismatches.append(row)
    if len(mismatches) != 30 or any(not r['source_name'].endswith('.self_attention.linear_attn.A_log')
                                  for r in mismatches):
        raise ValueError('The mismatch set differs from the observed 30 A_log dtype differences.')
    state = {}
    for row in mismatches:
        if row['dtype'] != 'torch.float32' or row['reference_dtype'] != 'torch.bfloat16':
            raise ValueError('Unexpected dtype difference.')
        state[row['source_name'].removeprefix('module.module.')] = torch.empty((32,), dtype=torch.float32)
    reader = dcp.FileSystemReader('/checkpoint/release')
    dcp.load(state, storage_reader=reader, planner=dcp.DefaultLoadPlanner(flatten_state_dict=False), no_dist=True)
    index = json.loads(Path('/model/model.safetensors.index.json').read_text())['weight_map']

    def digest(tensor):
        return hashlib.sha256(memoryview(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()

    results = []
    for row in mismatches:
        actual = state[row['source_name'].removeprefix('module.module.')]
        with safe_open(Path('/model') / index[row['reference_name']], framework='pt', device='cpu') as f:
            reference = f.get_tensor(row['reference_name'])
        if digest(actual) != row['actual_sha256'] or digest(reference) != row['reference_sha256']:
            raise ValueError('Selected tensor bytes differ from the preceding full parity evidence.')
        lifted = reference.float()
        restored = actual.to(torch.bfloat16)
        exact = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()
                     and torch.equal(actual, lifted) and torch.equal(restored, reference)
                     and digest(actual) == digest(lifted) and digest(restored) == digest(reference))
        results.append({'name': row['reference_name'], 'lossless_widening': exact,
                        'max_absolute_difference_in_fp32': float((actual - lifted).abs().max()),
                        'actual_fp32_sha256': digest(actual), 'reference_bf16_sha256': digest(reference),
                        'reference_lifted_fp32_sha256': digest(lifted), 'roundtrip_bf16_sha256': digest(restored)})
    # Reject an FP32 change too small to survive conversion back to BF16.
    reference = torch.tensor([1.0], dtype=torch.bfloat16)
    altered = torch.nextafter(reference.float(), torch.tensor([float('inf')]))
    sensitive = bool(torch.equal(altered.to(torch.bfloat16), reference)
                     and not torch.equal(altered, reference.float()))
    if not sensitive or not all(r['lossless_widening'] for r in results):
        raise ValueError('A_log is not an exact, lossless FP32 widening of the pinned BF16 values.')
    return {'status': 'ok', 'lossless_widening_count': len(results), 'scalar_count': 32 * len(results),
            'sub_bf16_perturbation_negative_control': sensitive, 'results': results,
            'scope': 'Targeted numeric diagnosis only. Original strict dtype parity remains failed; no model or gate repair.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir')
    ap.add_argument('--inside', action='store_true')
    args = ap.parse_args()
    if args.inside:
        print('PTX_DTYPE=' + json.dumps(probe(), allow_nan=False), flush=True)
        return 0
    from enroot_run_config import prepare

    run = Run(args.run_dir)
    phase = run.phase('02-checkpoint-dtype-diagnostic-v1')
    runtime = run.root / 'images/checkpoint-dtype-diagnostic-v1'
    runtime.mkdir(exist_ok=False)
    env = prepare(runtime)
    os.environ.update({key: value for key, value in env.items() if key.startswith('ENROOT_')})
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'void'
    command = ['timeout', '--signal=TERM', '--kill-after=10s', '120s',
        'enroot', 'start', '--net', '--pid', '--ipc', '--env', 'NVIDIA_VISIBLE_DEVICES=void',
        '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'OMP_NUM_THREADS=2']
    model = run.root / 'models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0'
    checkpoint = run.root / 'models/qwen3.6-35b-a3b-torch-dist-v2'
    for source, target in [(run.root, '/run-artifacts'), (Path(__file__).parent, '/ptx'),
                            (model, '/model'), (checkpoint, '/checkpoint')]:
        command += ['--mount', str(source) + ':' + target + ':none:bind,ro,x-create=dir']
    command += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
                'python3', '/ptx/checkpoint_dtype_probe.py', '--inside']
    rc, out, _ = phase.command(command, timeout=140)
    records = [line.removeprefix('PTX_DTYPE=') for line in out.splitlines() if line.startswith('PTX_DTYPE=')]
    errors = []
    if rc or len(records) != 1:
        errors.append('CPU dtype probe failed or did not produce one unambiguous result.')
        data = {'findings': errors}
    else:
        data = json.loads(records[0])
    data['code_sha256'] = sha256(__file__)
    atomic(phase.path / 'diagnosis.json', data)
    phase.finish('fail' if errors else 'ok', results=[] if errors else [
        metric('lossless_bf16_to_fp32_widenings', data['lossless_widening_count'], 'count')],
        failure_summary='; '.join(errors) or None, metadata=data, refresh=False)
    print(json.dumps({'status': 'fail' if errors else 'ok', 'findings': errors,
                      'widening_count': data.get('lossless_widening_count')}), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
