"""Read-only, full text/MTP weight parity using pinned Miles conversion functions.

The current Megatron format embeds common state in DCP, not common.pt. Use its
own common-state loader and Torch's tensor reader; use the unmodified Miles
name/weight conversion for every model tensor. Only the reference lookup adapts
unfused expert outputs to the public checkpoint's fused expert representation.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import runpy
import shutil
import time
import traceback

from evidence import Run, atomic, metric, sha256
from model_conversion import validate_imports


def reference_part(name, weight_map, num_experts):
    if name in weight_map:
        return name, None, None
    match = re.fullmatch(r'(.+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight', name)
    if match is None:
        raise ValueError('No pinned HF reference for ' + name)
    prefix, index, part = match.groups()
    index = int(index)
    if not 0 <= index < num_experts:
        raise ValueError('Expert index is out of range: ' + name)
    target = prefix + ('.down_proj' if part == 'down_proj' else '.gate_up_proj')
    if target not in weight_map:
        raise ValueError('Fused expert reference is absent: ' + target)
    return target, index, part


def required_parts(name, num_experts):
    if name.endswith('.mlp.experts.gate_up_proj'):
        return {(i, p) for i in range(num_experts) for p in ('gate_proj', 'up_proj')}
    if name.endswith('.mlp.experts.down_proj'):
        return {(i, 'down_proj') for i in range(num_experts)}
    return {(None, None)}


def check_coverage(weight_map, seen, num_experts):
    errors = []
    for name in sorted(weight_map):
        if name.startswith('model.visual.'):
            if name in seen:
                errors.append('Unexpected vision weight in text-only conversion: ' + name)
            continue
        parts = seen.get(name, set())
        # A directly converted fused tensor is also complete coverage.
        if parts not in (required_parts(name, num_experts), {(None, None)}):
            errors.append('Missing or overlapping reference coverage: ' + name)
    errors.extend('Unknown reference tensor: ' + name for name in seen if name not in weight_map)
    return errors


def compare_tensors(actual, expected):
    import torch
    result = {'shape': list(actual.shape), 'reference_shape': list(expected.shape),
              'dtype': str(actual.dtype), 'reference_dtype': str(expected.dtype)}
    same_shape = actual.shape == expected.shape
    same_dtype = actual.dtype == expected.dtype
    result['equal'] = bool(same_shape and same_dtype and torch.equal(actual, expected))
    for label, tensor in [('actual', actual), ('reference', expected)]:
        # uint8 is also supported for BF16, unlike a direct numpy() conversion.
        view = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        result[label + '_sha256'] = hashlib.sha256(memoryview(view).cast('B')).hexdigest()
    if result['equal'] and result['actual_sha256'] != result['reference_sha256']:
        result['equal'] = False  # e.g. numeric +0/-0 equality is not bitwise parity.
    return result


def self_test():
    import torch
    sample = torch.arange(24, dtype=torch.float32).reshape(4, 6).to(torch.bfloat16)
    assert compare_tensors(sample, sample.clone())['equal']
    assert not compare_tensors(sample, sample.flip(0))['equal']
    assert not compare_tensors(sample, sample.float())['equal']
    assert not compare_tensors(sample, sample.reshape(6, 4))['equal']
    bad = sample.clone()
    bad[0, 0] = float('nan')
    assert not compare_tensors(bad, bad)['equal']
    assert not compare_tensors(torch.tensor([0.0]), torch.tensor([-0.0]))['equal']
    return {'real_torch_positive_and_negative_cases': 6, 'status': 'ok'}


def verify_files(root, files):
    for item in files:
        path = root / item['path']
        if Path(item['path']).is_absolute() or '..' in Path(item['path']).parts:
            raise ValueError('Unsafe pinned file path.')
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError('Pinned input changed type or size: ' + item['path'])
        if sha256(path) != item['sha256']:
            raise ValueError('Pinned input hash mismatch: ' + item['path'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-checkpoint-parity-child-v{args.attempt}')
    source, model, checkpoint = Path('/miles-source'), Path('/model'), Path('/checkpoint')
    errors, result, metrics = [], {}, []
    try:
        import torch
        import torch.distributed.checkpoint as dcp
        from megatron.core.dist_checkpointing.serialization import load_common_state_dict
        from safetensors import safe_open
        from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf

        torch.set_num_threads(8)
        torch.set_grad_enabled(False)
        result['imports'] = validate_imports(run, source, model)
        result['self_test'] = self_test()
        atomic(phase.path / 'self-test.json', result['self_test'])
        conversion_lock = json.loads((Path(__file__).parent / 'converted.lock.json').read_text())
        hf_lock = json.loads((Path(__file__).parent / 'hf.lock.json').read_text())
        if sha256(checkpoint / 'conversion.manifest.json') != conversion_lock['manifest_sha256']:
            raise ValueError('Checkpoint conversion manifest differs from the pinned candidate.')
        started = time.monotonic()
        verify_files(model, hf_lock['files'])
        verify_files(checkpoint, conversion_lock['manifest']['files'])
        result['input_rehash_duration_s'] = time.monotonic() - started
        result['manifest_sha256'] = conversion_lock['manifest_sha256']
        result['model_revision'] = hf_lock['revision']
        print(json.dumps({'event': 'all_input_hashes_verified', 'duration_s': result['input_rehash_duration_s']}), flush=True)

        release = checkpoint / 'release'
        common = load_common_state_dict(release)
        saved_args = common['args']
        fields = ['num_layers', 'num_experts', 'mtp_num_layers', 'hidden_size', 'num_attention_heads',
                  'num_query_groups', 'kv_channels', 'vocab_size', 'moe_router_dtype',
                  'tensor_model_parallel_size', 'pipeline_model_parallel_size', 'expert_model_parallel_size']
        result['saved_recipe'] = {key: getattr(saved_args, key, None) for key in fields}
        if (saved_args.num_layers, saved_args.num_experts, saved_args.mtp_num_layers) != (40, 256, 1):
            raise ValueError('Saved checkpoint model/MTP recipe differs from the frozen architecture.')
        # Same setting as upstream save_tensors; does not mutate the checkpoint.
        saved_args.sglang_enable_ep_moe = False
        atomic(phase.path / 'saved-recipe.json', result['saved_recipe'])
        reader = dcp.FileSystemReader(release)
        metadata = reader.read_metadata()
        tensor_metadata = {k: v for k, v in metadata.state_dict_metadata.items()
                           if isinstance(v, dcp.TensorStorageMetadata)}
        if any('optimizer' in k or '_state' in k for k in tensor_metadata):
            raise ValueError('Unexpected optimizer or state tensor in the candidate model checkpoint.')
        tensor_bytes = sum(v.size.numel() * torch.empty((), dtype=v.properties.dtype).element_size()
                           for v in tensor_metadata.values())
        meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(meminfo['MemAvailable'].strip().split()[0]) * 1024
        if available < tensor_bytes * 3 + 16 * 1024**3:
            raise ValueError('Insufficient host RAM for a bounded full CPU tensor comparison.')
        result['tensor_metadata_count'] = len(tensor_metadata)
        result['mtp_weight_tensor_count'] = sum('mtp.' in k for k in tensor_metadata)
        result['non_tensor_metadata_count'] = len(metadata.state_dict_metadata) - len(tensor_metadata)
        result['checkpoint_tensor_bytes'] = tensor_bytes
        atomic(phase.path / 'tensor-metadata.json', {
            k: {'shape': list(v.size), 'dtype': str(v.properties.dtype)} for k, v in tensor_metadata.items()})
        started = time.monotonic()
        state = {k: torch.empty(v.size, dtype=v.properties.dtype) for k, v in tensor_metadata.items()}
        # Flat DCP keys are intentional. No optimizer, RNG or extra_state payload is compared as a weight.
        dcp.load(state, storage_reader=reader,
                 planner=dcp.DefaultLoadPlanner(flatten_state_dict=False), no_dist=True)
        result['checkpoint_read_duration_s'] = time.monotonic() - started
        print(json.dumps({'event': 'all_model_tensors_loaded', 'tensor_count': len(state),
                          'duration_s': result['checkpoint_read_duration_s']}), flush=True)

        # The upstream script installs a pickle wrapper for its legacy loader.
        # Restore it immediately; only its real parameter-expansion function is reused.
        original_unpickler = pickle.Unpickler
        try:
            upstream = runpy.run_path(str(source / 'tools/convert_torch_dist_to_hf.py'), run_name='parity_helpers')
        finally:
            pickle.Unpickler = original_unpickler
        index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
        seen, output_names = collections.defaultdict(set), set()
        counts = collections.Counter()
        partial = phase.path / 'tensor-comparisons.jsonl.partial'
        started = time.monotonic()
        with partial.open('x') as evidence:
            for name, value in upstream['get_named_params'](saved_args, state):
                for converted_name, actual in convert_to_hf(saved_args, 'qwen3_5_moe', name, value):
                    if converted_name in output_names:
                        raise ValueError('Duplicate converted weight: ' + converted_name)
                    output_names.add(converted_name)
                    target, expert, part = reference_part(converted_name, index, saved_args.num_experts)
                    token = (expert, part)
                    if token in seen[target] or ((None, None) in seen[target]):
                        raise ValueError('Duplicate/overlapping reference coverage: ' + converted_name)
                    seen[target].add(token)
                    with safe_open(model / index[target], framework='pt', device='cpu') as f:
                        view = f.get_slice(target)
                        if expert is None:
                            expected = view[:]
                        elif part == 'down_proj':
                            expected = view[expert]
                        else:
                            shape = view.get_shape()
                            if len(shape) != 3 or shape[0] != saved_args.num_experts or shape[1] % 2:
                                raise ValueError('Unexpected fused gate/up reference shape: ' + target)
                            half = shape[1] // 2
                            expected = view[expert, :half] if part == 'gate_proj' else view[expert, half:]
                        row = dict(compare_tensors(actual, expected), source_name=name, converted_name=converted_name,
                                   reference_name=target, expert_id=expert, reference_part=part)
                    counts['compared'] += 1
                    counts['mtp_compared'] += int(converted_name.startswith('mtp.'))
                    counts['equal' if row['equal'] else 'mismatched'] += 1
                    evidence.write(json.dumps(row, allow_nan=False) + '\n')
                    if not row['equal'] and len(errors) < 25:
                        errors.append('Weight parity mismatch: ' + converted_name)
                    if counts['compared'] % 256 == 0:
                        evidence.flush()
                        if shutil.disk_usage(phase.path).free < 128 * 1024**3:
                            raise ValueError('Evidence free-space reserve fell below 128 GiB.')
                        if (run.root / f'control/checkpoint-parity-v{args.attempt}.stop').exists():
                            raise RuntimeError('Explicit parity stop marker received.')
            evidence.flush()
            os.fsync(evidence.fileno())
        os.rename(partial, phase.path / 'tensor-comparisons.jsonl')
        result['comparison_duration_s'] = time.monotonic() - started
        result['counts'] = dict(counts)
        errors.extend(check_coverage(index, seen, saved_args.num_experts))
        if counts['mtp_compared'] == 0 or not counts['compared']:
            errors.append('No complete text/MTP weight comparison was made.')
        result['reference_weight_count'] = len(seen)
        result['excluded_vision_weights'] = sorted(k for k in index if k.startswith('model.visual.'))
        atomic(phase.path / 'reference-coverage.json', {k: sorted(v, key=str) for k, v in seen.items()})
        for key in ('input_rehash_duration_s', 'checkpoint_read_duration_s', 'comparison_duration_s'):
            metrics.append(metric(key, result[key], 's'))
        for key, value in counts.items():
            metrics.append(metric(key, value, 'count'))
    except Exception as exc:
        errors.append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    result['findings'] = errors
    result['scope'] = ('Exact CPU round-trip weight parity for text and MTP only. Vision is deliberately excluded. '
                       'Not a forward/logit, EP8 reshard, optimizer, resume, policy or quality test.')
    atomic(phase.path / 'parity-result.json', result)
    phase.finish('fail' if errors else 'ok', results=metrics, metadata=result,
                 failure_summary='; '.join(errors) or None, refresh=False)
    print(json.dumps({'findings': errors, 'counts': result.get('counts')}), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
