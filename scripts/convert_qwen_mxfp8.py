"""Isolated Qwen3.6 MXFP8 candidate; never changes a source/training checkpoint.

Uses the pinned Miles MXFP8 kernel, unpacks routed experts, and retains GDN,
vision, routers, embeddings, norms and MTP projection glue in source precision.
Conversion is not authorization to train: loader/broadcast/numerical gates remain.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys

from evidence import atomic, sha256


PACKED = re.compile(r'^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$')
QUANTIZED = re.compile(
    r'^(?:model\.language_model\.layers\.\d+|mtp\.layers\.\d+)\.'
    r'(?:mlp\.(?:experts\.\d+|shared_expert)\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)\.weight$')
METADATA_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
                  'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja')


def precision(name, shape, dtype):
    selected = bool(PACKED.fullmatch(name) or QUANTIZED.fullmatch(name))
    if selected and (dtype != 'BF16' or len(shape) not in (2, 3) or shape[-1] % 32):
        raise ValueError('Unsupported quantized tensor shape/dtype: ' + name)
    if '.experts.' in name and not selected:
        raise ValueError('Unrecognized expert format: ' + name)
    return 'mxfp8' if selected else 'source'


def unpack(name, tensor):
    """Same expert and gate/up order as Qwen's native SGLang/Miles loaders."""
    match = PACKED.fullmatch(name)
    if not match:
        yield name, tensor
        return
    prefix, kind = match.groups()
    if tensor.ndim != 3 or (kind == 'gate_up_proj' and tensor.shape[-2] % 2):
        raise ValueError('Packed experts require [expert, output, input], with even gate/up rows.')
    for expert, weight in enumerate(tensor.unbind(0)):
        parts = zip(('gate', 'up'), weight.chunk(2, dim=0)) if kind == 'gate_up_proj' else [('down', weight)]
        for projection, part in parts:
            yield f'{prefix}.{expert}.{projection}_proj.weight', part


def configuration(config):
    if config.get('model_type') != 'qwen3_5_moe' or config.get('quantization_config'):
        raise ValueError('Only the unquantized Qwen3.6 MoE checkpoint is accepted.')
    text = config['text_config']
    if text['num_hidden_layers'] != 40 or text['num_experts'] != 256:
        raise ValueError('Unexpected Qwen3.6-35B-A3B layer/expert topology.')
    result = copy.deepcopy(config)
    # Dotted module-boundary matching is verified against the installed SGLang.
    result['quantization_config'] = {
        'quant_method': 'mxfp8', 'activation_scheme': 'dynamic', 'fmt': 'e4m3',
        'weight_block_size': [1, 32], 'scale_fmt': 'ue8m0',
        'modules_to_not_convert': ['linear_attn', 'visual', 'mlp.gate', 'mlp.shared_expert_gate',
                                   'embed_tokens', 'lm_head', 'eh_proj', 'mtp.fc', 'norm'],
    }
    return result


def inspect_source(model):
    config_bytes = (model / 'config.json').read_bytes()
    config = configuration(json.loads(config_bytes))
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    rows, files = [], []
    for filename in sorted(set(index.values())):
        if Path(filename).name != filename or not filename.endswith('.safetensors'):
            raise ValueError('Unsafe source shard path.')
        path = model / filename
        if path.is_symlink():
            raise ValueError('Source shard symlinks are not accepted.')
        with path.open('rb') as handle:
            size = struct.unpack('<Q', handle.read(8))[0]
            if size > 64 * 1024**2:
                raise ValueError('Safetensors header exceeds guard.')
            header = json.loads(handle.read(size))
        for name, row in header.items():
            if name == '__metadata__':
                continue
            if index.get(name) != filename:
                raise ValueError('Index/header mismatch: ' + name)
            rows.append(dict(name=name, shape=row['shape'], dtype=row['dtype'], shard=filename,
                             payload_bytes=row['data_offsets'][1] - row['data_offsets'][0],
                             precision=precision(name, row['shape'], row['dtype'])))
        files.append(dict(path=filename, bytes=path.stat().st_size))
    if len(rows) != len(index) or len({row['name'] for row in rows}) != len(rows):
        raise ValueError('Source tensor coverage differs from index.')
    packed = [row for row in rows if PACKED.fullmatch(row['name'])]
    if len(packed) != 82 or any(row['shape'][0] != 256 for row in packed):
        raise ValueError('Expected 82 packed tensors including the MTP layer.')
    return config, {'schema_version': 1, 'status': 'candidate_not_runtime_qualified',
        'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'tensor_count': len(rows), 'packed_experts': len(packed), 'files': files, 'tensors': rows,
        'source_payload_bytes': sum(row['payload_bytes'] for row in rows),
        'quantized_source_bytes': sum(row['payload_bytes'] for row in rows if row['precision'] == 'mxfp8'),
        'optimizer_steps_enabled': False}


def convert_shard(model, destination, filename, rows, device):
    # Optional pinned CUDA-stack imports: header planning works without that stack.
    import safetensors
    import safetensors.torch
    import torch
    from miles.utils.mxfp8 import mxfp8_quantize

    output, metrics = {}, []
    with safetensors.safe_open(model / filename, framework='pt', device='cpu') as source:
        for row in rows:
            name = row['name']
            weight = source.get_tensor(name)
            if row['precision'] == 'source':
                output[name] = weight
                continue
            gpu_weight = weight.to(device)
            qweight, scale = mxfp8_quantize(gpu_weight)
            if qweight.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8:
                raise ValueError('MXFP8 data/scale dtype contract failed.')
            # UE8M0 is an exponent byte. Check actual conversion, including zeros.
            multiplier = torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)
            if (scale == 255).any():
                raise ValueError('NaN UE8M0 scale: ' + name)
            restored = qweight.float() * multiplier
            if not torch.isfinite(restored).all():
                raise ValueError('Nonfinite dequantized MXFP8 tensor: ' + name)
            delta = restored - gpu_weight.float()
            denominator = gpu_weight.float().norm().clamp_min(1e-30)
            metrics.append(dict(name=name, relative_l2=float(delta.norm() / denominator),
                                max_abs_error=float(delta.abs().max())))
            for key, value in unpack(name, qweight.cpu()):
                output[key] = value.contiguous()
            for key, value in unpack(name, scale.cpu()):
                output[key.removesuffix('.weight') + '.weight_scale_inv'] = value.contiguous()
            del gpu_weight, qweight, scale, multiplier, restored, delta
    temporary = destination / ('.' + filename + '.partial')
    safetensors.torch.save_file(output, temporary, metadata={'format': 'pt'})
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination / filename)
    return ({key: filename for key in output}, sum(t.numel() * t.element_size() for t in output.values()), metrics)


def execute(model, destination, plan, config, device):
    index, metrics, payload_bytes, input_hashes = {}, [], 0, {}
    for item in plan['files']:
        # Input + output guard includes a 256 GiB reserve; no cleanup on failure.
        if shutil.disk_usage(destination).free < 256 * 1024**3 + 2 * item['bytes']:
            raise RuntimeError('Conversion free-space reserve reached.')
        filename = item['path']
        input_hashes[filename] = sha256(model / filename)
        rows = [row for row in plan['tensors'] if row['shard'] == filename]
        mapping, size, errors = convert_shard(model, destination, filename, rows, device)
        if set(index).intersection(mapping):
            raise ValueError('Duplicate converted tensor name.')
        index.update(mapping)
        payload_bytes += size
        metrics.extend(errors)
        atomic(destination / 'progress.json', {'completed_shards': len(input_hashes), 'input_sha256': input_hashes})
    for name in METADATA_FILES:
        source = model / name
        if name != 'config.json' and source.is_file():
            if source.is_symlink():
                raise ValueError('Metadata symlink not accepted.')
            atomic(destination / name, source.read_text())
    atomic(destination / 'model.safetensors.index.json', {'metadata': {'total_size': payload_bytes}, 'weight_map': index})
    atomic(destination / 'config.json', config)
    atomic(destination / 'conversion.json', {'schema_version': 1, 'status': 'converted_not_runtime_qualified',
        'input_sha256': input_hashes, 'metrics': metrics, 'payload_bytes': payload_bytes,
        'optimizer_steps_enabled': False})
    checksums = ''.join(f'{sha256(path)}  {path.name}\n' for path in sorted(destination.iterdir()) if path.is_file())
    atomic(destination / 'checksums.sha256', checksums)
    # Published last, and explicitly NOT a serving/training acceptance marker.
    atomic(destination / 'CONVERSION_COMPLETE.json', {'status': 'converted_not_runtime_qualified',
        'checksums_sha256': sha256(destination / 'checksums.sha256'), 'optimizer_steps_enabled': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--save-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--miles-source', required=True, type=Path)
    parser.add_argument('--miles-kernel-sha256', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    run, model, destination = args.run_dir.resolve(), args.model_dir.resolve(), args.save_dir.resolve()
    if not (run / 'run.json').is_file() or not destination.is_relative_to(run) or destination == run:
        raise ValueError('Destination must be a new directory inside the current evidence run.')
    if destination.is_relative_to(model) or model.is_relative_to(destination):
        raise ValueError('Input/output trees must be disjoint.')
    kernel = args.miles_source / 'miles/utils/mxfp8.py'
    if sha256(kernel) != args.miles_kernel_sha256:
        raise ValueError('Pinned Miles MXFP8 kernel hash differs.')
    config, plan = inspect_source(model)
    plan['miles_kernel_sha256'] = args.miles_kernel_sha256
    destination.mkdir(parents=True, exist_ok=False)
    atomic(destination / 'plan.json', plan)
    if not args.plan_only:
        sys.path.insert(0, str(args.miles_source.resolve()))
        try:
            execute(model, destination, plan, config, args.device)
        except Exception as exc:
            atomic(destination / 'CONVERSION_FAILED.json', {'status': 'fail', 'reason': str(exc)})
            raise
    print(json.dumps({k: v for k, v in plan.items() if k not in ('files', 'tensors')}))


if __name__ == '__main__':
    main()
