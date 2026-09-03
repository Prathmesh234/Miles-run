"""Bounded actual-B200 MXFP8 kernel/export probe; no optimizer or serving claim."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import traceback

from convert_qwen_mxfp8 import inspect_source, unpack
from evidence import Run, atomic, metric, sha256
from model_conversion import validate_imports


def exercise(weight, name, quant_config):
    import torch
    from miles.utils.mxfp8 import mxfp8_quantize
    from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import quantize_params_mxfp8

    original_hash = hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    weight = weight.cuda()
    start = time.monotonic()
    qweight, scale = mxfp8_quantize(weight)
    torch.cuda.synchronize()
    duration = time.monotonic() - start
    if qweight.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8 or (scale == 255).any():
        raise ValueError('Invalid MXFP8 dtype/scale.')
    restored = qweight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)
    error = float((restored - weight.float()).norm() / weight.float().norm().clamp_min(1e-30))
    if not torch.isfinite(restored).all() or error > 0.06:
        raise ValueError('MXFP8 reconstruction exceeds pre-registered relative L2 <= 0.06.')
    expected = dict(unpack(name, qweight))
    expected.update({key.removesuffix('.weight') + '.weight_scale_inv': value for key, value in unpack(name, scale)})
    params = dict(unpack(name, weight))
    args = SimpleNamespace()
    compared = 0
    if '.experts.' in name:
        prefix = name.split('.mlp.experts.')[0]
        layer = prefix.rsplit('.', 1)[-1]
        mega_prefix = (f'module.module.mtp.layers.{layer}.transformer_layer' if prefix.startswith('mtp.')
                       else f'module.module.decoder.layers.{layer}')
        kind = 'linear_fc1' if name.endswith('gate_up_proj') else 'linear_fc2'
        for expert in range(weight.shape[0]):
            values = [(key, val) for key, val in params.items() if f'.experts.{expert}.' in key]
            exported = quantize_params_mxfp8(args, f'{mega_prefix}.mlp.experts.{kind}.weight{expert}', values, quant_config)
            for key, value in exported:
                if value.dtype != expected[key].dtype or not torch.equal(value.view(torch.uint8), expected[key].contiguous().view(torch.uint8)):
                    raise ValueError('Packed conversion disagrees with live per-expert export: ' + key)
                compared += 1
        if compared != len(expected):
            raise ValueError('Incomplete expert export comparison.')
    return dict(name=name, shape=list(weight.shape), source_sha256=original_hash, relative_l2=error,
                kernel_duration_s=duration, exact_export_tensors=compared,
                output_bytes=qweight.numel() + scale.numel(), source_bytes=weight.numel() * weight.element_size())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', required=True, type=int)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-qwen-mxfp8-probe-child-v{args.attempt}')
    result, findings = {}, []
    try:
        import safetensors
        import torch
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.layers.quantization.utils import is_layer_skipped

        result['imports'] = validate_imports(run, Path('/miles-source'), Path('/model'))
        config, plan = inspect_source(Path('/model'))
        atomic(phase.path / 'header-plan.json', plan)
        qconfig = config['quantization_config']
        installed = Fp8Config.from_config(qconfig)
        if installed.weight_block_size != [1, 32] or not installed.use_mxfp8:
            raise ValueError('Installed SGLang did not resolve MXFP8.')
        for prefix in ('model.layers.0.linear_attn.in_proj_qkvz', 'model.layers.0.linear_attn.in_proj_ba',
                       'model.layers.0.linear_attn.out_proj', 'model.visual.blocks.0.mlp.fc1', 'mtp.fc'):
            if not is_layer_skipped(prefix, installed.ignored_layers):
                raise ValueError('SGLang BF16 exception did not match: ' + prefix)
        for prefix in ('model.layers.0.mlp.experts', 'model.layers.0.mlp.shared_expert.gate_up_proj',
                       'model.layers.3.qkv_proj'):
            if is_layer_skipped(prefix, installed.ignored_layers):
                raise ValueError('SGLang incorrectly excludes intended quantization: ' + prefix)
        names = [f'{prefix}.mlp.experts.{projection}'
                 for prefix in ('model.language_model.layers.0', 'mtp.layers.0')
                 for projection in ('gate_up_proj', 'down_proj')]
        rows = []
        for name in names:
            row = next(row for row in plan['tensors'] if row['name'] == name)
            with safetensors.safe_open(Path('/model') / row['shard'], framework='pt', device='cpu') as source:
                # Two actual experts, preserving all rows/columns and gate/up order.
                weight = source.get_slice(name)[:2]
            rows.append(exercise(weight, name, qconfig))
        rows.append(exercise(torch.zeros((32, 64), dtype=torch.bfloat16), 'zero-weight-control', qconfig))
        result.update(cases=rows, quantized_source_bytes=plan['quantized_source_bytes'],
                      source_payload_bytes=plan['source_payload_bytes'],
                      sglang_exception_matching_passed=True,
                      kernel_sha256=sha256('/miles-source/miles/utils/mxfp8.py'),
                      acceptance={'maximum_relative_l2': 0.06, 'expert_export': 'byte-exact'},
                      gpu_memory_peak_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        findings.append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    result.update(findings=findings, optimizer_steps_enabled=False,
        scope='Representative real expert slices plus zero control on GPU0 of an 8-GPU allocation. Not full model conversion, SGLang activation, gradient equivalence, throughput or quality validation.')
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if findings else 'ok', metadata=result, failure_summary='; '.join(findings) or None,
                 results=[metric('exact_export_tensors', sum(row['exact_export_tensors'] for row in result.get('cases', [])), 'count')], refresh=False)
    print(json.dumps(result), flush=True)
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
