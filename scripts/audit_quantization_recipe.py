"""Audit the pinned Miles converters against real tensor headers without loading weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from evidence import Run, atomic, metric, sha256


class HeaderTensor:
    def __init__(self, row):
        self.shape = tuple(row['shape'])
        self.dtype = row['dtype']

    def dim(self):
        return len(self.shape)


def selectors(miles):
    fp8_path = miles / 'tools/convert_hf_to_fp8.py'
    tree = ast.parse(fp8_path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'process_file')
    conditions = [node.test for node in ast.walk(function) if isinstance(node, ast.If)
                  and any(isinstance(item, ast.Name) and item.id == 'key' for item in ast.walk(node.test))]
    if len(conditions) != 1:
        raise ValueError('Pinned FP8 converter selection structure changed.')
    expression = compile(ast.Expression(conditions[0]), str(fp8_path), 'eval')
    mxfp8_path = miles / 'tools/convert_hf_to_mxfp8.py'
    tree = ast.parse(mxfp8_path.read_text())
    constant = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'SKIP_WEIGHT_SUBSTRINGS' for target in node.targets))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'should_quantize')
    namespace = {'SKIP_WEIGHT_SUBSTRINGS': ast.literal_eval(constant), 'SOURCE_FP8_DTYPES': (),
                 'torch': SimpleNamespace(Tensor=HeaderTensor, float16='F16', bfloat16='BF16', float32='F32')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(mxfp8_path), 'exec'), namespace)
    return (lambda row: eval(expression, {'__builtins__': {}}, {'key': row['name']}),
            lambda row: namespace['should_quantize'](row['name'], HeaderTensor(row)))


def audit(inventory, config, miles):
    if hashlib.sha256(config).hexdigest() != inventory['config_sha256']:
        raise ValueError('Configuration differs from the inspected model headers.')
    config = json.loads(config)
    tensors = inventory['tensors']
    fp8, mxfp8 = selectors(miles)
    packed = [row for row in tensors if '.experts.' in row['name'] and len(row['shape']) == 3]
    total_bytes = sum(row['payload_bytes'] for row in tensors)
    reports = []
    for name, select in [('fp8_blockwise', fp8), ('mxfp8', mxfp8)]:
        chosen = [row for row in tensors if select(row)]
        non_2d = [dict(name=row['name'], shape=row['shape']) for row in chosen if len(row['shape']) != 2]
        packed_selected = [row for row in packed if select(row)]
        problems = []
        if name == 'fp8_blockwise' and non_2d:
            problems.append('The block converter unpacks exactly two dimensions but selects non-2D tensors.')
        if name == 'mxfp8' and 'num_hidden_layers' not in config:
            problems.append('The MXFP8 converter requires top-level num_hidden_layers; this model nests it under text_config.')
        if len(packed_selected) != len(packed):
            problems.append('Packed MoE expert names do not end in .weight and are left unquantized by the stock selector.')
        gdn = [row['name'] for row in chosen if '.linear_attn.' in row['name']]
        visual = [row['name'] for row in chosen if '.visual.' in row['name']]
        reports.append({'mode': name, 'status': 'blocked_stock_recipe' if problems else 'needs_runtime_qualification',
            'selected_tensors': len(chosen), 'selected_payload_bytes': sum(row['payload_bytes'] for row in chosen),
            'packed_expert_tensors_selected': len(packed_selected), 'non_2d_selected': non_2d,
            'gdn_tensors_selected': len(gdn), 'vision_tensors_selected': len(visual), 'problems': problems})
    return {'schema_version': 1, 'model_type': inventory['model_type'], 'tensor_count': len(tensors),
        'payload_bytes': total_bytes, 'packed_expert_tensor_count': len(packed),
        'packed_expert_payload_bytes': sum(row['payload_bytes'] for row in packed),
        'packed_expert_payload_fraction': sum(row['payload_bytes'] for row in packed) / total_bytes,
        'converters': reports,
        'source_sha256': {str(path.relative_to(miles)): sha256(path) for path in [
            miles / 'tools/convert_hf_to_fp8.py', miles / 'tools/convert_hf_to_mxfp8.py',
            miles / 'miles_plugins/models/qwen3_5.py', miles / 'docs/advanced/low-precision.md']},
        'scope': 'Exact stock selection logic replayed on real headers. No weights read, quantized, replaced, or trained.',
        'candidate': 'Qwen-aware expert unpacking, nested text configuration, explicit BF16 GDN/vision exceptions, and matching export/loader precision; not enabled until conversion, hot reload, logprob and gradient checks pass.'}


def render(data):
    lines = ['# Qwen3.6 low-precision preflight', '', data['scope'], '',
        f"Packed expert tensors: **{data['packed_expert_tensor_count']}**, "
        f"**{100 * data['packed_expert_payload_fraction']:.2f}%** of checkpoint tensor bytes.", '',
        '| Stock mode | Selected tensors | Packed experts selected | GDN / vision selected | State |',
        '|---|---:|---:|---:|---|']
    for row in data['converters']:
        lines.append(f"| {row['mode']} | {row['selected_tensors']} | {row['packed_expert_tensors_selected']} | "
                     f"{row['gdn_tensors_selected']} / {row['vision_tensors_selected']} | {row['status']} |")
    lines += ['', '## Findings', '']
    for row in data['converters']:
        lines.extend('- ' + row['mode'] + ': ' + reason for reason in row['problems'])
    lines += ['', '## Next candidate', '', data['candidate'], '',
        'This failure applies to the stock conversion recipe, not to B200 low-precision capability. The BF16 run remains unchanged. Higher-precision master weights and optimizer state remain in the documented recipes; quantization does not imply proportional full-checkpoint shrinkage.', '']
    if 'qualification' in data:
        lines += ['## Qualification', ''] + ['- ' + item for item in data['qualification']]
        lines += ['', '## Reproduce the bounded GPU probe', '', '```sh', data['reproducer'], '```', '']
        lines += ['## Provenance', ''] + [f'- {key}: `{value}`' for key, value in data['provenance'].items()]
        lines += ['', f"[Pinned Miles low-precision documentation]({data['documentation_url']})", '']
    if 'gpu_probe' in data:
        probe = data['gpu_probe']
        lines += ['## Executed B200 kernel probe', '',
                  f"Slurm {probe['slurm_job_id']}: {probe['status']}. "
                  f"{probe['exact_export_tensors']} exported weight/scale tensors matched byte-for-byte; "
                  f"maximum relative L2 error {probe['max_relative_l2']:.4%} (limit 6%).", '',
                  f"Telemetry: {probe['telemetry_streams']} finalized streams, "
                  f"{probe['collector_errors']} collector errors; maximum sample gap {probe['max_gap_s']:.3f}s.", '',
                  probe['scope'], '', f"Raw evidence: `{probe['evidence']}`.", '']
    if 'conversion' in data:
        row = data['conversion']
        lines += ['## Full converted candidate', '',
                  f"Slurm {row['slurm_job_id']}: {row['status']}. Conversion took {row['duration_s']:.2f}s.", '',
                  f"Tensor payload: **{row['payload_bytes'] / 1e9:.2f} GB**, from {row['source_payload_bytes'] / 1e9:.2f} GB. "
                  f"All {row['tensor_count']:,} serialized tensors passed names/shapes/dtypes/coverage checks. "
                  f"{row['unchanged_tensors_byte_exact']} higher-precision tensors and tokenizer metadata are byte-exact.", '',
                  f"{row['checksummed_files']} file checksums verified; maximum quantization relative L2 {row['max_relative_l2']:.4%}. "
                  f"{row['telemetry_streams']} finalized infrastructure streams, {row['collector_errors']} collector errors.", '',
                  'This reduces inference-weight storage, not the full optimizer checkpoint. No training speed or held-out quality claim.', '',
                  f"Checkpoint: `{row['checkpoint']}`. Audit: `{row['evidence']}`.", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--attempt', required=True, type=int)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-quantization-recipe-audit-v{args.attempt}')
    miles = Path(__file__).resolve().parents[1] / 'vendor/miles'
    try:
        data = audit(json.loads(Path(args.inventory).read_text()), Path(args.config).read_bytes(), miles)
        atomic(phase.path / 'result.json', data)
        atomic(phase.path / 'report.md', render(json.loads((phase.path / 'result.json').read_text())))
        failed = any(row['problems'] for row in data['converters'])
        phase.finish('fail' if failed else 'ok', failure_summary='Stock quantization recipes do not cover this checkpoint; retain BF16 until a Qwen-specific candidate passes.' if failed else None,
            results=[metric('packed_expert_payload_fraction', data['packed_expert_payload_fraction'], 'ratio')],
            metadata={'scope': data['scope'], 'optimizer_steps_enabled': False})
        print(json.dumps({key: data[key] for key in ('tensor_count','packed_expert_tensor_count','packed_expert_payload_fraction','converters')}, indent=2))
        return int(failed)
    except Exception as exc:
        phase.finish('fail', failure_summary=str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
