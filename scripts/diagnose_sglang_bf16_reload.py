"""Reproduce the pinned SGLang BF16 reload shape failure using actual functions.

CPU tensors exercise the exact parsed loader bodies; no replacement numerical
implementation, policy execution, GPU allocation, or optimizer is used.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evidence import Run, atomic, metric, sha256


def extract(path, class_name, method, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    code = ast.Module(body=[node], type_ignores=[])
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace[method]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    a = ap.parse_args()
    run = Run(a.run_dir)
    phase = run.phase('02-sglang-bf16-broadcast-diagnosis-v1')
    source = run.root / 'provenance/sglang-broadcast-diagnostic-v1'
    rows = json.loads((source / 'manifest.json').read_text())['files']
    rows.append(json.loads((source / 'unquant.manifest.json').read_text()))
    for row in rows:
        if sha256(source / row['path']) != row['sha256']:
            raise ValueError('Pinned source changed: ' + row['path'])
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    worker = ['kubectl', '--kubeconfig', a.kubeconfig, '-n', 'slurm', 'exec', 'slurm-worker-gpu-nodes-0', '--']
    verify = ('import subprocess,hashlib,json,sys\nrows=json.loads(sys.argv[2]);result=[]\n'
        'for row in rows:\n'
        ' data=subprocess.check_output(["unsquashfs","-cat",sys.argv[1],"sgl-workspace/sglang/"+row["path"]])\n'
        ' actual=hashlib.sha256(data).hexdigest()\n'
        ' if actual!=row["sha256"]:raise ValueError("Pinned image/source mismatch: "+row["path"])\n'
        ' result.append({"path":row["path"],"image_file_sha256":actual})\n'
        'print(json.dumps(result))\n')
    rc, out, _ = phase.command(worker + ['python3', '-c', verify,
        remote + '/images/enroot-import-v2/miles-amd64.sqsh', json.dumps(rows)], timeout=60)
    if rc:
        phase.finish('fail', failure_summary='Runtime-image and diagnostic source identity could not be established.')
        return 1
    atomic(phase.path / 'image-source-identity.json', json.loads(out))
    backend = SimpleNamespace(value='flashinfer_trtllm')
    backend.is_flashinfer_trtllm_routed = lambda: backend.value == 'flashinfer_trtllm_routed'
    namespace = {'torch': torch, 'get_moe_runner_backend': lambda: backend, '_is_cpu': False,
                 '_maybe_copy_weight_view_before_h2d': lambda value: value}
    restore = extract(source / 'python/sglang/srt/layers/quantization/unquant.py',
                      'UnquantizedFusedMoEMethod', 'maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load', namespace)
    copy = extract(source / 'python/sglang/srt/layers/moe/fused_moe_triton/layer.py', 'FusedMoE', '_load_w13', namespace)
    layer = SimpleNamespace(num_local_experts=1, intermediate_size_per_partition=512, hidden_size=2048,
        moe_runner_config=SimpleNamespace(is_gated=True), quant_method=SimpleNamespace(),
        use_padded_loading=False, use_presharded_weights=False, use_triton_kernels=False, moe_tp_size=1)
    loaded = torch.arange(512*2048).remainder(307).reshape(512, 2048).to(torch.bfloat16)
    records = []
    for mode in ('flashinfer_trtllm', 'flashinfer_trtllm_routed', 'triton'):
        backend.value = mode
        shape = (1, 1024, 2048) if mode == 'triton' else (1, 32, 1024, 64)
        param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)
        restore(None, layer, param, 'model.layers.0.mlp.experts.w13_weight')
        error = None
        try:
            copy(layer, param.data[0], 0, 'w1', loaded, 0)
        except RuntimeError as exc:
            error = str(exc)
        if mode == 'flashinfer_trtllm':
            if not error or '64' not in error or '2048' not in error:
                raise AssertionError('The recorded broadcast shape error was not reproduced.')
        elif error or not torch.equal(param.data[0, :512], loaded):
            raise AssertionError('Canonical shape control failed: ' + str(error))
        records.append({'backend': mode, 'initial_shape': shape, 'shape_after_restore': list(param.shape),
            'copy_error': error, 'canonical_values_exact': error is None})
    result = {'cases': records, 'sglang_revision': '98bb1455b9bfc922eaf199ea31945144e2f90fef',
        'failure_job': '137', 'classification': 'SGLang FlashInfer BF16 hot-reload layout incompatibility',
        'fix': 'Pin Triton MoE with canonical expert order; verify target and MTP weights before the first optimizer step.',
        'scope': 'CPU reproduction of actual image loader functions. Not an end-to-end GPU broadcast or policy result.'}
    atomic(phase.path / 'result.json', result)
    phase.finish('ok', metadata=result, results=[metric('loader_cases_verified', len(records), 'count')])
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
