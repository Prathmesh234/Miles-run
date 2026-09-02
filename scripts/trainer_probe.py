"""Pinned EP8 trainer reshard and forward/backward probe; never constructs an optimizer.

Uses the Miles model provider, production packed batching, DDP and Megatron
pipeline schedule. The main diagnostic loss is next-token cross entropy, not
GRPO. MTP uses the recipe's native auxiliary loss with detached shared heads.
This does not qualify policy logprobs against SGLang or GRPO/resume correctness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from evidence import Run, atomic, sha256, utcnow


MILES_SHA = '346946ae870be97e9cb6f4e8b7214c7fcf66c041'


def trainer_command(output, model_args):
    tokens = shlex.split(model_args)
    flag = '--moe-token-dispatcher-type'
    if tokens.count(flag) != 1:
        raise ValueError('Expected exactly one model dispatcher setting.')
    tokens[tokens.index(flag) + 1] = 'flex'
    if tokens.count('--mtp-num-layers') != 1 or tokens[tokens.index('--mtp-num-layers') + 1] != '1':
        raise ValueError('The probe must retain the cookbook MTP layer.')
    return [sys.executable, '-m', 'torch.distributed.run', '--rdzv-backend=static',
            '--master-addr=127.0.0.1', '--master-port=31875', '--nnodes=1', '--node-rank=0',
            '--nproc-per-node=8', str(Path(__file__).resolve()), '--worker', *tokens,
            '--hf-checkpoint', '/model', '--load', '/checkpoint', '--no-load-optim', '--no-load-rng',
            '--finetune', '--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1',
            '--context-parallel-size', '1', '--expert-model-parallel-size', '8',
            '--expert-tensor-parallel-size', '1', '--sequence-parallel',
            '--recompute-granularity', 'full', '--recompute-method', 'uniform', '--recompute-num-layers', '1',
            '--attention-dropout', '0', '--hidden-dropout', '0', '--accumulate-allreduce-grads-in-fp32',
            '--attention-softmax-in-fp32', '--attention-backend', 'flash',
            '--enable-mtp-training', '--mtp-loss-scaling-factor', '0.2',
            '--micro-batch-size', '1', '--global-batch-size', '8', '--seq-length', '128', '--seed', '1234',
            '--probe-output-dir', str(output)]


def parameter_hashes(model):
    import torch
    rows = []
    for chunk, module in enumerate(model):
        for name, parameter in module.named_parameters():
            data = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            rows.append({'chunk': chunk, 'name': name, 'shape': list(parameter.shape),
                         'dtype': str(parameter.dtype), 'bytes': parameter.numel() * parameter.element_size(),
                         'sha256': hashlib.sha256(memoryview(data).cast('B')).hexdigest()})
    return rows


def gradient_statistics(model):
    import torch
    rows = []
    for chunk, module in enumerate(model):
        for name, parameter in module.named_parameters():
            grad = getattr(parameter, 'main_grad', None)
            if grad is None:
                grad = parameter.grad
            row = {'chunk': chunk, 'name': name, 'present': grad is not None, 'mtp': '.mtp.' in name}
            if grad is not None:
                row.update(finite=bool(torch.isfinite(grad).all()),
                           nonzero=bool(torch.count_nonzero(grad)),
                           max_abs=float(grad.detach().abs().max()),
                           local_buffer_l2=float(torch.linalg.vector_norm(grad.detach().float())))
            rows.append(row)
    if any(row.get('finite') is False for row in rows):
        raise ValueError('Nonfinite gradient buffer.')
    for mtp in (False, True):
        if not any(row['mtp'] == mtp and row.get('nonzero') for row in rows):
            raise ValueError('No nonzero gradient in ' + ('MTP' if mtp else 'main model'))
    return rows


def build_probe_batch(rank, model_path):
    import torch
    from transformers import AutoTokenizer
    from miles.backends.training_utils.data import DataIterator, get_batch
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
    text = (f'Worker {rank} checks a local machine. A file contains one two three four. '
            'The machine reads the file and writes a checksum. ') * 6
    ids = tokenizer.encode(text, add_special_tokens=False)[:128]
    if len(ids) != 128:
        raise ValueError('The deterministic probe did not produce its fixed 128-token input.')
    tokens = torch.tensor(ids, dtype=torch.long, device='cuda')
    data = {'tokens': [tokens], 'total_lengths': [len(ids)], 'response_lengths': [len(ids) - 1],
            'loss_masks': [torch.ones(len(ids) - 1, device='cuda')]}
    batch = get_batch(DataIterator(data, micro_batch_size=1),
                      ['tokens', 'total_lengths', 'response_lengths', 'loss_masks', 'multimodal_train_inputs'],
                      pad_multiplier=128, qkv_format='thd')
    return ids, batch


def forward_backward(args, model, batch, output_dir):
    import torch
    import torch.nn.functional as F
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.utils import get_model_config
    from miles.backends.megatron_utils.parallel import get_packed_seq_params
    config = get_model_config(model[0])
    config.grad_scale_func = None
    config.timers = None
    config.finalize_model_grads_func = finalize_model_grads
    # The cookbook does not request gradient/parameter overlap. Stop if the
    # pinned resolver begins enabling it instead of silently omitting its hooks.
    if args.overlap_grad_reduce or args.overlap_param_gather:
        raise ValueError('Unexpected overlap hooks require explicit probe support.')
    for module in model:
        module.train()
        module.zero_grad_buffer()
    captured = {}

    def forward_step(iterator, module):
        data = next(iterator)
        logits = module(input_ids=data['tokens'], position_ids=None, attention_mask=None, labels=None,
                        packed_seq_params=get_packed_seq_params(data, SimpleNamespace(qkv_format='thd')),
                        loss_mask=data['full_loss_masks'], fp32_output=False)
        if tuple(logits.shape) != (1, 128, 248320) or not torch.isfinite(logits).all():
            raise ValueError('Unexpected or nonfinite Qwen logits.')
        detached = logits.detach().float()
        log_probs = detached[:, :-1].log_softmax(-1).gather(-1, data['tokens'][:, 1:, None]).squeeze(-1)
        captured.update(logit_shape=list(logits.shape), logit_dtype=str(logits.dtype),
                        teacher_forced_log_probs=log_probs.cpu().tolist(),
                        top_token_ids=detached.argmax(-1).cpu().tolist())

        def loss(output):
            value = F.cross_entropy(output[:, :-1].float().reshape(-1, output.shape[-1]),
                                    data['tokens'][:, 1:].reshape(-1), reduction='mean')
            if not torch.isfinite(value):
                raise ValueError('Nonfinite diagnostic CE loss.')
            captured['main_cross_entropy'] = float(value.detach())
            return value, {'diagnostic_cross_entropy': value.detach()}
        return logits, loss

    started = time.monotonic()
    get_forward_backward_func()(forward_step_func=forward_step, data_iterator=iter([batch]), model=model,
                               num_microbatches=1, seq_length=128, micro_batch_size=1, forward_only=False)
    torch.cuda.synchronize()
    captured['forward_backward_duration_s'] = time.monotonic() - started
    atomic(output_dir / 'forward.json', captured)
    return captured


def worker():
    import torch
    import torch.distributed as dist
    from megatron.core.enums import ModelType
    from megatron.core import mpu
    from megatron.core.utils import get_model_config
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.training import get_model
    from miles.backends.megatron_utils.arguments import set_default_megatron_args
    from miles.backends.megatron_utils.initialize import init
    from miles.backends.megatron_utils.model_provider import get_model_provider_func
    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    converter = runpy.run_path('/miles-source/tools/convert_hf_to_torch_dist.py', run_name='probe_import')

    def add_args(parser):
        converter['add_conversion_args'](parser)
        parser.add_argument('--enable-mtp-training', action='store_true', required=True)
        parser.add_argument('--probe-output-dir', required=True)
        return parser

    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    args = parse_args(add_args)
    args.debug_deterministic_collective = False
    args.enable_witness = False
    set_default_megatron_args(args)
    validate_args(args)
    output = Path(args.probe_output_dir) / f'rank-{rank:02d}'
    output.mkdir(exist_ok=False)
    result = {'rank': rank, 'hostname': socket.gethostname(), 'started_at': utcnow(),
              'gpu_uuid': str(torch.cuda.get_device_properties(local).uuid),
              'optimizer_constructed': False, 'optimizer_steps': 0}
    try:
        init(args)
        topology = {'tp': mpu.get_tensor_model_parallel_world_size(), 'pp': mpu.get_pipeline_model_parallel_world_size(),
                    'cp': mpu.get_context_parallel_world_size(), 'ep': mpu.get_expert_model_parallel_world_size(),
                    'etp': mpu.get_expert_tensor_parallel_world_size(), 'dense_dp': mpu.get_data_parallel_world_size(),
                    'expert_dp': mpu.get_expert_data_parallel_world_size()}
        if topology != {'tp': 1, 'pp': 1, 'cp': 1, 'ep': 8, 'etp': 1, 'dense_dp': 8, 'expert_dp': 1}:
            raise ValueError('Resolved trainer topology differs from the one-node EP8 contract.')
        result['topology'] = topology
        atomic(output / 'progress.json', dict(result, stage='building_model'))
        started = time.monotonic()
        model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=True)
        result['model_build_duration_s'] = time.monotonic() - started
        config = get_model_config(model[0])
        result['recipe'] = {key: getattr(config, key) for key in (
            'num_layers', 'hidden_size', 'num_moe_experts', 'moe_router_topk', 'moe_token_dispatcher_type',
            'mtp_num_layers', 'mtp_loss_scaling_factor', 'mtp_detach_heads', 'sequence_parallel',
            'recompute_granularity', 'recompute_method', 'recompute_num_layers')}
        if not config.mtp_detach_heads or config.mtp_num_layers != 1 or config.moe_token_dispatcher_type != 'flex':
            raise ValueError('MTP or dispatcher recipe changed.')
        atomic(output / 'progress.json', dict(result, stage='loading_checkpoint'))
        started = time.monotonic()
        load_checkpoint(model, None, None, None, False)
        torch.cuda.synchronize()
        result['checkpoint_load_duration_s'] = time.monotonic() - started
        before = parameter_hashes(model)
        atomic(output / 'parameters-before.json', before)
        ids, batch = build_probe_batch(rank, '/model')
        atomic(output / 'input.json', {'token_ids': ids, 'cu_seqlens': batch['cu_seqlens'].cpu().tolist(),
                                      'source': 'Fixed diagnostic prose; not a task, rollout, reward or policy sample.'})
        atomic(output / 'progress.json', dict(result, stage='forward_backward'))
        result['forward'] = forward_backward(args, model, batch, output)
        gradients = gradient_statistics(model)
        atomic(output / 'gradients.json', gradients)
        after = parameter_hashes(model)
        atomic(output / 'parameters-after.json', after)
        if before != after:
            raise ValueError('Model parameters changed despite no optimizer step.')
        result.update(status='ok', parameters_unchanged=True,
                      gradient_tensors_present=sum(r['present'] for r in gradients),
                      gradient_tensors_nonzero=sum(r.get('nonzero', False) for r in gradients),
                      mtp_gradient_tensors_nonzero=sum(r['mtp'] and r.get('nonzero', False) for r in gradients),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        result.update(status='fail', failure_summary=str(exc))
        atomic(output / 'exception.txt', traceback.format_exc())
        raise
    finally:
        result['ended_at'] = utcnow()
        atomic(output / 'result.json', result)
    dist.barrier()
    dist.destroy_process_group()


def main():
    from checkpoint_parity import verify_files
    from model_conversion import validate_imports
    from run_checkpoint_parity import run_child
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-trainer-probe-child-v{args.attempt}')
    result, errors = {}, []
    try:
        result['imports'] = validate_imports(run, Path('/miles-source'), Path('/model'), miles_sha=MILES_SHA)
        code = Path(__file__).parent
        hf = json.loads((code / 'hf.lock.json').read_text())
        ckpt = json.loads((code / 'converted.lock.json').read_text())
        if sha256(Path('/checkpoint/conversion.manifest.json')) != ckpt['manifest_sha256']:
            raise ValueError('Converted checkpoint manifest changed.')
        started = time.monotonic()
        verify_files(Path('/model'), hf['files'])
        verify_files(Path('/checkpoint'), ckpt['manifest']['files'])
        result['input_rehash_duration_s'] = time.monotonic() - started
        from miles.utils.external_utils.model_args_utils import load_model_args
        output = phase.path / 'ranks'
        output.mkdir()
        command = trainer_command(output, load_model_args('qwen3.6-35B-A3B'))
        atomic(phase.path / 'command.json', {'argv': command, 'scope': __doc__})
        if run_child(phase, command, args.attempt, stage='trainer-probe', execution_limit_s=900):
            raise RuntimeError('EP8 worker failed; inspect per-rank and raw torchrun evidence.')
        rows = [json.loads((output / f'rank-{rank:02d}/result.json').read_text()) for rank in range(8)]
        if any(row['status'] != 'ok' or row['optimizer_steps'] != 0 for row in rows):
            raise ValueError('At least one rank did not pass the no-optimizer contract.')
        if len({row['gpu_uuid'] for row in rows}) != 8:
            raise ValueError('Worker GPU UUIDs are not unique.')
        normalize_uuid = lambda value: value.lower().removeprefix('gpu-').replace('-', '')
        if {normalize_uuid(row['gpu_uuid']) for row in rows} != {
                normalize_uuid(value) for value in result['imports']['gpu_uuids']}:
            raise ValueError('CUDA worker UUIDs do not match the reconciled physical inventory.')
        result['ranks'] = rows
    except Exception as exc:
        errors.append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    result.update(findings=errors, scope=__doc__)
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None, metadata=result, refresh=False)
    print(json.dumps({'findings': errors, 'passed_ranks': len(result.get('ranks', []))}), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    if '--worker' in sys.argv:
        sys.argv.remove('--worker')
        worker()
    else:
        raise SystemExit(main())
