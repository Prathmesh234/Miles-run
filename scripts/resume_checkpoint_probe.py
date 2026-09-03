"""Strict native checkpoint reload and frozen next-update comparison.

Two independent 16-rank trainer replicas may run on the four-node allocation.
This is a checkpoint diagnostic, not a replacement for the async role sweep.
No serving, new trajectories, checkpoint payload writes, or tolerance relaxation.
"""
import argparse
import copy
import gc
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import socket
import time
import traceback

from evidence import atomic, sha256, utcnow


def compare_values(actual, expected, path='state'):
    """Return dense leaf evidence; bitwise differences are failures, not repairs."""
    import numpy as np
    import torch

    if isinstance(actual, dict) and isinstance(expected, dict):
        rows = []
        if actual.keys() != expected.keys():
            rows.append(dict(path=path, equal=False, reason='dictionary_keys_differ'))
        for key in sorted(actual.keys() & expected.keys(), key=str):
            rows.extend(compare_values(actual[key], expected[key], path + '/' + str(key)))
        return rows
    if isinstance(actual, (list, tuple)) and type(actual) is type(expected):
        if len(actual) != len(expected):
            return [dict(path=path, equal=False, reason='sequence_lengths_differ')]
        return [row for i, (a, b) in enumerate(zip(actual, expected))
                for row in compare_values(a, b, path + '/' + str(i))]
    row = dict(path=path, equal=False, actual_type=type(actual).__name__, expected_type=type(expected).__name__)
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        a, b = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
        row.update(shape=list(a.shape), dtype=str(a.dtype), bytes=a.numel() * a.element_size())
        for key, value in [('actual_sha256', a), ('expected_sha256', b)]:
            row[key] = hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        row['equal'] = a.shape == b.shape and a.dtype == b.dtype and row['actual_sha256'] == row['expected_sha256']
        if a.is_floating_point() and b.is_floating_point():
            row['finite'] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            row['equal'] = row['equal'] and row['finite']
            if a.shape == b.shape and row['finite'] and not row['equal']:
                row['max_absolute_difference'] = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
        return [row]
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        row.update(actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                   expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
        row['equal'] = actual.dtype == expected.dtype and actual.shape == expected.shape and row['actual_sha256'] == row['expected_sha256']
    elif isinstance(actual, io.BytesIO) and isinstance(expected, io.BytesIO):
        row['equal'] = actual.getvalue() == expected.getvalue()
    elif type(actual) is type(expected) and isinstance(actual, (str, int, float, bool, bytes, type(None))):
        row['equal'] = actual == expected
        # Only scalar state is retained, never arbitrary repr/args/environment.
        if not isinstance(actual, (bytes, str)):
            row.update(actual=actual, expected=expected)
    return [row]


def expand_shards(value):
    from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory

    if isinstance(value, ShardedTensorFactory):
        return expand_shards(value.build())
    if isinstance(value, dict):
        return {k: expand_shards(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(expand_shards(v) for v in value)
    return value


def cpu_reference(value):
    import torch
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor

    if isinstance(value, ShardedTensor):
        result = copy.copy(value)
        result.data = torch.empty_like(value.data, device='cpu')
        if result.data.is_floating_point():
            result.data.fill_(float('nan'))
        else:
            result.data.zero_()
        return result
    if isinstance(value, ShardedObject):
        result = copy.copy(value)
        result.data = copy.deepcopy(value.data)
        return result
    if isinstance(value, dict):
        return {k: cpu_reference(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_reference(v) for v in value)
    return copy.deepcopy(value)


def unwrap_shards(value):
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject, ShardedTensor

    if isinstance(value, (ShardedTensor, ShardedObject)):
        return value.data
    if isinstance(value, LocalNonpersistentObject):
        return value.unwrap()
    if isinstance(value, dict):
        return {k: unwrap_shards(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(unwrap_shards(v) for v in value)
    return value


def verify_state(args, model, optimizer, scheduler, checkpoint, iteration, output, label):
    import torch
    import torch.distributed as dist
    from megatron.core import dist_checkpointing, mpu
    from megatron.core.rerun_state_machine import get_rerun_state_machine
    from megatron.training.checkpointing import _build_sharded_state_dict_metadata, generate_state_dict, get_rng_state
    from megatron.training.utils import unwrap_model

    started = time.monotonic()
    metadata = _build_sharded_state_dict_metadata(args)
    rng = get_rng_state(args.ckpt_format, mpu.get_tensor_model_parallel_group(),
                        mpu.get_pipeline_model_parallel_group())
    state = generate_state_dict(args, unwrap_model(model), optimizer, scheduler, rng,
        iteration=iteration, optim_sd_kwargs=dict(metadata=metadata), model_sd_kwargs=dict(metadata=metadata),
        rerun_state=get_rerun_state_machine().state_dict(data_iterator=None, ckpt_format=args.ckpt_format))
    state = expand_shards(state)
    expected, missing, unexpected = dist_checkpointing.load(cpu_reference(state), str(checkpoint), strict='return_all')
    components = ('model', 'optimizer', 'opt_param_scheduler', 'rng_state', 'iteration')
    actual = unwrap_shards(state)
    rows = compare_values({k: actual[k] for k in components}, {k: expected[k] for k in components})
    failures = [row for row in rows if not row['equal']]
    result = dict(label=label, checkpoint=str(checkpoint), iteration=iteration,
        duration_s=time.monotonic()-started, leaves=len(rows), failed_leaves=len(failures),
        missing_keys=sorted(missing), unexpected_keys=sorted(unexpected),
        tensor_bytes=sum(row.get('bytes', 0) for row in rows),
        findings=[] if not failures and not missing and not unexpected else ['Strict checkpoint state comparison failed.'])
    atomic(output / (label + '.jsonl'), ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    atomic(output / (label + '.json'), result)
    okay = torch.tensor(int(not result['findings']), device='cuda')
    dist.all_reduce(okay, op=dist.ReduceOp.MIN)
    del state, actual, expected, rows
    gc.collect()
    if not okay.item():
        raise ValueError(label + ': at least one rank failed; optimizer continuation prohibited.')
    return result


def move_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {k: move_tensors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_tensors(v) for v in value)
    return value


def verify_payload_identity(root, expected):
    for name, identity in expected.items():
        path = root / name
        if path.is_symlink():
            raise ValueError('Checkpoint payload symlink refused.')
        stat = path.stat()
        if dict(bytes=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns) != identity:
            raise ValueError('Frozen checkpoint file identity changed: ' + name)


def worker(config):
    import torch
    import torch.distributed as dist
    from megatron.core import mpu
    from megatron.core.dist_checkpointing.serialization import load_common_state_dict
    from miles.backends.megatron_utils.initialize import init
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer, train, TrainStepOutcome
    from miles.backends.training_utils.data import get_data_iterator, get_num_rollouts

    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    replica = os.environ['PTX_RESUME_REPLICA']
    output = Path('/probe-output') / replica / f'rank-{rank:02d}'
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, hostname=socket.gethostname(), rank=rank, replica=replica,
        slurm_job_id=os.environ['SLURM_JOB_ID'], started_at=utcnow(), optimizer_steps=0, findings=[])
    try:
        logging.basicConfig(level=logging.INFO)
        verify_payload_identity(Path('/run-artifacts'), config['payload_stat'])
        for name, digest in config['small_inputs'].items():
            if sha256(Path('/run-artifacts') / name) != digest:
                raise ValueError('Frozen input checksum differs: ' + name)
        checkpoint_root = Path('/run-artifacts') / config['checkpoint_root']
        before = checkpoint_root / 'iter_0000000'
        after = checkpoint_root / 'iter_0000001'
        args = copy.deepcopy(load_common_state_dict(str(before))['args'])
        args.rank, args.local_rank, args.world_size = rank, local, 16
        # Native Megatron treats ckpt_step=0 as false. An isolated read-only
        # root with its own tracker selects step zero without editing job167.
        args.load, args.ckpt_step = '/reload-root', None
        args.no_load_optim = args.no_load_rng = args.finetune = False
        args.use_checkpoint_opt_param_scheduler = True
        args.hf_checkpoint = '/model'
        args.save = args.save_hf = args.save_debug_train_data = args.save_debug_rollout_data = None
        args.save_debug_trajectory_data = args.save_debug_event_data = args.dump_details = None
        args.custom_megatron_before_train_step_hook_path = args.custom_megatron_post_save_hook_path = None
        if args.stream_optimizer_state_to_disk or args.reset_optimizer_states or args.debug_disable_optimizer:
            raise ValueError('Replay cannot stream, reset, or disable optimizer state.')
        if args.custom_megatron_init_path or args.enable_witness:
            raise ValueError('Unexpected custom initialization or witness path.')
        if args.dumper_enable or args.save_local_weight_checksum:
            raise ValueError('Unexpected auxiliary dump path in replay settings.')
        result['saved_settings'] = {key: getattr(args, key) for key in (
            'global_batch_size', 'seed', 'data_parallel_random_init', 'deterministic_mode',
            'enable_mtp_training', 'use_distributed_optimizer', 'bf16', 'lr', 'num_rollout')}
        torch.cuda.set_device(local)
        from container_fabric_probe import verify_rdma
        result['opened_hcas'] = verify_rdma()
        if int(os.environ['WORLD_SIZE']) != 16 or torch.cuda.device_count() != 8:
            raise ValueError('Replay requires two whole eight-GPU nodes per replica.')
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
        init(args)
        topology = dict(tp=mpu.get_tensor_model_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size(),
            cp=mpu.get_context_parallel_world_size(), ep=mpu.get_expert_model_parallel_world_size(),
            etp=mpu.get_expert_tensor_parallel_world_size(), dense_dp=mpu.get_data_parallel_world_size(),
            expert_dp=mpu.get_expert_data_parallel_world_size())
        if topology != dict(tp=1, pp=1, cp=1, ep=8, etp=1, dense_dp=16, expert_dp=2):
            raise ValueError('Resume topology differs from the uninterrupted trainer.')
        result['topology'] = topology
        result['gpu_uuid'] = str(torch.cuda.get_device_properties(local).uuid)
        started = time.monotonic()
        model, optimizer, scheduler, iteration = initialize_model_and_optimizer(args)
        result['load_duration_s'] = time.monotonic()-started
        if iteration != 0 or optimizer is None or scheduler is None:
            raise ValueError('Checkpoint load did not restore the required trainer objects.')
        result['loaded_state'] = verify_state(args, model, optimizer, scheduler, before, 0, output, 'loaded-state')
        # Neither independent replica may step until all 32 ranks pass reload.
        ready = Path('/probe-output/load-ready')
        atomic(ready / f'{replica}-{rank:02d}.json', dict(rank=rank, replica=replica, time=utcnow()))
        deadline = time.monotonic() + 180
        expected_ready = {f'{rep}-{r:02d}.json' for rep in ('a', 'b') for r in range(16)}
        while {p.name for p in ready.glob('*.json')} != expected_ready:
            if time.monotonic() > deadline:
                raise ValueError('All-rank reload barrier timed out; no optimizer step permitted.')
            time.sleep(0.5)
        source = Path('/run-artifacts') / config['dump_root'] / 'train_data' / f'1_{rank}.pt'
        dump = torch.load(source, map_location='cpu', weights_only=False)
        if dump['rank'] != rank or dump['rollout_id'] != 1 or dump['cp_size'] != 1:
            raise ValueError('Frozen trainer input identity mismatch.')
        data = move_tensors(dump['rollout_data'])
        result['input_sha256'] = sha256(source)
        result['sample_indices'] = [int(index) for index in data['sample_indices']]
        iterator, microbatches = get_data_iterator(args, model, data)
        if len(microbatches) != 1:
            raise ValueError('Replay must perform exactly one optimizer step.')
        started = time.monotonic()
        outcome = train(1, model, optimizer, scheduler, iterator, microbatches,
                        get_num_rollouts(args, data, 1), witness_info=None, attempt=0)
        if outcome != TrainStepOutcome.NORMAL:
            raise ValueError('Replay optimizer step was not normal.')
        result.update(optimizer_steps=1, step_duration_s=time.monotonic()-started)
        result['next_state'] = verify_state(args, model, optimizer, scheduler, after, 1, output, 'next-state')
        verify_payload_identity(Path('/run-artifacts'), config['payload_stat'])
        dist.barrier()
    except Exception as exc:
        result['findings'].append(type(exc).__name__ + ': ' + str(exc))
        atomic(output / 'exception.txt', traceback.format_exc())
    finally:
        result['ended_at'] = utcnow()
        result['scope'] = 'Strict trainer checkpoint/replay only; no online policy-version, data-cursor, async-buffer or held-out-quality claim.'
        atomic(output / 'result.json', result)
    return int(bool(result['findings']))


def cpu_self_test(output):
    import torch
    import torch.distributed as dist
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.serialization import load_common_state_dict
    # Import the exact replay entrypoints while CUDA remains hidden. This
    # catches package/API mismatches without building a model or optimizer.
    from megatron.training.utils import unwrap_model
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer, train, TrainStepOutcome
    from miles.backends.training_utils.data import get_data_iterator, get_num_rollouts
    assert all(callable(f) for f in (unwrap_model, initialize_model_and_optimizer, train,
                                   get_data_iterator, get_num_rollouts))
    assert TrainStepOutcome.NORMAL is not None
    from megatron.core.dist_checkpointing.strategies.torch import (
        MCoreSavePlanner, TorchDistSaveShardedStrategy,
        _replace_state_dict_keys_with_sharded_keys, mcore_to_pyt_state_dict,
    )

    class CPUFixtureWriter(TorchDistSaveShardedStrategy):
        """Use the native format/planner with Torch's CPU-safe sync writer.

        The default MCore writer calls CUDA synchronize even for CPU tensors.
        This adapter is fixture-only; it never replaces the training writer.
        """
        def save(self, state, checkpoint_dir):
            from torch.distributed import checkpoint
            grouped, _, _ = _replace_state_dict_keys_with_sharded_keys(state, True)
            checkpoint.save(mcore_to_pyt_state_dict(grouped, False),
                storage_writer=checkpoint.FileSystemWriter(checkpoint_dir),
                planner=MCoreSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False))

    if torch.cuda.device_count() != 0:
        raise ValueError('CPU validation unexpectedly exposes GPUs.')
    output.mkdir(parents=True, exist_ok=False)
    dist.init_process_group('gloo', init_method=(output / 'rendezvous').as_uri(), rank=0, world_size=1)
    try:
        live = dict(model={'weight': ShardedTensor.from_rank_offsets('weight', torch.arange(12.).reshape(3, 4))},
                    rng=ShardedObject('rng', [dict(state=torch.arange(8, dtype=torch.uint8))], (1,), (0,)),
                    opt_param_scheduler={'num_steps': 16}, iteration=0)
        frozen = expand_shards(live)
        target = cpu_reference(frozen)
        assert target['model']['weight'].data.data_ptr() != live['model']['weight'].data.data_ptr()
        assert torch.isnan(target['model']['weight'].data).all()
        assert target['rng'].data is not live['rng'].data
        checkpoint = output / 'checkpoint'
        checkpoint.mkdir()
        dist_checkpointing.save(live, str(checkpoint), sharded_strategy=CPUFixtureWriter())
        loaded, missing, unexpected = dist_checkpointing.load(target, str(checkpoint), strict='return_all')
        assert not missing and not unexpected
        actual = unwrap_shards(frozen)
        rows = compare_values(actual, loaded)
        assert rows and all(row['equal'] for row in rows)
        loaded['model']['weight'][0, 0] += 1
        assert any(not row['equal'] for row in compare_values(actual, loaded))
        assert not compare_values(torch.zeros(1), torch.zeros(1, dtype=torch.bfloat16))[0]['equal']
        assert not compare_values(torch.tensor([float('nan')]), torch.tensor([float('nan')]))[0]['equal']
        assert not compare_values({'a': 1}, {})[0]['equal']
        common = load_common_state_dict(str(checkpoint))
        assert common['opt_param_scheduler']['num_steps'] == 16
        result = dict(status='ok', checks=9, cuda_device_count=0, torch=torch.__version__,
                      fixture_writer='Native MCore conversion/planner with Torch synchronous CPU writer',
                      scope='Native DCP loader and strict comparison with corruption controls; fixture-only writer adapter, no model or optimizer execution.')
        atomic(output / 'result.json', result)
        print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cpu-self-test', type=Path)
    args = parser.parse_args()
    if args.cpu_self_test:
        cpu_self_test(args.cpu_self_test)
    elif args.config:
        raise SystemExit(worker(json.loads(args.config.read_text())))
    else:
        parser.error('Either --config or --cpu-self-test is required.')
