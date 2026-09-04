"""Observational metrics. Host wall time is NOT PCIe time or wire bandwidth.

CPU Adam state residency and parameter/gradient offloading are separate metrics.
No tensor values are copied to CPU, no state_dict/gathers, no added CUDA barriers.
"""
import functools
import os
from pathlib import Path
import time

from async_runtime import record


def memory():
    import torch
    result = {'rss_bytes': int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')}
    if torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info()
        result.update(cuda_free_bytes=free, cuda_total_bytes=total,
                      cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved())
    return result


def optimizer_inventory(optimizer):
    """Count unique referenced storages, not physical resident TMS pages."""
    import torch
    seen_objects, seen_storages, classes = set(), set(), set()
    totals = {}

    def tensor_walk(value, kind):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key not in seen_storages:
                seen_storages.add(key)
                label = kind+':'+str(value.device)
                totals[label] = totals.get(label, 0)+storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                tensor_walk(item, kind)
        elif isinstance(value, (list, tuple)):
            for item in value:
                tensor_walk(item, kind)

    def visit(obj):
        if obj is None or id(obj) in seen_objects:
            return
        seen_objects.add(id(obj))
        if isinstance(obj, (list, tuple)):
            for child in obj:
                visit(child)
            return
        if isinstance(obj, dict):
            for child in obj.values():
                visit(child)
            return
        classes.add(type(obj).__module__+'.'+type(obj).__name__)
        attrs = vars(obj) if hasattr(obj, '__dict__') else {}
        if 'state' in attrs:
            tensor_walk(attrs['state'], 'state')
        if 'param_groups' in attrs:
            for group in attrs['param_groups']:
                tensor_walk(group.get('params', []), 'parameter')
        # Only optimizer-owned children, never arbitrary model graphs.
        for key, child in attrs.items():
            if 'optim' in key and (hasattr(child, '__dict__') or isinstance(child, (list, tuple, dict))):
                visit(child)
    visit(optimizer)
    return {'unique_storage_bytes': totals, 'optimizer_classes': sorted(classes),
            'interpretation': 'referenced storage sizes by device, not transfer bytes or resident pages'}


def install(args):
    from miles.backends.megatron_utils.actor import MegatronTrainRayActor
    cls = MegatronTrainRayActor
    if getattr(cls, '_async_metrics_installed', False):
        return
    for name in ('sleep', 'wake_up', 'update_weights'):
        original = getattr(cls, name)

        def wrap(original, name):
            @functools.wraps(original)
            def observed(self, *a, **kw):
                fields = {'role': self.role, 'rank': os.environ.get('RANK'),
                          'memory_before': memory()}
                start = time.perf_counter()
                ok = False
                try:
                    result = original(self, *a, **kw)
                    ok = True
                    return result
                finally:
                    elapsed = time.perf_counter()-start
                    record('rank_'+name, 'end', elapsed_seconds=elapsed, ok=ok,
                           memory_after=memory(), **fields)
            return observed
        setattr(cls, name, wrap(original, name))
    cls._async_metrics_installed = True
    record('instrumentation', 'installed', optimizer_cpu_offload=args.optimizer_cpu_offload,
           overlap_cpu_optimizer=args.overlap_cpu_optimizer_d2h_h2d)


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    optimizer._async_metric_step = (rollout_id, step_id)
    if getattr(optimizer, '_async_metric_wrapped', False):
        return
    original = optimizer.step

    @functools.wraps(original)
    def step(*a, **kw):
        rollout, batch = optimizer._async_metric_step
        inv_start = time.perf_counter()
        before = optimizer_inventory(optimizer)
        inventory_before_seconds = time.perf_counter()-inv_start
        mem_before = memory()
        start = time.perf_counter()
        ok = False
        try:
            result = original(*a, **kw)
            ok = True
            return result
        finally:
            elapsed = time.perf_counter()-start
            inv_start = time.perf_counter()
            after = optimizer_inventory(optimizer)
            inv_seconds = time.perf_counter()-inv_start
            record('optimizer_step', 'end', rollout_id=rollout, step_id=batch,
                   rank=os.environ.get('RANK'), role=getattr(args, 'role', 'unknown'),
                   elapsed_seconds=elapsed, ok=ok, before=before, after=after,
                   memory_before=mem_before, memory_after=memory(),
                   inventory_overhead_seconds=inventory_before_seconds+inv_seconds)
    optimizer.step = step
    optimizer._async_metric_wrapped = True


def tis(args, **kwargs):
    """Native clamp [0,2], with detached behavior correction and extra scalars."""
    import torch
    from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function
    assert kwargs['train_log_probs'] is not None and kwargs['rollout_log_probs'] is not None
    assert all(not x.requires_grad for x in kwargs['train_log_probs'])
    loss, masks, metrics = vanilla_tis_function(args, **kwargs)
    ratio = metrics['tis']
    weights = ratio.clamp(args.tis_clip_low, args.tis_clip)
    metrics['tis_weight'] = weights
    metrics['tis_weight_squared'] = weights.square()
    metrics['tis_upper_clipfrac'] = (ratio > args.tis_clip).float()
    # Miles applies its existing valid-token reducer to these vectors.
    return loss, masks, metrics
