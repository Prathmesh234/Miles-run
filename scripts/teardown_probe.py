"""Bounded CUDA/NCCL teardown control, optionally with pinned host buffers.

No model, checkpoint, optimizer, device reset, or allocator configuration change.
The owning node runner keeps telemetry live through exit and for 15 seconds after.
"""
import datetime
import json
import os
from pathlib import Path
import socket
import time


def available_host_bytes(meminfo, cgroup_root=Path('/sys/fs/cgroup')):
    """Respect both host availability and finite enclosing cgroup limits."""
    fields = dict(line.split(':', 1) for line in meminfo.splitlines() if ':' in line)
    available = int(fields['MemAvailable'].split()[0]) * 1024
    # Enroot inherits the Slurm pod's memory cgroup. Check v2 and v1 mounts;
    # absence of a finite cgroup cap does not replace the host capacity guard.
    for maximum, current in [('memory.max', 'memory.current'),
                              ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes')]:
        limit_path, current_path = cgroup_root / maximum, cgroup_root / current
        if limit_path.exists():
            limit = limit_path.read_text().strip()
            if limit != 'max':
                available = min(available, int(limit) - int(current_path.read_text()))
    return available


def host_allocation_guard(available, pinned_bytes_per_rank, local_world_size):
    required = local_world_size * pinned_bytes_per_rank + 128 * 1024**3
    if available < required:
        raise RuntimeError(f'Host memory reserve: need {required} bytes, available {available}.')
    return required


def emit(event, rank, **values):
    print(json.dumps(dict(event=event, rank=rank, hostname=socket.gethostname(),
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(), monotonic_s=time.monotonic(),
        **values)), flush=True)


def release_pinned_buffers(torch, buffers):
    """Release only this probe's pinned tensors, then prove allocator accounting."""
    torch.cuda.synchronize()
    keys = ('active_bytes.current', 'allocated_bytes.current')
    before = {k: v for k, v in torch.cuda.memory.host_memory_stats().items() if k in keys}
    expected = sum(t.numel() * t.element_size() for t in buffers)
    if before['active_bytes.current'] < expected:
        raise RuntimeError('Pinned allocator accounting is smaller than the live buffers.')
    started = time.monotonic()
    buffers.clear()
    # Private API is used explicitly in upstream PyTorch's CUDA tests. Missing
    # support or unexpected accounting is a failed diagnostic, never a fallback.
    torch._C._host_emptyCache()
    elapsed = time.monotonic() - started
    after = {k: v for k, v in torch.cuda.memory.host_memory_stats().items() if k in keys}
    for key in keys:
        if after[key] > before[key] - expected:
            raise RuntimeError('Pinned buffers were not fully released: ' + key)
    return dict(expected_released_bytes=expected, duration_s=elapsed, before=before, after=after)


def main():
    # Torch is a pinned native runtime dependency, not needed by guard unit tests.
    import torch
    import torch.distributed as dist

    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['LOCAL_WORLD_SIZE']) == 8
    torch.cuda.set_device(rank)
    control = None
    if os.environ.get('PTX_PROBE_NCCL') == '1':
        dist.init_process_group(backend='nccl', device_id=torch.device('cuda', rank))
        control = torch.ones(1024, device='cuda')
        dist.all_reduce(control)
        assert torch.all(control == 8).item()
    free, total = torch.cuda.mem_get_info()
    assert free >= 96 * 1024**3, 'Require 96 GiB free HBM before bounded 64 GiB allocation'
    pinned_gib = int(os.environ.get('PTX_PROBE_PINNED_GIB', '0'))
    assert pinned_gib in (0, 24)
    release_pinned = os.environ.get('PTX_PROBE_RELEASE_PINNED', '0') == '1'
    assert not release_pinned or pinned_gib == 24
    if pinned_gib:
        available = available_host_bytes(Path('/proc/meminfo').read_text())
        required = host_allocation_guard(available, pinned_gib * 1024**3, 8)
        emit('host_capacity_guard', rank, available_bytes=available, required_bytes=required)
    emit('before_allocate', rank, free_bytes=free, total_bytes=total, torch=torch.__version__)
    chunk_mib = int(os.environ.get('PTX_PROBE_CHUNK_MIB', '65536'))
    assert chunk_mib in (16, 65536)
    payload = [torch.empty(chunk_mib * 1024**2, dtype=torch.uint8, device='cuda')
               for _ in range(65536 // chunk_mib)]
    for tensor in payload:
        tensor.zero_()
    torch.cuda.synchronize()
    assert payload[0][0].item() == 0 and payload[-1][-1].item() == 0
    # Power-of-two chunks avoid rounding the 24 GiB budget up in the host cache.
    pinned = []
    for index in range(pinned_gib * 128):
        tensor = torch.empty(8 * 1024**2, dtype=torch.uint8, device='cpu', pin_memory=True)
        tensor.copy_(payload[index % len(payload)][:tensor.numel()], non_blocking=True)
        pinned.append(tensor)
    del tensor  # Do not leave an extra reference to the last pinned tensor.
    torch.cuda.synchronize()
    if pinned:
        assert all(t.is_pinned() for t in pinned)
        assert pinned[0][0].item() == 0 and pinned[-1][-1].item() == 0
    emit('allocated', rank, allocated_bytes=torch.cuda.memory_allocated(),
         allocation_count=len(payload), chunk_mib=chunk_mib,
         pinned_bytes=sum(t.numel() for t in pinned), pinned_allocation_count=len(pinned))
    time.sleep(10)
    if release_pinned:
        emit('before_pinned_release', rank)
        released = release_pinned_buffers(torch, pinned)
        emit('pinned_released', rank, **released)
    emit('exit_with_live_context', rank, pinned_bytes=sum(t.numel() for t in pinned))
    # Deliberately retain the buffers and communicator through ordinary process
    # exit, matching the preceding diagnostic. Explicit cleanup is a separate test.
    return payload, pinned, control


if __name__ == '__main__':
    _live_resources = main()
