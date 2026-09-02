"""Bounded in-allocation CUDA/NCCL gate using the actual training container."""
import ctypes
import datetime
import os
from pathlib import Path
import socket
import time

from evidence import Run, atomic, metric
from fabric_probe import active_training_ports


def verify_rdma():
    ports = active_training_ports()
    expected = {hca for hca, _ in ports}
    if os.environ.get('NCCL_NET') != 'IB':
        raise RuntimeError('Container preflight must fail closed on non-IB transport.')
    requested = '=' + ','.join(hca + ':' + port for hca, port in ports)
    if os.environ.get('NCCL_IB_HCA') != requested:
        raise RuntimeError('NCCL HCA selection differs from the eight active400G rails.')
    lib = ctypes.CDLL('libibverbs.so.1', use_errno=True)
    lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    lib.ibv_get_device_name.restype = ctypes.c_char_p
    lib.ibv_open_device.argtypes = [ctypes.c_void_p]
    lib.ibv_open_device.restype = ctypes.c_void_p
    lib.ibv_close_device.argtypes = [ctypes.c_void_p]
    lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    count = ctypes.c_int()
    devices = lib.ibv_get_device_list(ctypes.byref(count))
    opened = set()
    try:
        for index in range(count.value):
            name = lib.ibv_get_device_name(devices[index]).decode()
            if name not in expected:
                continue
            context = lib.ibv_open_device(devices[index])
            if not context:
                raise RuntimeError(f'Cannot open required RDMA device {name}: errno{ctypes.get_errno()}')
            lib.ibv_close_device(context)
            opened.add(name)
    finally:
        if devices:
            lib.ibv_free_device_list(devices)
    if opened != expected:
        raise RuntimeError(f'Container verbs devices missing: {sorted(expected - opened)}')
    return ports


def main():
    import torch
    import torch.distributed as dist

    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 32 or torch.cuda.device_count() != 8:
        raise RuntimeError('Fabric qualification requires four whole8-GPU nodes.')
    root = Path('/run-artifacts')
    label = os.environ['PTX_FABRIC_LABEL']
    phase = Run(root).phase(f'01-container-fabric-{label}-rank{rank}')
    results = []
    try:
        ports = verify_rdma()
        torch.cuda.set_device(local)
        dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120), device_id=torch.device('cuda', local))
        groups = [dist.new_group(list(range(base, base + 8)), backend='nccl') for base in range(0, 32, 8)]
        group = groups[rank // 8]
        for size in (1024**2, 16 * 1024**2):
            tensor = torch.full((size // 4,), float(rank + 1), device='cuda', dtype=torch.float32)
            torch.cuda.synchronize()
            start = time.monotonic()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            if not torch.all(tensor == world * (world + 1) / 2).item():
                raise RuntimeError('In-container all-reduce returned incorrect values.')
            results.append(metric('all_reduce_smoke_latency', elapsed, 's', socket.gethostname(), bytes=size, rank=rank))
        send = torch.full((8 * 32768,), float(rank), device='cuda')
        receive = torch.empty_like(send)
        dist.all_to_all_single(receive, send, group=group)
        expected = torch.arange(rank // 8 * 8, rank // 8 * 8 + 8, device='cuda').repeat_interleave(32768)
        if not torch.equal(receive, expected):
            raise RuntimeError('Node-local EP8 all-to-all smoke returned incorrect values.')
        dist.barrier()
        phase.finish('ok', results=results, metadata={'rank':rank, 'hostname':socket.gethostname(), 'rails':ports,
            'nccl_net':os.environ['NCCL_NET'], 'nccl_ib_hca':os.environ['NCCL_IB_HCA'], 'slurm_job_id':os.environ['SLURM_JOB_ID'],
            'scope':'Two cold all-reduce sizes across32 GPUs and one node-local EP8 all-to-all correctness case. Not a throughput benchmark.'}, refresh=False)
    except Exception as exc:
        phase.finish('fail', failure_summary=str(exc), refresh=False)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
