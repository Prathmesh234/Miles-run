"""Short CPU-only rendezvous/all-reduce check inside the isolated GPU image."""
import datetime
import json
import os

import torch
import torch.distributed as dist


if __name__ == '__main__':
    if os.environ.get('MASTER_ADDR') != '127.0.0.1':
        raise ValueError('Torchrun did not preserve the explicit loopback address.')
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=30))
    rank, size = dist.get_rank(), dist.get_world_size()
    try:
        value = torch.tensor(rank + 1, dtype=torch.int64)
        dist.all_reduce(value)
        if value.item() != size * (size + 1) // 2:
            raise ValueError('CPU rendezvous all-reduce result mismatch.')
        print(json.dumps({'rank': rank, 'world_size': size, 'master_addr': os.environ['MASTER_ADDR'],
                          'sum': value.item(), 'scope': 'CPU rendezvous validation, not GPU collectives.'}), flush=True)
    finally:
        dist.destroy_process_group()
