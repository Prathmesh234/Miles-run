"""Job-local driver timing; no changes to optimization or distributed barriers."""
import json
import os
from pathlib import Path
import socket
import time


def record(operation, phase, **fields):
    root = Path(os.environ['MILES_RUN_DIR'])/'infra'
    root.mkdir(parents=True, exist_ok=True)
    row = dict(time=time.time(), host=socket.gethostname(), pid=os.getpid(),
               operation=operation, phase=phase, **fields)
    with (root/f'async-events-{socket.gethostname()}-{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')


async def measured(operation, awaitable, **fields):
    start = time.perf_counter()
    record(operation, 'start', **fields)
    ok = False
    try:
        result = await awaitable
        ok = True
        return result
    finally:
        record(operation, 'end', elapsed_seconds=time.perf_counter()-start, ok=ok, **fields)


def preserve_actor_backup(args):
    if args.use_critic and args.offload_train and not args.colocate:
        args.disable_param_buffers_cpu_backup = False


async def update_actor_weights(args, actor_model, **kwargs):
    restore = args.use_critic and args.offload_train and not args.colocate
    fields = {'rollout_id': kwargs.get('rollout_id', -1)}
    if restore:
        await measured('broadcast_actor_onload', actor_model.onload(), **fields)
    try:
        await measured('weight_transfer', actor_model.update_weights(**kwargs), **fields)
    finally:
        if restore:
            await measured('broadcast_actor_offload', actor_model.offload(), **fields)
