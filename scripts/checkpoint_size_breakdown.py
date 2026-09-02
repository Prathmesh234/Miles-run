"""Attribute a completed distributed checkpoint's bytes without loading tensors."""
import argparse
import inspect
import json

from evidence import Run, atomic


def breakdown(path):
    import collections
    import hashlib
    import math
    import pickle
    from pathlib import Path
    import torch

    path = Path(path)
    metadata_path = path / '.metadata'
    with metadata_path.open('rb') as handle:
        metadata = pickle.load(handle)
    categories = collections.defaultdict(lambda: {'logical_tensor_bytes': 0, 'tensor_count': 0, 'elements': 0})
    dtypes = collections.Counter()
    for name, value in metadata.state_dict_metadata.items():
        if not hasattr(value, 'size') or not hasattr(value, 'properties'):
            continue
        if name.startswith('chained_'):
            category = {'exp_avg': 'adam_first_moment', 'exp_avg_sq': 'adam_second_moment',
                        'param': 'fp32_master_weights'}.get(name.rsplit('.', 1)[-1], 'other_optimizer_tensors')
        else:
            category = 'model_weights'
        elements = math.prod(value.size)
        size = elements * torch.empty(0, dtype=value.properties.dtype).element_size()
        categories[category]['logical_tensor_bytes'] += size
        categories[category]['tensor_count'] += 1
        categories[category]['elements'] += elements
        dtypes[str(value.properties.dtype)] += size
    files = [p for p in path.rglob('*') if p.is_file()]
    disk_bytes = sum(p.stat().st_size for p in files)
    logical_bytes = sum(row['logical_tensor_bytes'] for row in categories.values())
    return {'schema_version': 1, 'categories': dict(categories), 'logical_tensor_bytes_by_dtype': dict(dtypes),
            'disk_bytes': disk_bytes, 'logical_tensor_bytes': logical_bytes,
            'serialization_metadata_and_padding_bytes': disk_bytes - logical_bytes,
            'files': len(files), 'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            'cuda_device_count': torch.cuda.device_count(),
            'scope': 'Read-only logical tensor sizes from trusted run-owned DCP metadata plus physical file sizes. Not payload checksum or resume verification.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--training-attempt', required=True, type=int)
    parser.add_argument('--checkpoint-step', required=True, type=int)
    args = parser.parse_args()
    run = Run(args.run_dir)
    label = f'checkpoint-size-v{args.training_attempt}-step{args.checkpoint_step}'
    phase = run.phase('02-' + label)
    inner = inspect.getsource(breakdown) + '\nimport json\nprint(json.dumps(breakdown("/checkpoint")))\n'
    outer = '''import pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); attempt=sys.argv[4];step=int(sys.argv[5]);code=root/('provenance/sync-grpo-code-v'+attempt)
sys.path.insert(0,str(code));from enroot_run_config import prepare
runtime=root/('images/'+sys.argv[3]);runtime.mkdir(exist_ok=False)
env=prepare(runtime);env['NVIDIA_VISIBLE_DEVICES']='void'
checkpoint=root/('training/sync-grpo-v'+attempt+'/checkpoints/iter_'+f'{step:07d}')
cmd=['enroot','start','--pid','--ipc','--rw','--env','NVIDIA_VISIBLE_DEVICES=void','--env','PYTHONDONTWRITEBYTECODE=1','--mount',str(checkpoint)+':/checkpoint:none:bind,ro,x-create=dir',str(root/'images/enroot-import-v2/miles-amd64.sqsh'),'python3','-c',sys.argv[2]]
raise SystemExit(subprocess.call(cmd,env=env))
'''
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', outer, remote, inner, label,
        str(args.training_attempt), str(args.checkpoint_step)], timeout=120)
    result = json.loads(out.splitlines()[-1]) if not rc else {'error': 'Metadata size inspection failed; inspect stderr.'}
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if rc else 'ok', metadata=result, failure_summary=result.get('error'), refresh=False)
    print(json.dumps(result))
    return int(bool(rc))


if __name__ == '__main__':
    raise SystemExit(main())
