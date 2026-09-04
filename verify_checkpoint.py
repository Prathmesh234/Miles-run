"""Structural checkpoint validation plus actual CPU reads of small tensors."""
import datetime
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
parser.add_argument('--critic',action='store_true')
parser.add_argument('--base',type=Path)
args=parser.parse_args()
root=args.root
folder='checkpoints_critic' if args.critic else 'checkpoints'
iteration=int((root/folder/'latest_checkpointed_iteration.txt').read_text())
assert iteration==1, 'Miles saves the zero-based rollout ID: 1 denotes the second update'
checkpoint=root/folder/f'iter_{iteration:07d}'
reader=dcp.FileSystemReader(str(checkpoint))
metadata=reader.read_metadata()
errors=[]
files={}
for index,location in metadata.storage_data.items():
    path=checkpoint/location.relative_path
    if not path.is_file() or location.offset+location.length>path.stat().st_size:
        errors.append(str(index))
    files[str(path.relative_to(checkpoint))]=path.stat().st_size if path.exists() else None
tensors={k:v for k,v in metadata.state_dict_metadata.items() if isinstance(v,TensorStorageMetadata)}
for key,value in tensors.items():
    if sum(math.prod(c.sizes) for c in value.chunks)!=math.prod(value.size):
        errors.append('chunk-volume:'+key)
assert not errors,errors[:10]
small=[(k,v) for k,v in tensors.items() if 0<math.prod(v.size)<=8192]
small.sort(key=lambda pair:(pair[1].properties.dtype!=torch.float32,pair[0]))
selected={key:torch.empty(value.size,dtype=value.properties.dtype) for key,value in small[:6]}
assert len(selected)>=3
dcp.load(selected,storage_reader=reader)
assert all(torch.isfinite(x).all().item() for x in selected.values())
comparison={}
if args.base:
    base_reader=dcp.FileSystemReader(str(args.base))
    base_metadata=base_reader.read_metadata().state_dict_metadata
    initial={k:torch.empty_like(v) for k,v in selected.items() if k in base_metadata and list(base_metadata[k].size)==list(v.shape)}
    dcp.load(initial,storage_reader=base_reader)
    for k,v in initial.items():
        delta=selected[k].float()-v.float()
        comparison[k]={'changed_elements':int((delta!=0).sum()),'max_abs_change':float(delta.abs().max()),
                       'base_sha256':hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'role':'critic' if args.critic else 'actor',
        'checkpoint':str(checkpoint),'zero_based_rollout_id':iteration,'completed_optimizer_updates':2,
        'metadata_entries':len(metadata.state_dict_metadata),'tensor_entries':len(tensors),
        'storage_entries_checked':len(metadata.storage_data),'shard_files':files,
        'tensor_elements':sum(math.prod(v.size) for v in tensors.values()),
        'sampled_tensor_reads':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':True,
            'sha256':hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()} for k,v in selected.items()},
        'selected_tensors_vs_base':comparison,
        'verification':'All referenced byte ranges exist and tensor chunk volumes match; up to six small tensors loaded on CPU. Not a full distributed resume test.'}
out=root/('critic-checkpoint-verification.json' if args.critic else 'checkpoint-verification.json')
assert not out.exists()
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
