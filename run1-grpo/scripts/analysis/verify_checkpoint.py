"""Structural checkpoint validation plus actual CPU reads of small tensors."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

root=Path(sys.argv[1])
iteration=int((root/'checkpoints/latest_checkpointed_iteration.txt').read_text())
assert iteration==1, 'Miles saves the zero-based rollout ID: 1 denotes the second update'
checkpoint=root/'checkpoints'/f'iter_{iteration:07d}'
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
selected={key:torch.empty(value.size,dtype=value.properties.dtype) for key,value in list(
    (k,v) for k,v in tensors.items() if 0<math.prod(v.size)<=8192)[:3]}
assert len(selected)==3
dcp.load(selected,storage_reader=reader)
assert all(torch.isfinite(x).all().item() for x in selected.values())
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'checkpoint':str(checkpoint),'zero_based_rollout_id':iteration,'completed_optimizer_updates':2,
        'metadata_entries':len(metadata.state_dict_metadata),'tensor_entries':len(tensors),
        'storage_entries_checked':len(metadata.storage_data),'shard_files':files,
        'tensor_elements':sum(math.prod(v.size) for v in tensors.values()),
        'sampled_tensor_reads':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':True,
            'sha256':hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()} for k,v in selected.items()},
        'verification':'All referenced byte ranges exist and tensor chunk volumes match; three small tensors loaded on CPU. Not a full distributed resume test.'}
out=root/'checkpoint-verification.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
