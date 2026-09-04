"""Patch only this job's disposable image layer; save before/after evidence."""
import hashlib
import json
import os
from pathlib import Path
import socket

source = Path('/sgl-workspace/sglang/python/sglang/srt/models/qwen2_moe.py')
before = source.read_text()
needle = '        self.gate = ReplicatedLinear(\n'
assert before.count(needle) == 1, 'Pinned Qwen router source contract changed'
after = before.replace(needle, '        from sglang_precision import FP32RouterLinear\n\n        self.gate = FP32RouterLinear(\n')
out = Path(os.environ['MILES_RUN_DIR'])/'infra'/('precision-patch-'+socket.gethostname())
out.mkdir(exist_ok=False)
(out/'qwen2_moe.before.py').write_text(before)
(out/'qwen2_moe.after.py').write_text(after)
(out/'manifest.json').write_text(json.dumps({
    'path':str(source), 'before_sha256':hashlib.sha256(before.encode()).hexdigest(),
    'after_sha256':hashlib.sha256(after.encode()).hexdigest(),
    'scope':'isolated job container only; BF16 weights/input, FP32 router logits',
}, indent=2))
source.write_text(after)
print('Installed job-local FP32 router output patch', flush=True)
