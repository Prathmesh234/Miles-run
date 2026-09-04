"""Apply the recorded Miles patch to a job-local driver; upstream stays read-only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

code = Path(os.environ.get('MILES_PATCH_OUTPUT_DIR',str(Path(__file__).resolve().parent)))
upstream = Path(os.environ.get('MILES_SOURCE_ROOT','/campaign/miles'))
assert hashlib.sha256((upstream/'train.py').read_bytes()).hexdigest() == '34693c11173e69a402228bd1d732122eef9fdb6e5d105660dfdd5eae0b1b954c'
stage = code/'driver-patch'
stage.mkdir(exist_ok=False)
shutil.copy2(upstream/'train.py',stage/'train.py')
patch = Path(__file__).resolve().parent/'ppo-resident-broadcast.patch'
subprocess.run(['git','apply','--check',str(patch)],cwd=stage,check=True)
subprocess.run(['git','apply',str(patch)],cwd=stage,check=True)
destination = code/'train_ppo.py'
assert not destination.exists()
(stage/'train.py').replace(destination)
compile(destination.read_text(),str(destination),'exec')
manifest = {'base_miles_sha':'70b89e11770fc9bac984e22cfff89c51cca44203',
            'patch_commits':['6ef4ae5581fa9e4558df22f39e92a24b67687a4b','90c5eb3f272b736f5c91d97fe8fcb6f03605e277'], 'scope':'job-local synchronous driver only; raw baseline is read-only',
            'reason':'Keep the disjoint PPO actor parameter backup; restore before broadcasting live tensors and offload afterward',
            'base_driver_sha256':hashlib.sha256((upstream/'train.py').read_bytes()).hexdigest(),
            'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),
            'patched_driver_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
(code.parent/'ppo-driver-patch.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
