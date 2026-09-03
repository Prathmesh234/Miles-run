"""Wait for the already-running pull, then validate CPU-only in the pinned image."""
import json
import os
from pathlib import Path
import subprocess
import time

C = Path('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030')
MODEL = '/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00/model-fetch/models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0'
DOCKER = ['docker','--host=unix:///tmp/miles-terminal-lego-20260903-2030/docker.sock']
REFERENCE = json.loads((C/'image-digest.json').read_text())[0]

for _ in range(180):
    result = subprocess.run([*DOCKER,'image','inspect',REFERENCE],capture_output=True,text=True)
    if result.returncode == 0:
        break
    time.sleep(10)
else:
    raise TimeoutError('Image pull did not complete within the preflight wait budget')
image = json.loads(result.stdout)[0]
(C/'image-digest.json').write_text(json.dumps(image['RepoDigests']))
(C/'preflight/image-gpu-nodes-0.json').write_text(result.stdout)
cmd = [*DOCKER,'run','--name','miles-terminal-lego-cpu-preflight','--runtime=nvidia',
       '-e','NVIDIA_VISIBLE_DEVICES=none','-e','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
       '-e','PYTHONPATH=/campaign/miles:/root/Megatron-LM:/campaign/code',
       '-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1',
       '-v',str(C)+':/campaign','-v',MODEL+':'+MODEL+':ro','-w','/campaign/miles',
       image['RepoDigests'][0],'python','/campaign/code/training_entry.py','validate']
(C/'preflight/cpu-preflight-argv.json').write_text(json.dumps(cmd,indent=2))
result = subprocess.run(cmd,timeout=240)
(C/'preflight/cpu-preflight-exit-code.txt').write_text(str(result.returncode))
raise SystemExit(result.returncode)
