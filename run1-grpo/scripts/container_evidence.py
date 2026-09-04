"""Preserve package versions and Ray logs from this job's isolated container."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

out = Path(sys.argv[1]) / "infra" / socket.gethostname()
out.mkdir(exist_ok=True, parents=True)
versions = {d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
(out/"python-packages.json").write_text(json.dumps(versions,indent=2))
commands = [("nvidia",["nvidia-smi","-q"]),("nvcc",["nvcc","--version"])]
if Path('/tmp/ray/session_latest').exists():
    commands.append(("ray-status",["ray","status"]))
for label,cmd in commands:
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        (out/(label+".txt")).write_text(p.stdout+p.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:
        (out/(label+".txt")).write_text(str(e))
src=Path("/tmp/ray/session_latest/logs")
if src.exists():
    shutil.copytree(src,out/"ray-logs",dirs_exist_ok=True)
