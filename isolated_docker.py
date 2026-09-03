"""Isolated copy-on-write Docker daemon for this campaign, without network changes."""
import json
from pathlib import Path
import socket
import subprocess
import time

C = Path('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030')
LOCAL = Path('/tmp/miles-terminal-lego-20260903-2030')
LOCAL.mkdir(exist_ok=True)
SOCKET = 'unix://' + str(LOCAL/'docker.sock')
HOST = socket.gethostname()
config = LOCAL/'daemon.json'
config.write_text(json.dumps({'runtimes':{'nvidia':{'path':'nvidia-container-runtime','args':[]}}}))
argv = ['dockerd','--config-file='+str(config),'--host='+SOCKET,
        '--data-root='+str(LOCAL/'data'),'--exec-root='+str(LOCAL/'exec'),'--pidfile='+str(LOCAL/'dockerd.pid'),
        '--storage-driver=fuse-overlayfs','--bridge=none','--iptables=false','--ip6tables=false',
        '--ip-forward=false','--ip-masq=false','--userland-proxy=false']
log = (C/'preflight'/('isolated-dockerd-'+HOST+'.log')).open('w')
p = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(C/'preflight'/('isolated-dockerd-'+HOST+'.json')).write_text(json.dumps({'pid':p.pid,'argv':argv},indent=2))
for _ in range(60):
    result=subprocess.run(['docker','--host='+SOCKET,'info','--format','{{.Driver}}'],capture_output=True,text=True)
    if result.returncode==0:
        assert result.stdout.strip()=='fuse-overlayfs', result.stdout
        break
    if p.poll() is not None:
        raise RuntimeError('Isolated dockerd failed to start; see its log')
    time.sleep(1)
else:
    raise TimeoutError('Isolated dockerd readiness timed out')
manifest=json.loads((C/'preflight/image-manifest.json').read_text())
entry=next(x for x in manifest['manifests'] if x.get('platform',{}).get('architecture')=='amd64')
image='radixark/miles@'+entry['digest']
(C/'image-digest.json').write_text(json.dumps([image]))
with (C/'preflight'/('isolated-pull-'+HOST+'.log')).open('w') as output:
    result=subprocess.run(['docker','--host='+SOCKET,'pull',image],stdout=output,stderr=subprocess.STDOUT,timeout=1800)
if result.returncode:
    raise RuntimeError('Isolated image pull failed')
result=subprocess.run(['docker','--host='+SOCKET,'image','inspect',image],capture_output=True,text=True,check=True)
(C/'preflight'/('image-'+HOST+'.json')).write_text(result.stdout)
print('Isolated image ready',HOST,image,flush=True)
