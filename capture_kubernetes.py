"""Capture assigned worker placement and resources without credential or env dumps."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

KUBE = '/Users/prathmeshbhatt/.kube/vultr-vke.yaml'


def get(*args):
    return json.loads(subprocess.check_output(['kubectl','--kubeconfig='+KUBE,'--request-timeout=20s',*args,'-o','json']))


pods = get('get','pods','-n','slurm')['items']
workers = [p for p in pods if p['metadata']['name'].startswith('slurm-worker-gpu-nodes-')]
node_names = {p['spec']['nodeName'] for p in workers}
nodes = get('get','nodes')['items']
result = {'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':[],'nodes':[]}
for p in workers:
    spec = p['spec']
    result['workers'].append({
        'name':p['metadata']['name'], 'uid':p['metadata']['uid'],
        'node_name':spec['nodeName'], 'hostNetwork':spec.get('hostNetwork',False),
        'hostPID':spec.get('hostPID',False),'hostIPC':spec.get('hostIPC',False),
        'podIP':p['status'].get('podIP'), 'hostIP':p['status'].get('hostIP'),
        'containers':[{k:c.get(k) for k in ['name','image','resources','securityContext','volumeMounts']} for c in spec['containers']],
        'container_status':p['status'].get('containerStatuses'),
        'conditions':p['status'].get('conditions'),
        'volumes':[{k:v for k,v in vol.items() if k not in ['secret','projected']} for vol in spec.get('volumes',[])]})
for n in nodes:
    if n['metadata']['name'] not in node_names:
        continue
    result['nodes'].append({'name':n['metadata']['name'],'labels':n['metadata']['labels'],
                            'spec':{k:n['spec'].get(k) for k in ['podCIDR','podCIDRs','taints','unschedulable']},
                            'status':n['status']})
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('kubernetes-infra.json')
if out.exists():
    raise FileExistsError(f'Preserving earlier snapshot: {out}')
out.write_text(json.dumps(result,indent=2)+'\n')
print(f'{len(workers)} assigned workers, {len(result["nodes"])} backing nodes captured in {out}')
