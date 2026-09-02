"""Inspect host Lustre statistics through read-only mounts in a run-owned pod."""
import argparse
import json
from pathlib import Path
import sys

from evidence import Run, atomic


IMAGE = 'python@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49'
PROBE = r'''
import json,pathlib,time,socket
paths=[pathlib.Path('/host/proc-fs/lustre/llite'),pathlib.Path('/host/sys/kernel/debug/lustre/llite')]
result={'time_unix_s':time.time(),'hostname':socket.gethostname(),'roots':[],'files':[],'errors':[]}
for root in paths:
 try:
  result['roots'].append({'path':str(root),'exists':root.exists(),'entries':[p.name for p in root.iterdir()] if root.exists() else []})
  for name in ('stats','read_ahead_stats'):
   for path in root.glob('*/'+name):
    text=path.read_text()
    if len(text)>1024*1024: raise ValueError('Unexpectedly large statistics file')
    result['files'].append({'path':str(path),'text':text})
 except (OSError,ValueError) as exc:
  result['errors'].append({'path':str(root),'error':str(exc)})
print(json.dumps(result),flush=True)
'''


def pod_manifest(name, node, run_id):
    return {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': name, 'namespace': 'slurm',
        'labels': {'app': 'posttrainingx-diagnostic', 'posttrainingx-run': run_id}},
        'spec': {'nodeName': node, 'restartPolicy': 'Never', 'activeDeadlineSeconds': 120,
            'automountServiceAccountToken': False, 'terminationGracePeriodSeconds': 5,
            'securityContext': {'seccompProfile': {'type': 'RuntimeDefault'}},
            'containers': [{'name': 'probe', 'image': IMAGE, 'imagePullPolicy': 'IfNotPresent',
                'command': ['python3', '-c', PROBE],
                'resources': {'requests': {'cpu': '100m', 'memory': '64Mi'}, 'limits': {'cpu': '1', 'memory': '128Mi'}},
                'securityContext': {'allowPrivilegeEscalation': False, 'readOnlyRootFilesystem': True,
                                    'capabilities': {'drop': ['ALL']}},
                'volumeMounts': [{'name': 'host-proc-fs', 'mountPath': '/host/proc-fs', 'readOnly': True},
                                 {'name': 'host-sys', 'mountPath': '/host/sys', 'readOnly': True}]}],
            'volumes': [{'name': 'host-proc-fs', 'hostPath': {'path': '/proc/fs', 'type': 'Directory'}},
                        {'name': 'host-sys', 'hostPath': {'path': '/sys', 'type': 'Directory'}}]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('01-host-lustre-readonly-probe')
    name = 'ptx-lustre-' + run.root.name
    manifest = pod_manifest(name, 'b200-nodepool-ac23753e6cfa', run.root.name)
    atomic(phase.path / 'pod.json', manifest)
    k = ['kubectl', '--kubeconfig', str(Path(args.kubeconfig).resolve()), '-n', 'slurm']
    rc, _, _ = phase.command(k + ['create', '-f', '-'], stdin=json.dumps(manifest), timeout=45)
    if rc:
        phase.finish('fail', failure_summary='Diagnostic pod creation failed or was ambiguous; inspect by exact name before retry.')
        return 1
    rc, _, _ = phase.command(k + ['wait', '--for=jsonpath={.status.phase}=Succeeded', 'pod/' + name, '--timeout=130s'], timeout=140)
    log_rc, out, _ = phase.command(k + ['logs', name], timeout=30)
    phase.command(k + ['get', 'pod', name, '-o', 'json'], timeout=30)
    try:
        data = json.loads(out) if not log_rc else {}
        atomic(phase.path / 'host-statistics.json', data)
        okay = not rc and not log_rc and bool(data.get('files')) and not data.get('errors')
    except ValueError:
        okay = False
    phase.finish('ok' if okay else 'fail',
        failure_summary=None if okay else 'Host-mounted Lustre statistics were absent, inaccessible, or diagnostic pod failed.',
        metadata={'pod': name, 'node': manifest['spec']['nodeName'], 'image': IMAGE,
                  'scope': 'Read-only host statistics discovery, no load or host/cluster configuration changes.',
                  'artifacts': [str(phase.path.relative_to(run.root)) + '/pod.json',
                                str(phase.path.relative_to(run.root)) + '/host-statistics.json']})
    print(json.dumps({'pod': name, 'status': 'ok' if okay else 'fail'}))
    return int(not okay)


if __name__ == '__main__':
    sys.exit(main())
