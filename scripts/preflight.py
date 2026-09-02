"""Collect all four workers without allocation, load, or mutation on the cluster."""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse

from evidence import Run, atomic, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    k = ['kubectl', '--kubeconfig', str(Path(args.kubeconfig).resolve()), '--request-timeout=45s']
    phase = run.phase('01-readonly-preflight')
    code, text, err = phase.command(k + ['get', 'nodes', '-o', 'json'])
    if code:
        phase.finish('fail', failure_summary='The Kubernetes inventory could not be read.')
        return 1
    raw_nodes = json.loads(text)['items']
    nodes = [{'name': n['metadata']['name'], 'uid': n['metadata']['uid'], 'labels': n['metadata'].get('labels', {}),
              'capacity': n['status'].get('capacity'), 'allocatable': n['status'].get('allocatable'),
              'node_info': n['status'].get('nodeInfo'), 'conditions': n['status'].get('conditions')} for n in raw_nodes]
    # Node responses contain no credentials. Pod responses below are projected server-side.
    atomic(run.root / 'inventory/cluster.values.json', {'nodes': nodes})
    pod_template = '{range .items[*]}{.metadata.name}{"\\t"}{.spec.nodeName}{"\\t"}{.status.phase}{"\\t"}{.status.containerStatuses[*].imageID}{"\\n"}{end}'
    code, pods_text, err = phase.command(k + ['-n', 'slurm', 'get', 'pods', '-o', 'jsonpath=' + pod_template])
    if code:
        phase.finish('fail', failure_summary='The Slurm pod inventory could not be read.')
        return 1
    pods = [dict(zip(['pod', 'node', 'phase', 'image_id'], row.split('\t'))) for row in pods_text.splitlines()]
    workers = [p for p in pods if p['pod'].startswith('slurm-worker-gpu-nodes-')]
    atomic(run.root / 'inventory/scheduler.values.json', {'pods': pods})
    probe = (Path(__file__).parent / 'inventory_probe.py').read_text()
    atomic(run.root / 'provenance/inventory_probe.py', probe)
    # A distinct phase owns each worker so its command log is never shared between threads.
    def collect(pod):
        wp = run.phase('01-worker-' + pod['pod'].rsplit('-', 1)[-1])
        code, out, err = wp.command(k + ['-n', 'slurm', 'exec', '-i', pod['pod'], '--', 'python3', '-'], stdin=probe, timeout=240)
        if code:
            return pod, wp, None, 'The worker inventory probe did not complete.'
        try:
            data = json.loads(out)
        except ValueError:
            return pod, wp, None, 'The worker inventory probe did not return valid JSON.'
        atomic(run.root / 'inventory' / (pod['pod'] + '.raw.json'), data)
        return pod, wp, data, None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        collected = list(pool.map(collect, workers))
    all_gpu = []
    ib = []
    mounts = []
    findings = []
    if len(workers) != 4:
        findings.append('Expected four worker pods, found ' + str(len(workers)))
    for pod, wp, data, error in collected:
        if error:
            wp.finish('fail', failure_summary=error)
            findings.append(error)
            continue
        errors = []
        gpu = data['commands']['gpu_csv']
        rows = list(csv.reader(gpu['stdout'].splitlines(), skipinitialspace=True)) if gpu['exit_code'] == 0 else []
        if len(rows) != 8:
            errors.append(f"Physical GPU count is {len(rows)}, expected 8.")
        node = next(n for n in nodes if n['name'] == pod['node'])
        if int(node['allocatable'].get('nvidia.com/gpu', 0)) != 8:
            errors.append('Kubernetes allocatable GPU count differs from eight.')
        for row in rows:
            if len(row) != 8:
                errors.append('The nvidia-smi CSV schema differs from the expected schema.')
                continue
            all_gpu.append(dict(hostname=data['hostname'], kubernetes_node=pod['node'], index=int(row[0]),
                                uuid=row[1], pci_bdf=row[2], model=row[3], driver=row[4],
                                hbm_total_bytes=int(row[5])*1024**2, hbm_free_bytes=int(row[6])*1024**2,
                                hbm_used_bytes=int(row[7])*1024**2))
        active_rails = [h['name'] for h in data['hcas'] for p in h['ports']
                        if isinstance(p['state'], str) and p['state'].endswith('ACTIVE') and
                        isinstance(p['rate'], str) and p['rate'].startswith('400')]
        if len(active_rails) != 8:
            errors.append(f'Expected eight active 400G rails, found {len(active_rails)}.')
        ib.append({'hostname': data['hostname'], 'active_400g_rails': active_rails, 'devices': data['hcas']})
        mount = data['commands']['mount']
        if mount['exit_code'] == 0:
            mount_data = json.loads(mount['stdout'])['filesystems'][0]
            mounts.append({'hostname': data['hostname'], **mount_data})
            if mount_data['fstype'] != 'lustre' or mount_data['target'] != '/shared':
                errors.append('The worker /shared path is not the expected Lustre mount.')
        else:
            errors.append('The /shared mount could not be inspected.')
        # Slurm JSON must describe this worker with exactly eight configured GPUs.
        sj = data['commands']['slurm_nodes']
        slurm_nodes = json.loads(sj['stdout']).get('nodes', []) if sj['exit_code'] == 0 else []
        this_node = next((n for n in slurm_nodes if n.get('name') == data['hostname']), None)
        if this_node is None or 'gpu:8' not in str(this_node.get('gres', '')):
            errors.append('Slurm GRES does not reconcile to eight GPUs on this host.')
        collector_errors = {name: c['stderr'] for name, c in data['commands'].items() if c['exit_code']}
        meta = {'pod': pod, 'environment': data['environment'], 'tools': data['tools'],
                'collector_errors': collector_errors, 'findings': errors,
                'artifacts': ['inventory/' + pod['pod'] + '.raw.json']}
        wp.finish('fail' if errors else 'ok', results=[metric('physical_gpu_count', len(rows), 'count', data['hostname']),
                  metric('active_400g_ib_rail_count', len(active_rails), 'count', data['hostname'])],
                  metadata=meta, failure_summary='; '.join(errors) if errors else None)
        findings.extend(errors)
    if len({g['uuid'] for g in all_gpu}) != 32:
        findings.append('The fleet does not expose 32 distinct physical GPU UUIDs.')
    if len({m['source'] for m in mounts}) != 1:
        findings.append('Worker mounts do not identify one shared Lustre filesystem.')
    atomic(run.root / 'inventory/gpu.values.json', {'gpus': all_gpu})
    atomic(run.root / 'inventory/hbm.values.json', {'gpus': [{k: v for k, v in g.items() if k.startswith('hbm_') or k in ('uuid', 'hostname')} for g in all_gpu]})
    atomic(run.root / 'inventory/ib.values.json', {'hosts': ib})
    atomic(run.root / 'inventory/storage.values.json', {'mounts': mounts})
    atomic(run.root / 'inventory/nvlink.values.json', {'hosts': [{'hostname': d['hostname'], 'status': d['commands']['nvlink_status'], 'errors': d['commands']['nvlink_errors']} for _, _, d, _ in collected if d]})
    atomic(run.root / 'inventory/topology.raw.out', '\n'.join(d['commands']['gpu_topology']['stdout'] for _, _, d, _ in collected if d))
    phase.finish('fail' if findings else 'ok', results=[metric('fleet_gpu_count', len(all_gpu), 'count')],
                 metadata={'findings': findings, 'scope': 'Static inventory only. Allocation reconciliation, loaded telemetry, and performance gates remain pending.'},
                 failure_summary='; '.join(findings) if findings else None)
    print(json.dumps({'run_dir': str(run.root), 'gpu_count': len(all_gpu), 'findings': findings}))
    return int(bool(findings))


if __name__ == '__main__':
    sys.exit(main())
