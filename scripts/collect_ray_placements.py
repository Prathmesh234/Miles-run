"""Read-only, bounded Ray actor placement sampling for an existing allocation.

Never persist serialized runtime environments, call sites, reprs or arbitrary
API error messages. This observer does not create, stop or modify Ray actors.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import socket
import time
import urllib.request

from evidence import Run, atomic, sha256, utcnow
from telemetry_native import Streams


def placements(packet, node_hosts):
    if packet.get('result') is not True:
        raise ValueError('Ray API reported an unsuccessful response.')
    data = packet['data']['result']
    rows = data['result']
    if data.get('partial_failure_warning') or data.get('warnings'):
        raise ValueError('Ray API reported partial or uncertain actor state.')
    if len(rows) != data['num_filtered'] or data['num_after_truncation'] < data['num_filtered']:
        raise ValueError('Ray actor state was truncated.')
    result, seen = [], set()
    for row in rows:
        identity = row['actor_id']
        if not identity or identity in seen:
            raise ValueError('Duplicate or empty actor identity.')
        seen.add(identity)
        node = row['node_id']
        host = node_hosts.get(node)
        if row['state'] == 'ALIVE' and host is None:
            raise ValueError('Live actor is placed outside the recorded Ray nodes.')
        resources = row.get('required_resources')
        if resources is None and row['state'] == 'ALIVE':
            raise ValueError('Live actor resource requirements are missing.')
        if resources is not None and any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in resources.values()):
            raise ValueError('Invalid actor resource requirements.')
        result.append({key: row.get(key) for key in (
            'actor_id', 'class_name', 'state', 'job_id', 'node_id', 'pid', 'name',
            'placement_group_id', 'required_resources', 'num_restarts')} | {'assigned_hostname': host})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--job-id', type=int, required=True)
    args = parser.parse_args()
    run = Run(args.run_dir)
    label = f'sync-grpo-v{args.attempt}'
    output = run.root / 'training' / label
    config = json.loads((run.root / f'provenance/sync-grpo-code-v{args.attempt}/launch.json').read_text())
    phase = run.phase(f'02-ray-placement-observer-{label}')
    directory = run.root / 'telemetry' / ('ray-placement-' + label) / socket.gethostname()
    directory.mkdir(parents=True, exist_ok=False)
    streams = Streams(directory)
    started, ticks, observed, findings = time.monotonic(), 0, {}, []
    common = dict(hostname=socket.gethostname(), source='ray-state-api', slurm_job_id=str(args.job_id))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{config['dashboard_port']}/api/v0/actors?detail=true&limit=1000&timeout=5"
    try:
        marker = json.loads((run.root / 'control' / (label + '-job.json')).read_text())
        if str(marker['slurm_job_id']) != str(args.job_id):
            raise ValueError('Allocation identity does not match the observer.')
        nodes_path = output / 'ray-nodes.start.json'
        while not nodes_path.exists():
            if (output / 'driver.finished.json').exists() or time.monotonic() - started > 300:
                raise RuntimeError('Ray node inventory did not become ready within the bounded startup.')
            time.sleep(1)
        hosts = {n['ray_node_ip']: n['hostname'] for n in config['host_map']['nodes']}
        node_hosts = {n['NodeID']: hosts[n['NodeManagerAddress']] for n in json.loads(nodes_path.read_text())}
        if len(node_hosts) != 4:
            raise ValueError('Expected four explicitly mapped Ray nodes.')
        while not (output / 'driver.finished.json').exists():
            tick = time.monotonic()
            common.update(time=utcnow(), monotonic_s=tick)
            if tick - started > 5100 or shutil.disk_usage(run.root).free < 256 * 1024**2:
                raise RuntimeError('Ray observer duration or evidence-space guard reached.')
            with opener.open(url, timeout=8) as response:
                packet = json.load(response)
            rows = placements(packet, node_hosts)
            for row in rows:
                streams.write('actor-placements', dict(common, metric='actor_observed', value=1, unit='count', **row))
                if row['state'] == 'ALIVE':
                    observed[row['actor_id']] = row
            streams.write('ray', dict(common, metric='actor_count', value=len(rows), unit='count'))
            streams.write('ray', dict(common, metric='actor_state_query_duration', value=time.monotonic()-tick, unit='s'))
            streams.flush()
            ticks += 1
            time.sleep(max(0, 5 - (time.monotonic() - tick)))
        actual = Counter((row['class_name'], row['assigned_hostname']) for row in observed.values())
        for node in config['host_map']['nodes']:
            klass, count = ('MegatronTrainRayActor', 8) if node['role'] == 'trainer' else ('SGLangEngine', 1)
            if actual[(klass, node['hostname'])] != count:
                findings.append(f"{node['hostname']}: actual {klass} placement count differs from {count}.")
    except Exception as exc:
        findings.append(type(exc).__name__ + ': ' + str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        streams.write('ray', dict(common, time=utcnow(), monotonic_s=time.monotonic(),
            metric='collector_error', value=None, unit='event', error=findings[-1]))
    finally:
        streams.close()
    result = dict(findings=findings, ticks=ticks, observed_alive_actors=list(observed.values()),
        scope='Periodic actual Ray actor/node/PID/resource placement, not GPU worker-process or lifetime completeness.',
        source_sha256=sha256(__file__), artifacts=[str(directory.relative_to(run.root))])
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if findings else 'ok', metadata=result, failure_summary='; '.join(findings) or None, refresh=False)
    print(json.dumps({'ticks': ticks, 'actors': len(observed), 'findings': findings}), flush=True)
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
