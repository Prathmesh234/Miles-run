"""Validate all four host collectors without reclassifying the original failure."""
import argparse
import json
from pathlib import Path
import sys

from evidence import Run, metric, sha256


def validate(run):
    phase = run.phase('01-host-lustre-load-coverage')
    errors, results, files = [], [], {}
    for i in range(4):
        host = f'gpu-nodes-{i}'
        path = run.root / 'telemetry/lustre-host-validation-v1' / host / 'lustre.jsonl'
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            errors.extend(host + ': ' + r.get('error', 'collector error') for r in rows if r['metric'] == 'collector_error')
            files[str(path.relative_to(run.root))] = sha256(path)
            raw = path.with_name('raw-lustre.jsonl')
            files[str(raw.relative_to(run.root))] = sha256(raw)
            times = sorted({r['monotonic_s'] for r in rows})
            if len(times) < 150 or max(b-a for a, b in zip(times, times[1:])) > 3:
                errors.append(host + ': insufficient coverage or sampling gap over three seconds')
            results.append(metric('lustre_collector_ticks', len(times), 'count', host))
            for direction in ('read', 'write'):
                grouped = {}
                for row in rows:
                    if row['metric'] == direction + '_bytes.sum' and row.get('slurm_job_id'):
                        grouped.setdefault(row['lustre_client'], []).append(row['value'])
                delta = 0
                for client, values in grouped.items():
                    if any(b < a for a, b in zip(values, values[1:])):
                        errors.append(host + '/' + client + ': counter reset during allocation')
                    delta += values[-1] - values[0]
                if delta < 2*1024**3:
                    errors.append(host + ': less than 2 GiB observed ' + direction + ' counter increase during load')
                results.append(metric('observed_lustre_' + direction + '_bytes_delta', delta, 'B', host,
                                      scope='all host clients during allocation, not isolated workload throughput'))
            load = run.root / 'tests' / ('01-lustre-collector-storage-' + host)
            if not (load / (load.name + '.values.json')).exists():
                errors.append(host + ': missing successful fio verification result')
        except (OSError, ValueError, KeyError) as exc:
            errors.append(host + ': ' + str(exc))
    phase.finish('fail' if errors else 'ok', results=results, failure_summary='; '.join(errors) or None,
        metadata={'findings': errors, 'telemetry_sha256': files,
                  'scope': 'Four-node 1s host Lustre collector validation with verified 2 GiB I/O per node. Not a storage benchmark or full RL telemetry gate.',
                  'artifacts': ['telemetry/lustre-host-validation-v1']}, refresh=False)
    print(json.dumps({'status': 'fail' if errors else 'ok', 'findings': errors, 'results': results}), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    sys.exit(validate(Run(args.run_dir)))
