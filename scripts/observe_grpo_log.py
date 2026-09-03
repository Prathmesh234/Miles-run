"""Extract deduplicated training and checkpoint receipts from a completed log."""
import argparse
import ast
import datetime as dt
import json
import re

from evidence import Run, atomic, metric, sha256


def parse_log(text):
    clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
    steps, performance, saves = {}, {}, {}
    for line in clean.splitlines():
        timestamp = re.search(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)', line)
        if timestamp is None:
            continue
        time = timestamp[1].replace(' ', 'T') + 'Z'
        found = re.search(r' - (step|perf) (\d+): (\{.*\})', line)
        if found:
            values = ast.literal_eval(found[3])
            number = int(found[2])
            if found[1] == 'step' and 'train/grad_norm' in values:
                if values.get('train/step') != number:
                    raise ValueError('Training metric step does not match its log index.')
                origin = re.search(r' ([a-z0-9_]+)\]\s+(\w+\.py):\d+ - step ', line)
                actor, emitter = origin.groups() if origin else (None, None)
                row = {'time': time, 'step': number, 'metrics': values, 'actor': actor,
                       'receipt_times': [time], 'emitters': [emitter]}
                if number in steps:
                    previous = steps[number]
                    gap = abs((dt.datetime.fromisoformat(time.replace('Z', '+00:00')) -
                               dt.datetime.fromisoformat(previous['time'].replace('Z', '+00:00'))).total_seconds())
                    same_call = (gap <= 0.1 and set(previous['emitters'] + [emitter]) == {'log_utils.py', 'model.py'})
                    if (previous['metrics'] != values or previous['actor'] != actor
                            or (time not in previous['receipt_times'] and not same_call)):
                        raise ValueError('Conflicting duplicate optimizer metric receipt.')
                    previous['receipt_times'] = sorted(set(previous['receipt_times'] + [time]))
                    previous['emitters'] = sorted(set(previous['emitters'] + [emitter]), key=str)
                    previous['time'] = previous['receipt_times'][0]
                else:
                    steps[number] = row
            elif found[1] == 'perf':
                role = 'rollout' if 'rollout_manager]' in line else 'trainer' if 'actor_cell' in line else None
                if role is None:
                    raise ValueError('Performance receipt has no recognized role.')
                key = (role, number)
                row = {'time': time, 'role': role, 'rollout_id': number, 'metrics': values}
                if key in performance and performance[key] != row:
                    raise ValueError('Conflicting duplicate performance receipt.')
                performance[key] = row
        found = re.search(r'successfully saved checkpoint from iteration\s+(\d+)', line)
        if found:
            number = int(found[1])
            row = {'time': time, 'iteration': number}
            if number in saves and saves[number] != row:
                raise ValueError('Conflicting checkpoint-save receipt.')
            saves[number] = row
    if not steps:
        raise ValueError('No optimizer metric receipts found; never infer zero steps from a parser miss.')
    return {'steps': [steps[k] for k in sorted(steps)],
            'performance': [performance[k] for k in sorted(performance)],
            'checkpoint_saves': [saves[k] for k in sorted(saves)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', required=True, type=int)
    parser.add_argument('--job-id', required=True, type=int)
    parser.add_argument('--source-log', required=True)
    parser.add_argument('--expected-steps', required=True, type=int)
    parser.add_argument('--observation-attempt', type=int, required=True)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-sync-grpo-v{args.attempt}-optimizer-observation-a{args.observation_attempt}')
    source = run.root / args.source_log
    try:
        source.resolve().relative_to(run.root)
        if source.is_symlink():
            raise ValueError('Linked source log refused.')
        data = parse_log(source.read_text())
        if len(data['steps']) != args.expected_steps:
            raise ValueError('Optimizer receipt count does not match the expected completed run.')
        for row in data['steps']:
            row['source'] = args.source_log
        data.update(schema_version=1, slurm_job_id=str(args.job_id),
            optimizer_steps_observed=len(data['steps']), source_log=args.source_log,
            log_sha256=sha256(source),
            qualification_status='optimizer_receipts_verified_full_benchmark_not_qualified',
            limitations=['Metric receipts are not independent optimizer-state or gradient verification.',
                         'Full telemetry qualification, trajectory accounting, resume and held-out quality require their separate audits.'],
            scope='Deduplicated optimizer/performance/checkpoint log receipts; not an async or quality result.')
        atomic(phase.path / 'result.json', data)
        atomic(run.root / f'rl/optimizer-steps-job{args.job_id}.jsonl',
               ''.join(json.dumps(row, allow_nan=False) + '\n' for row in data['steps']))
        phase.finish('ok', results=[metric('optimizer_steps_observed', len(data['steps']), 'steps')],
                     metadata={'source_log': args.source_log, 'log_sha256': data['log_sha256'],
                               'artifacts': [str((phase.path / 'result.json').relative_to(run.root))]}, refresh=False)
        print(json.dumps({'optimizer_steps': len(data['steps']), 'checkpoint_saves': len(data['checkpoint_saves'])}))
        return 0
    except Exception as exc:
        phase.finish('fail', failure_summary=str(exc), refresh=False)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
