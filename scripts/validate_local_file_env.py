"""Live CPU-only oracle, isolation, timeout and OpenEnv protocol checks."""
import argparse
import asyncio
import io
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

from evidence import Run, atomic, metric
from local_file_env import FileTaskSession
from local_openenv_client import Action, LocalOpenEnvClient


def check(value, message):
    if not value:
        raise ValueError(message)


def main():
    def terminated(signum, frame):
        raise TimeoutError('Validation received termination; cleaning owned sessions.')
    signal.signal(signal.SIGTERM, terminated)
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--images-manifest', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-local-file-runtime-validation-v{args.attempt}')
    records, findings = [], []
    manifest = Path(args.images_manifest)
    server = None

    def session(task, purpose='isolation-test'):
        return FileTaskSession(run.root, manifest, task, purpose=purpose)

    def record(name, **data):
        records.append({'case': name, **data})
        atomic(phase.path / 'cases.json', records)
        print(json.dumps({'event': 'case_passed', 'case': name}), flush=True)

    try:
        # Run trusted references only in explicitly oracle-labelled containers.
        # Their transcripts never become policy samples or training prompts.
        for row in json.loads(manifest.read_text())['images']:
            current = session(row['task_id'], 'oracle')
            try:
                reference = current.source / 'solution/solve.sh'
                code, _ = current.run_command(reference.read_text(), timeout_s=120)
                check(code == 0, 'Reference solution failed: ' + row['task_id'])
                result = current.evaluate()
                check(result['reward'] == 1.0 and not result['error'], 'Known-good oracle did not pass: ' + row['task_id'])
                record('oracle-' + row['task_id'], reward=result['reward'], episode=current.episode)
            finally:
                current.close()

        current = session('task_00000')
        try:
            command = (
                'test ! -e /tests && test ! -e /solution && test ! -e /shared && '
                'test ! -e /var/run/docker.sock && '
                '! touch /usr/.posttrainingx-write-probe 2>/dev/null && '
                '! timeout 2 /bin/bash -c "echo >/dev/tcp/1.1.1.1/443" 2>/dev/null && '
                'grep -q "CapEff:[[:space:]]*0000000000000000" /proc/self/status')
            code, _ = current.run_command(command)
            check(code == 0, 'Read-only root, credential-path, capability or network isolation probe failed.')
            # A background watcher must not run while hidden tests are staged.
            code, _ = current.run_command(
                'mkdir -p output; (while true; do if [ -d /tests ]; then '
                'echo unexpected-tests > /app/task_file/output/leak; fi; sleep 0.05; done) '
                '> /tmp/watcher.out 2>&1 < /dev/null & '
                'find /app/task_file/input -type f -name "*.txt" > /app/task_file/output/found_files.txt')
            check(code == 0, 'Background watcher setup failed.')
            result = current.evaluate()
            check(result['reward'] == 1.0 and result['output'] == '' and not result['error'], 'Sealed grading did not return an opaque successful verdict.')
            with tarfile.open(current.directory / 'task-snapshot.tar') as archive:
                check(not any(member.name.endswith('/leak') for member in archive), 'Background watcher observed tests before sealing.')
            events = [json.loads(line) for line in (current.directory / 'events.jsonl').read_text().splitlines()]
            names = [row['event'] for row in events]
            check(names.index('policy_stopped') < names.index('grader_assets_staged'), 'Grader assets staged before all policy processes stopped.')
            try:
                current.run_command('cat /tests/test_outputs.py')
            except RuntimeError as exc:
                check('permanently sealed' in str(exc), 'Wrong post-grade rejection.')
            else:
                raise ValueError('Post-grading policy command was accepted.')
            record('sealed-background-process-isolation', episode=current.episode, reward=result['reward'])
        finally:
            current.close()

        current = session('task_00000')
        try:
            result = current.evaluate()
            check(result['reward'] == 0.0 and not result['error'], 'An untouched task was not a genuine negative verdict.')
            record('task-failure-is-not-grader-error', episode=current.episode, reward=0.0)
        finally:
            current.close()

        current = session('task_00000')
        try:
            current.task = dict(current.task, offline_harness_sha256='0' * 64)
            result = current.evaluate()
            check(result['reward'] is None and result['done'] and result['error'] == 'grader_error' and not result['output'],
                  'Changed grader input did not fail closed without a synthetic negative reward.')
            record('changed-harness-is-operational-error', episode=current.episode, reward=None)
        finally:
            current.close()

        current = session('task_00000')
        try:
            started = time.monotonic()
            try:
                current.run_command('sleep 30', timeout_s=1)
            except TimeoutError:
                check(current.sealed and time.monotonic() - started < 15, 'Timeout did not bound/close the session.')
            else:
                raise ValueError('Command timeout was not enforced.')
            record('command-timeout-stops-policy', episode=current.episode, duration_s=time.monotonic() - started)
        finally:
            current.close()

        # Exercise the real maintained OpenEnv WebSocket server with the small
        # GPU-compatible environment-only client, not a fake HTTP substitute.
        env = dict(os.environ, PTX_RUN_DIR=str(run.root), PTX_TASK_IMAGES=str(manifest), MAX_CONCURRENT_ENVS='4')
        with (phase.path / 'logs/server.out').open('w') as out, (phase.path / 'logs/server.err').open('w') as err:
            server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'local_openenv_app:app', '--host', '127.0.0.1', '--port', '18243'],
                                      env=env, stdout=out, stderr=err)
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('OpenEnv server did not become ready.')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18243/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.25)

            async def protocol():
                async with LocalOpenEnvClient('http://127.0.0.1:18243') as client:
                    reset = await client.reset(task_id='task_00000')
                    check(reset.observation.task_id == 'task_00000' and reset.observation.instruction,
                          'OpenEnv reset instruction/task identity mismatch.')
                    check(client.task_id == 'task_00000' and client.episode_id is not None,
                          'OpenEnv controller episode identity missing after reset/state roundtrip.')
                    episode_id = client.episode_id
                    reply = await client.step(Action(action_type='exec', command='printf protocol-ok'))
                    check(reply.observation.output == 'protocol-ok', 'OpenEnv command/observation roundtrip mismatch.')
                    verdict = await client.step(Action(action_type='evaluate'))
                    check(verdict.reward == 0.0 and verdict.done and not verdict.observation.output,
                          'OpenEnv verdict/isolation roundtrip mismatch.')
                    check(verdict.observation.info['harness'] == 'tests/test.sh', 'Missing explicit harness identity.')
                events_path = manifest.parent / 'episodes' / episode_id / 'events.jsonl'
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                check(events and all(e['episode_id'] == episode_id and e['task_id'] == 'task_00000' for e in events),
                      'Client episode identity does not join the durable controller events.')
                record('openenv-live-websocket-roundtrip', reward=0.0, episode=episode_id)
            asyncio.run(protocol())
    except Exception as exc:
        findings.append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
                findings.append('OpenEnv server required forced cleanup.')
        import docker
        client = docker.from_env()
        containers = client.containers.list(all=True, filters={'label': 'posttrainingx.run=' + run.root.name})
        leaked = [c.id for c in containers if c.labels.get('posttrainingx.role') in ('policy', 'oracle', 'grader', 'isolation-test')]
        client.close()
        if leaked:
            findings.append('Leaked run-owned task containers remain; IDs retained for diagnosis.')
    result = {'findings': findings, 'cases': records, 'leaked_container_ids': leaked,
              'scope': 'CPU-only reference/isolation/timeout/protocol qualification. No model trajectories, optimizer steps or quality improvement.'}
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if findings else 'ok', failure_summary='; '.join(findings) or None, metadata=result,
                 results=[metric('live_environment_cases_passed', len(records), 'count')], refresh=False)
    print(json.dumps({'findings': findings, 'cases_passed': len(records)}), flush=True)
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
