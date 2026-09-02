"""Bounded read-only DCGM queries through a separate, run-owned hostengine.

Never stops or reconfigures the delivered hostengine and never runs diagnostics,
GPU resets, policy changes, or field profiling. The child is always reaped.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from evidence import Run, atomic, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    run = Run(args.run_dir)
    host = socket.gethostname()
    phase = run.phase(f'01-owned-dcgm-{host}-v{args.attempt}')
    result = {'hostname': host, 'findings': [], 'scope': __doc__, 'shared_daemon_modified': False}
    process = None
    port = 19443
    try:
        # Refuse to reuse any listener; do not send it termination or commands.
        with socket.socket() as guard:
            guard.bind(('127.0.0.1', port))
        command = ['nv-hostengine', '--no-daemon', '--port', str(port), '--bind-interface', '127.0.0.1',
                   '--pid', str(phase.path / 'owned-hostengine.pid'), '--log-level', 'ERROR',
                   '--log-filename', str(phase.path / 'logs/owned-hostengine.log')]
        atomic(phase.path / 'hostengine-command.json', command)
        env = dict(os.environ, NVIDIA_VISIBLE_DEVICES='all')
        with (phase.path / 'logs/hostengine.out').open('x') as out, (phase.path / 'logs/hostengine.err').open('x') as err:
            process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Run-owned hostengine exited or did not become ready.')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        break
                except OSError:
                    time.sleep(.2)
            rc, discovery, _ = phase.command(['dcgmi', 'discovery', '--host', f'127.0.0.1:{port}', '--list'], timeout=20)
            result['null_identity_fields'] = discovery.count('<<<NULL>>>')
            if rc or '8 GPUs found (Active).' not in discovery or result['null_identity_fields']:
                raise RuntimeError('Run-owned DCGM does not report all eight real GPU identities.')
            rc, samples, _ = phase.command(['dcgmi', 'dmon', '--host', f'127.0.0.1:{port}',
                '--field-id', '100,101,150,155,203', '--count', '3', '--delay', '1000'], timeout=20)
            result['raw_sample_rows'] = sum(line.lstrip().startswith('GPU ') for line in samples.splitlines())
            if rc or result['raw_sample_rows'] != 24 or any(x in samples for x in ('N/A', '<<<NULL>>>')):
                raise RuntimeError('Basic DCGM watch did not yield three complete numeric samples per GPU.')
    except Exception as exc:
        result['findings'].append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    result['findings'].append('Run-owned hostengine required forced cleanup.')
            result['owned_child_reaped'] = process.returncode is not None
        result['artifacts'] = [{'path': str(p.relative_to(run.root)), 'sha256': sha256(p)}
                               for p in phase.path.glob('logs/hostengine.*') if p.is_file()]
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if result['findings'] else 'ok', metadata=result,
                 failure_summary='; '.join(result['findings']) or None, refresh=False)
    print(json.dumps(result))
    return int(bool(result['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
