"""Run-owned Ray process tree and real synchronous Miles GRPO entrypoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback

from evidence import atomic, utcnow


def main():
    def terminated(signum, frame):
        raise RuntimeError('Training container termination requested.')
    signal.signal(signal.SIGTERM, terminated)
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True)
    a = ap.parse_args()
    root = Path('/run-artifacts')
    config = json.loads((Path('/ptx') / 'launch.json').read_text())
    host = socket.gethostname()
    plan = next(n for n in config['host_map']['nodes'] if n['hostname'] == host)
    head = config['host_map']['nodes'][0]['ray_node_ip']
    label = f'sync-grpo-v{a.attempt}'
    output = root / 'training' / label
    logs = output / 'logs' / host
    logs.mkdir(parents=True, exist_ok=True)
    port = config['ray_port']
    ray_process = None
    code = 1
    try:
        import torch
        if torch.cuda.device_count() != 8:
            raise RuntimeError('Container does not expose the complete eight-GPU node.')
        # Exercise the actual training image/devices before any model loads.
        # IB is mandatory; NCCL may not silently select TCP for this gate.
        node_rank = next(i for i, n in enumerate(config['host_map']['nodes']) if n['hostname'] == host)
        fabric_env = dict(os.environ, PTX_FABRIC_LABEL=label,
                         NCCL_DEBUG_FILE=str(root / 'telemetry/nccl' / label / host / 'probe.%h.%p.log'))
        fabric_command = ['torchrun', '--nnodes=4', '--nproc-per-node=8', '--node-rank=' + str(node_rank),
            '--master-addr=' + head, '--master-port=19378', '/ptx/container_fabric_probe.py']
        atomic(logs / 'fabric-smoke-command.json', fabric_command)
        with (logs / 'fabric-smoke.out').open('x') as out, (logs / 'fabric-smoke.err').open('x') as err:
            subprocess.run(fabric_command, env=fabric_env, stdout=out, stderr=err, check=True, timeout=240)
        command = ['ray', 'start', '--block', '--node-ip-address=' + plan['ray_node_ip'], '--num-gpus=8',
            '--num-cpus=32', '--disable-usage-stats', '--object-store-memory=17179869184',
            '--min-worker-port=24000', '--max-worker-port=24999']
        if host == config['host_map']['nodes'][0]['hostname']:
            command += ['--head', '--port=' + str(port), '--include-dashboard=true', '--dashboard-host=127.0.0.1',
                '--dashboard-port=' + str(config['dashboard_port']), '--temp-dir=/tmp/posttrainingx-ray']
        else:
            deadline = time.monotonic() + 240
            while True:
                try:
                    with socket.create_connection((head, port), timeout=2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError('Run-owned Ray head did not become reachable.')
                    time.sleep(1)
            command += ['--address=' + head + ':' + str(port)]
        atomic(logs / 'ray-start-command.json', command)
        with (logs / 'ray-start.out').open('x') as out, (logs / 'ray-start.err').open('x') as err:
            ray_process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            if host != config['host_map']['nodes'][0]['hostname']:
                while not (output / 'driver.finished.json').exists():
                    if ray_process.poll() is not None:
                        raise RuntimeError('Ray worker exited before training driver finished.')
                    time.sleep(1)
                code = json.loads((output / 'driver.finished.json').read_text())['exit_code']
            else:
                import ray
                ray.init(address=head + ':' + str(port), logging_level='WARNING')
                deadline = time.monotonic() + 300
                while True:
                    nodes = [n for n in ray.nodes() if n['Alive'] and n.get('Resources', {}).get('GPU', 0)]
                    if len(nodes) == 4:
                        break
                    if ray_process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Four-node Ray convergence failed.')
                    time.sleep(1)
                if sum(n['Resources']['GPU'] for n in nodes) != 32:
                    raise RuntimeError('Ray does not report exactly 32 GPUs.')
                atomic(output / 'ray-nodes.start.json', nodes)
                ray.shutdown()
                os.environ['MASTER_ADDR'] = head
                os.environ['RAY_ADDRESS'] = 'http://127.0.0.1:' + str(config['dashboard_port'])
                launch = ['python3', '/miles-source/scripts/run_qwen3_6_35b_a3b_posttrainingx.py',
                    '--hardware', 'B200', '--output-dir', '/run-artifacts/training', '--run-id', label,
                    '--execution', 'sync', '--layout', '2t2r', '--num-rollout', str(config['optimizer_steps_requested']), '--save-interval', '1',
                    '--rollout-batch-size', '2', '--n-samples-per-prompt', '8', '--global-batch-size', '16',
                    '--hf-checkpoint', '/model', '--ref-load', '/checkpoint',
                    '--prompt-data', '/ptx/train.prompts.jsonl', '--host-map', '/ptx/host-map.json',
                    '--placement-record-path', str(output / 'ray-placement.json'),
                    '--rollout-journal-dir', str(output / 'trajectory-journal'),
                    '--max-seq-len', '8192', '--rollout-max-response-len', '2048', '--max-tokens-per-gpu', '8192',
                    '--openenv-env-url', 'http://' + head + ':' + str(config['env_port']),
                    '--openenv-max-turns', '8', '--openenv-max-rollout-time-seconds', '900',
                    '--openenv-message-timeout-s', '360', '--sglang-max-running-requests', '16',
                    '--session-server-workers', '4']
                launch += ['--verify-initial-weight-broadcast']
                if config['release_pinned_backups_on_exit']:
                    launch += ['--release-pinned-backups-on-exit']
                atomic(output / 'training-command.json', {'argv': launch, 'time': utcnow(),
                    'scope': f"{config['optimizer_steps_requested']} real synchronous GRPO optimizer steps, initial validation only; not a quality result."})
                with (logs / 'miles.out').open('x') as out2, (logs / 'miles.err').open('x') as err2:
                    code = subprocess.call(launch, stdout=out2, stderr=err2)
                atomic(output / 'driver.finished.json', {'exit_code': code, 'time': utcnow()})
    except Exception:
        atomic(logs / 'exception.txt', traceback.format_exc())
        if host == config['host_map']['nodes'][0]['hostname']:
            atomic(output / 'driver.finished.json', {'exit_code': 1, 'time': utcnow()})
    finally:
        if ray_process is not None and ray_process.poll() is None:
            os.killpg(ray_process.pid, signal.SIGTERM)
            try:
                ray_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(ray_process.pid, signal.SIGKILL)
                ray_process.wait(timeout=10)
        # Persist regular log files only; never copy Ray session symlinks.
        logroot = Path('/tmp/posttrainingx-ray') if host == config['host_map']['nodes'][0]['hostname'] else Path('/tmp/ray')
        for path in logroot.glob('session_*/logs/**/*'):
            if path.is_file() and not path.is_symlink():
                target = logs / 'ray' / path.relative_to(logroot)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
