"""One complete role node: reconcile inventory, collect telemetry, launch Miles."""
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
import urllib.request

from enroot_run_config import prepare
from evidence import atomic, utcnow
from infra_node import allocated_run, read_inventory


def cleanup_actions(actions, findings):
    """One cleanup failure must not suppress subsequent evidence or cleanup."""
    for name, action in actions:
        try:
            action()
        except Exception as exc:
            findings.append(f'Cleanup {name} failed: {type(exc).__name__}: {exc}')


def main():
    cleaning_up = False
    def terminated(signum, frame):
        if cleaning_up:
            return  # Finish bounded cleanup; Slurm's hard kill remains effective.
        raise RuntimeError('Slurm terminated this run-owned node process; preserving artifacts and cleaning children.')
    signal.signal(signal.SIGTERM, terminated)
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    a = ap.parse_args()
    run = allocated_run(a.run_dir)
    code = Path(__file__).resolve().parent
    config = json.loads((code / 'launch.json').read_text())
    host = socket.gethostname()
    plan = next(n for n in config['host_map']['nodes'] if n['hostname'] == host)
    label = f'sync-grpo-v{a.attempt}'
    phase = run.phase('02-' + label + '-' + host)
    logs = phase.path / 'logs'
    collector = controller = child = None
    findings = []
    handles = []
    def spawn(argv, name, env=None):
        out, err = (logs / (name + '.out')).open('x'), (logs / (name + '.err')).open('x')
        handles.extend([out, err])
        atomic(logs / (name + '.command.json'), {'argv': argv, 'time': utcnow()})
        return subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)
    def stop(p):
        if p is not None and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=10)
    controller_name = 'ptx-grpo-env-' + run.root.name + '-v' + str(a.attempt)
    try:
        if read_inventory(run, label + '-start'):
            raise RuntimeError('Physical GPUs and Slurm GRES failed allocation reconciliation.')
        validation = json.loads((run.root / config['environment_gate']).read_text())
        if validation['findings'] or len(validation['cases']) < 10:
            raise RuntimeError('Live environment qualification has not passed.')
        atomic(run.root / f'control/{label}-job.json', {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'active': True})
        collector = spawn(['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root),
            '--limit-s', '5300', '--ib-backend', 'perfquery', '--stream-label', label,
            '--role', plan['role'], '--lustre-backend', 'host-debugfs-pod'], 'native-collector')
        deadline = time.monotonic() + 45
        while True:
            gpu = run.root / 'telemetry' / label / host / 'nvidia-smi.jsonl.partial'
            lustre = run.root / 'telemetry' / ('lustre-' + label) / host / 'lustre.jsonl.partial'
            if gpu.exists() and gpu.stat().st_size and lustre.exists() and lustre.stat().st_size:
                break
            if collector.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Native GPU and host Lustre telemetry are not both live.')
            time.sleep(1)
        if host == config['host_map']['nodes'][0]['hostname']:
            argv = ['docker', 'run', '--rm', '--runtime=runc', '--network=host', '--cpus=8', '--memory=16g',
                '--pids-limit=512', '--cap-drop=ALL', '--security-opt=no-new-privileges', '--name=' + controller_name,
                '--label=posttrainingx.run=' + run.root.name, '--label=posttrainingx.role=grpo-controller',
                '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'PYTHONPATH=/ptx:/opt/openenv/src:/opt/openenv/envs',
                '-e', 'PTX_RUN_DIR=' + str(run.root), '-e', 'PTX_TASK_IMAGES=' + str(run.root / config['images_manifest']),
                '-e', 'MAX_CONCURRENT_ENVS=16', '-v', str(run.root) + ':' + str(run.root) + ':rw',
                '-v', str(code) + ':/ptx:ro', '-v', '/var/run/docker.sock:/var/run/docker.sock', config['controller_image'],
                'python3', '-m', 'uvicorn', 'local_openenv_app:app', '--host', plan['ray_node_ip'], '--port', str(config['env_port'])]
            controller = spawn(argv, 'openenv-controller')
        env_url = 'http://' + config['host_map']['nodes'][0]['ray_node_ip'] + ':' + str(config['env_port']) + '/health'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 90
        while True:
            try:
                with opener.open(env_url, timeout=3) as r:
                    if r.status == 200:
                        break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Local OpenEnv controller not reachable.')
                time.sleep(1)
        runtime = run.root / 'images' / label / host
        runtime.mkdir(parents=True, exist_ok=False)
        env = prepare(runtime)
        env['NVIDIA_VISIBLE_DEVICES'] = 'all'
        nccl = run.root / 'telemetry/nccl' / label / host
        nccl.mkdir(parents=True, exist_ok=False)
        envs = {'NVIDIA_VISIBLE_DEVICES': 'all', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
            'SLURM_JOB_ID': os.environ['SLURM_JOB_ID'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTHONPATH': '/ptx:/miles-source:/root/Megatron-LM', 'OMP_NUM_THREADS': '1',
            'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '0', 'NCCL_DEBUG': 'INFO',
            'NCCL_DEBUG_SUBSYS': 'INIT,NET,GRAPH,COLL', 'GLOO_SOCKET_IFNAME': config['network_interface'],
            'NCCL_SOCKET_IFNAME': config['network_interface'],
            'NCCL_DEBUG_FILE': '/run-artifacts/telemetry/nccl/' + label + '/' + host + '/nccl.%h.%p.log',
            'RAY_USAGE_STATS_ENABLED': '0', 'WANDB_MODE': 'disabled'}
        command = ['enroot', 'start', '--pid', '--ipc', '--rw']
        for key, value in envs.items():
            command += ['--env', key + '=' + value]
        for source, target, mode in [(run.root, '/run-artifacts', 'rw'), (code, '/ptx', 'ro'),
            (run.root / config['miles_source'], '/miles-source', 'ro'),
            (run.root / config['hf_model'], '/model', 'ro'), (run.root / config['converted_model'], '/checkpoint', 'ro')]:
            command += ['--mount', str(source) + ':' + target + ':none:bind,' + mode + ',x-create=dir']
        command += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'),
                    'python3', '/ptx/grpo_container.py', '--attempt', str(a.attempt)]
        child = spawn(command, 'training-container', env=env)
        deadline = time.monotonic() + 5100
        while child.poll() is None:
            if any((run.root / 'control').glob(label + '-failure-*.json')):
                raise RuntimeError('A peer node failed; stopping this run-owned child and retaining evidence.')
            if shutil.disk_usage(run.root).free < 256*1024**3:
                raise RuntimeError('256 GiB free-space reserve reached; stopping current run only.')
            if collector.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Telemetry stopped or bounded training time expired.')
            time.sleep(1)
        if child.returncode:
            raise RuntimeError('Miles training process exited with code ' + str(child.returncode))
    except Exception as exc:
        findings.append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
        atomic(run.root / 'control' / (label + '-failure-' + host + '.json'),
               {'time': utcnow(), 'hostname': host, 'failure': str(exc)})
    finally:
        cleaning_up = True
        def stop_controller():
            if controller is None:
                return
            # Exact container name was created by this process; no global cleanup.
            rc, _, _ = phase.command(['docker', 'stop', '--time=20', controller_name], timeout=30)
            if rc:
                findings.append('Run-owned OpenEnv controller stop returned an error.')
            stop(controller)
        def stop_collector():
            if collector is None:
                return
            try:
                collector.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stop(collector)
            if collector.returncode:
                findings.append('Native telemetry collector failed to finalize.')
        def end_inventory():
            if read_inventory(run, label + '-end'):
                findings.append('Post-allocation GPU reconciliation failed.')
        cleanup_actions([
            ('training-child', lambda: stop(child)),
            ('controller', stop_controller),
            ('native-stop-marker', lambda: atomic(run.root / 'control' / (label + '-telemetry.stop'), {'time': utcnow()})),
            ('lustre-stop-marker', lambda: atomic(run.root / 'control' / (label + '-lustre.stop'), {'time': utcnow()})),
            ('collector', stop_collector),
            ('post-allocation-inventory', end_inventory),
        ], findings)
        for handle in handles:
            handle.close()
    phase.finish('fail' if findings else 'ok', failure_summary='; '.join(findings) or None,
                 metadata={'findings': findings, 'role': plan['role'], 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                           'scope': 'Real synchronous GRPO execution; exit zero is not a quality or resume claim.'}, refresh=False)
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
