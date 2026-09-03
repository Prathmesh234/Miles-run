"""Run one whole-node member of the bounded, two-replica checkpoint replay."""
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

from enroot_run_config import prepare
from evidence import atomic, sha256, utcnow
from fabric_probe import active_training_ports
from infra_node import allocated_run, read_inventory
from telemetry_health import assert_healthy, require_healthy


def main(args):
    run = allocated_run(args.run_dir)
    code = Path(__file__).resolve().parent
    config = json.loads((code / 'launch.json').read_text())
    host, job = socket.gethostname(), os.environ['SLURM_JOB_ID']
    plan = next(row for row in config['hosts'] if row['hostname'] == host)
    label = f'resume-replay-v{args.attempt}'
    phase = run.phase('02-' + label + '-' + host)
    streams = [run.root / 'telemetry' / name / host for name in (label, 'lustre-' + label)]
    children, findings = [], []
    collector = child = None
    cleaning = False

    def terminated(signum, frame):
        if not cleaning:
            raise RuntimeError('Allocation terminated; stopping owned children and preserving evidence.')

    signal.signal(signal.SIGTERM, terminated)

    def spawn(argv, name, env=None):
        out = phase.path / 'logs' / (name + '.out')
        err = phase.path / 'logs' / (name + '.err')
        handles = [out.open('x'), err.open('x')]
        record = dict(argv=argv, started_at=utcnow(), stdout=str(out.relative_to(run.root)),
                      stderr=str(err.relative_to(run.root)))
        atomic(phase.path / 'logs' / (name + '.command.json'), record)
        proc = subprocess.Popen(argv, stdout=handles[0], stderr=handles[1], env=env, start_new_session=True)
        children.append((proc, handles, record, time.monotonic()))
        return proc

    def stop(proc):
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)

    try:
        for name, digest in config['code_files'].items():
            if sha256(code / name) != digest:
                raise ValueError('Frozen node code differs: ' + name)
        if shutil.disk_usage(run.root).free < 256 * 1024**3:
            raise RuntimeError('The 256 GiB reserve is not available.')
        if read_inventory(run, label + '-start'):
            raise RuntimeError('Allocation inventory failed reconciliation.')
        atomic(run.root / f'control/{label}-job.json', dict(slurm_job_id=job, active=True))
        collector = spawn(['python3', str(code / 'telemetry_native.py'), '--run-dir', str(run.root),
            '--limit-s', '1700', '--ib-backend', 'perfquery', '--stream-label', label,
            '--role', 'trainer', '--lustre-backend', 'host-debugfs-pod', '--gpu-backend', 'nvml',
            '--nvml-binding', str(code / 'pynvml.py'),
            '--stop-marker', 'control/' + label + '-' + host + '-telemetry.stop'], 'native-collector')
        deadline = time.monotonic() + 45
        while True:
            heartbeat = streams[1] / 'heartbeat.json'
            unallocated = heartbeat.exists() and json.loads(heartbeat.read_text()).get('slurm_job_id') == 'unallocated'
            if not unallocated and all([assert_healthy(path, host, job) for path in streams]):
                break
            if collector.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Both telemetry collectors must be live before model loading.')
            time.sleep(1)
        runtime = run.root / 'images' / label / host
        runtime.mkdir(parents=True, exist_ok=False)
        env = prepare(runtime)
        env['NVIDIA_VISIBLE_DEVICES'] = 'all'
        output = run.root / 'training' / label
        output.mkdir(parents=True, exist_ok=True)
        (output / 'nccl' / host).mkdir(parents=True, exist_ok=False)
        variables = dict(NVIDIA_VISIBLE_DEVICES='all', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
            SLURM_JOB_ID=job, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
            PYTHONPATH='/ptx:/miles-source:/root/Megatron-LM', OMP_NUM_THREADS='1',
            CUDA_DEVICE_MAX_CONNECTIONS='1', NCCL_NVLS_ENABLE='0', NCCL_DEBUG='INFO',
            NCCL_DEBUG_SUBSYS='INIT,NET,GRAPH,COLL', GLOO_SOCKET_IFNAME='eth0', NCCL_SOCKET_IFNAME='eth0',
            NCCL_NET='IB', NCCL_IB_HCA='=' + ','.join(hca + ':' + port for hca, port in active_training_ports()),
            NCCL_DEBUG_FILE='/probe-output/nccl/' + host + '/nccl.%h.%p.log', PTX_RESUME_REPLICA=plan['replica'])
        command = ['enroot', 'start', '--pid', '--ipc', '--rw']
        for key, value in variables.items():
            command += ['--env', key + '=' + value]
        for source, target, mode in [(run.root, '/run-artifacts', 'ro'), (output, '/probe-output', 'rw'),
            (code, '/ptx', 'ro'), (run.root / config['miles_source'], '/miles-source', 'ro'),
            (code / 'load-root', '/reload-root', 'ro'),
            (run.root / config['checkpoint_root'] / 'iter_0000000', '/reload-root/iter_0000000', 'ro'),
            (run.root / config['hf_model'], '/model', 'ro'), (Path('/dev/infiniband'), '/dev/infiniband', 'rw')]:
            command += ['--mount', str(source) + ':' + target + ':none:bind,' + mode + ',x-create=dir']
        command += [str(run.root / 'images/enroot-import-v2/miles-amd64.sqsh'), 'torchrun',
            '--nnodes=2', '--nproc-per-node=8', '--node-rank=' + str(plan['node_rank']),
            '--master-addr=' + plan['master_ip'], '--master-port=19471',
            '/ptx/resume_checkpoint_probe.py', '--config', '/ptx/launch.json']
        child = spawn(command, 'replay', env)
        deadline = time.monotonic() + 1500
        while child.poll() is None:
            for directory in streams:
                require_healthy(directory, host, job)
            if collector.poll() is not None:
                raise RuntimeError('Native telemetry stopped while replay was active.')
            if any((run.root / 'control').glob(label + '-failure-*.json')):
                raise RuntimeError('A peer failed; stopping this replay, without a retry.')
            if shutil.disk_usage(run.root).free < 256 * 1024**3 or time.monotonic() > deadline:
                raise RuntimeError('Replay storage or elapsed-time guard reached.')
            time.sleep(1)
        if child.returncode:
            raise RuntimeError('Replay process exited with code ' + str(child.returncode))
    except Exception as exc:
        findings.append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
        atomic(run.root / 'control' / (label + '-failure-' + host + '.json'), dict(time=utcnow(), failure=str(exc)))
    finally:
        cleaning = True
        def finalize_collectors():
            for kind in ('telemetry', 'lustre'):
                atomic(run.root / 'control' / (label + '-' + host + '-' + kind + '.stop'), dict(time=utcnow()))
            if collector is not None:
                try:
                    collector.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    stop(collector)
                if collector.returncode:
                    raise RuntimeError('Native collector did not finalize cleanly.')
                deadline = time.monotonic() + 15
                while not (streams[1] / 'lustre.jsonl').exists() and time.monotonic() < deadline:
                    time.sleep(0.25)
                if not (streams[1] / 'lustre.jsonl').exists() or not (streams[0] / 'nvml-validation.json').exists():
                    raise RuntimeError('Finalized Lustre stream or NVML parity proof is missing.')
                if any((path / 'failure.json').exists() for path in streams):
                    raise RuntimeError('A telemetry collector recorded a failure.')
        for name, action in [('replay', lambda: stop(child)), ('collectors', finalize_collectors),
                              ('inventory', lambda: read_inventory(run, label + '-end'))]:
            try:
                if action():
                    findings.append('Final ' + name + ' check failed.')
            except Exception as exc:
                findings.append(name + ': ' + str(exc))
        for proc, handles, record, started in children:
            for handle in handles:
                handle.close()
            record.update(ended_at=utcnow(), duration_s=time.monotonic()-started,
                          exit_code=proc.poll(), timeout=False)
            phase.commands.append(record)
            if proc.poll() != 0 and not findings:
                findings.append('Child process did not exit cleanly.')
        atomic(phase.path / 'logs/commands.json', phase.commands)
    phase.finish('fail' if findings else 'ok', failure_summary='; '.join(findings) or None,
        metadata=dict(slurm_job_id=job, hostname=host, replica=plan['replica'], findings=findings,
                      scope='Frozen checkpoint replay only, not fresh training trajectories or full async resume.'), refresh=False)
    return int(bool(findings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    raise SystemExit(main(parser.parse_args()))
