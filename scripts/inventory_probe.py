"""Read-only worker probe. It prints JSON and never changes cluster state."""
import datetime as dt
import glob
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time


def command(argv, timeout=30):
    start = time.monotonic()
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        return dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr, duration_s=time.monotonic() - start)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(argv=argv, exit_code=124 if isinstance(e, subprocess.TimeoutExpired) else 127,
                    stdout='', stderr=str(e), duration_s=time.monotonic() - start, collector_error=True)


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        return {'collector_error': str(e)}


checks = {
    'gpu_csv': ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
    'gpu_xml': ['nvidia-smi', '-q', '-x'],
    'gpu_topology': ['nvidia-smi', 'topo', '-m'],
    'nvlink_status': ['nvidia-smi', 'nvlink', '-s'],
    'nvlink_capabilities': ['nvidia-smi', 'nvlink', '-c'],
    'nvlink_errors': ['nvidia-smi', 'nvlink', '-e'],
    'cpu': ['lscpu', '--json'],
    'numa': ['numactl', '--hardware'],
    'mount': ['findmnt', '-J', '-T', '/shared', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'],
    'disk': ['df', '-B1', '/shared'],
    'inodes': ['df', '-i', '/shared'],
    'block_devices': ['lsblk', '-J', '-b', '-o', 'NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL'],
    'network': ['ip', '-j', 'address'],
    'ib_devices': ['ibstat'],
    'ib_devices_verbose': ['ibv_devinfo', '-v'],
    'cuda_compiler': ['nvcc', '--version'],
    'slurm_version': ['srun', '--version'],
    'slurm_config': ['scontrol', 'show', 'config'],
    'slurm_nodes': ['scontrol', 'show', 'nodes', '--json'],
    'slurm_partitions': ['scontrol', 'show', 'partition', '--json'],
    'slurm_queue': ['squeue', '--json'],
    'packages': ['dpkg-query', '-W', '-f=${Package}\t${Version}\n'],
    'clock': ['timedatectl', 'show'],
    'clustermax_sha': ['git', '-C', '/shared/ClusterMAX-internal', 'rev-parse', 'HEAD'],
    'clustermax_status': ['git', '-C', '/shared/ClusterMAX-internal', 'status', '--short', '--untracked-files=no'],
    'ofed': ['ofed_info', '-s'],
    'lnet': ['lctl', 'list_nids'],
}
payload = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), monotonic_s=time.monotonic(),
               hostname=socket.gethostname(), kernel=platform.release(), python=platform.python_version(),
               os_release=read('/etc/os-release'), meminfo=read('/proc/meminfo'),
               environment={k: os.environ.get(k) for k in ['NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                   'NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA', 'GLOO_SOCKET_IFNAME', 'SLURM_JOB_ID',
                   'SLURM_JOB_NODELIST', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'PYTHONPATH']},
               credential_presence={k: bool(os.environ.get(k)) for k in ['HF_TOKEN', 'DAYTONA_API_KEY']},
               tools={k: shutil.which(k) for k in ['srun', 'sbatch', 'enroot', 'docker', 'uv', 'python3', 'fio', 'dcgmi', 'nhc', 'nvidia-smi', 'all_reduce_perf']},
               commands={}, hcas=[])
for name, argv in checks.items():
    payload['commands'][name] = command(argv)
for device in sorted(glob.glob('/sys/class/infiniband/*')):
    p = Path(device)
    hca = dict(name=p.name, pci_bdf=Path(os.path.realpath(p / 'device')).name,
               firmware=read(p / 'fw_ver'), numa_node=read(p / 'device/numa_node'), ports=[])
    for port in sorted((p / 'ports').glob('*')):
        hca['ports'].append(dict(port=port.name, **{k: read(port / k) for k in ['state', 'phys_state', 'rate', 'lid', 'link_layer']},
                                gids={f.name: read(f) for f in sorted((port / 'gids').glob('*'))},
                                counters={f.name: read(f) for f in sorted((port / 'counters').glob('*'))},
                                hw_counters={f.name: read(f) for f in sorted((port / 'hw_counters').glob('*'))}))
    payload['hcas'].append(hca)
print(json.dumps(payload, sort_keys=True))
