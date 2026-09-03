"""Read-only one-second GPU, NVLink, IB, CPU, NUMA, and Lustre sampling.

Each worker owns its streams. The controller merges them after all writers stop.
Counter values remain cumulative; reset-aware rates are derived during analysis.
"""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import traceback

from evidence import atomic, utcnow
from telemetry_health import heartbeat
from infra_node import allocated_run
from fabric_probe import active_training_ports, capture_port, write_port_capture


GPU_FIELDS = [
    ('index', None), ('uuid', None), ('utilization.gpu', '%'), ('utilization.memory', '%'),
    ('memory.total', 'MiB'), ('memory.free', 'MiB'), ('memory.used', 'MiB'),
    ('temperature.gpu', 'degC'), ('power.draw', 'W'), ('clocks.current.sm', 'MHz'),
    ('clocks.current.memory', 'MHz'), ('ecc.errors.corrected.volatile.total', 'count'),
    ('ecc.errors.uncorrected.volatile.total', 'count'),
]


def gpu_records(text):
    records = []
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    if len(rows) != 8:
        raise ValueError(f'Expected eight GPUs, found {len(rows)}.')
    for row in rows:
        if len(row) != len(GPU_FIELDS):
            raise ValueError('Unexpected GPU CSV shape.')
        for (field, unit), value in zip(GPU_FIELDS[2:], row[2:]):
            try:
                parsed = float(value)
            except ValueError:
                records.append(dict(metric='collector_error', value=None, unit='event',
                                    gpu_uuid=row[1], requested_metric=field, error=value))
                continue
            records.append(dict(metric=field, value=parsed, unit=unit, gpu_uuid=row[1]))
    return records


def nvlink_records(text):
    records, uuid = [], None
    for line in text.splitlines():
        header = re.search(r'UUID: (GPU-[^)]+)', line)
        if header:
            uuid = header.group(1)
        found = re.search(r'Link\s+(\d+):\s+Data (Tx|Rx):\s+(\d+) KiB', line)
        if found and uuid:
            link, direction, kib = found.groups()
            records.append(dict(metric='nvlink_data_' + direction.lower() + '_bytes_total',
                                value=int(kib)*1024, unit='B', gpu_uuid=uuid, link=int(link)))
    if len(records) != 8 * 18 * 2:
        raise ValueError(f'Expected 288 NVLink counters, found {len(records)}.')
    return records


def capture(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        return {'argv': argv, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'exit_code': 124 if isinstance(exc, subprocess.TimeoutExpired) else 127,
                'stdout': '', 'stderr': str(exc)}


class Streams:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.errors = 0

    def write(self, name, record):
        if name not in self.files:
            self.files[name] = (self.root / (name + '.jsonl.partial')).open('x')
        self.files[name].write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        if record.get('metric') == 'collector_error':
            self.errors += 1
            # Sticky marker is visible even if a later driver call hangs.
            if not (self.root / 'failure.json').exists():
                atomic(self.root / 'failure.json', record)

    def flush(self):
        for f in self.files.values():
            f.flush()
            os.fsync(f.fileno())

    def close(self):
        self.flush()
        for name, f in self.files.items():
            f.close()
            os.replace(self.root / (name + '.jsonl.partial'), self.root / (name + '.jsonl'))


def sample_files(streams, common, collect_lustre_stats=True):
    raw = {}
    paths = ['/proc/stat', '/proc/meminfo', '/proc/vmstat', '/proc/loadavg', '/proc/softirqs']
    paths += [str(p) for p in Path('/sys/devices/system/node').glob('node*/numastat')]
    for name in paths:
        try:
            raw[name] = Path(name).read_text()
        except OSError as exc:
            streams.write('cpu-memory-numa', dict(common, source='proc-sysfs', metric='collector_error',
                                                value=None, unit='event', path=name, error=str(exc)))
    streams.write('raw-system', dict(common, source='proc-sysfs', files=raw))
    for line in raw.get('/proc/meminfo', '').splitlines():
        cells = line.split()
        if len(cells) >= 2:
            streams.write('cpu-memory-numa', dict(common, source='proc-meminfo', metric=cells[0].rstrip(':'),
                                                value=int(cells[1]) * (1024 if len(cells) == 3 else 1),
                                                unit='B' if len(cells) == 3 else 'count'))
    for line in raw.get('/proc/stat', '').splitlines():
        cells = line.split()
        if cells and cells[0] == 'cpu':
            names = ['user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'guest', 'guest_nice']
            for key, value in zip(names, cells[1:]):
                streams.write('cpu-memory-numa', dict(common, source='proc-stat', metric='cpu_' + key,
                                                    value=int(value), unit='USER_HZ_ticks'))
    for line in raw.get('/proc/vmstat', '').splitlines():
        name, value = line.split()
        streams.write('cpu-memory-numa', dict(common, source='proc-vmstat', metric=name, value=int(value), unit='count'))
    v = os.statvfs(streams.root)
    for name, value, unit in [('capacity', v.f_blocks*v.f_frsize, 'B'),
                              ('available', v.f_bavail*v.f_frsize, 'B'), ('free_inodes', v.f_favail, 'count')]:
        streams.write('lustre', dict(common, source='statvfs', metric=name, value=value, unit=unit))
    if not collect_lustre_stats:
        return
    stats = list(Path('/proc/fs/lustre/llite').glob('*/stats'))
    if not stats:
        streams.write('lustre', dict(common, source='lustre-llite', metric='collector_error',
                                  value=None, unit='event', error='Lustre client stats unavailable in this namespace.'))
    for p in stats:
        try:
            streams.write('raw-lustre', dict(common, source='lustre-llite', path=str(p), text=p.read_text()))
        except OSError as exc:
            streams.write('lustre', dict(common, source='lustre-llite', metric='collector_error',
                                      value=None, unit='event', path=str(p), error=str(exc)))


def sample_sysfs_ib(streams, common):
    ib_paths = sorted(Path('/sys/class/infiniband').glob('*/ports/*/counters/*'))
    ib_paths += sorted(Path('/sys/class/infiniband').glob('*/ports/*/hw_counters/*'))
    if not ib_paths:
        streams.write('infiniband', dict(common, source='ib-sysfs', metric='collector_error',
                                        value=None, unit='event', error='No IB counters found.'))
    for p in ib_paths:
        attrs = dict(common, source='ib-sysfs', hca=p.parts[-5], hca_port=p.parts[-3],
                     counter_group=p.parts[-2], metric=p.name)
        try:
            value = int(p.read_text().strip())
            unit = '4_octets' if p.name in ('port_xmit_data', 'port_rcv_data') else 'counter_units'
            streams.write('infiniband', dict(attrs, value=value, unit=unit))
        except (OSError, ValueError) as exc:
            streams.write('infiniband', dict(attrs, metric='collector_error', value=None, unit='event', error=str(exc)))
def collect(run, stop_file, limit_s, ib_backend='sysfs', stream_label='native',
            role='infrastructure-preflight', lustre_backend='namespace',
            gpu_backend='cli', nvml_binding=None):
    if ib_backend not in ('sysfs', 'perfquery') or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', stream_label):
        raise ValueError('Invalid explicit collector backend or stream label.')
    host = socket.gethostname()
    root = run.root / 'telemetry' / stream_label / host
    root.mkdir(parents=True, exist_ok=False)
    streams = Streams(root)
    start = time.monotonic()
    ticks, sampler, findings = 0, None, []
    common = {'time': utcnow(), 'monotonic_s': start, 'hostname': host,
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'role': role}
    commands = {
        'nvidia-smi': (['nvidia-smi', '--query-gpu=' + ','.join(x[0] for x in GPU_FIELDS),
                        '--format=csv,noheader,nounits'], gpu_records),
        'nvlink': (['nvidia-smi', 'nvlink', '-gt', 'd'], nvlink_records),
    }
    try:
        if gpu_backend == 'nvml':
            from telemetry_nvml import NVMLSampler
            inventory = json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']
            uuids = [row['uuid'] for row in inventory if row['hostname'] == host]
            sampler = NVMLSampler(nvml_binding, uuids, streams, common)
            commands = {}
        elif gpu_backend != 'cli':
            raise ValueError('Unsupported GPU telemetry backend.')
        ports = active_training_ports() if ib_backend == 'perfquery' else []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            while time.monotonic() - start < limit_s and not stop_file.exists():
                tick = time.monotonic()
                common = {'time': utcnow(), 'monotonic_s': tick, 'hostname': host,
                          'slurm_job_id': os.environ['SLURM_JOB_ID'], 'role': role}
                pending = {name: pool.submit(capture, argv) for name, (argv, _) in commands.items()}
                fabric = [pool.submit(capture_port, hca, port, common) for hca, port in ports]
                if sampler:
                    sampler.sample(common)
                sample_files(streams, common, collect_lustre_stats=lustre_backend == 'namespace')
                if ib_backend == 'sysfs':
                    sample_sysfs_ib(streams, common)
                for future in fabric:
                    write_port_capture(streams, future.result())
                for name, future in pending.items():
                    raw = future.result()
                    streams.write('raw-' + name, dict(common, source=name, **raw))
                    try:
                        if raw['exit_code']:
                            raise ValueError(raw['stderr'])
                        for row in commands[name][1](raw['stdout']):
                            streams.write(name, dict(common, source=name, **row))
                    except ValueError as exc:
                        streams.write(name, dict(common, source=name, metric='collector_error',
                                                value=None, unit='event', error=str(exc)))
                streams.write('cpu-memory-numa', dict(common, source='collector', metric='collection_duration',
                                                    value=time.monotonic()-tick, unit='s'))
                streams.flush()
                elapsed = time.monotonic() - tick
                if elapsed > 12:
                    streams.write('cpu-memory-numa', dict(common, time=utcnow(), monotonic_s=time.monotonic(),
                        source='collector', metric='collector_error', value=None, unit='event',
                        error=f'Collector tick including flush exceeded 12 seconds: {elapsed:.6f}s.'))
                ticks += 1
                heartbeat(root, host, os.environ['SLURM_JOB_ID'], ticks, streams.errors, time.monotonic())
                if streams.errors:
                    raise RuntimeError('Collector error recorded; stopping instead of concealing missing samples.')
                time.sleep(max(0, 1 - (time.monotonic()-tick)))
        if sampler:
            atomic(root / 'nvml-validation.json', sampler.finish())
    except Exception as exc:
        findings.append(str(exc))
        atomic(root / 'collector-exception.txt', traceback.format_exc())
        streams.write('cpu-memory-numa', dict(common, time=utcnow(), monotonic_s=time.monotonic(),
            source='collector', metric='collector_error', value=None, unit='event', error=str(exc)))
    finally:
        # Finalize streams even if NVML shutdown raises. If it blocks, the owner
        # kills this process after a bounded grace period and retains .partial files.
        if sampler:
            try:
                sampler.shutdown()
            except Exception as exc:
                findings.append('NVML shutdown: ' + str(exc))
                streams.write('cpu-memory-numa', dict(common, time=utcnow(), monotonic_s=time.monotonic(),
                    source='collector', metric='collector_error', value=None, unit='event', error=str(exc)))
        streams.close()
    if findings:
        raise RuntimeError('; '.join(findings))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--limit-s', type=int, default=840)
    ap.add_argument('--ib-backend', choices=['sysfs', 'perfquery'], default='sysfs')
    ap.add_argument('--stream-label', default='native')
    ap.add_argument('--role', default='infrastructure-preflight')
    ap.add_argument('--lustre-backend', choices=['namespace', 'host-debugfs-pod'], default='namespace')
    ap.add_argument('--gpu-backend', choices=['cli', 'nvml'], default='cli')
    ap.add_argument('--nvml-binding')
    ap.add_argument('--stop-marker', help='Run-relative, node-owned stop marker.')
    args = ap.parse_args()
    run = allocated_run(args.run_dir)
    marker = args.stop_marker or ('control/' + args.stream_label + '-telemetry.stop')
    if Path(marker).is_absolute() or '..' in Path(marker).parts:
        raise ValueError('Stop marker must be a relative run path.')
    collect(run, run.root / marker, args.limit_s,
            args.ib_backend, args.stream_label, args.role, args.lustre_backend,
            args.gpu_backend, args.nvml_binding)


if __name__ == '__main__':
    main()
