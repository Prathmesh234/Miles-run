"""Read eight active 400G InfiniBand rails without resetting hardware counters."""
import concurrent.futures
import json
from pathlib import Path
import re
import socket
import subprocess
import time

from evidence import utcnow


def active_training_ports(root=Path('/sys/class/infiniband')):
    ports = []
    for path in sorted(root.glob('*/ports/*')):
        if (path / 'link_layer').read_text().strip() != 'InfiniBand':
            continue
        if not (path / 'state').read_text().strip().startswith('4: ACTIVE'):
            continue
        if (path / 'rate').read_text().split()[0] != '400':
            continue
        ports.append((path.parts[-3], path.name))
    if len(ports) != 8:
        raise ValueError(f'Expected eight active 400G IB ports, found {ports}.')
    return ports


def perfquery_command(hca, port):
    if not re.fullmatch(r'mlx5_\d+', hca) or not re.fullmatch(r'\d+', str(port)):
        raise ValueError('Invalid local HCA/port identifier.')
    # -r, -R, reset masks, and remote destination arguments are never accepted.
    return ['perfquery', '-x', '-C', hca, '-P', str(port)]


def perfquery_records(text, hca, port):
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Za-z0-9]+):\.+(\d+)', line.strip())
        if match and match[1] not in ('PortSelect', 'CounterSelect', 'CounterSelect2'):
            values[match[1]] = int(match[2])
    required = {'PortXmitData', 'PortRcvData', 'PortXmitPkts', 'PortRcvPkts'}
    if not required.issubset(values):
        raise ValueError('Missing required extended port counters: ' + str(sorted(required - values.keys())))
    records = []
    for name, value in values.items():
        data = name in ('PortXmitData', 'PortRcvData')
        unit = 'B' if data else ('pma_ticks' if name == 'PortXmitWait' else 'count')
        records.append({'metric': name, 'value': value * 4 if data else value,
                        'unit': unit, 'hca': hca, 'hca_port': str(port),
                        'raw_value': value, 'raw_unit': '4_octets' if data else unit})
    return records


def capture_port(hca, port, context=None):
    start = time.monotonic()
    common = {'role': 'read-only-diagnostic', **(context or {}),
              'time': utcnow(), 'monotonic_s': start, 'hostname': socket.gethostname(),
              'source': 'perfquery', 'hca': hca, 'hca_port': str(port)}
    argv = perfquery_command(hca, port)
    raw = dict(common, argv=argv)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        raw = dict(common, argv=argv, stdout=result.stdout, stderr=result.stderr,
                   exit_code=result.returncode, duration_s=time.monotonic() - start)
        if result.returncode:
            raise ValueError(result.stderr)
        rows = [dict(common, **row) for row in perfquery_records(result.stdout, hca, port)]
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raw.update(error=str(exc), duration_s=time.monotonic() - start)
        rows = [dict(common, metric='collector_error', value=None, unit='event', error=str(exc))]
    return {'raw': raw, 'records': rows}


def write_port_capture(streams, capture):
    streams.write('raw-infiniband', capture['raw'])
    for row in capture['records']:
        streams.write('infiniband', row)


def main():
    ports = active_training_ports()
    print(json.dumps({'ports': ports, 'scope': 'Read-only counters; no load generation or resets.'}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for tick in range(3):
            start = time.monotonic()
            for result in pool.map(lambda port: capture_port(*port), ports):
                print(json.dumps(result), flush=True)
            if tick < 2:
                time.sleep(max(0, 1 - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
