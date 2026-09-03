"""Persistent read-only NVML GPU/NVLink sampling; no device configuration calls.

The pure Python binding is copied, hash-verified, from the pinned training image.
Errors stay explicit and create a run-owned failure marker. An owning process
must also supervise the heartbeat because a blocked driver call cannot self-timeout.
"""
import hashlib
import importlib.util
from pathlib import Path
import time

from evidence import utcnow
from telemetry_native import nvlink_records


def field_records(values, uuid, expected_links=18):
    """NVML field IDs 138/139 are per-link cumulative data payload in KiB."""
    expected = {(field, link) for field in (138, 139) for link in range(expected_links)}
    seen, rows = set(), []
    for field in values:
        key = (field.fieldId, field.scopeId)
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate NVLink field identity.')
        seen.add(key)
        attrs = {'gpu_uuid': uuid, 'link': field.scopeId, 'nvml_field_id': field.fieldId,
                 'nvml_sample_timestamp_us': field.timestamp, 'nvml_query_latency_us': field.latencyUsec}
        if field.nvmlReturn != 0:
            rows.append(dict(attrs, metric='collector_error', value=None, unit='event',
                             error='NVML field status ' + str(field.nvmlReturn)))
            continue
        # The installed API declares these integer counters as unsigned long long.
        if field.valueType != 3 or field.timestamp <= 0:
            raise ValueError('Unexpected NVLink counter type or missing field timestamp.')
        direction = 'tx' if field.fieldId == 138 else 'rx'
        rows.append(dict(attrs, metric='nvlink_data_' + direction + '_bytes_total',
                         value=int(field.value.ullVal) * 1024, unit='B'))
    if seen != expected:
        raise ValueError('Incomplete per-link NVML field response.')
    return rows


def gpu_snapshot(nvml, handle, uuid):
    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
    values = [
        ('utilization.gpu', utilization.gpu, '%'),
        ('utilization.memory', utilization.memory, '%'),
        ('memory.total', memory.total / 1024**2, 'MiB'),
        ('memory.free', memory.free / 1024**2, 'MiB'),
        ('memory.used', memory.used / 1024**2, 'MiB'),
        ('temperature.gpu', nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU), 'degC'),
        ('power.draw', nvml.nvmlDeviceGetPowerUsage(handle) / 1000, 'W'),
        ('clocks.current.sm', nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM), 'MHz'),
        ('clocks.current.memory', nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM), 'MHz'),
        ('ecc.errors.corrected.volatile.total', nvml.nvmlDeviceGetTotalEccErrors(
            handle, nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, nvml.NVML_VOLATILE_ECC), 'count'),
        ('ecc.errors.uncorrected.volatile.total', nvml.nvmlDeviceGetTotalEccErrors(
            handle, nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nvml.NVML_VOLATILE_ECC), 'count'),
    ]
    return [dict(gpu_uuid=uuid, metric=name, value=value, unit=unit) for name, value, unit in values]


def load_binding(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Pinned NVML binding hash mismatch or symlink.')
    spec = importlib.util.spec_from_file_location('posttrainingx_pinned_nvml', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



BINDING_SHA256 = 'd3233c78cd3f17bda97850346eba30fa88f1d6b295cee90fa7ec1c6cbf291e5b'


class NVMLSampler:
    """Read-only sampler. The owner must supervise heartbeat and shutdown deadlines."""
    def __init__(self, binding, expected_uuids, streams, common):
        self.nvml = load_binding(binding, BINDING_SHA256)
        self.streams, self.common = streams, common
        self.initialized = False
        self.counters = {}
        self.nvml.nvmlInit()
        self.initialized = True
        self.handles = [self.nvml.nvmlDeviceGetHandleByIndex(i)
                        for i in range(self.nvml.nvmlDeviceGetCount())]
        self.uuids = [self.nvml.nvmlDeviceGetUUID(handle) for handle in self.handles]
        if len(self.uuids) != 8 or len(set(self.uuids)) != 8 or set(self.uuids) != set(expected_uuids):
            raise ValueError('NVML UUIDs do not reconcile to the frozen eight physical GPUs.')
        self.before = self.reference('before')

    def reference(self, boundary):
        import subprocess
        argv = ['nvidia-smi', 'nvlink', '-gt', 'd']
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        self.streams.write('raw-nvlink', dict(self.common, time=utcnow(), monotonic_s=time.monotonic(),
            source='nvlink', boundary=boundary, argv=argv, exit_code=result.returncode,
            stdout=result.stdout, stderr=result.stderr))
        if result.returncode:
            raise RuntimeError('NVLink CLI boundary reference failed: ' + boundary)
        rows = nvlink_records(result.stdout)
        counters = {(r['gpu_uuid'], r['link'], r['metric']): r['value'] for r in rows}
        if len(counters) != 288 or {key[0] for key in counters} != set(self.uuids):
            raise ValueError('CLI reference has duplicate or mismatched NVLink identities.')
        return counters

    def sample(self, common):
        for handle, uuid in zip(self.handles, self.uuids):
            for row in gpu_snapshot(self.nvml, handle, uuid):
                self.streams.write('nvidia-smi', dict(common, source='persistent-nvml', **row))
            fields = self.nvml.nvmlDeviceGetFieldValues(handle,
                [(field, link) for field in (138, 139) for link in range(18)])
            for row in field_records(fields, uuid):
                self.streams.write('nvlink', dict(common, source='persistent-nvml', **row))
                if row['metric'] == 'collector_error':
                    raise ValueError('NVML returned an unsuccessful per-link field.')
                key = (uuid, row['link'], row['metric'])
                if row['value'] < self.counters.get(key, self.before[key]):
                    raise ValueError('NVLink counter reset or disagreement with CLI reference: ' + str(key))
                self.counters[key] = row['value']

    def finish(self):
        after = self.reference('after')
        if len(self.counters) != 288 or any(value > after[key] for key, value in self.counters.items()):
            raise ValueError('NVML counters do not lie between the CLI boundary snapshots.')
        return {'binding_sha256': BINDING_SHA256, 'gpu_uuids': self.uuids,
                'nvlink_counter_identities': len(self.counters), 'cli_counter_bracket_passed': True}

    def shutdown(self):
        if self.initialized:
            self.nvml.nvmlShutdown()
            self.initialized = False
