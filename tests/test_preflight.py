import json
import base64
import gzip
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evidence import Run, markdown, sha256
from infra_controller import parse_nccl, srun
from telemetry_native import GPU_FIELDS, gpu_records, nvlink_records
from submit_native_preflight import batches, entry
from fabric_probe import active_training_ports, capture_port, perfquery_command, perfquery_records, write_port_capture
from summarize_native import counter_rate, summary
from pull_pinned_model import download_file, validate_manifest
import io
import threading
from validate_fabric_under_load import validate_records
from telemetry_lustre_host import stats_records
from enroot_run_config import prepare as prepare_enroot_config
from runtime_inventory import parse_inventory_stdout


class EvidenceTests(unittest.TestCase):
    def test_failed_command_cannot_be_success_and_evidence_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Run.create(Path(tmp) / 'run', {})
            phase = run.phase('failure')
            phase.command([sys.executable, '-c', 'import sys; print("kept"); sys.exit(7)'])
            with self.assertRaises(ValueError):
                phase.finish('ok')
            result = phase.finish('fail', failure_summary='Expected nonzero command.')
            self.assertEqual(result['exit_code'], 7)
            self.assertEqual(sha256(run.root / result['log_relpath']), result['log_sha256'])
            self.assertEqual((phase.path / 'failure.md').read_text(), markdown(result))
            self.assertIn('kept', (run.root / result['log_relpath']).read_text())
            summary = json.loads((run.root / 'sweep.summary.json').read_text())
            self.assertEqual(summary['counts'], {'ok': 0, 'fail': 1, 'skip': 0})
            with self.assertRaises(FileExistsError):
                run.phase('failure')

    def test_timeout_preserves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Run.create(Path(tmp) / 'run', {})
            phase = run.phase('timeout')
            code, out, _ = phase.command([sys.executable, '-c', 'import time; print("before", flush=True); time.sleep(5)'], timeout=0.1)
            self.assertEqual(code, 124)
            self.assertIn('before', out)
            result = phase.finish('fail', failure_summary='Expected test timeout.')
            self.assertTrue(result['timeout'])


class ParserTests(unittest.TestCase):
    def test_runtime_inventory_accepts_entrypoint_banner_but_not_ambiguous_records(self):
        record = {'python': '3.12', 'packages': [], 'torch': {}, 'scope': 'CPU only'}
        text = 'CUDA banner\n{}\n' + json.dumps(record) + '\n'
        self.assertEqual(parse_inventory_stdout(text), record)
        with self.assertRaises(ValueError):
            parse_inventory_stdout(text + json.dumps(record))
        with self.assertRaises(ValueError):
            parse_inventory_stdout('no JSON inventory')

    def test_enroot_hook_patch_is_run_scoped_and_rejects_unexpected_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'system'
            hook = source / 'hooks.d/10-devices.sh'
            hook.parent.mkdir(parents=True)
            original = '#!/bin/bash\n/dev/log /dev/log none x-create=file,bind,rw,private\n'
            hook.write_text(original)
            env = prepare_enroot_config(Path(tmp) / 'run', source)
            self.assertEqual(hook.read_text(), original)
            self.assertEqual((Path(env['ENROOT_SYSCONF_PATH']) / 'hooks.d/10-devices.sh').read_text(),
                             original.replace('private', 'private,nofail,silent'))
            self.assertEqual(env['ENROOT_MOUNT_HOME'], 'n')
            hook.write_text('#!/bin/bash\n')
            with self.assertRaises(ValueError):
                prepare_enroot_config(Path(tmp) / 'invalid-run', source)

    def test_lustre_aggregate_units_and_saturated_moment_are_not_invented(self):
        text = 'snapshot_time 123 secs.nsecs\nread_bytes 4 samples [bytes] 1 8 16 9223372036854775807\nopen 2 samples [usecs] 1 3 4 10\nioctl 3 samples [reqs]\n'
        rows = {r['metric']: r for r in stats_records(text)}
        self.assertEqual(rows['read_bytes.sum']['value'], 16)
        self.assertEqual(rows['read_bytes.sum']['unit'], 'B')
        self.assertEqual(rows['open.sum']['unit'], 'us')
        self.assertEqual(rows['open.max']['kind'], 'lifetime_aggregate')
        self.assertNotIn('read_bytes.sum_squares', rows)
        with self.assertRaises(ValueError):
            stats_records(text.replace('[bytes]', '[unknown]'))
        with self.assertRaises(ValueError):
            stats_records('snapshot_time 123 secs.nsecs\n')

    def test_fabric_load_gate_requires_every_rail_and_rejects_reset(self):
        rows = [dict(hca=f'mlx5_{i}', hca_port='1', metric=name, value=value)
                for i in range(8) for name in ('PortXmitData', 'PortRcvData') for value in (1, 2, 3)]
        errors, metrics = validate_records(rows, 'test-node')
        self.assertFalse(errors)
        self.assertEqual(len(metrics), 16)
        self.assertTrue(validate_records(rows[6:], 'test-node')[0])
        rows[1]['value'] = 4
        self.assertTrue(validate_records(rows, 'test-node')[0])

    def test_perfquery_failure_preserves_raw_evidence_and_allocation_context(self):
        failed = subprocess.CompletedProcess(['perfquery'], 7, 'partial counters', 'port unavailable')
        with patch('fabric_probe.subprocess.run', return_value=failed):
            result = capture_port('mlx5_0', '1', {'slurm_job_id': '123', 'role': 'trainer'})
        self.assertEqual(result['raw']['stdout'], 'partial counters')
        self.assertEqual(result['raw']['exit_code'], 7)
        self.assertEqual(result['records'][0]['slurm_job_id'], '123')
        self.assertEqual(result['records'][0]['role'], 'trainer')
        self.assertIsNone(result['records'][0]['value'])
        writes = []
        class Sink:
            def write(self, name, value):
                writes.append((name, value))
        write_port_capture(Sink(), result)
        self.assertEqual([name for name, _ in writes], ['raw-infiniband', 'infiniband'])

    def test_model_download_checks_hash_size_and_never_overwrites(self):
        payload = b'pinned model data'
        class Response(io.BytesIO):
            status = 200
        item = {'path': 'model.safetensors', 'size': len(payload),
                'lfs': {'sha256': hashlib.sha256(payload).hexdigest()}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            result = download_file(item, 'https://example.invalid/pinned', root,
                                   threading.Event(), opener=lambda *a, **k: Response(payload))
            self.assertEqual(result['sha256'], item['lfs']['sha256'])
            self.assertEqual((root/item['path']).read_bytes(), payload)
            with self.assertRaises(ValueError):
                download_file(item, 'https://example.invalid/pinned', root, threading.Event())
            bad = dict(item, path='bad.safetensors')
            with self.assertRaises(ValueError):
                download_file(bad, 'https://example.invalid/pinned', root,
                              threading.Event(), opener=lambda *a, **k: Response(b'bad'))
            self.assertFalse((root/'bad.safetensors').exists())
            self.assertTrue((root/'bad.safetensors.partial').exists())

    def test_statistics_and_counter_discontinuities_are_not_zero_filled(self):
        result = summary([1, 2, 3, 4])
        self.assertEqual(result['median'], 2.5)
        self.assertAlmostEqual(result['p95'], 3.85)
        self.assertIsNone(summary([0, 0])['coefficient_of_variation'])
        before = {'monotonic_s': 1, 'value': 100}
        self.assertEqual(counter_rate(before, {'monotonic_s': 3, 'value': 140}), 20)
        self.assertIsNone(counter_rate(before, {'monotonic_s': 3, 'value': 1}))
        self.assertIsNone(counter_rate(before, {'monotonic_s': 8, 'value': 200}))
        self.assertIsNone(counter_rate(before, {'monotonic_s': 1, 'value': 200}))

    def test_perfquery_units_missing_data_and_no_reset_arguments(self):
        text = '\n'.join(f'{k}:....{v}' for k, v in {
            'PortSelect': 1, 'PortXmitData': 12, 'PortRcvData': 4,
            'PortXmitPkts': 2, 'PortRcvPkts': 3, 'PortXmitWait': 7,
        }.items())
        rows = {x['metric']: x for x in perfquery_records(text, 'mlx5_0', '1')}
        self.assertEqual(rows['PortXmitData']['value'], 48)
        self.assertEqual(rows['PortRcvData']['value'], 16)
        self.assertEqual(rows['PortXmitWait']['unit'], 'pma_ticks')
        self.assertNotIn('PortSelect', rows)
        with self.assertRaises(ValueError):
            perfquery_records(text.replace('PortRcvData', 'Absent'), 'mlx5_0', '1')
        self.assertEqual(perfquery_command('mlx5_0', 1), ['perfquery', '-x', '-C', 'mlx5_0', '-P', '1'])
        with self.assertRaises(ValueError):
            perfquery_command('mlx5_0', '1 -R')

    def test_fabric_discovery_excludes_storage_and_ethernet_ports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(10):
                p = root / f'mlx5_{i}/ports/1'
                p.mkdir(parents=True)
                (p/'rate').write_text('400 Gb/sec' if i != 8 else '100 Gb/sec')
                (p/'state').write_text('4: ACTIVE')
                (p/'link_layer').write_text('InfiniBand' if i != 9 else 'Ethernet')
            self.assertEqual(len(active_training_ports(root)), 8)
            (root/'mlx5_0/ports/1/state').write_text('1: DOWN')
            with self.assertRaises(ValueError):
                active_training_ports(root)

    def test_staging_batches_bound_payload_and_preserve_hashes(self):
        original = {str(i): entry(os.urandom(4000)) for i in range(5)}
        payloads = list(batches({'root': '/example', 'create': False}, original, limit=10000))
        self.assertGreater(len(payloads), 1)
        rebuilt = {}
        for payload in payloads:
            self.assertLessEqual(len(payload), 10000)
            rebuilt.update(json.loads(gzip.decompress(base64.b64decode(payload)))['files'])
        self.assertEqual(rebuilt, original)
        for item in rebuilt.values():
            self.assertEqual(hashlib.sha256(base64.b64decode(item['data'])).hexdigest(), item['sha256'])
        with self.assertRaises(ValueError):
            list(batches({}, {'too-large': entry(os.urandom(20000))}, limit=10000))

    def test_nccl_requires_success_and_preserves_modes_and_units(self):
        output = '8388608 2097152 float sum -1 10.0 800.0 1400.0 0 11.0 700.0 1300.0 0\n# Out of bounds values : 0 OK\n'
        rows = parse_nccl(output, 4)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['labels']['ranks'], 32)
        self.assertEqual(rows[3]['labels']['mode'], 'in_place')
        with self.assertRaises(ValueError):
            parse_nccl(output.replace('0 OK', '1 FAILED'), 4)
        with self.assertRaises(ValueError):
            parse_nccl(output.replace('1400.0 0', '1400.0 1'), 4)

    def test_missing_gpu_values_are_errors_not_zero(self):
        text = '\n'.join(', '.join([str(i), 'GPU-test' + str(i)] + ['1']*(len(GPU_FIELDS)-2)) for i in range(8))
        rows = gpu_records(text.replace('GPU-test0, 1', 'GPU-test0, [N/A]', 1))
        error = next(r for r in rows if r['metric'] == 'collector_error')
        self.assertIsNone(error['value'])
        self.assertEqual(error['requested_metric'], 'utilization.gpu')
        with self.assertRaises(ValueError):
            gpu_records(text.splitlines()[0])

    def test_nvlink_units_and_link_completeness(self):
        text = '\n'.join('GPU ' + str(g) + ': B200 (UUID: GPU-' + str(g) + ')\n' + '\n'.join(
            f' Link {link}: Data {direction}: 12 KiB' for link in range(18) for direction in ['Tx', 'Rx']) for g in range(8))
        rows = nvlink_records(text)
        self.assertEqual(rows[0]['value'], 12288)
        self.assertEqual(rows[-1]['gpu_uuid'], 'GPU-7')
        with self.assertRaises(ValueError):
            nvlink_records(text.replace('Data Tx: 12 KiB', 'Data Tx: [N/A]', 1))

    def test_explicit_whole_node_placement(self):
        command = srun(['gpu-nodes-1', 'gpu-nodes-3'], mpi='pmix')
        self.assertIn('--nodelist=gpu-nodes-1,gpu-nodes-3', command)
        self.assertIn('--ntasks=2', command)
        self.assertIn('--mpi=pmix', command)


if __name__ == '__main__':
    unittest.main()
