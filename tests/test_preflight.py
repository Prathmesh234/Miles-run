import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evidence import Run, markdown, sha256
from infra_controller import parse_nccl, srun
from telemetry_native import GPU_FIELDS, gpu_records, nvlink_records


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
