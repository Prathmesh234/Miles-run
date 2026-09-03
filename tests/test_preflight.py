import json
import base64
import gzip
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import tarfile
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
from telemetry_lustre_host import stats_records, duration_seconds
from enroot_run_config import prepare as prepare_enroot_config
from runtime_inventory import parse_inventory_stdout
from qwen_serving_probe import server_command, prompt_token_ids, prometheus_rows, stop_owned_server
from model_conversion import conversion_command, checkpoint_files
from checkpoint_parity import ALOG_WIDENINGS, reference_part, check_coverage, required_parts, verify_files
from prepare_terminal_lego import CATALOG_URL, next_page, select_tasks
from materialize_terminal_lego import configure_git_environment, inventory_tasks
from materialize_terminal_lego_lfs import copy_verified_sources, lfs_item, tracked_files
from trainer_probe import trainer_command, parameter_hashes, gradient_statistics
from audit_trainer_probe import validate_rank_evidence
from report_trainer_probe import analyze_streams
from prepare_local_task_images import offline_harness, pin_dockerfile
from local_file_env import validate_archive, tree_archive, atomic_bytes, archive_contents, FileTaskSession
from grpo_node import cleanup_actions
from audit_grpo_attempt import audit_remote as audit_grpo_remote
from audit_local_episodes import summarize_episode
from container_fabric_probe import verify_rdma


class EvidenceTests(unittest.TestCase):
    def test_resume_state_comparison_is_bitwise_and_rejects_missing_or_nonfinite_state(self):
        import torch
        from resume_checkpoint_probe import compare_values, move_tensors
        value = {'model': torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
                 'scheduler': {'num_steps': 16}, 'rng': (b'bytes', [1, 2, 3])}
        self.assertTrue(all(row['equal'] for row in compare_values(value, value)))
        changed = dict(value, scheduler={'num_steps': 32})
        self.assertFalse(all(row['equal'] for row in compare_values(value, changed)))
        self.assertFalse(compare_values(torch.tensor([0.]), torch.tensor([-0.]))[0]['equal'])
        self.assertFalse(compare_values(torch.tensor([float('nan')]), torch.tensor([float('nan')]))[0]['equal'])
        self.assertFalse(compare_values({'optimizer': [1]}, {'optimizer': [1, 2]})[0]['equal'])
        self.assertFalse(compare_values({'state': 1}, {})[0]['equal'])
        self.assertEqual(move_tensors({'metadata': ['unchanged', 3]}), {'metadata': ['unchanged', 3]})

    def test_resume_checkpoint_identity_rejects_modified_and_symlinked_payloads(self):
        from resume_checkpoint_probe import verify_payload_identity
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / 'rank.distcp'
            payload.write_bytes(b'fixture')
            info = payload.stat()
            expected = {'rank.distcp': dict(bytes=info.st_size, inode=info.st_ino, mtime_ns=info.st_mtime_ns)}
            self.assertEqual(verify_payload_identity(root, expected), expected)
            payload.write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError, 'identity changed.*expected.*observed'):
                verify_payload_identity(root, expected)
            payload.unlink()
            other = root / 'other'
            other.write_bytes(b'fixture')
            payload.symlink_to(other)
            with self.assertRaisesRegex(ValueError, 'symlink refused'):
                verify_payload_identity(root, expected)

    def test_sync_digest_reports_current_findings_without_historical_failure_claims(self):
        from report_journal_validation import render
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / 'docs/grpo-validation-job161.json').read_text())
        data.update(findings=[], qualification='component_validation_only', slurm_state='COMPLETED',
                    slurm_exit_code='0:0', slow_nvml_calls=[], host_cleanup=None)
        text = render(data)
        self.assertIn('No findings in these component audits', text)
        self.assertNotIn('overlong collector ticks occurred', text)
        data['findings'] = ['Synthetic missing metric']
        self.assertIn('Synthetic missing metric', render(data))

    def test_full_trainer_host_cleanup_requires_identity_coverage_and_allocator_release(self):
        import copy
        from audit_grpo_attempt import audit_host_cleanup
        events = []
        for rank in range(16):
            common = dict(rank=rank, hostname='trainer-' + str(rank // 8), pid=100+rank,
                          slurm_job_id='167', policy_version=2)
            events.append(dict(common, event='training_host_cleanup_started', monotonic_s=10.0))
            events.append(dict(common, event='training_host_cleanup_completed', monotonic_s=12.0,
                duration_s=1.0, tensor_bytes=1024,
                before={'active_bytes.current': 2048, 'allocated_bytes.current': 4096},
                after={'active_bytes.current': 1024, 'allocated_bytes.current': 1024}))
        hosts = ['trainer-0', 'trainer-1']
        self.assertFalse(audit_host_cleanup(events, hosts, '167')['findings'])
        for changed in (events[:-2], events + [events[-1]]):
            self.assertTrue(audit_host_cleanup(changed, hosts, '167')['findings'])
        for key, value in [('policy_version', 3), ('hostname', 'wrong'), ('slurm_job_id', '168'),
                           ('duration_s', float('nan')), ('tensor_bytes', 0),
                           ('after', {'active_bytes.current': 2048, 'allocated_bytes.current': 4096})]:
            changed = copy.deepcopy(events)
            changed[-1][key] = value
            self.assertTrue(audit_host_cleanup(changed, hosts, '167')['findings'], key)

    def test_ray_placement_capture_excludes_runtime_secrets_and_rejects_partial_state(self):
        from collect_ray_placements import placements
        row = dict(actor_id='a', node_id='n', state='ALIVE', class_name='Trainer',
            required_resources={'GPU': 1}, serialized_runtime_env='do-not-persist', call_site='private')
        body = dict(result=[row], num_filtered=1, num_after_truncation=1, warnings=[], partial_failure_warning='')
        packet = dict(result=True, data={'result': body})
        result = placements(packet, {'n': 'node'})
        self.assertEqual(result[0]['assigned_hostname'], 'node')
        self.assertNotIn('serialized_runtime_env', result[0])
        self.assertNotIn('call_site', result[0])
        with self.assertRaisesRegex(ValueError, 'outside'):
            placements(packet, {})
        body['num_filtered'] = 2
        with self.assertRaisesRegex(ValueError, 'truncated'):
            placements(packet, {'n': 'node'})
        body['num_filtered'] = 1
        body['partial_failure_warning'] = 'missing node'
        with self.assertRaisesRegex(ValueError, 'partial'):
            placements(packet, {'n': 'node'})

    def test_teardown_audit_requires_every_rank_and_full_host_release(self):
        from audit_nvml_validation import audit_teardown_log
        events = []
        host_bytes = 24 * 1024**3
        for rank in range(8):
            common = dict(rank=rank, hostname='node', time='2026-09-03T03:00:00Z')
            for index, kind in enumerate(['host_capacity_guard', 'before_allocate', 'allocated',
                                         'before_pinned_release', 'pinned_released', 'exit_with_live_context']):
                row = dict(event=kind, **common, monotonic_s=index)
                if kind == 'allocated':
                    row.update(allocation_count=4096, chunk_mib=16, pinned_bytes=host_bytes, pinned_allocation_count=3072)
                if kind == 'pinned_released':
                    row.update(expected_released_bytes=host_bytes, duration_s=1.0,
                        before={'active_bytes.current': host_bytes, 'allocated_bytes.current': host_bytes},
                        after={'active_bytes.current': 0, 'allocated_bytes.current': 0})
                if kind == 'exit_with_live_context':
                    row['pinned_bytes'] = 0
                events.append(row)
        # The real stream has NCCL text between JSON events.
        def stream():
            return '\nNCCL INFO control\n'.join(json.dumps(e) for e in events)
        rows = audit_teardown_log(stream(), 'node', 'pinned-host-clean-teardown')
        self.assertEqual(len(rows), 8)
        events[4]['after']['allocated_bytes.current'] = 1
        with self.assertRaisesRegex(ValueError, 'full control payload'):
            audit_teardown_log(stream(), 'node', 'pinned-host-clean-teardown')
        events.pop()
        with self.assertRaisesRegex(ValueError, 'rank teardown evidence'):
            audit_teardown_log(stream(), 'node', 'pinned-host-clean-teardown')

    def test_teardown_probe_respects_host_and_cgroup_memory_reserves(self):
        from teardown_probe import available_host_bytes, host_allocation_guard
        gib = 1024**3
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            info = f'MemAvailable: {1024 * 1024**2} kB\n'
            self.assertEqual(available_host_bytes(info, root), 1024 * gib)
            (root / 'memory.max').write_text(str(512 * gib))
            (root / 'memory.current').write_text(str(64 * gib))
            available = available_host_bytes(info, root)
            self.assertEqual(available, 448 * gib)
            self.assertEqual(host_allocation_guard(available, 24 * gib, 8), 320 * gib)
            with self.assertRaisesRegex(RuntimeError, 'Host memory reserve'):
                host_allocation_guard(319 * gib, 24 * gib, 8)
            (root / 'memory.max').write_text('max')
            self.assertEqual(available_host_bytes(info, root), 1024 * gib)

    def test_pinned_teardown_profile_is_opt_in_and_keeps_whole_node(self):
        from types import SimpleNamespace
        from validate_nvml_under_load import teardown_command
        with tempfile.TemporaryDirectory() as temporary, patch('enroot_run_config.prepare', return_value={}):
            run = SimpleNamespace(root=Path(temporary))
            code = Path(temporary)
            pinned, _ = teardown_command(run, code, 'pinned', 'node', 'pinned-host-nccl-teardown')
            old, _ = teardown_command(run, code, 'control', 'node', 'fragmented-nccl-teardown')
            clean, _ = teardown_command(run, code, 'clean', 'node', 'pinned-host-clean-teardown')
            self.assertIn('PTX_PROBE_PINNED_GIB=24', pinned)
            self.assertIn('PTX_PROBE_PINNED_GIB=0', old)
            self.assertIn('PTX_PROBE_RELEASE_PINNED=0', pinned)
            self.assertIn('PTX_PROBE_RELEASE_PINNED=1', clean)
            self.assertIn('PTX_PROBE_PINNED_GIB=24', clean)
            for command in [pinned, old, clean]:
                self.assertIn('PTX_PROBE_NCCL=1', command)
                self.assertIn('PTX_PROBE_CHUNK_MIB=16', command)
                self.assertIn('--nproc-per-node=8', command)
            with self.assertRaisesRegex(ValueError, 'Unknown teardown'):
                teardown_command(run, code, 'invalid', 'node', 'typo')

    def test_pinned_buffer_cleanup_checks_release_and_keeps_other_resources(self):
        from types import SimpleNamespace
        from teardown_probe import release_pinned_buffers
        buffers = [SimpleNamespace(numel=lambda: 8, element_size=lambda: 1)]
        operations = []
        stats = [{'active_bytes.current': 8, 'allocated_bytes.current': 8},
                 {'active_bytes.current': 0, 'allocated_bytes.current': 0}]
        def empty_cache():
            self.assertEqual(buffers, [])
            operations.append('host_cache')
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: operations.append('sync'),
            memory=SimpleNamespace(host_memory_stats=lambda: stats.pop(0))),
            _C=SimpleNamespace(_host_emptyCache=empty_cache))
        result = release_pinned_buffers(torch, buffers)
        self.assertEqual(operations, ['sync', 'host_cache'])
        self.assertEqual(result['expected_released_bytes'], 8)
        buffers.append(SimpleNamespace(numel=lambda: 8, element_size=lambda: 1))
        stats.extend([{'active_bytes.current': 8, 'allocated_bytes.current': 8}] * 2)
        with self.assertRaisesRegex(RuntimeError, 'not fully released'):
            release_pinned_buffers(torch, buffers)

    def test_nvml_call_audit_keeps_stalls_errors_and_rejects_missing_gpu_or_invalid_timing(self):
        from audit_nvml_calls import analyze_calls
        rows = [dict(time='2026-09-03T00:00:00Z', monotonic_s=1.0, hostname='node',
            slurm_job_id='161', gpu_uuid=gpu, metric='nvml_api_duration', source='persistent-nvml',
            unit='s', api='memory', value=value, error=None) for gpu, value in [('a', 0.1), ('b', 15.0)]]
        result = analyze_calls(rows, 'node', 161, {'a', 'b'})
        self.assertIn('12s', result['findings'][0])
        self.assertEqual(result['slowest_calls'][0]['gpu_uuid'], 'b')
        self.assertEqual(result['by_api'][0]['statistics']['max'], 15.0)
        rows[1]['error'] = 'driver error'
        self.assertEqual(analyze_calls(rows, 'node', 161, {'a', 'b'})['error_count'], 1)
        with self.assertRaisesRegex(ValueError, 'every expected GPU'):
            analyze_calls(rows[:1], 'node', 161, {'a', 'b'})
        with self.assertRaisesRegex(ValueError, 'identity'):
            analyze_calls(rows, 'wrong-node', 161, {'a', 'b'})
        rows[1]['value'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid NVML call timing'):
            analyze_calls(rows, 'node', 161, {'a', 'b'})

    def test_journal_joins_discarded_and_cancelled_work_without_reward_inference(self):
        from audit_trajectory_journal import audit_journal
        from copy import deepcopy
        selected = dict(attempt_id='a', sample_index=7, group_index=1, task_id='task',
            tokens=[1, 2], rollout_log_probs=[-0.1], loss_mask=[1], response_length=1,
            reward=1.0, weight_versions=[2], environment_attempts=[dict(episode_id='e1', task_id='task')])
        discarded = dict(selected, sample_index=8, environment_attempts=[dict(episode_id='e2', task_id='task')])
        cancelled = dict(selected, attempt_id='b', sample_index=9, environment_attempts=[])
        def group(kind, *samples):
            return dict(event=kind, samples=list(samples))
        events = [group('group_submitted', selected, discarded), group('group_submitted', cancelled),
            group('group_returned', selected, discarded), group('selected_for_training', selected),
            group('sync_excess_discarded', discarded), group('group_cancelled', cancelled)]
        for sample, eid in [(selected, 'e1'), (discarded, 'e2'), (cancelled, 'e3')]:
            events.append(dict(event='environment_attempt', episode_id=eid,
                **{key: sample[key] for key in ('attempt_id', 'sample_index', 'group_index', 'task_id')}))
        episodes = [dict(episode_id=eid, task_id='task') for eid in ('e1', 'e2', 'e3')]
        native = [dict(index=7, **{k: selected[k] for k in ('tokens', 'rollout_log_probs', 'loss_mask', 'response_length', 'reward')},
            metadata={'_miles_journal_attempt_id': 'a'})]
        result = audit_journal(events, episodes, native, [7])
        self.assertEqual(result['findings'], [])
        self.assertEqual(result['counts']['environment_attempts'], 3)
        self.assertEqual(result['counts']['dispositions']['group_cancelled'], 1)
        self.assertEqual(result['counts']['unjoined_episodes'], 0)
        changed = deepcopy(events)
        changed[2]['samples'][0]['tokens'] = [1, 3]
        self.assertTrue(any('native tokens changed' in f for f in audit_journal(changed, episodes, native, [7])['findings']))
        missing = [event for event in events if event['event'] != 'sync_excess_discarded']
        self.assertTrue(any('ambiguous submission/disposition' in f for f in audit_journal(missing, episodes, native, [7])['findings']))
        unjoined = audit_journal(events[:-1], episodes, native, [7])
        self.assertEqual(unjoined['unjoined_episode_ids'], ['e3'])
        self.assertTrue(any('trainer inputs' in f for f in audit_journal(events, episodes, native, [])['findings']))
        with self.assertRaisesRegex(ValueError, 'Repeated durable environment'):
            audit_journal(events + [events[-1]], episodes, native, [7])

    def test_journal_reader_rejects_mutation_and_unfinished_writes(self):
        from audit_trajectory_journal import read_journal
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'training/trajectory-journal'
            directory.mkdir(parents=True)
            raw = json.dumps(dict(schema_version=1, event='group_submitted')).encode()
            path = directory / ('group_submitted-id-' + hashlib.sha256(raw).hexdigest() + '.json')
            path.write_bytes(raw)
            events, hashes = read_journal(root, directory.parent)
            self.assertEqual(len(events), 1)
            self.assertEqual(hashes[str(path.relative_to(root))], sha256(path))
            path.write_bytes(raw + b' ')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                read_journal(root, directory.parent)
            path.unlink()
            (directory / '.event-incomplete').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                read_journal(root, directory.parent)

    def test_episode_join_requires_native_ids_rewards_and_unique_trainer_membership(self):
        from audit_local_episodes import join_episode_ids
        episodes = [dict(episode_id='e1', task_id='task', category='graded', reward=1.0, event_span_s=3)]
        samples = [dict(index=7, group_index=1, reward=1.0, metadata=dict(task_id='task',
            posttrainingx_environment_attempts=[dict(episode_id='e1', task_id='task')]))]
        result = join_episode_ids(episodes, samples, [7])
        self.assertEqual(result['findings'], [])
        self.assertEqual(result['counts']['joined_episodes'], 1)
        with self.assertRaisesRegex(ValueError, 'more than one'):
            join_episode_ids(episodes, samples, [7, 7])
        samples[0]['reward'] = 0.0
        self.assertIn('canonical reward differs', join_episode_ids(episodes, samples, [7])['findings'][0])
        missing = join_episode_ids(episodes, [], [7])
        self.assertEqual(missing['unjoined_episode_ids'], ['e1'])
        self.assertEqual(missing['missing_trainer_sample_ids'], [7])
        self.assertEqual(missing['counts']['retry_attempts'], 0)

    def test_pinned_binding_import_registers_module_before_defining_exports(self):
        from telemetry_nvml import load_binding
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'binding.py'
            source.write_text('import sys\nassert sys.modules[__name__].__file__ == __file__\nvalue = 138\n')
            module = load_binding(source, sha256(source))
            self.assertEqual(module.value, 138)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_binding(source, '0' * 64)
            source.write_text('raise ValueError("import fixture")\n')
            with self.assertRaisesRegex(ValueError, 'import fixture'):
                load_binding(source, sha256(source))
            self.assertIs(sys.modules['posttrainingx_pinned_nvml'], module)

    def test_telemetry_heartbeat_rejects_stall_error_missing_and_wrong_identity(self):
        from telemetry_health import heartbeat, assert_healthy, require_healthy
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(assert_healthy(root, 'node', '1', 100))
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                require_healthy(root, 'node', '1', 100)
            heartbeat(root, 'node', '1', 3, 0, 99)
            self.assertTrue(assert_healthy(root, 'node', '1', 100))
            for now in (112, 98, float('nan')):
                with self.assertRaisesRegex(RuntimeError, 'stalled'):
                    assert_healthy(root, 'node', '1', now)
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                assert_healthy(root, 'other', '1', 100)
            heartbeat(root, 'node', '1', 4, 1, 100)
            with self.assertRaisesRegex(RuntimeError, 'errors'):
                assert_healthy(root, 'node', '1', 100)
            (root / 'failure.json').write_text('{}')
            heartbeat(root, 'node', '1', 5, 0, 100)
            with self.assertRaisesRegex(RuntimeError, 'failure marker'):
                assert_healthy(root, 'node', '1', 100)

    def test_nvml_counter_fields_validate_units_types_identities_and_failures(self):
        from types import SimpleNamespace as S
        from telemetry_nvml import field_records
        fields = [S(fieldId=f, scopeId=l, timestamp=1000000, latencyUsec=2,
                    nvmlReturn=0, valueType=3, value=S(ullVal=7))
                  for f in (138, 139) for l in range(18)]
        rows = field_records(fields, 'GPU-fixture')
        self.assertEqual(len(rows), 36)
        self.assertTrue(all(r['value'] == 7168 and r['unit'] == 'B' for r in rows))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            field_records(fields[:-1], 'GPU-fixture')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            field_records(fields + [fields[0]], 'GPU-fixture')
        fields[0].valueType = 0
        with self.assertRaisesRegex(ValueError, 'type'):
            field_records(fields, 'GPU-fixture')
        fields[0].nvmlReturn = 3
        error = field_records(fields, 'GPU-fixture')[0]
        self.assertEqual(error['metric'], 'collector_error')
        self.assertIsNone(error['value'])

    def test_heartbeat_reads_clock_after_file_and_preserves_supervisor_error(self):
        from telemetry_health import heartbeat, assert_healthy, require_healthy
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heartbeat(root, 'node', '1', 3, 0, 100.01)
            order = []
            original = Path.read_text
            def read(path, *args, **kwargs):
                order.append('read')
                return original(path, *args, **kwargs)
            def clock():
                order.append('clock')
                return 100.02
            with patch.object(Path, 'read_text', read), patch('telemetry_health.time.monotonic', clock):
                self.assertTrue(assert_healthy(root, 'node', '1'))
            self.assertEqual(order, ['read', 'clock'])
            with self.assertRaisesRegex(RuntimeError, 'limit_s=12'):
                require_healthy(root, 'node', '1', 113)
            path = root / 'supervisor-error.jsonl'
            before = path.read_bytes()
            row = json.loads(before)
            self.assertEqual(row['metric'], 'collector_error')
            self.assertIsNone(row['value'])
            self.assertIn(str(root), row['error'])
            with self.assertRaises(RuntimeError):
                require_healthy(root, 'node', '1', 114)
            self.assertEqual(before, path.read_bytes())

    def test_nvml_api_timing_preserves_return_and_error(self):
        from telemetry_nvml import NVMLSampler
        from telemetry_native import Streams
        with tempfile.TemporaryDirectory() as temporary:
            sampler = NVMLSampler.__new__(NVMLSampler)
            sampler.streams = Streams(Path(temporary))
            def success(value): return value
            def failure(): raise ValueError('fixture')
            with patch('telemetry_nvml.time.monotonic', side_effect=[1, 15.5, 20, 20.2]):
                self.assertEqual(sampler.timed_call({}, 'GPU-test', success, 7), 7)
                with self.assertRaisesRegex(ValueError, 'fixture'):
                    sampler.timed_call({}, 'GPU-test', failure)
            sampler.streams.close()
            rows = [json.loads(line) for line in (Path(temporary) / 'nvml-api.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['value'], 14.5)
            self.assertEqual(rows[0]['api'], 'success')
            self.assertIsNone(rows[0]['error'])
            self.assertEqual(rows[1]['error'], 'ValueError: fixture')

    def test_stream_error_is_durable_and_visible_to_publication(self):
        from telemetry_native import Streams
        from telemetry_health import assert_healthy
        from publish_telemetry import validate_rows
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            streams = Streams(root)
            row = dict(time='2026-09-03T00:00:00Z', monotonic_s=1, hostname='gpu-nodes-0',
                slurm_job_id='152', source='persistent-nvml', metric='collector_error',
                value=None, unit='event', error='Fixture driver timeout')
            streams.write('nvidia-smi', row)
            self.assertEqual(streams.errors, 1)
            with self.assertRaisesRegex(RuntimeError, 'failure marker'):
                assert_healthy(root, 'gpu-nodes-0', '152', 1)
            streams.close()
            validated = validate_rows((root / 'nvidia-smi.jsonl').read_bytes(), 'gpu-nodes-0', '152')
            self.assertEqual(validated['collector_errors'], 1)

    def test_nvml_finalization_preserves_streams_when_shutdown_raises(self):
        from telemetry_native import collect
        class Sampler:
            def __init__(self, *args): pass
            def finish(self): return {'fixture': True}
            def shutdown(self): raise RuntimeError('shutdown fixture')
        with tempfile.TemporaryDirectory() as temporary:
            run = Run.create(Path(temporary) / 'run', {})
            (run.root / 'inventory/gpu.values.json').write_text('{"gpus":[]}')
            with patch('telemetry_nvml.NVMLSampler', Sampler), patch.dict(os.environ, {'SLURM_JOB_ID': '152'}):
                with self.assertRaisesRegex(RuntimeError, 'shutdown fixture'):
                    collect(run, run.root / 'stop', 0, gpu_backend='nvml')
            files = list((run.root / 'telemetry/native').glob('*/cpu-memory-numa.jsonl'))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())['metric'], 'collector_error')

    def test_candidate_repackage_preserves_parent_and_links_only_weight_payloads(self):
        from repackage_qwen_candidate import repackage

        with tempfile.TemporaryDirectory() as temporary:
            run = Run.create(Path(temporary) / 'run', {})
            parent, source, target = [run.root / 'models' / name for name in ('parent', 'source', 'candidate')]
            parent.mkdir(parents=True); source.mkdir()
            (parent / 'model.safetensors').write_bytes(b'unchanged-weight-fixture')
            (parent / 'config.json').write_text('{}')
            frozen = {}
            for name in ('preprocessor_config.json', 'video_preprocessor_config.json'):
                (source / name).write_text('{"fixture":true}')
                frozen[name] = sha256(source / name)
            (parent / 'checksums.sha256').write_text(''.join(sha256(parent / name) + '  ' + name + '\n'
                for name in ('model.safetensors', 'config.json')))
            checksum = sha256(parent / 'checksums.sha256')
            (parent / 'CONVERSION_COMPLETE.json').write_text(json.dumps({'checksums_sha256': checksum}))
            result = repackage(run.root, source, parent, target, checksum, frozen)
            self.assertEqual(result['hardlinked_weight_shards'], 1)
            self.assertEqual((parent / 'model.safetensors').stat().st_ino, (target / 'model.safetensors').stat().st_ino)
            self.assertEqual(sha256(parent / 'checksums.sha256'), checksum)
            self.assertEqual(sha256(target / 'preprocessor_config.json'), frozen['preprocessor_config.json'])
            with self.assertRaisesRegex(ValueError, 'Destination exists'):
                repackage(run.root, source, parent, target, checksum, frozen)

    def test_mxfp8_serving_keeps_ep8_mtp_and_requires_checksum_pin(self):
        from run_qwen_serving_validation import model_path

        command = server_command('/model', True, 'mxfp8')
        self.assertEqual(command[command.index('--ep-size') + 1], '8')
        self.assertEqual(command[command.index('--tp-size') + 1], '8')
        self.assertIn('--speculative-algorithm', command)
        self.assertEqual(command[command.index('--moe-runner-backend') + 1], 'flashinfer_trtllm_routed')
        self.assertEqual(command[command.index('--fp8-gemm-backend') + 1], 'triton')
        with tempfile.TemporaryDirectory() as temporary:
            run = Run.create(Path(temporary) / 'run', {})
            code = Path(temporary) / 'code'; code.mkdir()
            model = run.root / 'models/candidate'; model.mkdir(parents=True)
            (model / 'config.json').write_text('{"quantization_config":{"quant_method":"mxfp8"}}')
            (model / 'checksums.sha256').write_text(sha256(model / 'config.json') + '  config.json\n')
            digest = sha256(model / 'checksums.sha256')
            (model / 'CONVERSION_COMPLETE.json').write_text(json.dumps({'checksums_sha256': digest}))
            (code / 'candidate.json').write_text(json.dumps({'model_relpath': 'models/candidate', 'checksums_sha256': digest}))
            self.assertEqual(model_path(run, code), model)
            (model / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'file checksum differs'):
                model_path(run, code)

    def test_serialized_mxfp8_audit_checks_scale_shape_and_payload_boundaries(self):
        import struct
        from audit_mxfp8_checkpoint import expected_tensors, headers, tensor_hash

        rows = [dict(name='mtp.layers.0.mlp.experts.gate_up_proj', shape=[2, 4, 64], dtype='BF16', precision='mxfp8'),
                dict(name='mtp.fc.weight', shape=[1, 2], dtype='BF16', precision='source')]
        expected = expected_tensors(rows)
        self.assertEqual(expected['mtp.layers.0.mlp.experts.1.up_proj.weight'], ([2, 64], 'F8_E4M3'))
        self.assertEqual(expected['mtp.layers.0.mlp.experts.1.up_proj.weight_scale_inv'], ([2, 2], 'U8'))
        self.assertEqual(expected['mtp.fc.weight'], ([1, 2], 'BF16'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            header = json.dumps({'mtp.fc.weight': dict(shape=[1, 2], dtype='BF16', data_offsets=[0, 4])}).encode()
            path = root / 'model.safetensors'
            path.write_bytes(struct.pack('<Q', len(header)) + header + b'abcd')
            entry = headers(root, [path.name])['mtp.fc.weight']
            self.assertEqual(tensor_hash(root, entry), hashlib.sha256(b'abcd').hexdigest())
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                tensor_hash(root, entry)

    def test_mxfp8_probe_uses_separate_paths_and_preserves_whole_node_visibility(self):
        from run_model_conversion import container_command

        with tempfile.TemporaryDirectory() as temporary:
            run = Run.create(Path(temporary) / 'run', {})
            (run.root / 'images').mkdir()
            with patch('run_model_conversion.prepare', return_value={}), patch.dict(os.environ, SLURM_JOB_ID='999'):
                command = container_command(run, Path('/code'), 1, Path('/pinned/miles'), 'mxfp8-probe')
            self.assertIn('/ptx/qwen_mxfp8_probe.py', command)
            self.assertNotIn('/ptx/model_conversion.py', command)
            self.assertIn('NVIDIA_VISIBLE_DEVICES=all', command)
            self.assertTrue((run.root / 'images/qwen-mxfp8-probe-runtime-v1').is_dir())
            self.assertIn('/pinned/miles:/miles-source:none:bind,ro,x-create=dir', command)
            with patch('run_model_conversion.prepare', return_value={}), patch.dict(os.environ, SLURM_JOB_ID='999'):
                baseline = container_command(run, Path('/code'), 1, Path('/pinned/miles'))
            self.assertIn('/ptx/model_conversion.py', baseline)
            self.assertTrue((run.root / 'images/model-conversion-runtime-v1').is_dir())
            with patch('run_model_conversion.prepare', return_value={}), patch.dict(os.environ, SLURM_JOB_ID='999'):
                full = container_command(run, Path('/code'), 1, Path('/pinned/miles'), 'mxfp8')
            self.assertIn('--convert', full)
            self.assertNotIn('--convert', command)
            self.assertTrue((run.root / 'images/qwen-mxfp8-conversion-runtime-v1').is_dir())

    def test_qwen_mxfp8_expert_unpacking_preserves_expert_gate_up_and_scale_order(self):
        import torch
        from convert_qwen_mxfp8 import unpack

        name = 'model.language_model.layers.0.mlp.experts.gate_up_proj'
        weights = torch.arange(2 * 4 * 32).reshape(2, 4, 32)
        parts = dict(unpack(name, weights))
        prefix = 'model.language_model.layers.0.mlp.experts.'
        restored = torch.stack([torch.cat([parts[f'{prefix}{i}.{p}_proj.weight'] for p in ('gate', 'up')])
                                for i in range(2)])
        self.assertTrue(torch.equal(weights, restored))
        scales = torch.arange(8, dtype=torch.uint8).reshape(2, 4, 1)
        scale_parts = dict(unpack(name, scales))
        self.assertEqual(set(parts), set(scale_parts))
        self.assertTrue(torch.equal(scale_parts[prefix + '1.up_proj.weight'], scales[1, 2:]))
        down = dict(unpack('mtp.layers.0.mlp.experts.down_proj', weights))
        self.assertTrue(torch.equal(down['mtp.layers.0.mlp.experts.1.down_proj.weight'], weights[1]))
        with self.assertRaisesRegex(ValueError, 'even gate/up'):
            list(unpack(name, torch.zeros(2, 3, 32)))

    def test_qwen_mxfp8_candidate_preserves_gdn_vision_router_and_mtp_glue(self):
        from convert_qwen_mxfp8 import configuration, precision

        for name in ('model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
                     'model.visual.blocks.0.mlp.fc1.weight', 'mtp.fc.weight',
                     'model.language_model.layers.0.mlp.shared_expert_gate.weight'):
            self.assertEqual(precision(name, [32, 32], 'BF16'), 'source')
        for name in ('model.language_model.layers.0.mlp.experts.gate_up_proj',
                     'mtp.layers.0.mlp.experts.down_proj',
                     'model.language_model.layers.3.self_attn.q_proj.weight'):
            self.assertEqual(precision(name, [32, 32], 'BF16'), 'mxfp8')
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                precision(name, [32, 31], 'BF16')
        with self.assertRaisesRegex(ValueError, 'Unrecognized expert'):
            precision('model.layers.0.mlp.experts.unknown', [32, 32], 'BF16')
        original = {'model_type': 'qwen3_5_moe', 'text_config': {'num_hidden_layers': 40, 'num_experts': 256}}
        converted = configuration(original)
        self.assertNotIn('quantization_config', original)
        self.assertIn('linear_attn', converted['quantization_config']['modules_to_not_convert'])
        with self.assertRaisesRegex(ValueError, 'unquantized'):
            configuration(converted)

    def test_quantization_audit_uses_actual_stock_selectors_and_model_layout(self):
        from audit_quantization_recipe import audit, selectors, render

        miles = Path(__file__).resolve().parents[1] / 'vendor/miles'
        config = json.dumps({'text_config': {'num_hidden_layers': 40}}).encode()
        def tensor(name, shape, size):
            return dict(name=name, shape=shape, dtype='BF16', payload_bytes=size)
        packed = tensor('model.language_model.layers.0.mlp.experts.gate_up_proj', [256,1024,2048],1024)
        conv = tensor('model.language_model.layers.0.linear_attn.conv1d.weight', [8192,1,4],64)
        dense = tensor('model.language_model.layers.0.self_attn.q_proj.weight', [2048,2048],128)
        rows = [packed,conv,dense]
        fp8,mxfp8 = selectors(miles)
        self.assertFalse(fp8(packed))
        self.assertFalse(mxfp8(packed))
        self.assertTrue(fp8(conv))
        self.assertFalse(mxfp8(conv))
        self.assertTrue(fp8(dense) and mxfp8(dense))
        inventory = dict(config_sha256=hashlib.sha256(config).hexdigest(),model_type='qwen3_5_moe',tensors=rows)
        report = audit(inventory,config,miles)
        self.assertEqual(report['packed_expert_tensor_count'],1)
        self.assertEqual(report['converters'][0]['non_2d_selected'][0]['shape'],[8192,1,4])
        self.assertTrue(any('top-level num_hidden_layers' in p for p in report['converters'][1]['problems']))
        self.assertTrue(all(row['status']=='blocked_stock_recipe' for row in report['converters']))
        report.update(qualification=[], reproducer='example', provenance={'z': 'last', 'a': 'first'}, documentation_url='https://example.test')
        self.assertEqual(render(report), render(json.loads(json.dumps(report, sort_keys=True))))
        with self.assertRaisesRegex(ValueError,'Configuration differs'):
            audit(inventory,b'{}',miles)

    def test_public_telemetry_preserves_errors_and_rejects_unreviewed_data(self):
        from publish_telemetry import validate_rows

        record = dict(time='2026-09-02T22:00:00Z', monotonic_s=1.0, hostname='gpu-nodes-0',
                      source='nvidia-smi', metric='utilization.gpu', value=12.5, unit='percent', slurm_job_id='143')
        error = dict(record, metric='collector_error', value=None, error='Read timed out.')
        raw = ''.join(json.dumps(row) + '\n' for row in (record, error)).encode()
        self.assertEqual(validate_rows(raw, 'gpu-nodes-0', 143)['collector_errors'], 1)
        self.assertEqual(validate_rows(raw, 'gpu-nodes-0', 143)['records'], 2)
        for bad in (dict(record, transcript='not allowed'), dict(record, value={'nested': 1}),
                    dict(record, hostname='gpu-nodes-1'), dict(record, slurm_job_id='144'),
                    dict(record, value=float('nan')), dict(error, value=0),
                    dict(error, error='gh' + 'p_' + 'a' * 36)):
            with self.subTest(bad_keys=sorted(bad)), self.assertRaises(ValueError):
                validate_rows((json.dumps(bad) + '\n').encode(), 'gpu-nodes-0', 143)
        with self.assertRaisesRegex(ValueError, 'inside a JSONL'):
            validate_rows(raw[:-1], 'gpu-nodes-0', 143)

    def test_public_gate_rejects_gaps_and_failed_allocations_without_error_rows(self):
        from summarize_telemetry import assess_gate
        coverage = [dict(path='gpu.jsonl', state='complete', max_observed_interval_s=1.2)]
        self.assertEqual(assess_gate(coverage, 0, 'COMPLETED', '0:0'), ('ok', []))
        self.assertEqual(assess_gate(coverage, 0, 'RUNNING', '0:0'), ('partial', []))
        status, findings = assess_gate(coverage, 0, 'FAILED', '1:0')
        self.assertEqual(status, 'fail')
        self.assertIn('Slurm', findings[0])
        coverage[0]['max_observed_interval_s'] = 15.48
        status, findings = assess_gate(coverage, 0, 'COMPLETED', '0:0')
        self.assertEqual(status, 'fail')
        self.assertIn('15.48s', findings[0])
        coverage[0].update(state='partial', max_observed_interval_s=None)
        self.assertEqual(assess_gate(coverage, 0, 'COMPLETED', '0:0')[0], 'fail')

    def test_public_telemetry_chunk_resume_finalization_and_mutation_guard(self):
        from publish_telemetry import export_chunk

        remote = '/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210'
        relative = 'telemetry/sync-grpo-v9/gpu-nodes-0/nvidia-smi.jsonl'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            final = root / relative
            final.parent.mkdir(parents=True)
            partial = final.with_suffix('.jsonl.partial')
            partial.write_bytes(b'{"value":1}\n{"value":')
            with patch('pathlib.Path', return_value=root):
                first = export_chunk(remote, relative, 0, None, None)
                self.assertFalse(first['complete'])
                self.assertEqual(gzip.decompress(base64.b64decode(first['gzip_base64'])), b'{"value":1}\n')
                with partial.open('ab') as output:
                    output.write(b'2}\n')
                second = export_chunk(remote, relative, first['end'], first['inode'], first['anchor'])
                self.assertEqual(gzip.decompress(base64.b64decode(second['gzip_base64'])), b'{"value":2}\n')
                partial.rename(final)
                last = export_chunk(remote, relative, second['end'], second['inode'], second['anchor'])
                self.assertTrue(last['complete'])
                self.assertEqual(last['source_sha256'], hashlib.sha256(final.read_bytes()).hexdigest())
                with self.assertRaisesRegex(ValueError, 'boundary changed'):
                    export_chunk(remote, relative, second['end'], second['inode'], 'bad-anchor')
                with self.assertRaisesRegex(ValueError, 'truncated or replaced'):
                    export_chunk(remote, relative, 0, second['inode'] + 1, None)
                with self.assertRaisesRegex(ValueError, 'explicitly allowed'):
                    export_chunk(remote, 'rl/trajectories.jsonl', 0, None, None)

    def test_public_telemetry_watcher_is_bounded_and_run_owned(self):
        from publish_telemetry import start_watcher

        with tempfile.TemporaryDirectory() as temporary, patch('publish_telemetry.subprocess.Popen') as popen:
            popen.return_value.pid = 12345
            receipt = start_watcher(temporary, '/example/config', 'sync-grpo-v9', 143)
            command = popen.call_args.args[0]
            self.assertIn('--watch', command)
            self.assertIn('--push', command)
            self.assertEqual(command[command.index('--interval-seconds') + 1], '300')
            self.assertEqual(command[command.index('--max-seconds') + 1], '6000')
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(receipt['status'], 'spawned')
            self.assertTrue((Path(temporary) / receipt['log_directory'] / 'started.json').is_file())
            with self.assertRaises(FileExistsError):
                start_watcher(temporary, '/example/config', 'sync-grpo-v9', 143)

    def test_dense_telemetry_keeps_spikes_failures_and_rejects_counter_gaps(self):
        from summarize_telemetry import summarize_sources, build_summary, render, timeline_csv

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def row(name, value, second=0, source='nvidia-smi', **extra):
                return dict(time=f'2026-09-02T00:00:{second:02d}Z', monotonic_s=second,
                    hostname='gpu-nodes-0', role='trainer', source=source, metric=name,
                    value=value, unit='count', **extra)
            gpu = [row('utilization.gpu', value, second, gpu_uuid='GPU-a') for second,value in ((0,0),(1,100))]
            gpu += [row('ecc.errors.corrected.volatile.total', 0, second, gpu_uuid='GPU-a') for second in (0,1)]
            gpu += [row('memory.total', 123, second, gpu_uuid='GPU-a') for second in (0,1)]
            gpu += [row('collector_error', None, 2, error='command timed out')]
            link = [row('nvlink_data_tx_bytes_total', value, second, source='nvlink', gpu_uuid='GPU-a', link=i)
                    for second,value in ((0,0),(1,10**10),(10,10**12)) for i in range(18)]
            fabric = [row('PortXmitData', value, second, source='perfquery', hca='mlx5_0', hca_port='1')
                      for second,value in ((0,0),(1,2*10**9))]
            fabric += [row('PortRcvErrors', 0, second, source='perfquery', hca='mlx5_0', hca_port='1') for second in (0,1)]
            fabric += [row('SymbolErrorCounter', value, second, source='perfquery', hca='mlx5_0', hca_port='1')
                       for second,value in ((0,2),(1,5))]
            streams = []
            for name,rows in (('nvidia-smi',gpu),('nvlink',link),('infiniband',fabric)):
                raw = ''.join(json.dumps(r)+'\n' for r in rows).encode()
                payload = gzip.compress(raw,mtime=0)
                path = name + '.gz'
                (root/path).write_bytes(payload)
                streams.append(dict(path=f'telemetry/sync-grpo-v9/gpu-nodes-0/{name}.jsonl',
                    status='complete', complete=True, end=len(raw), source_sha256=hashlib.sha256(raw).hexdigest(),
                    chunks=[dict(path=path, offset=0, end=len(raw), records=len(rows),
                        collector_errors=sum(r['metric']=='collector_error' for r in rows),
                        raw_sha256=hashlib.sha256(raw).hexdigest(), gzip_sha256=hashlib.sha256(payload).hexdigest())]))
            data = summarize_sources(root,streams)
            metrics = {r['metric']:r for r in data['results']}
            self.assertEqual(metrics['gpu_utilization']['value'],50)
            self.assertEqual(metrics['gpu_utilization']['statistics']['p95'],95)
            self.assertEqual(metrics['nvlink_gpu_tx']['value'],180)
            self.assertEqual(metrics['ib_rail_tx']['value'],2)
            self.assertEqual(sum(r['count'] for r in data['counter_gaps']),18)
            self.assertEqual(data['health'][0]['all_zero_series'],2)
            self.assertEqual(data['health'][0]['exceptions'][0]['last'],5)
            self.assertEqual(data['collector_errors'][0]['count'],1)
            self.assertEqual(data['inventory_values'][0]['samples_collapsed'],2)
            timeline = next(r for r in data['timeline'] if r['metric']=='gpu_utilization')
            self.assertEqual((timeline['n'],timeline['min'],timeline['mean'],timeline['max']),(2,0,50,100))
            self.assertNotIn('00:01:00',timeline_csv(data['timeline']))
            result,_ = build_summary(root,root,streams,143,'COMPLETED','0:0')
            self.assertEqual(result['status'],'fail')
            self.assertTrue(result['timeout'])
            self.assertIn('Telemetry gate: FAIL',render(result))
            (root/'nvidia-smi.gz').write_bytes(gzip.compress(b'changed\n'))
            with self.assertRaisesRegex(ValueError,'checksum/offset'):
                summarize_sources(root,streams)

    def test_optimizer_log_parser_deduplicates_actual_log_shape_and_rejects_conflicts(self):
        from observe_grpo_log import parse_log

        first = "[2026-09-02 22:57:57.204 actor_cell0_rank0] log_utils.py:544 - step 0: {'train/step': 0, 'train/grad_norm': 0.4}"
        duplicate = first.replace('log_utils.py:544', 'model.py:861')
        data = parse_log(first + '\n' + duplicate)
        self.assertEqual(len(data['steps']), 1)
        self.assertEqual(data['steps'][0]['time'], '2026-09-02T22:57:57.204Z')
        delayed = parse_log(first + '\n' + duplicate.replace('57.204', '57.205'))['steps'][0]
        self.assertEqual(delayed['receipt_times'], ['2026-09-02T22:57:57.204Z', '2026-09-02T22:57:57.205Z'])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            parse_log(first + '\n' + duplicate.replace('57.204', '58.204'))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            parse_log(first + '\n' + duplicate.replace('rank0', 'rank1'))
        rollout = "[2026-09-02 22:55:01.707 rollout_manager] metrics.py:89 - perf 0: {'rollout/episode_raw_reward': 0.5}"
        trainer = "[2026-09-02 22:57:57.480 actor_cell0_rank0] train_metric_utils.py:56 - perf 0: {'perf/train_time': 175.4}"
        performance = parse_log('\n'.join([first, rollout, trainer]))['performance']
        self.assertEqual({row['role'] for row in performance}, {'rollout', 'trainer'})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            parse_log(first + '\n' + duplicate.replace('0.4', '0.5'))
        with self.assertRaisesRegex(ValueError, 'No optimizer'):
            parse_log('A completed job without usable metric records')

    def test_openenv_state_identity_is_separate_from_policy_observation(self):
        import asyncio
        from unittest.mock import AsyncMock
        from local_openenv_client import LocalOpenEnvClient

        for fault in (None, 'missing_episode', 'bad_episode', 'wrong_task', 'wrong_reply'):
            with self.subTest(fault=fault):
                observation = {'instruction': 'Solve this task.', 'task_id': 'task_00000'}
                state = {'episode_id': 'a' * 32, 'task_id': 'task_00000'}
                if fault == 'missing_episode':
                    state.pop('episode_id')
                elif fault == 'bad_episode':
                    state['episode_id'] = '../../outside'
                elif fault == 'wrong_task':
                    state['task_id'] = 'task_00001'
                replies = [dict(type='observation', data={'observation': observation}),
                           dict(type='observation' if fault == 'wrong_reply' else 'state', data=state)]
                client = LocalOpenEnvClient('http://localhost:8000')
                client.ws = type('Socket', (), {'send': AsyncMock(),
                    'recv': AsyncMock(side_effect=[json.dumps(reply) for reply in replies])})()
                client.episode_id = 'previous-episode'
                if fault:
                    with self.assertRaises(RuntimeError):
                        asyncio.run(client.reset(task_id='task_00000'))
                    self.assertIsNone(client.episode_id)
                else:
                    result = asyncio.run(client.reset(task_id='task_00000'))
                    self.assertEqual(vars(result.observation), observation)
                    self.assertEqual((client.episode_id, client.task_id), ('a' * 32, 'task_00000'))
                    sent = [json.loads(call.args[0]) for call in client.ws.send.await_args_list]
                    self.assertEqual(sent, [{'type': 'reset', 'data': {'task_id': 'task_00000'}},
                                            {'type': 'state'}])
        source = Path(__file__).resolve().parents[1]
        self.assertEqual((source / 'scripts/local_openenv_client.py').read_bytes(),
                         (source / 'vendor/miles/examples/experimental/openenv/posttrainingx_openenv_client.py').read_bytes())

    def test_grpo_tensor_audit_detects_sample_corruption_and_incomplete_rank_coverage(self):
        import torch
        from audit_grpo_tensors import audit_tensors

        for fault in (None, 'tokens', 'rollout_log_probs', 'loss_masks', 'advantages', 'missing_rank', 'duplicate_sample'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for folder in ('qualification-groups', 'rollout_data', 'train_data'):
                    (root / folder).mkdir()
                samples = [{'index': i, 'group_index': 0, 'tokens': [1, 2, 3],
                            'response_length': 2, 'rollout_log_probs': [-0.5, -0.25],
                            'loss_mask': [1, 1], 'reward': float(i < 4), 'weight_versions': ['1']}
                           for i in range(8)]
                (root / 'qualification-groups/group.json').write_text(json.dumps({'samples': samples}))
                torch.save({'rollout_id': 0, 'samples': samples}, root / 'rollout_data/0.pt')
                for rank in range(2):
                    indices = list(range(rank * 4, rank * 4 + 4))
                    # Match the recipe's float32 arithmetic, independently of the audit implementation.
                    rewards = [(torch.tensor(samples[i]['reward'] - 0.5) /
                                (torch.sqrt(torch.tensor(2 / 7)) + 1e-6)).item() for i in indices]
                    data = {'sample_indices': indices, 'raw_reward': [s['reward'] for s in samples],
                            'rewards': rewards, 'weight_versions': [['1'] for _ in indices]}
                    for key, source, dtype in (('tokens', 'tokens', torch.int64),
                                               ('loss_masks', 'loss_mask', torch.int32),
                                               ('rollout_log_probs', 'rollout_log_probs', torch.float32)):
                        data[key] = [torch.tensor(samples[i][source], dtype=dtype) for i in indices]
                    for key in ('advantages', 'returns'):
                        data[key] = [torch.full((2,), reward) for reward in rewards]
                    for key in ('log_probs', 'ref_log_probs'):
                        data[key] = [torch.tensor([-0.5, -0.25]) for _ in indices]
                    if rank == 0 and fault in ('tokens', 'rollout_log_probs', 'loss_masks', 'advantages'):
                        data[fault][0][0] += 1
                    if rank == 0 and fault == 'duplicate_sample':
                        data['sample_indices'][0] = data['sample_indices'][1]
                    if rank == 1 and fault == 'missing_rank':
                        continue
                    torch.save({'rollout_id': 0, 'rank': rank, 'cp_rank': 0, 'cp_size': 1,
                                'rollout_data': data}, root / f'train_data/0_{rank}.pt')
                if fault:
                    with self.assertRaises(ValueError):
                        audit_tensors(root, 2)
                else:
                    result = audit_tensors(root, 2)
                    self.assertEqual(result['findings'], [])
                    self.assertEqual(result['trained_input_samples'], 8)

    def test_container_fabric_gate_refuses_tcp_before_verbs_or_gpu_calls(self):
        ports = [(f'mlx5_{i}', '1') for i in range(8)]
        with patch('container_fabric_probe.active_training_ports', return_value=ports), patch.dict(os.environ, {'NCCL_NET':'Socket'}):
            with self.assertRaisesRegex(RuntimeError, 'non-IB'):
                verify_rdma()

    def test_container_fabric_gate_refuses_a_different_hca_selection(self):
        ports = [(f'mlx5_{i}', '1') for i in range(8)]
        with patch('container_fabric_probe.active_training_ports', return_value=ports), patch.dict(os.environ, {'NCCL_NET':'IB','NCCL_IB_HCA':'mlx5_0'}):
            with self.assertRaisesRegex(RuntimeError, 'selection differs'):
                verify_rdma()

    def test_episode_audit_keeps_missing_verdict_distinct_and_rejects_leaky_order(self):
        def rows(events):
            return [dict(event=e, episode_id='episode', task_id='task', time=str(i), monotonic_s=i,
                         **({'reward': 0.0} if e == 'graded' else {})) for i, e in enumerate(events)]
        missing = summarize_episode(rows(['created', 'workspace_volume_created', 'workspace_volume_removed']))
        self.assertEqual((missing['category'], missing['reward'], missing['findings']), ('no_verdict', None, []))
        failed = summarize_episode(rows(['created', 'policy_stopped', 'grader_assets_staged', 'graded']))
        self.assertEqual((failed['category'], failed['reward'], failed['findings']), ('graded', 0.0, []))
        unsafe = summarize_episode(rows(['created', 'grader_assets_staged', 'policy_stopped', 'graded']))
        self.assertTrue(unsafe['findings'])

    def test_grpo_audit_refuses_live_allocation(self):
        with patch('subprocess.check_output', return_value='138|RUNNING|0:0|00:01:00|start|Unknown|nodes\n'):
            with self.assertRaisesRegex(ValueError, 'not unambiguously terminal'):
                audit_grpo_remote('/unused', 5, 138)

    def test_grpo_audit_does_not_infer_training_success_from_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'provenance/sync-grpo-code-v5/launch.json'
            path.parent.mkdir(parents=True)
            nodes = [{'hostname': f'gpu-nodes-{i}', 'role': 'trainer' if i < 2 else 'rollout',
                      'gpu_uuids': [f'GPU-{i}-{j}' for j in range(8)]} for i in range(4)]
            path.write_text(json.dumps({'root_sha': 'root', 'miles_sha': 'miles', 'layout': '2t2r',
                'task_ids': ['task_1'], 'optimizer_steps_requested': 3, 'host_map': {'nodes': nodes}}))
            with patch('subprocess.check_output', return_value='138|COMPLETED|0:0|00:01:00|start|end|nodes\n'):
                data = audit_grpo_remote(root, 5, 138)
            self.assertIn('Missing finalized driver.finished.json', data['findings'])
            self.assertTrue(any('optimizer execution' in value for value in data['unverified']))
            self.assertNotIn('optimizer_steps', data)
            self.assertTrue(all(not node['phase_finalized'] for node in data['nodes']))

    def test_node_cleanup_retains_failures_and_continues_to_inventory(self):
        findings, events = ['original training failure'], []
        def broken():
            events.append('stop')
            raise RuntimeError('stop failed')
        cleanup_actions([('stop', broken), ('inventory', lambda: events.append('inventory'))], findings)
        self.assertEqual(events, ['stop', 'inventory'])
        self.assertEqual(findings, ['original training failure', 'Cleanup stop failed: RuntimeError: stop failed'])

    def test_bundle_checksums_include_hidden_artifacts_and_nested_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Run.create(Path(tmp) / 'run', {})
            (run.root / 'provenance/.pinned-metadata').write_text('record')
            (run.root / 'provenance/checksums.sha256').write_text('nested manifest')
            run.refresh()
            names = {line.split('  ', 1)[1] for line in (run.root / 'checksums.sha256').read_text().splitlines()}
            self.assertIn('provenance/.pinned-metadata', names)
            self.assertIn('provenance/checksums.sha256', names)
            self.assertNotIn('checksums.sha256', names)

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
    def test_training_collector_duration_is_bounded_but_allows_full_allocation(self):
        import argparse
        self.assertEqual(duration_seconds('5460'), 5460)
        for duration in ('0', '86401'):
            with self.assertRaises(argparse.ArgumentTypeError):
                duration_seconds(duration)

    def test_file_runtime_rejects_links_special_files_and_escaping_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'answer').write_text('candidate output')
            payload = tree_archive(source, 'task_file')
            validate_archive(payload, 'task_file')
            with tarfile.open(fileobj=io.BytesIO(archive_contents(payload, 'task_file'))) as inside:
                self.assertEqual(inside.getnames(), ['answer'])
                self.assertEqual(inside.extractfile('answer').read(), b'candidate output')
            atomic_bytes(root / 'snapshot.tar', payload)
            self.assertEqual((root / 'snapshot.tar').read_bytes(), payload)
            (source / 'link').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                tree_archive(source, 'task_file')
        for name, kind in [('task_file/../../escape', tarfile.REGTYPE), ('/task_file/escape', tarfile.REGTYPE),
                           ('task_file/link', tarfile.SYMTYPE), ('task_file/hardlink', tarfile.LNKTYPE),
                           ('task_file/fifo', tarfile.FIFOTYPE), ('tests/hidden', tarfile.REGTYPE)]:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode='w') as archive:
                member = tarfile.TarInfo(name)
                member.type = kind
                archive.addfile(member)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_archive(buffer.getvalue(), 'task_file')

    def test_sealed_policy_rejects_commands_before_any_docker_call(self):
        session = FileTaskSession.__new__(FileTaskSession)
        session.lock = threading.RLock()
        session.sealed = True
        with self.assertRaisesRegex(RuntimeError, 'permanently sealed'):
            session.run_command('cat /tests/test_outputs.py')

    def test_offline_harness_preserves_scoring_and_refuses_unknown_setup(self):
        setup = ('#!/bin/bash\napt-get update\napt-get install -y curl\n'
                 'curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n'
                 'source $HOME/.local/bin/env\n')
        scoring = ('# Run pytest tests\nuvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 '
                   'pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA\n'
                   'if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; fi\n')
        self.assertTrue(offline_harness(setup + scoring).endswith(scoring))
        with self.assertRaises(ValueError):
            offline_harness(setup + 'echo unexpected mutation\n' + scoring)
        with self.assertRaises(ValueError):
            offline_harness((setup + scoring).replace('0.9.5/install', 'latest/install'))
        original = 'FROM ubuntu:22.04\nWORKDIR /app\n'
        pinned = 'ubuntu@sha256:' + 'a' * 64
        self.assertEqual(pin_dockerfile(original, {'ubuntu:22.04': pinned}), 'FROM ' + pinned + '\nWORKDIR /app\n')
        with self.assertRaises(ValueError):
            pin_dockerfile(original + 'FROM node:20\n', {'ubuntu:22.04': pinned})

    def test_trainer_report_preserves_counter_gaps_and_rechecks_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'infiniband.jsonl'
            rows = [dict(time=f'2026-09-02T00:00:{t:02d}Z', monotonic_s=t, hostname='node',
                         source='perfquery', metric='PortXmitData', value=value, unit='B',
                         hca='mlx5_0', hca_port='1') for t, value in [(1, 10), (2, 12), (9, 19), (10, 22)]]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            coverage = [{'path': path.name, 'sha256': sha256(path)}]
            result = analyze_streams(root, coverage)
            self.assertEqual(result['findings'], [])
            self.assertEqual([row['value'] for row in result['timeline']], [2.0, 3.0])
            self.assertEqual(len(result['counter_discontinuities']), 1)
            self.assertEqual(result['distributions'][0]['statistics']['mean'], 2.5)
            coverage[0]['sha256'] = '0' * 64
            self.assertTrue(analyze_streams(root, coverage)['findings'])

    def test_trainer_probe_keeps_ep8_mtp_and_prohibits_optimizer_loading(self):
        command = trainer_command('/evidence', '--mtp-num-layers 1 --moe-token-dispatcher-type alltoall')
        for flag, value in {'--tensor-model-parallel-size': '1', '--pipeline-model-parallel-size': '1',
                            '--expert-model-parallel-size': '8', '--expert-tensor-parallel-size': '1',
                            '--context-parallel-size': '1', '--moe-token-dispatcher-type': 'flex',
                            '--global-batch-size': '8', '--seq-length': '128'}.items():
            self.assertEqual(command.count(flag), 1)
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('--no-load-optim', command)
        self.assertIn('--enable-mtp-training', command)
        self.assertNotIn('--save', command)
        with self.assertRaises(ValueError):
            trainer_command('/evidence', '--mtp-num-layers 0 --moe-token-dispatcher-type alltoall')

    def test_parameter_and_gradient_audit_rejects_mutation_nonfinite_and_missing_mtp(self):
        import torch
        model = torch.nn.Module()
        model.base = torch.nn.Linear(2, 2, bias=False)
        model.mtp = torch.nn.Linear(2, 2, bias=False)
        wrapper = torch.nn.Module()
        wrapper.module = model
        before = parameter_hashes([wrapper])
        self.assertEqual(before, parameter_hashes([wrapper]))
        for param in wrapper.parameters():
            param.grad = torch.ones_like(param)
        self.assertEqual(len(gradient_statistics([wrapper])), 2)
        model.mtp.weight.grad.zero_()
        with self.assertRaises(ValueError):
            gradient_statistics([wrapper])
        model.mtp.weight.grad.fill_(float('nan'))
        with self.assertRaises(ValueError):
            gradient_statistics([wrapper])
        with torch.no_grad():
            model.base.weight[0, 0] += 1
        self.assertNotEqual(before, parameter_hashes([wrapper]))

    def test_finalized_trainer_audit_cross_checks_counts_mtp_loss_and_parameter_records(self):
        import copy
        import torch
        model = torch.nn.Module()
        model.base = torch.nn.Linear(2, 2, bias=False)
        model.mtp = torch.nn.Linear(2, 2, bias=False)
        wrapper = torch.nn.Module()
        wrapper.module = model
        for param in wrapper.parameters():
            param.grad = torch.ones_like(param)
        hashes = parameter_hashes([wrapper])
        records = [hashes, copy.deepcopy(hashes), gradient_statistics([wrapper]),
                   {'logit_shape': [1, 128, 248320], 'teacher_forced_log_probs': [[-2.0] * 127],
                    'main_cross_entropy': 2.0}, {'token_ids': [1] * 128, 'cu_seqlens': [0, 128]},
                   {'gradient_tensors_present': 2, 'gradient_tensors_nonzero': 2,
                    'mtp_gradient_tensors_nonzero': 1, 'parameters_unchanged': True}]
        self.assertEqual(validate_rank_evidence(*records)[0], [])
        mutations = [
            lambda r: r[1][0].update(sha256='f' * 64),
            lambda r: r[2].append(copy.deepcopy(r[2][0])),
            lambda r: r[2][0].update(mtp=True),
            lambda r: r[2][0].update(present=False),
            lambda r: r[2][0].update(max_abs=float('nan')),
            lambda r: r[2][0].update(local_buffer_l2=0),
            lambda r: r[3].update(main_cross_entropy=3.0),
            lambda r: r[3].update(teacher_forced_log_probs=[[float('-inf')] * 127]),
            lambda r: r[4].update(token_ids=[True] * 128),
            lambda r: r[5].update(gradient_tensors_nonzero=3),
            lambda r: r[5].update(parameters_unchanged=False),
        ]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(records)
            mutate(changed)
            with self.subTest(index=index):
                self.assertTrue(validate_rank_evidence(*changed)[0])

    def test_task_lfs_completion_checks_original_git_blobs_and_preserves_failed_source(self):
        payload = b'version https://git-lfs.github.com/spec/v1\noid sha256:' + b'a' * 64 + b'\nsize 12\n'
        item = lfs_item(payload, 'task_00000/environment/data')
        self.assertEqual(item['size'], 12)
        with self.assertRaises(ValueError):
            lfs_item(payload.replace(b'size 12', b'size 99999999999'), item['path'])
        blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        text = f'100644 blob {blob}\t{item["path"]}\0'
        records = tracked_files(text, {'task_00000'})
        for bad in (text.replace('100644', '120000'), text.replace('environment/data', '../data'), text + text):
            with self.assertRaises(ValueError):
                tracked_files(bad, {'task_00000'})
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp).resolve() / 'source', Path(tmp).resolve() / 'new'
            path = source / item['path']
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            self.assertEqual(copy_verified_sources(source, destination, records), [item])
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(destination.exists())
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                copy_verified_sources(source, destination, records)

    def test_task_materialization_labels_grader_assets_and_refuses_linked_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / 'task_00001'
            for name in ('instruction.md', 'task.toml', 'environment/Dockerfile', 'tests/test.sh', 'solution/solve.sh'):
                path = task / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('[metadata]\ncategory="code"\n' if name == 'task.toml' else 'fixture')
            result = inventory_tasks(root, ['task_00001'])
            private = [x['path'] for x in result['files'] if x['access'] == 'grader_only']
            self.assertEqual(set(private), {'task_00001/tests/test.sh', 'task_00001/solution/solve.sh'})
            self.assertEqual(result['categories'], {'code': 1})
            (task / 'environment/link').symlink_to(task / 'solution/solve.sh')
            with self.assertRaises(ValueError):
                inventory_tasks(root, ['task_00001'])
        with patch.dict(os.environ, {'GIT_CURL_VERBOSE': '0', 'GIT_TRACE_CURL': '1'}):
            configure_git_environment()
            self.assertNotIn('GIT_CURL_VERBOSE', os.environ)
            self.assertNotIn('GIT_TRACE_CURL', os.environ)
            self.assertEqual(os.environ['GIT_TERMINAL_PROMPT'], '0')

    def test_clean_split_is_disjoint_deterministic_and_outcome_independent(self):
        tasks = [f'task_{i:05d}' for i in range(20)]
        train, dev = select_tasks(tasks, train_count=8, dev_count=4)
        self.assertEqual((train, dev), select_tasks(list(reversed(tasks)), 8, 4))
        self.assertFalse(set(train) & set(dev))
        self.assertNotIn('task_00000', train + dev)
        with self.assertRaises(ValueError):
            select_tasks(tasks + ['task_00001'], 8, 4)
        with self.assertRaises(ValueError):
            select_tasks(tasks, 20, 4)
        self.assertEqual(next_page('<' + CATALOG_URL + '?cursor=public>; rel="next"'), CATALOG_URL + '?cursor=public')
        with self.assertRaises(ValueError):
            next_page('<https://example.invalid/other>; rel="next"')

    def test_parity_requires_complete_nonoverlapping_expert_and_mtp_coverage(self):
        self.assertEqual(len(ALOG_WIDENINGS), 30)
        self.assertNotIn('model.language_model.layers.3.linear_attn.A_log', ALOG_WIDENINGS)
        self.assertNotIn('mtp.layers.0.linear_attn.A_log', ALOG_WIDENINGS)
        weights = {'model.language_model.layers.0.mlp.experts.gate_up_proj': 'part.safetensors',
                   'mtp.layers.0.mlp.experts.down_proj': 'part.safetensors',
                   'model.language_model.norm.weight': 'part.safetensors',
                   'model.visual.patch_embed.proj.weight': 'vision.safetensors'}
        name = 'model.language_model.layers.0.mlp.experts.1.up_proj.weight'
        self.assertEqual(reference_part(name, weights, 2),
                         ('model.language_model.layers.0.mlp.experts.gate_up_proj', 1, 'up_proj'))
        with self.assertRaises(ValueError):
            reference_part(name.replace('.1.', '.2.'), weights, 2)
        with self.assertRaises(ValueError):
            reference_part('unknown.weight', weights, 2)
        seen = {k: required_parts(k, 2) for k in weights if not k.startswith('model.visual.')}
        self.assertEqual(check_coverage(weights, seen, 2), [])
        seen['mtp.layers.0.mlp.experts.down_proj'].remove((1, 'down_proj'))
        self.assertTrue(check_coverage(weights, seen, 2))
        seen['mtp.layers.0.mlp.experts.down_proj'] = {(None, None), (0, 'down_proj')}
        self.assertTrue(check_coverage(weights, seen, 2))
        seen['mtp.layers.0.mlp.experts.down_proj'] = {(None, None)}
        self.assertEqual(check_coverage(weights, seen, 2), [])
        seen['model.visual.patch_embed.proj.weight'] = {(None, None)}
        self.assertTrue(check_coverage(weights, seen, 2))

    def test_parity_rehashes_inputs_and_rejects_changed_payloads_and_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / 'weights'
            payload.write_bytes(b'base')
            files = [{'path': 'weights', 'bytes': 4, 'sha256': sha256(payload)}]
            verify_files(root, files)
            payload.write_bytes(b'edit')
            with self.assertRaises(ValueError):
                verify_files(root, files)
            files[0]['path'] = '../outside'
            with self.assertRaises(ValueError):
                verify_files(root, files)
            (root / 'link').symlink_to(payload)
            files[0]['path'] = 'link'
            with self.assertRaises(ValueError):
                verify_files(root, files)

    def test_conversion_preserves_mtp_and_refuses_unhashed_links(self):
        command = conversion_command(Path('/source'), Path('/model'), Path('/output.partial'),
                                     '--mtp-num-layers 1 --num-layers 40')
        self.assertIn('--nproc-per-node=8', command)
        self.assertNotIn('--standalone', command)
        self.assertIn('--rdzv-backend=static', command)
        self.assertIn('--master-addr=127.0.0.1', command)
        self.assertEqual(command[-2:], ['--save', '/output.partial'])
        for wrong in ('--num-layers 40', '--mtp-num-layers 0', '--mtp-num-layers 1 --mtp-num-layers 1'):
            with self.assertRaises(ValueError):
                conversion_command(Path('/source'), Path('/model'), Path('/output.partial'), wrong)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / 'weights.distcp'
            payload.write_bytes(b'x' * (1024**2 + 1))
            entries = checkpoint_files(root)
            self.assertEqual(entries[0]['sha256'], sha256(payload))
            (root/'alias').symlink_to(payload)
            with self.assertRaises(ValueError):
                checkpoint_files(root)

    def test_server_cleanup_signals_parent_and_reports_surviving_workers(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        server, child = Mock(pid=22, returncode=-9), Mock(pid=23)
        server.poll.return_value = None
        child.is_running.return_value = True
        child.status.return_value = 'running'
        ps = SimpleNamespace(Process=lambda _: SimpleNamespace(children=lambda **_: [child]),
                             wait_procs=Mock(side_effect=[([], [child]), ([child], [])]),
                             NoSuchProcess=ProcessLookupError, STATUS_ZOMBIE='zombie')
        with patch.dict(sys.modules, {'psutil': ps}), patch('qwen_serving_probe.os.killpg') as killpg:
            result = stop_owned_server(server)
        server.terminate.assert_called_once()
        killpg.assert_not_called()
        child.kill.assert_called_once()
        self.assertTrue(result['forced_cleanup'])
        self.assertTrue(result['errors'])
        self.assertEqual(result['descendant_pids_after_grace'], [23])

    def test_prometheus_nonfinite_values_are_preserved_not_zero_filled(self):
        # A fake parser isolates normalization from optional monitoring packages.
        from types import SimpleNamespace
        samples = [SimpleNamespace(name=name, value=value, labels={'node': 'n'}) for name, value in
                   [('sglang:fwd_occupancy', float('nan')), ('sglang:queue', float('nan')),
                    ('sglang:fwd_occupancy', float('inf')), ('sglang:queue', 3)]]
        parser = SimpleNamespace(text_string_to_metric_families=lambda _: [SimpleNamespace(samples=samples)])
        with patch.dict(sys.modules, {'prometheus_client': SimpleNamespace(), 'prometheus_client.parser': parser}):
            rows = prometheus_rows('retained raw document')
        self.assertFalse(rows[0]['fatal'])
        self.assertEqual(rows[0]['reason'], 'upstream_timer_window_unavailable')
        self.assertTrue(rows[1]['fatal'])
        self.assertTrue(rows[2]['fatal'])
        self.assertIsNone(rows[0]['value'])
        self.assertEqual(rows[3]['value'], 3)
        json.dumps(rows, allow_nan=False)

    def test_chat_renderer_requests_and_preserves_exact_unbatched_ids(self):
        from unittest.mock import Mock
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [151644, 872, 198, 19]
        self.assertEqual(prompt_token_ids(tokenizer, 'text'), [151644, 872, 198, 19])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{'role': 'user', 'content': 'text'}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        for invalid in ({'input_ids': [1]}, [[1]], [True], [], [1.5], [-1]):
            tokenizer.apply_chat_template.return_value = invalid
            with self.assertRaises(ValueError):
                prompt_token_ids(tokenizer, 'text')

    def test_serving_smoke_preserves_node_local_ep8_and_mtp_control(self):
        off, on = server_command('/model', False), server_command('/model', True)
        for argv in (off, on):
            self.assertEqual(argv[argv.index('--tp-size')+1], '8')
            self.assertEqual(argv[argv.index('--ep-size')+1], '8')
            self.assertEqual(argv[argv.index('--nnodes')+1], '1')
            self.assertNotIn('--language-model-only', argv)
            self.assertNotIn('--language-only', argv)
            self.assertIn('--enable-metrics', argv)
        self.assertNotIn('--speculative-algorithm', off)
        self.assertEqual(on[on.index('--speculative-num-steps')+1], '2')
        self.assertEqual(on[on.index('--speculative-num-draft-tokens')+1], '3')
        self.assertEqual(on[on.index('--mamba-scheduler-strategy')+1], 'extra_buffer')

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
