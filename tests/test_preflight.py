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
    def test_optimizer_log_parser_deduplicates_actual_log_shape_and_rejects_conflicts(self):
        from observe_grpo_log import parse_log

        first = "[2026-09-02 22:57:57.204 actor_cell0_rank0] log_utils.py:544 - step 0: {'train/step': 0, 'train/grad_norm': 0.4}"
        duplicate = first.replace('log_utils.py:544', 'model.py:861')
        data = parse_log(first + '\n' + duplicate)
        self.assertEqual(len(data['steps']), 1)
        self.assertEqual(data['steps'][0]['time'], '2026-09-02T22:57:57.204Z')
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
