"""Regression checks for evidence units, missing-data handling, and phase status."""
import json
import math
import csv
from pathlib import Path
import tempfile
import unittest

from analyze_ppo import epoch, phase_results, sha, stats, telemetry


class EvidenceTests(unittest.TestCase):
    def test_quantiles_and_missing_are_not_zero(self):
        self.assertIsNone(stats([None,float('nan')]))
        self.assertEqual(stats([7])['p99'],7)
        self.assertEqual(stats([0,10])['p90'],9)
        self.assertIsNone(stats([0,0])['cv'])
        self.assertEqual(epoch('2026-09-03T15:24:22.87057993Z'),epoch('2026-09-03T15:24:22.870579Z'))

    def test_failed_phase_retains_log_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'training.log').write_text('the actual failure\n')
            events=[{'time':'2026-09-04T00:00:00Z','event':'training_start','argv':['train']},
                    {'time':'2026-09-04T00:00:03Z','event':'training_end','exit_code':1}]
            (root/'timeline.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
            result=phase_results(root)[0]
            self.assertEqual(result['status'],'fail')
            self.assertEqual(result['duration_s'],3)
            self.assertEqual(result['log_sha256'],sha(root/'training.log'))
            events[1]={'time':'2026-09-04T00:00:04Z','event':'coordinator_exit','exit_code':1}
            (root/'timeline.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
            incomplete=phase_results(root)[0]
            self.assertEqual(incomplete['status'],'fail')
            self.assertIsNone(incomplete['exit_code'])
            self.assertEqual(incomplete['duration_s'],4)

    def test_pma_four_byte_units_and_counter_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'infra').mkdir()
            rows=[]
            for second,count in [(0,1000),(10,1250),(20,100)]:
                rows.append({'utc':f'2026-09-04T00:00:{second:02d}Z','ports':[
                    {'argv':['perfquery','-C','mlx5_0'],'status':0,
                     'stdout':f'PortXmitData:....{count}\nPortRcvData:....{count}\n'}]})
            (root/'infra/gpu-nodes-0-rdma-counters.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            data=telemetry(root,epoch(rows[0]['utc']),epoch(rows[-1]['utc']))
            tx=[x for x in data['series'] if x['metric']=='ib_tx_gbps']
            self.assertEqual(len(tx),1)
            self.assertAlmostEqual(tx[0]['value'],250*4*8/10/1e9)
            self.assertEqual(data['collector_errors']['gpu-nodes-0:counter_reset:mlx5_0'],2)
            self.assertNotIn('gpu-nodes-1/gpu_util_percent_mean',data['node_statistics'])


@unittest.skipUnless(Path('results/comparison.json').exists(), 'Final artifacts not generated yet')
class PublishedRunTests(unittest.TestCase):
    def test_optimizer_and_checkpoint_evidence(self):
        data=json.loads(Path('results/comparison.json').read_text())
        run=next(r for r in data['runs'] if r['algorithm']=='ppo')
        self.assertEqual(run['status'],'ok')
        self.assertEqual(run['completed_actor_updates'],2)
        self.assertEqual(run['completed_critic_updates'],2)
        self.assertEqual(run['positive_actor_gradient_steps'],2)
        for rows in run['all_scalars'].values():
            self.assertTrue(all(math.isfinite(r['value']) for r in rows))
        for key in ['checkpoints','critic_checkpoint']:
            checkpoint=run[key]
            self.assertIsNotNone(checkpoint)
            self.assertTrue(all(r['finite'] for r in checkpoint['sampled_tensor_reads'].values()))
            self.assertGreater(sum(r['changed_elements'] for r in checkpoint['selected_tensors_vs_base'].values()),0)

    def test_trajectory_and_gpu_accounting(self):
        data=json.loads(Path('results/comparison.json').read_text())
        run=next(r for r in data['runs'] if r['algorithm']=='ppo')
        self.assertEqual(run['accounting']['accepted'],32)
        self.assertEqual(run['accounting']['consumed_by_logged_actor_step'],32)
        self.assertEqual(len({e['trace_id'] for e in run['episodes'] if e['accepted']}),32)
        self.assertEqual(run['accounting']['errors'],sum(not e['ok'] for e in run['episodes']))
        self.assertEqual(len(run['telemetry']['gpu_statistics']),32)
        self.assertGreater(len(run['telemetry']['prometheus_last_sample']),0)
        provenance=json.loads(Path('results/provenance.json').read_text())
        self.assertEqual(len(provenance['ray_placement']),32)
        self.assertEqual(provenance['native_tests']['status'],'passed')
        self.assertEqual(provenance['transport_tests']['status'],'passed')
        self.assertEqual([r['status'] for r in provenance['policy_validity']],['passed','passed'])
        self.assertTrue(all(r['unchanged'] for r in provenance['final_runtime']['baseline_source_preservation']))
        nodes=provenance['node_post_runtime']
        self.assertEqual(len(nodes),4)
        uuids=[u for n in nodes for u in n['gpu_uuids']]
        self.assertEqual(len(set(uuids)),32)
        before={row.split(',')[0] for n in provenance['launch_manifest']['inventory'] for row in n['gpus']['stdout'].strip().splitlines()}
        self.assertEqual(set(uuids),before)
        for node in nodes:
            self.assertEqual(node['summary']['remaining_run_containers'],'')
        commands=provenance['final_runtime']['commands']
        queue=next(c for c in commands if c['argv'][0]=='squeue')
        self.assertNotIn(str(run['job_id']),[line.split('|')[0] for line in queue['stdout'].splitlines()])
        accounting=next(c for c in commands if c['argv'][0]=='sacct')
        own=next(row for row in csv.DictReader(accounting['stdout'].splitlines(),delimiter='|') if row['JobID']==str(run['job_id']))
        self.assertEqual(own['State'],'COMPLETED')
        self.assertEqual(own['ExitCode'],'0:0')


if __name__=='__main__':
    unittest.main()
