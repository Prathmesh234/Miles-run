"""Small synthetic counter fixtures verify units/coverage, never used in report data."""
import json
from pathlib import Path
import tempfile
import unittest

import analyze as a
import capture_fabric_1s as capture


class AnalysisTests(unittest.TestCase):
    def test_naive_timestamps_are_utc(self):
        self.assertEqual(a.epoch('1970-01-01 00:00:01.000'),1)
        self.assertEqual(a.epoch('1970-01-01T01:00:01+01:00'),1)

    def test_overlap_does_not_double_count_actor_and_critic(self):
        self.assertEqual(a.union_seconds([(0,5),(2,7),(10,11)]),8)
        self.assertEqual(a.overlap(0,3,3,8),0)

    def test_incomplete_node_is_missing_not_zero_or_rescaled(self):
        hw={'n':{'ib_port_gbps':{'p0':400,'p1':400,'storage':100},'nvlink_reported_line_GBps':{}}}
        rows=[dict(fabric='IB',host='n',device='p0',start=0,end=1,direction='TX',gbps=200,
                   capacity_gbps=400,utilization_pct=50)]
        partial=a.aggregate_fabric(rows,hw)[0]
        self.assertIsNone(partial['utilization_pct'])
        self.assertFalse(partial['complete'])
        full=a.aggregate_fabric(rows+[dict(rows[0],device='p1',gbps=0,utilization_pct=0),
                                     dict(rows[0],device='storage',capacity_gbps=100)],hw)[0]
        self.assertEqual(full['utilization_pct'],25)
        self.assertEqual(full['hottest_link_utilization_pct'],50)
        self.assertEqual(full['collected_links'],2)

    def test_highrate_units_interval_and_reset(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'infra').mkdir()
            rows=[]
            for t,ib,nv in [(10,100,200),(12,200,400),(13,1,1),(14,11,21)]:
                item=dict(status=0,end=t,monotonic=t+100)
                rows.append(dict(time=t,collector_wall_seconds=.1,
                    ib=[dict(item,device='mlx5_0',counters={'PortXmitData':ib})],
                    nvlink=dict(item,counters={'gpu/0':{'Tx':nv}})))
            (root/'infra/gpu-nodes-0-fabric-1s.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result,errors,cost=a.highrate_fabric(root)
            first=[r for r in result if r['end']==12]
            ib=next(r for r in first if r['fabric']=='IB')
            nv=next(r for r in first if r['fabric']=='NVLink')
            self.assertEqual(ib['bytes'],400)
            self.assertAlmostEqual(ib['gbps'],400*8/2/1e9)
            self.assertEqual(nv['bytes'],200*1024)
            self.assertEqual(sum(errors.values()),2)
            self.assertEqual(len(result),4)
            self.assertEqual(cost['count'],4)

    def test_native_timer_and_scalars_keep_critic_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'training.log').write_text(
                '[2026-09-04 01:00:00.000 actor_cell0_rank0] Timer actor_train start\n'
                '[2026-09-04 01:00:01.500 actor_cell0_rank0] Timer actor_train end\n'
                "[2026-09-04 01:00:01.501 actor_cell0_rank0] step 0: {'train/grad_norm': 2.0}\n"
                "[2026-09-04 01:00:02.000 critic_cell0_rank0] step 0: {'train/grad_norm': 4.0}\n")
            timers,scalars=a.timers_and_scalars(root)
            self.assertEqual(timers[0]['seconds'],1.5)
            self.assertEqual(scalars[('actor',0)]['train/grad_norm'],2)
            self.assertEqual(scalars[('critic',0)]['train/grad_norm'],4)

    def test_sampler_extracts_counters(self):
        original=capture.command
        try:
            capture.command=lambda _:dict(status=0,stdout='PortXmitData:..........42\n')
            self.assertEqual(capture.sample_port('mlx5_0')['counters']['PortXmitData'],42)
            capture.command=lambda _:dict(status=0,stdout='GPU 0: B200 (UUID: GPU-test)\n Link 0: Data Tx: 123 KiB\n Link 0: Data Rx: 456 KiB\n')
            self.assertEqual(capture.sample_nvlink()['counters'],{'GPU-test/0':{'Tx':123,'Rx':456}})
        finally: capture.command=original

    def test_native_lifecycle_retains_quantized_rank_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'training.log').write_text('[2026-09-04 01:31:16.508 actor_cell0_rank8] structured_log.py:27 - ft cls=MegatronTrainRayActor fn=wake_up phase=end ok=true elapsed_s=4.6\n')
            rows=a.native_lifecycle(root)
            self.assertEqual((rows[0]['rank'],rows[0]['role'],rows[0]['elapsed_seconds']),(8,'actor',4.6))

    def test_role_inference_requires_unambiguous_enclosing_train_span(self):
        row=dict(time=10,elapsed_seconds=2,host='n',pid=1,role='unknown')
        phase=dict(name='critic_train',start=1,end=11)
        self.assertEqual(a.optimizer_role(row,{},[phase])[0],'critic')
        self.assertEqual(a.optimizer_role(row,{},[phase,dict(phase,name='actor_train')])[0],'unknown')
        self.assertEqual(a.optimizer_role(dict(row,elapsed_seconds=20),{},[phase])[0],'unknown')


if __name__=='__main__':unittest.main()
