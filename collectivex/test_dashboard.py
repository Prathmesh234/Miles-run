"""Check the real imported dataset and reject misleading/malformed measurements."""
import copy,json,unittest
from pathlib import Path
import render

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=json.loads((Path(__file__).parent/'metrics.json').read_text())

    def test_generated_html_matches_reviewed_data(self):
        self.assertEqual(render.render(self.data),(Path(__file__).parent/'index.html').read_text())

    def test_rejects_duplicate_point_and_nonfinite_latency(self):
        duplicate=copy.deepcopy(self.data);duplicate['points'][1]=duplicate['points'][0]
        with self.assertRaises(AssertionError):render.validate(duplicate)
        bad=copy.deepcopy(self.data);bad['points'][0]['components']['dispatch']['latency_us']['p50']=float('nan')
        with self.assertRaises(AssertionError):render.validate(bad)

    def test_payload_contains_no_private_inventory_or_remote_dependencies(self):
        serialized=json.dumps(self.data)
        for excluded in ['gpu_uuid','kubeconfig','Authorization','slack.com/archives','/shared/','/Users/']:
            self.assertNotIn(excluded,serialized)
        html=render.render(self.data)
        self.assertNotIn('<script src=',html)
        self.assertNotIn('fetch(',html)

    def test_missing_infrastructure_values_stay_null(self):
        data=copy.deepcopy(self.data)
        data['infrastructure']['gpu-nodes-0']['power.draw'][0][1]=None
        html=render.render(data)
        embedded=html.split('<script id="data" type="application/json">',1)[1].split('</script>',1)[0]
        self.assertIsNone(json.loads(embedded)['infrastructure']['gpu-nodes-0']['power.draw'][0][1])

if __name__=='__main__':unittest.main()
