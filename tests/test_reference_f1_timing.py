import tempfile
import unittest
import zipfile
from pathlib import Path
from reference_f1_timing import stats,select_images,export,TAG
from refnet_f1.protocol import write


class TimingTests(unittest.TestCase):
    def test_statistics_units_and_nonfinite(self):
        r=stats([1,2,3,4,5]);self.assertEqual(r['median_ms'],3);self.assertAlmostEqual(r['p90_ms'],4.6)
        for values in [[],[float('nan')]]:
            with self.assertRaises(ValueError):stats(values)

    def test_image_selection_independent_of_input_order(self):
        self.assertEqual(select_images(['a','b','c'],2),select_images(['c','b','a'],2))
        self.assertEqual(len(select_images(['a','b'],30)),2)

    def test_summary_uses_online_and_e2e_not_stage_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'reports/reference_f1_timing'/TAG/'Aquatic/task1/seed_42'
            s=dict(online=stats([10,20,30]),end_to_end=stats([100,200,300]),stages=stats([1000,2000,3000]))
            write(folder/'RESULT.json',dict(scene='Aquatic',task='task1',seed=42,summary={'F1':s}))
            write(folder/'SAMPLES.json',[]);write(folder/'LOCK.json',{})
            r=export(root)
            with zipfile.ZipFile(r['path']) as z:
                self.assertIsNone(z.testzip())
                text=z.read('reports/reference_f1_timing/'+TAG+'/SUMMARY.md').decode()
                self.assertIn('20.000',text);self.assertIn('200.000',text);self.assertNotIn('2000.000',text)


if __name__=='__main__':unittest.main()
