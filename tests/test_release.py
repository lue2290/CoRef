from pathlib import Path
import importlib.util,json,tempfile,unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vrdd_public',ROOT/'run_vrdd.py')
release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_scene_task_index_is_unique_and_names_are_portable(self):
        rows=release.read(ROOT/'checkpoints/index.json')['models']
        self.assertEqual(len(rows),10)
        self.assertEqual(len({(r['scene'],r['task']) for r in rows}),10)
        for row in rows:
            self.assertEqual(row['path'],f"checkpoints/{row['scene']}/{row['task']}/model.pt")
            self.assertEqual(len(row['sha256']),64)
            self.assertTrue((ROOT/f"protocols/{row['scene']}/{row['task']}.json").is_file())

    def test_same_basename_different_image_is_not_silently_reused(self):
        base=ROOT/'.cache/test-release';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            folder=Path(tmp);a=folder/'a/x.jpg';b=folder/'b/x.jpg'
            for p in [a,b]:p.parent.mkdir(parents=True,exist_ok=True)
            a.write_bytes(b'first');b.write_bytes(b'second')
            release.link_image(folder/'view',a)
            self.assertEqual(release.link_image(folder/'view',a),'x.jpg')
            with self.assertRaisesRegex(ValueError,'collision'):release.link_image(folder/'view',b)


if __name__=='__main__':unittest.main()
