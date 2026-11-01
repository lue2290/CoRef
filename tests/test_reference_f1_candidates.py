import unittest
from pathlib import Path
import tempfile
import zipfile
from reference_f1_candidates import attribution, percentages, save_gzip, load_gzip, export, home, verify_bank_audit, diagnostic_lock, EMPTY_ATTEMPT_SOURCE, replay_difference
from refnet_f1.protocol import write, sha, lock, fp


class CandidateAttributionTests(unittest.TestCase):
    def test_candidate_only_archive_preserves_json_bytes_and_excludes_images(self):
        from reference_f1_candidates import export_candidates_only
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=home(root)/'Aquatic/task1/seed_42';p=folder/'images/one.json.gz'
            save_gzip(p,dict(evidence=self.evidence([1])))
            write(folder/'SUMMARY.json',dict(complete=True,files=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p))]))
            (folder/'picture.jpg').write_bytes(b'not included')
            receipt=export_candidates_only(root)
            with zipfile.ZipFile(receipt['path']) as z:
                self.assertEqual(z.namelist(),[p.relative_to(root).as_posix()])
                self.assertEqual(z.read(z.namelist()[0]),p.read_bytes())
            self.assertFalse(receipt['contains_images']);self.assertFalse(receipt['complete'])

    def test_replay_report_separates_hash_from_real_output_changes(self):
        import torch, hashlib
        values=dict(ids=torch.tensor([3]),boxes=torch.tensor([[0.,0.,10.,10.]]),joint=torch.tensor([.5]),logits=torch.tensor([[1.,2.]]))
        h=hashlib.sha256()
        for k in ['ids','boxes','joint','logits']:h.update(values[k].numpy().tobytes())
        p=dict(boxes=[[0.,0.,10.,10.]],labels=[1],scores=[.5]);diag=dict(selected_proposal_ids=[3])
        old=dict(predictions={'F1':p},diagnostics={'F1':diag},shared_candidate_evidence_sha256=h.hexdigest())
        self.assertTrue(replay_difference(values,p,diag,old)['evidence_exact'])
        values['logits'][0,0]+=.00001
        d=replay_difference(values,p,diag,old)
        self.assertFalse(d['evidence_exact']);self.assertTrue(d['predictions_exact'])
        changed=dict(p,labels=[0]);d=replay_difference(values,changed,diag,old)
        self.assertEqual(d['common_identity_changes'],1)

    def test_original157_nested_identity_is_verified_without_ignoring_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            body=dict(rows=[dict(id='one',class_id=0)],class_counts={'0':1})
            audit=dict(body,identity=fp(body));saved=lock(Path(tmp)/'bank.json',audit)
            verify_bank_audit(audit,saved)
            changed=dict(saved,rows=[dict(id='one',class_id=1)])
            with self.assertRaisesRegex(RuntimeError,'rows'):verify_bank_audit(audit,changed)
            with self.assertRaisesRegex(RuntimeError,'identity'):verify_bank_audit(audit,dict(saved,identity='wrong'))

    def test_only_empty_known_failed_attempt_can_migrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);old=dict(source={'src/reference_f1_candidates.py':EMPTY_ATTEMPT_SOURCE},fixed='same')
            lock(folder/'LOCK.json',old)
            new=dict(source={'src/reference_f1_candidates.py':'patched'},fixed='same')
            diagnostic_lock(folder,new)
            self.assertEqual(len(list(folder.glob('EMPTY_ATTEMPT_LOCK_*.json'))),1)
            diagnostic_lock(folder,new)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);lock(folder/'LOCK.json',old);(folder/'images').mkdir();(folder/'images'/'one.json.gz').write_bytes(b'preserve')
            with self.assertRaisesRegex(RuntimeError,'candidate outputs'):diagnostic_lock(folder,new)

    def evidence(self,labels,gate=None,kept=None):
        n=len(labels)
        return dict(boxes=[[0,0,10,10]]*n,proposal_ids=list(range(100,100+n)),F1_identity=labels,
            passed_gate=gate or [True]*n,final_output=kept or [False]*n)

    def gt(self,label=0,difficult=False):
        return dict(gt_id=0,original_class='known' if label==0 else 'novel',class_id=label,difficult=difficult,box=[0,0,10,10])

    def test_duplicate_candidates_not_duplicate_gt(self):
        r=attribution(self.evidence([1,1,1]),[self.gt()],1)
        self.assertEqual(r['counts']['known_rejected_as_unknown'],3)
        self.assertEqual(r['counts']['known_covered_gt_all_identity_error'],1)
        self.assertEqual(r['candidate_matches'][0]['proposal_id'],100)

    def test_one_correct_candidate_prevents_all_rejected(self):
        r=attribution(self.evidence([1,0]),[self.gt()],1)
        self.assertEqual(r['counts']['known_covered_gt_all_identity_error'],0)
        self.assertEqual(r['counts']['known_covered_gt_with_any_identity_error'],1)

    def test_gate_is_separate_from_identity(self):
        r=attribution(self.evidence([0,1],[False,True]),[self.gt()],1)
        self.assertEqual(r['gt_records'][0]['correctly_identified_candidates'],1)
        self.assertTrue(r['gt_records'][0]['all_gate_surviving_candidates_identity_error'])
        self.assertEqual(r['counts']['known_rejected_as_unknown_passed_gate'],1)

    def test_unknown_absorption_and_known_wrong_class(self):
        r=attribution(self.evidence([0,0]),[self.gt(1)],1)
        self.assertEqual(r['counts']['unknown_accepted_as_known'],2)
        self.assertEqual(r['counts']['unknown_covered_gt_all_identity_error'],1)
        r=attribution(self.evidence([1]),[self.gt()],2)
        self.assertEqual(r['counts']['known_wrong_class'],1)
        self.assertEqual(r['counts'].get('known_rejected_as_unknown',0),0)

    def test_difficult_and_empty_gt_are_not_false_rejections(self):
        for gt in [[],[self.gt(difficult=True)]]:
            r=attribution(self.evidence([1]),gt,1)
            self.assertEqual(r['counts']['no_nondifficult_gt_iou50'],1)
            self.assertIsNone(r['candidate_matches'][0]['matched_gt_id'])
            self.assertIsNone(percentages(r['counts'])['known_candidate_false_reject_rate']['percent'])

    def test_exact_iou_boundary_and_multiple_targets(self):
        e=self.evidence([0]);e['boxes']=[[0,0,5,10]]
        r=attribution(e,[self.gt(),dict(self.gt(),gt_id=1)],1)
        self.assertEqual(r['counts']['known_localized_candidates'],1)
        self.assertEqual(r['counts']['known_covered_gt'],2)

    def test_compressed_records_export_and_corruption_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=home(root)/'Aquatic/task1/seed_42';p=folder/'images/one.json.gz'
            record=dict(evidence=self.evidence([1]),analysis=attribution(self.evidence([1]),[self.gt()],1))
            save_gzip(p,record);self.assertEqual(load_gzip(p),record)
            write(folder/'LOCK.json',dict(frozen=True))
            write(folder/'SUMMARY.json',dict(complete=True,scene='Aquatic',task='task1',seed=42,counts=record['analysis']['counts'],files=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p))]))
            receipt=export(root,dict(min_free_gib=0))
            self.assertFalse(receipt['complete']);self.assertEqual(receipt['cases'],1)
            with zipfile.ZipFile(receipt['path']) as z:
                self.assertIsNone(z.testzip());self.assertIn(p.relative_to(root).as_posix(),z.namelist())
            p.write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):export(root,dict(min_free_gib=0))


if __name__=='__main__':unittest.main()
