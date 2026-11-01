import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from PIL import Image
from refnet.protocol import read, write, sha, fp
from refnet_f1.protocol import inputs, remap_cached, summary, export_results
from refnet_f1.runtime import load_references, calibrate_threshold, decode, calibrate, predict, official_evaluation
from refnet_f1.experiment import validate_config, make_plans

ROOT = Path(__file__).resolve().parents[1]


class FakeParent(nn.Module):
    def __init__(self):
        super().__init__(); self.anchor=nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.active=[0, 1]; self.calls=[]

    def relation(self, raw, refs, labels, active):
        assert active == [0, 1]
        return raw

    def forward_image(self, view, name, objects, device):
        assert objects == []
        self.calls.append(name)
        raw = torch.tensor([[3., 1., -2.], [2., 0., -2.]]) if name=='cal.jpg' else torch.tensor([[1., 0., 4.], [3., 0., -3.]])
        return dict(cls=torch.ones(1, 2, 3)*3, ctr=torch.ones(1, 2)*2, raw=raw,
                    refined=torch.tensor([[.25, .25, .3, .3], [.75, .75, .3, .3]]),
                    quality_logit=torch.ones(2)*3, meta=dict(ori_shape=(100,100)))


class FormalF1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg=read(ROOT/'configs/reference_f1.json')
        cls.parent_cfg=read(ROOT/'configs/reference_net.json')['model']

    def test_all30_parents_full_bank_corrected_game_and_task2(self):
        validate_config(self.cfg)
        ids=[]
        for scene in self.cfg['scenes']:
            for task in self.cfg['tasks']:
                for seed in self.cfg['seeds']:
                    spec, parent, binding=inputs(ROOT, scene, task, seed)
                    self.assertEqual(len(spec['classes']),len(set(o['class_id'] for o in spec['fit_objects'])))
                    self.assertEqual(parent['terminal']['step'],500)
                    self.assertEqual(parent['task'],task)
                    self.assertNotIn('held',spec)
                    if task=='task2': self.assertEqual(spec['classes'],spec['PK']+spec['CK'])
                    self.assertEqual(binding['corrected_reference_labels'],scene=='Game')
                    if scene=='Game': self.assertIn('annotation_revisions',binding['split_path'])
                    ids.append(parent['id'])
        self.assertEqual(len(set(ids)),30)

    def test_negative_threshold_ties_known_and_no_gt_or_test_fallback(self):
        self.assertEqual(calibrate_threshold([-5.,-4.]),-4.)
        with self.assertRaises(RuntimeError): calibrate_threshold([])
        with self.assertRaises(RuntimeError): calibrate_threshold([np.nan])
        v=dict(logits=torch.tensor([[0.,-1.,-4.],[0.,-1.,-3.]]),joint=torch.ones(2),
               boxes=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]]),ids=torch.arange(2))
        cal=dict(threshold_margin=-4.,object_threshold=0.,f0_reject_shift=0.)
        pred,_=decode(v,'F1',cal,self.parent_cfg)
        labels=dict(zip([tuple(b) for b in pred['boxes']],pred['labels']))
        self.assertEqual(labels[(0.,0.,10.,10.)],0)
        self.assertEqual(labels[(20.,20.,30.,30.)],2)

    def test_f1_score_is_raw_sigmoid_not_threshold_shifted_or_dual(self):
        logits=torch.tensor([[2.,0.,3.],[2.,0.,-1.]])
        v=dict(logits=logits,joint=torch.tensor([.8,.9]),boxes=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]]),ids=torch.arange(2))
        cal=dict(threshold_margin=-1.,object_threshold=.1,f0_reject_shift=.5)
        p,_=decode(v,'F1',cal,self.parent_cfg)
        expected=[.8*torch.sigmoid(torch.tensor(1.)).item(),.9*(1-torch.sigmoid(torch.tensor(-3.)).item())*torch.softmax(logits[1,:2],0).max().item()]
        by_box={tuple(b):(s,y) for b,s,y in zip(p['boxes'],p['scores'],p['labels'])}
        for j,box in enumerate(v['boxes'].tolist()): self.assertAlmostEqual(by_box[tuple(box)][0],expected[j],places=6)
        self.assertEqual(len(p['boxes']),2)

    def test_f0_original_softmax_nms_and_top100_budget(self):
        n=130; x=torch.arange(n,dtype=torch.float32)*20
        v=dict(logits=torch.tensor([[2.,0.,3.]]*n),joint=torch.ones(n),
               boxes=torch.stack([x,x,x+10,x+10],1),ids=torch.arange(n))
        cal=dict(threshold_margin=0.,object_threshold=.1,f0_reject_shift=2.)
        p,_=decode(v,'F0',cal,self.parent_cfg)
        self.assertEqual(len(p['scores']),100)
        self.assertEqual(set(p['labels']),{0})
        self.assertAlmostEqual(p['scores'][0],torch.tensor([2.,0.,1.]).softmax(0).max().item(),places=6)
        v['joint'].zero_()
        self.assertEqual(decode(v,'F1',cal,self.parent_cfg)[0]['scores'],[])

    def test_decoding_matches_completed156_f0_and_f1(self):
        from reject_pair.evaluation import decode as old_decode
        torch.manual_seed(7)
        logits=torch.randn(24,4); x=torch.arange(24,dtype=torch.float32)*20
        v=dict(logits=logits,joint=torch.rand(24),boxes=torch.stack([x,x,x+10,x+10],1),ids=torch.arange(24))
        cal=dict(threshold_margin=-.7,object_threshold=.15,f0_reject_shift=.4)
        for arm in ['F0','F1']:
            z=logits[:,-1] if arm=='F0' else logits[:,-1]-logits[:,:-1].max(1).values
            scored=dict(known=logits[:,:-1].numpy(),z=z.numpy())
            expected,_=old_decode(scored,v['boxes'].numpy(),v['joint'].numpy(),[0,1,2],3,arm,
                dict(threshold_logit=.4 if arm=='F0' else -.7,object_threshold=.15))
            actual,_=decode(v,arm,cal,self.parent_cfg)
            for k in ['boxes','scores','labels']: self.assertEqual(actual[k],expected[k])

    def test_corrected_gt_relabels_identical_cached_identity_only(self):
        raw=np.arange(2*5120,dtype=np.float32).reshape(2,5120)
        old=[dict(id='a',class_id=0,box=[0,0,10,10]),dict(id='b',class_id=1,box=[10,10,20,20])]
        new=[dict(id='relabel_b',original_instance_id='b',class_id=0,box=[10,10,20,20]),old[0]]
        gt=np.array([o['box'] for o in old])/100
        got,y=remap_cached(raw,['a','b'],gt,[0,1],old,new,np.ones(4)*100)
        self.assertTrue(np.array_equal(got[0],raw[1]));self.assertEqual(y,[0,0])
        bad=copy.deepcopy(new);bad[0]['box'][0]=11
        with self.assertRaises(RuntimeError): remap_cached(raw,['a','b'],gt,[0,1],old,bad,np.ones(4)*100)

    def test_complete_reference_bank_loading_and_cal_separation(self):
        # Exercise the public loader, including the class-count audit which the
        # earlier per-instance remapping test did not reach.
        for scene in ['Aquatic', 'Game']:
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as td:
                root=Path(td); folder=root/'cache';folder.mkdir()
                objects=[dict(id='a',image='fit.jpg',class_id=0,box=[10,10,40,40]),
                         dict(id='b',image='fit.jpg',class_id=1,box=[60,60,90,90]),
                         dict(id='c',image='cal.jpg',class_id=0,box=[10,10,40,40])]
                original=dict(identity='original_split',objects=objects)
                write(root/'original.json',original)
                spec=dict(identity='corrected_split' if scene=='Game' else 'original_split',scene=scene,
                    task='task1',classes=['a','b'],fit_images=['fit.jpg'],cal_images=['cal.jpg'],
                    groups={'fit.jpg':'fit_group','cal.jpg':'cal_group'},objects=copy.deepcopy(objects))
                if scene=='Game':
                    for obj in spec['objects'][:2]:
                        obj.update(original_instance_id=obj['id'],id='relabel_'+obj['id'],class_id=1-obj['class_id'])
                manifest=dict(split='original_split',rows=[],files={});dataset=dict(files={})
                for role,name in [('fit','fit.jpg'),('cal','cal.jpg')]:
                    obs=[o for o in objects if o['image']==name]
                    image=root/'assets/rwd/JPEGImages'/scene/name;image.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(100,100)).save(image)
                    dataset['files'][f'JPEGImages/{scene}/{name}']=dict(sha256=sha(image))
                    path=folder/(role+'.npz')
                    np.savez_compressed(path,raw=np.full((len(obs),5120),1 if role=='fit' else 9,dtype=np.float32),
                        ids=np.array([o['id'] for o in obs]),labels=np.array([o['class_id'] for o in obs]),
                        gt=np.array([o['box'] for o in obs],np.float32)/100)
                    manifest['rows'].append(dict(role=role,image=name,file=path.name))
                    manifest['files'][path.name]=sha(path)
                write(folder/'MANIFEST.json',manifest)
                parent=dict(reference_cache='cache',reference_manifest_sha256=sha(folder/'MANIFEST.json'),split='original.json')
                bank,cal_gt,audit=load_references(root,spec,parent,'cpu',dataset,{})
                self.assertEqual(tuple(bank['raw'].shape),(2,5120))
                self.assertTrue(torch.all(bank['raw']==1))
                self.assertTrue(np.all(cal_gt['cal.jpg']==9))
                self.assertEqual(bank['labels'].tolist(),[1,0] if scene=='Game' else [0,1])
                self.assertEqual(audit['class_counts'],{'0':1,'1':1})
                self.assertEqual(audit['corrected_labels'],scene=='Game')
                self.assertEqual({row['image'] for row in audit['rows']},{'fit.jpg'})

    def test_macro_first_per_seed_then_sample_sd_and_incomplete(self):
        rows=[]
        for scene in self.cfg['scenes']:
            for task in self.cfg['tasks']:
                for j,seed in enumerate(self.cfg['seeds']):
                    metrics={a:dict(K_AP50=60+j,U_AP50=20+j+(2 if a=='F1' else 0),PK_AP50=62+j,CK_AP50=59+j) for a in ['F0','F1']}
                    rows.append(dict(scene=scene,task=task,seed=seed,metrics=metrics))
        result=summary(rows,self.cfg)
        self.assertTrue(result['complete'])
        self.assertEqual(result['macro']['task1/F1/U_AP50']['mean'],23.)
        self.assertEqual(result['macro']['task1/F1/U_AP50']['sample_std'],1.)
        self.assertEqual(result['paired_macro_delta']['task1/U_AP50']['mean'],2.)
        partial=summary(rows[:-1],self.cfg)
        self.assertFalse(partial['complete']);self.assertEqual(partial['macro']['task2/F1/U_AP50']['n'],2)
        with self.assertRaises(RuntimeError):summary(rows+[rows[0]],self.cfg)

    def test_calibration_prediction_resume_official_xml_and_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); report=root/'reports/reference_f1'/self.cfg['tag']/'Aquatic/task1/seed_42'
            spec=dict(scene='Aquatic',task='task1',classes=['a','b'],PK=['a','b'],CK=[],
                cal_images=['cal.jpg'],test_images=['test.jpg'],groups={'cal.jpg':'train_group'},rare_classes=[],
                cal_objects=[dict(image='cal.jpg',class_id=0,box=[10,10,40,40]),dict(image='cal.jpg',class_id=1,box=[60,60,90,90])])
            dataset=dict(files={})
            for name in ['cal.jpg','test.jpg']:
                path=root/'assets/rwd/JPEGImages/Aquatic'/name;path.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(100,100)).save(path)
                dataset['files'][f'JPEGImages/Aquatic/{name}']=dict(sha256=sha(path))
            model=FakeParent();bank=dict(raw=torch.empty(0,3),labels=torch.empty(0,dtype=torch.long));verified={}
            cal=calibrate(model,root,root,spec,bank,{},self.cfg,self.parent_cfg,dict(object_threshold=.1,reject_shift=0.),report,verified,dataset)
            self.assertEqual(cal['threshold_margin'],-4.)
            seal=predict(model,root,root,spec,bank,self.cfg,self.parent_cfg,cal,report,dataset,verified)
            self.assertEqual(model.calls,['cal.jpg','test.jpg'])
            # Atomic files are reused; a stray incomplete temp file is never consumed.
            (report/'test_images/ignored.json.tmp').write_text('{')
            cal2=calibrate(model,root,root,spec,bank,{},self.cfg,self.parent_cfg,dict(object_threshold=.1,reject_shift=0.),report,verified,dataset)
            self.assertEqual(cal,cal2)
            predict(model,root,root,spec,bank,self.cfg,self.parent_cfg,cal2,report,dataset,verified)
            self.assertEqual(model.calls,['cal.jpg','test.jpg'])
            # Real official evaluator, full XML includes an unknown outside the shot list.
            xml=root/'assets/rwd/Annotations/Aquatic/test.xml';xml.parent.mkdir(parents=True)
            xml.write_text('<annotation><object><name>novel</name><difficult>0</difficult><bndbox><xmin>11</xmin><ymin>11</ymin><xmax>41</xmax><ymax>41</ymax></bndbox></object><object><name>a</name><bndbox><xmin>61</xmin><ymin>61</ymin><xmax>91</xmax><ymax>91</ymax></bndbox></object></annotation>')
            dataset['files']['Annotations/Aquatic/test.xml']=dict(sha256=sha(xml))
            vendor=root/'vendor';vendor.mkdir();shutil.copy2(ROOT/'vendor/fomo_open_world_eval.py',vendor/'fomo_open_world_eval.py')
            metrics=official_evaluation(root,spec,self.cfg,report,dataset,verified)
            self.assertAlmostEqual(metrics['F1']['U_AP50'],100.)
            write(report/'RESULT.json',dict(metrics=metrics))
            cfg=copy.deepcopy(self.cfg);cfg['min_free_gib']=0
            receipt=export_results(root,cfg)
            self.assertTrue(receipt['crc_verified']);self.assertEqual(sha(receipt['path']),receipt['sha256'])
            self.assertEqual(seal['predictions']['F1'],sha(report/'F1/predictions.json'))


if __name__=='__main__':unittest.main()
