from pathlib import Path
import unittest,tempfile,copy,json
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from refnet.protocol import read,write,split_spec,recover_best,sha
from refnet.relation import ReferenceRelation,episode_loss
from refnet.loss import perturb,total_loss
from refnet.model import ReferenceDetector
from jointdet.geometry import xyxy,overlaps

ROOT=Path(__file__).resolve().parents[1]


class TinyDino(nn.Module):
    num_prefix_tokens=1
    def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(1))
    def forward_features(self,d):return torch.arange(40,dtype=d.dtype).reshape(1,5,8)*self.anchor/40


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__();self.cls=nn.Linear(256,3);self.box=nn.Linear(256,4);self.ctr=nn.Linear(256,1)
    def forward(self,h,*_):return self.cls(h),self.box(h).sigmoid(),self.ctr(h).sigmoid()


class TinyPF(nn.Module):
    def __init__(self):
        super().__init__();self.decoder=nn.Linear(256,256);self.query_embedding=nn.Embedding(12,256);self.bbox_head=TinyHead()
    def extract_feat(self,x):return x
    def forward_transformer(self,x,extra,samples):
        owner=self.decoder._joint_owner();q=self.query_embedding.weight[None]
        boxes=q.new_full((1,12,4),.4);h=owner.bridges['3'](q,owner._semantic_map,boxes)
        return dict(hidden_states=h[None],references=None,memory_text=None,text_token_mask=None)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_all_ten_splits_disjoint_and_rare_classes_retained(self):
        cfg=read(ROOT/'configs/reference_net.json')
        for scene in cfg['scenes']:
            for task in ['task1','task2']:
                s=split_spec(ROOT,scene,task);self.assertEqual(s['identity'],split_spec(ROOT,scene,task)['identity'])
                sets=[set(s[k+'_images']) for k in ['fit','cal','monitor']]+[set(s['test_images'])]
                for i,a in enumerate(sets):
                    for b in sets[i+1:]:self.assertFalse(a&b)
                groups=[{s['groups'][n] for n in s[k+'_images']} for k in ['fit','cal','monitor']]
                for i,g in enumerate(groups):
                    for h in groups[i+1:]:self.assertFalse(g&h)
                self.assertEqual(set(o['class_id'] for o in s['fit_objects']),set(range(len(s['classes']))))

    def test_support_mask_singleton_and_variable_class_count(self):
        torch.manual_seed(42);m=ReferenceRelation(10,8,3);r=torch.randn(7,10);y=torch.tensor([0,0,1,1,2,2,3]);q=torch.randn(2,10)
        allowed=torch.ones(2,7,dtype=torch.bool);allowed[0,:2]=False
        out=m(q,r,y,[0,1,2,3],allowed);self.assertEqual(out.shape,(2,5));self.assertLess(float(out[0,0]),-1000)
        self.assertTrue(torch.isfinite(out).all());self.assertEqual(m(q,r,y,[3]).shape,(2,2))
        changed=r.clone();changed[:2]=1e3
        self.assertTrue(torch.equal(out[0],m(q,changed,y,[0,1,2,3],allowed)[0]))

    def test_episode_updates_shared_embedding_without_encoder_gradient(self):
        torch.manual_seed(2);m=ReferenceRelation(10,8,3);opt=torch.optim.AdamW(m.parameters(),lr=.001)
        raw=torch.randn(4,10,requires_grad=True);refs=torch.randn(9,10);labels=torch.tensor([0]*3+[1]*3+[2]*3)
        y=torch.tensor([0,1,2,0]);allowed=torch.ones(4,9,dtype=torch.bool);before=m.embed[1].weight.detach().clone()
        loss,details=episode_loss(m,raw.detach(),y,refs,labels,allowed,np.random.default_rng(1));loss.backward();opt.step()
        self.assertIsNone(raw.grad);self.assertTrue(torch.isfinite(loss));self.assertFalse(torch.equal(before,m.embed[1].weight));self.assertNotIn(details['held_class'],details['support_classes'])

    def test_perturbation_targets_use_all_gt(self):
        gt=torch.tensor([[.1,.1,.5,.5],[.3,.1,.7,.5]])
        boxes,target=perturb(gt);self.assertTrue(torch.allclose(target,overlaps(xyxy(boxes),gt).max(1).values))
        self.assertTrue((xyxy(boxes)>=0).all());self.assertTrue((xyxy(boxes)<=1).all());self.assertTrue(torch.allclose(target[:2],torch.ones(2)))

    def test_real_new_heads_loss_and_stage_backward_on_tiny_injected_models(self):
        c=read(ROOT/'configs/reference_net.json')['model'];c=copy.deepcopy(c);c['gradient_checkpointing']=False
        torch.manual_seed(3);m=ReferenceDetector(ROOT,c,[0,1],pf=TinyPF(),encoder=TinyDino(),dino_dim=8)
        gt=torch.tensor([[.2,.2,.6,.6],[.45,.4,.8,.8]])
        sample=SimpleNamespace(gt_instances=SimpleNamespace(semantic_labels=torch.tensor([0,1])))
        bank=dict(raw=torch.randn(6,40),labels=torch.tensor([0,0,0,1,1,1]),images=['b','c','d','b','c','d'])
        t=dict(boxes=torch.tensor([[.8,.2,.1,.1],[.1,.8,.1,.1]]),p=torch.tensor([.9,.01]),ctr=torch.tensor([.8,.1]),reliable=torch.tensor([True,False]),thresholds=dict(ignore=.5,low=.05))
        for stage in [False,True]:
            m.configure_stage(stage);m.zero_grad(set_to_none=True)
            with patch('refnet.model.pixels',return_value=(torch.zeros(1,3,32,32),torch.zeros(1,3,32,32),{})),patch('refnet.model.make_sample',return_value=(sample,gt)):
                out=m.forward_image(ROOT,'dummy',[],torch.device('cpu'));loss,stats=total_loss(out,t,m,c,bank,'a',np.random.default_rng(4));loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None for p in m.dino.parameters()))
            for prefix in ['relation.','refine.','loc_quality.','bridges.','projection.']:
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(prefix)),prefix)
            self.assertEqual(any(p.grad is not None for n,p in m.named_parameters() if n.startswith('pf.bbox_head.')),stage)

    def test_best_checkpoint_transaction_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'best.pt').write_bytes(b'old');(p/'best.pending.pt').write_bytes(b'uncommitted')
            recover_best(p,sha(p/'best.pt'));self.assertEqual((p/'best.pt').read_bytes(),b'old')
            (p/'best.pending.pt').write_bytes(b'new');committed=sha(p/'best.pending.pt');recover_best(p,committed)
            self.assertEqual((p/'best.pt').read_bytes(),b'new');self.assertFalse((p/'best.pending.pt').exists())
            with self.assertRaises(RuntimeError):recover_best(p,'bad')

    def test_zero_test_gt_is_reported_without_nan_json(self):
        from refnet.evaluation import evaluate
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);d=p/'assets/rwd/Annotations/Surgical';d.mkdir(parents=True)
            (d/'a.xml').write_text('<annotation><object><name>present</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>11</xmax><ymax>11</ymax></bndbox></object></annotation>')
            (p/'vendor').mkdir();(p/'vendor/fomo_open_world_eval.py').write_text('fixture')
            write(p/'pred.json',[dict(image='a.jpg',boxes=[[0,0,10,10]],scores=[.8],labels=[1])])
            def ev(lines,ann,test,c,**kwargs):return np.array([np.nan]) if c=='missing' else np.array([0.]),np.array([0.]),0.,None
            s=dict(task='task2',scene='Surgical',test_images=['a.jpg'],classes=['present','missing'],PK=['present'],CK=['missing'])
            with patch('refnet.evaluation.official_functions',return_value={'voc_eval':ev}):m=evaluate(p,p/'pred.json',s,p/'out')
            self.assertEqual(m['zero_effective_gt_classes'],['missing']);self.assertEqual(read(p/'out/pr_curves.json')['missing']['recall'],[None])


if __name__=='__main__':unittest.main()
