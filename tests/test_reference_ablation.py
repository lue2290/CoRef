import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from refnet.protocol import read,write,sha,fp,lock
from refnet.model import ReferenceDetector
from refnet.relation import ReferenceRelation
from refnet.loss import total_loss as original_loss
from refnet_ablation.model import AblationDetector,RawRelation,common_initialization,planned_identity_loss,total_loss
from refnet_ablation.protocol import training_plan,case_paths,home,summarize,export_results,load_baselines
from refnet_ablation.training import train
from refnet_ablation.localization import coverage,aggregate,diagnose_case,summarize as diag_summary
from refnet_ablation.experiment import validate_config,run_variant,run_job
from test_reference_net import TinyDino,TinyHead
from test_reference_game import TinyParent,tiny_loss,tiny_monitor,fixture

ROOT=Path(__file__).resolve().parents[1]


class TinyPF(nn.Module):
    def __init__(self):
        super().__init__();self.decoder=nn.Linear(256,256);self.query_embedding=nn.Embedding(12,256);self.bbox_head=TinyHead()
    def extract_feat(self,x):return x
    def forward_transformer(self,x,extra,samples):
        owner=self.decoder._joint_owner();q=self.query_embedding.weight[None];boxes=q.new_full((1,12,4),.4)
        h=owner.bridges['3'](q,owner._semantic_map,boxes) if owner.feedback else q
        return dict(hidden_states=h[None],references=None,memory_text=None,text_token_mask=None)


def episode(step):
    return dict(step=step,microbatches=[dict(image='fit.jpg',flip=bool(step%2),episode=dict(held_class=0,support_classes=[1]),query_sampling_seed=step+10)])


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.cfg=read(ROOT/'configs/reference_ablation.json');cls.original=read(ROOT/'configs/reference_net.json')

    def tiny_model(self,variant='full',base=False):
        c=copy.deepcopy(self.original['model']);c['gradient_checkpointing']=False
        torch.manual_seed(9);pf=TinyPF();dino=TinyDino()
        cls=ReferenceDetector if base else AblationDetector
        return cls(ROOT,c,[0,1],pf=pf,encoder=dino,dino_dim=8,**({} if base else dict(variant=variant)))

    def out(self,m):
        gt=torch.tensor([[.2,.2,.6,.6],[.45,.4,.8,.8]])
        sample=SimpleNamespace(gt_instances=SimpleNamespace(semantic_labels=torch.tensor([0,1])))
        with patch('refnet.model.pixels',return_value=(torch.zeros(1,3,32,32),torch.zeros(1,3,32,32),{})),\
             patch('refnet.model.make_sample',return_value=(sample,gt)):
            return m.forward_image(ROOT,'a',[],torch.device('cpu'))

    def loss_inputs(self):
        bank=dict(raw=torch.randn(6,40),labels=torch.tensor([0,0,0,1,1,1]),images=['b','c','d','b','c','d'])
        teacher=dict(boxes=torch.tensor([[.8,.2,.1,.1],[.1,.8,.1,.1]]),p=torch.tensor([.9,.01]),ctr=torch.tensor([.8,.1]),
            reliable=torch.tensor([True,False]),thresholds=dict(ignore=.5,low=.05))
        return bank,teacher

    def small(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(updates=2,save_every=1,validate_every=1,warmup=1,min_free_gib=0,_source_identity='test')
        original=copy.deepcopy(self.original);original['model'].update(accumulate=1,unfreeze_at=1)
        return cfg,original

    def test_config_exact36_and_default_model_matches_original_outputs(self):
        validate_config(self.cfg,self.original)
        self.assertEqual(len(self.cfg['variants'])*len(self.cfg['scenes'])*len(self.cfg['seeds']),36)
        old=self.tiny_model(base=True);old_state=torch.get_rng_state().clone();new=self.tiny_model()
        self.assertTrue(torch.equal(old_state,torch.get_rng_state()))
        for n,p in old.named_parameters():self.assertTrue(torch.equal(p,dict(new.named_parameters())[n]),n)
        a,b=self.out(old),self.out(new)
        for key in ['cls','ctr','refined','raw','quality_logit']:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_common_initialization_and_rng_match_all_variants(self):
        full=self.tiny_model();wanted=common_initialization(full);state=torch.get_rng_state().clone()
        for variant in self.cfg['variants']:
            m=self.tiny_model(variant)
            self.assertEqual(common_initialization(m),wanted)
            self.assertTrue(torch.equal(torch.get_rng_state(),state))
            if variant=='no_preserve':self.assertEqual(m.c['preserve_weight'],0.)
            if variant=='raw_relation':self.assertFalse(hasattr(m.relation,'embed'));self.assertFalse(hasattr(m.relation,'pair'))

    def test_no_feedback_really_cuts_bridge_gradients_but_keeps_reference_loss(self):
        for variant in ['full','no_feedback']:
            m=self.tiny_model(variant);bank,t=self.loss_inputs();out=self.out(m)
            loss,stats=original_loss(out,t,m,m.c,bank,'a',np.random.default_rng(4));loss.backward()
            feedback_grads=[p.grad for n,p in m.named_parameters() if n.startswith(('projection.','bridges.'))]
            self.assertEqual(any(g is not None and bool(g.abs().sum()) for g in feedback_grads),variant=='full')
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith('relation.')))
            self.assertTrue(all(p.grad is None for p in m.dino.parameters()))

    def test_raw_relation_matches_normalized_cosine_without_pair_network(self):
        torch.manual_seed(1);old=ReferenceRelation(10,10,3)
        old.encode=lambda x:F.normalize(x,dim=-1)
        for p in old.pair.parameters():nn.init.zeros_(p)
        nn.init.constant_(old.reject[-1].weight,.05)
        raw=RawRelation(old);q=torch.randn(3,10);r=torch.randn(7,10);y=torch.tensor([0,0,0,1,1,2,2])
        mask=torch.ones(3,7,dtype=torch.bool);mask[0,:3]=False
        self.assertTrue(torch.allclose(raw(q,r,y,[0,1,2],mask),old(q,r,y,[0,1,2],mask),atol=1e-6))
        changed=r.clone();changed[:3]*=100
        self.assertTrue(torch.equal(raw(q,r,y,[0,1,2],mask)[0],raw(q,changed,y,[0,1,2],mask)[0]))

    def test_planned_loss_matches_original_and_teacher_weight_is_only_change(self):
        m=self.tiny_model();bank,t=self.loss_inputs();out=self.out(m)
        old,details=original_loss(out,t,m,m.c,bank,'a',np.random.default_rng(5))
        new,stats=total_loss(out,t,m,m.c,bank,'a',np.random.default_rng(5),details['episode'])
        self.assertTrue(torch.allclose(old,new,atol=1e-6))
        c={**m.c,'preserve_weight':0.}
        without,_=total_loss(out,t,m,c,bank,'a',np.random.default_rng(5),details['episode'])
        self.assertAlmostEqual(float(new-without),m.c['preserve_weight']*details['preserve'],places=5)

    def test_no_episode_is_exact_full_known_ce(self):
        m=ReferenceRelation(10,8,3);q=torch.randn(4,10);r=torch.randn(6,10);labels=torch.tensor([0,0,0,1,1,1]);y=torch.tensor([0,1,0,1])
        allowed=torch.ones(4,6,dtype=torch.bool);plan=dict(held_class=0,support_classes=[1])
        loss,stats=planned_identity_loss(m,q,y,r,labels,allowed,plan,disable=True)
        self.assertTrue(torch.equal(loss,F.cross_entropy(m(q,r,labels,[0,1],allowed),y)))
        self.assertTrue(stats['episode_disabled']);self.assertEqual(stats['episode_queries'],0)

    def test_training_plan_uses_original_images_flips_and_episodes_without_student_warmstart(self):
        from jointdet.protocol import atomic_save
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,original=self.small();write(root/'configs/reference_net.json',original)
            history=[dict(step=i,samples=[dict(image='fit.jpg',flip=bool(i%2))],stats=[dict(episode=dict(held_class=0,support_classes=[1]))]) for i in [1,2]]
            atomic_save(dict(step=2,complete=True,history=history,model={'MUST_NOT_INITIALIZE_STUDENT':torch.ones(1)}),root/'full.pt')
            case=dict(scene='Game',seed=42,spec=dict(fit_images=['fit.jpg'],classes=['a','b']),identity='full',parent=dict(terminal=dict(checkpoint='full.pt')))
            plan=training_plan(root,cfg,case)
            self.assertEqual([r['microbatches'][0]['flip'] for r in plan['steps']],[True,False])
            self.assertEqual([r['microbatches'][0]['episode']['held_class'] for r in plan['steps']],[0,0])
            self.assertEqual(plan,training_plan(root,cfg,case))

    def test_full_control_preflight_requires_corrected_game_and_matching_data(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=copy.deepcopy(self.cfg);dataset=root/'assets/rwd/DATASET_MANIFEST.json';write(dataset,dict(revision='test'))
            (root/'vendor').mkdir();(root/'vendor/fomo_open_world_eval.py').write_text('test')
            checkpoint=root/'full.pt';checkpoint.write_bytes(b'full metadata fixture')
            write(root/'cache/MANIFEST.json',dict(split='split'))
            cache_sha=sha(root/'cache/MANIFEST.json');rows={}
            for scene in cfg['scenes']:
                for seed in cfg['seeds']:
                    report=root/'full_results'/scene/f'seed_{seed}'
                    parent=dict(split_identity='split',terminal=dict(checkpoint='full.pt',sha256=sha(checkpoint),step=500),
                        reference_cache='cache',reference_manifest_sha256=cache_sha,lock=(report/'TRAIN_LOCK.json').relative_to(root).as_posix())
                    binding=dict(annotation=dict(dataset_manifest_sha256=sha(dataset)))
                    write(report/'LOCK.json',binding);write(report/'TRAIN_LOCK.json',binding)
                    write(report/'CORRECTED_PARENT.json',parent)
                    write(report/'RESULT.json',dict(complete=True,scene=scene,task='task1',seed=seed,
                        corrected_label_retraining=scene=='Game',parent_checkpoint=parent['terminal'],artifacts={},
                        calibration=dict(target=.95),metrics=dict(F1=dict(K_AP50=60.,U_AP50=20.,official_evaluator_sha256=sha(root/'vendor/fomo_open_world_eval.py')))))
                    rows[scene,seed]=dict(scene=scene,seed=seed,spec=dict(identity='split'),historical=parent,report=report,
                        annotation=dict(parent_retrained_on_corrections=False))
            with patch('refnet_ablation.protocol.baseline_metadata',side_effect=lambda r,c,s,z:copy.deepcopy(rows[s,z])):
                cases=load_baselines(root,cfg,{}, {},with_plans=False);self.assertEqual(len(cases),9)
                self.assertTrue(all(c['annotation']['parent_retrained_on_corrections'] for c in cases if c['scene']=='Game'))
                path=rows['Game',42]['report']/'RESULT.json';result=read(path);result['corrected_label_retraining']=False;write(path,result)
                with self.assertRaises(RuntimeError):load_baselines(root,cfg,{}, {},with_plans=False)
                result['corrected_label_retraining']=True;write(path,result)
                write(rows['Medical',42]['report']/'LOCK.json',dict(annotation=dict(dataset_manifest_sha256='different_revision')))
                with self.assertRaises(RuntimeError):load_baselines(root,cfg,{}, {},with_plans=False)

    def test_replay_training_is_exactly_resumable(self):
        from cil.common import configure_torch
        from jointdet.protocol import load_torch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,original=self.small();case,_=fixture(root,cfg)
            plan=lock(root/'plan.json',dict(steps=[episode(1),episode(2)]));states=[]
            for kind in ['continuous','resumed']:
                configure_torch(42,2);m=TinyParent(root,original['model'],[0,1]);m.variant='no_episode'
                binding=dict(seed=42,identity='new');run=root/kind;report=root/(kind+'_report')
                kwargs=dict(model=m,view=root,root=root,cfg=cfg,original=original,spec=case['spec'],bank=dict(manifest_sha256='bank'),
                    cache=root,run=run,report=report,binding=binding,plan=plan,device=torch.device('cpu'),
                    loss_fn=lambda *a:tiny_loss(*a[:-1]),teacher_fn=lambda *a:{},monitor_fn=tiny_monitor)
                if kind=='resumed':
                    train(**kwargs,stop_after=1);m=TinyParent(root,original['model'],[0,1]);m.variant='no_episode';kwargs['model']=m
                train(**kwargs);states.append(load_torch(run/'last.pt'))
            self.assertEqual(states[0]['history'],states[1]['history'])
            self.assertEqual(states[0]['history'][0]['samples'],[dict(image='fit.jpg',flip=True)])
            for name in states[0]['model']:self.assertTrue(torch.equal(states[0]['model'][name],states[1]['model'][name]))
            self.assertIsNone(states[1]['optimizer'])

    def test_public_variant_training_calibration_and_official_eval(self):
        import shutil
        from refnet_game.protocol import prepare_cache
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,original=self.small();case,dataset=fixture(root,cfg)
            cache=prepare_cache(root,cfg,case['spec'],case['historical'],{})
            case.update(scene='Game',seed=42,cache=cache,identity='control',plan=lock(root/'plan.json',dict(steps=[episode(1),episode(2)])))
            vendor=root/'vendor';vendor.mkdir();shutil.copy2(ROOT/'vendor/fomo_open_world_eval.py',vendor/'fomo_open_world_eval.py')
            xml=root/'assets/rwd/Annotations/Game/test.xml';xml.parent.mkdir(parents=True)
            xml.write_text('<annotation><object><name>novel</name><bndbox><xmin>11</xmin><ymin>11</ymin><xmax>41</xmax><ymax>41</ymax></bndbox></object><object><name>a</name><bndbox><xmin>61</xmin><ymin>61</ymin><xmax>91</xmax><ymax>91</ymax></bndbox></object></annotation>')
            dataset['files']['Annotations/Game/test.xml']=dict(sha256=sha(xml))
            def factory(root,c,active,variant):
                m=TinyParent(root,c,active);m.variant=variant;m.feedback=True;return m
            with patch('refnet_ablation.model.AblationDetector',factory),patch('scenepilot.protocol.view',return_value=root),\
                 patch('refnet_ablation.training.total_loss',side_effect=lambda *a:tiny_loss(*a[:-1])),patch('refnet_ablation.training.monitor',tiny_monitor):
                result=run_variant(root,cfg,original,{},case,'no_episode',torch.device('cpu'),dataset,{})
                self.assertAlmostEqual(result['metrics']['F1']['U_AP50'],100.)
                self.assertFalse(result['old_adapted_checkpoint_loaded'])
            receipt=export_results(root,cfg);self.assertTrue(receipt['crc_verified'])

    def test_localization_budget_gate_difficult_and_B0_not_repeated(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,original=self.small();case,dataset=fixture(root,cfg);cfg['scenes']=['Game']
            write(root/'MAINLINE_PARENT_INDEX.json',dict(base_weights={original['model']['pf_weights']:'frozen_pf'}))
            xml=root/'assets/rwd/Annotations/Game/test.xml';xml.parent.mkdir(parents=True)
            xml.write_text('<annotation><object><name>novel</name><bndbox><xmin>11</xmin><ymin>11</ymin><xmax>41</xmax><ymax>41</ymax></bndbox></object><object><name>a</name><bndbox><xmin>61</xmin><ymin>61</ymin><xmax>91</xmax><ymax>91</ymax></bndbox></object><object><name>a</name><difficult>1</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>3</xmax><ymax>3</ymax></bndbox></object></annotation>')
            dataset['files']['Annotations/Game/test.xml']=dict(sha256=sha(xml));write(root/'assets/rwd/DATASET_MANIFEST.json',dataset)
            case.update(scene='Game',seed=42,parent=dict(terminal=dict(sha256='full')),result=dict(calibration=dict(object_threshold=.5)))
            boxes=np.tile(np.array([0,0,1,1],np.float32),(600,1));boxes[0]=[10,10,40,40];boxes[1]=[60,60,90,90]
            values=dict(boxes=boxes,joint=None,proposal_ids=list(range(600)),pf_scores=list(range(600,0,-1)))
            with patch('refnet_ablation.localization.b0_values',return_value=values) as fn:
                b0=diagnose_case(root,cfg,case,'B0',None,root,original['model'],torch.device('cpu'),dataset,{})
                diagnose_case(root,cfg,{**case,'seed':43},'B0',None,root,original['model'],torch.device('cpu'),dataset,{})
                self.assertEqual(fn.call_count,1)
            joint=np.ones(600,np.float32)*.9;joint[0]=.2
            with patch('refnet_ablation.localization.full_values',return_value={**values,'joint':joint}):
                for seed in [42,43,44]:full=diagnose_case(root,cfg,{**case,'seed':seed},'full',None,root,original['model'],torch.device('cpu'),dataset,{})
            self.assertEqual(full['gt_counts'],dict(known=1,unknown=1));self.assertEqual(full['ignored_difficult'],1)
            self.assertEqual(full['metrics']['before']['unknown/R50@100']['recall'],100.)
            self.assertEqual(full['metrics']['after']['unknown/R50@100']['recall'],0.)
            self.assertEqual(full['metrics']['after']['known/R75@600']['recall'],100.)
            summary=diag_summary(root,cfg);self.assertTrue(summary['complete'])
            self.assertEqual(summary['table']['B0/Game/before/known/R50@100']['n'],1)
            self.assertIsNone(summary['table']['B0/Game/before/known/R50@100']['sample_std'])
            with self.assertRaises(RuntimeError):coverage(boxes[:300],[],budgets=(100,300,600))

    def test_core_macro_aggregates_seed_means_and_never_completes_partial_variant(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=self.cfg
            controls=[dict(scene=s,seed=seed,metrics=dict(K_AP50=60.,U_AP50=20.)) for s in cfg['scenes'] for seed in cfg['seeds']]
            write(home(root,cfg)/'FULL_BASELINES.json',dict(cases=controls))
            for variant in cfg['variants']:
                for scene in cfg['scenes']:
                    for j,seed in enumerate(cfg['seeds']):
                        _,p=case_paths(root,cfg,variant,scene,seed)
                        write(p/'RESULT.json',dict(variant=variant,scene=scene,seed=seed,metrics=dict(F1=dict(K_AP50=59.+j,U_AP50=18.+j))))
            r=summarize(root,cfg);self.assertTrue(r['complete']);self.assertEqual(r['new_training_cases'],36)
            self.assertEqual(r['macro']['no_episode/U_AP50']['sample_std'],1.)
            self.assertEqual(r['delta_from_full']['no_episode/U_AP50']['mean'],-1.)
            case_paths(root,cfg,'no_episode','Game',44)[1].joinpath('RESULT.json').unlink()
            self.assertFalse(summarize(root,cfg)['completion']['no_episode']['complete'])

    def test_batch_records_uncalibratable_case_and_fails_fast_for_code_errors(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);case=dict(scene='Medical',seed=42)
            with patch('refnet_ablation.experiment.run_variant',side_effect=[RuntimeError('Empty/nonfinite known calibration; no test fallback'),dict(complete=True)]):
                self.assertIsNone(run_job(root,self.cfg,self.original,{},case,'no_feedback',None,{},{}))
                self.assertEqual(run_job(root,self.cfg,self.original,{},case,'raw_relation',None,{},{}),dict(complete=True))
            row=read(case_paths(root,self.cfg,'no_feedback','Medical',42)[1]/'FAILURE.json')
            self.assertFalse(row['ap_defined']);self.assertFalse(row['test_fallback_used'])
            with patch('refnet_ablation.experiment.run_variant',side_effect=NameError('code issue')):
                with self.assertRaises(NameError):run_job(root,self.cfg,self.original,{},case,'no_feedback',None,{},{})


if __name__=='__main__':unittest.main()
