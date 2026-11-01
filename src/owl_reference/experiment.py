import argparse
import gc
import json
from pathlib import Path
import torch
from current.progress import stage,track
from cil.common import configure_torch
from .protocol import read,write,sha,fp,lock,locations,sources,verify_assets,make_cases,summary,dino_comparison,stage_upload,guard
from .cache import candidate_cache,build_plans,prepare_features,Data,native_candidates
from .model import Head,load_visual
from .training import train_head


def validate(cfg):
    fixed=dict(scenes=['Medical','Aquatic','Game'],task='task1',arms=['O0','O1'],seeds=[42,43,44],updates=500,
        accumulate=2,warmup=25,learning_rate=1e-4,weight_decay=.01,grad_clip=1.,identity_dim=128,
        neighbors=3,proposal_limit=600,detection_limit=100,nms_iou=.5,context_scale=1.25,
        known_retention_target=.95,calibration_min_iou=.5,pictures_per_case=10)
    for key,value in fixed.items():
        if cfg.get(key)!=value:raise ValueError('160 fixed protocol mismatch: '+key)


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


class LazyVisual:
    channels=768
    def __init__(self,root,cfg,device):self.root=root;self.cfg=cfg;self.device=device;self.visual=None;self.processor=None
    def load(self):
        if self.visual is None:
            with stage('160 load official frozen OWL detection visual encoder'):
                self.visual,self.processor=load_visual(self.root,self.cfg,self.device)
        return self.visual,self.processor
    def __call__(self,pixels):return self.load()[0](pixels)
    def process(self,**kwargs):return self.load()[1](**kwargs)
    def close(self):self.visual=None;self.processor=None;release()


def prepare(root,cfg,cases,device,dataset,verified):
    from jointdet.model import build_pf
    from scenepilot.protocol import view
    c=read(Path(root)/'configs/reference_net.json')['model'];pf=None;prepared=[]
    def extract(_,v,name,config,dev):
        nonlocal pf
        if pf is None:
            with stage('160 load original frozen PF-RPN (no DINO)'):
                pf=build_pf(root,config).to(dev).eval().requires_grad_(False)
        return native_candidates(pf,v,name,config,dev)
    try:
        for case in cases:
            folder=candidate_cache(root,cfg,case,None,view(root,case['scene']),c,device,dataset,verified,extract)
            plans=build_plans(root,cfg,case,folder);prepared.append((case,folder,plans))
    finally:pf=None;release()
    visual=LazyVisual(root,cfg,device)
    try:
        for case,folder,plans in prepared:
            prepare_features(root,cfg,case,folder,plans,visual,visual.process,device,dataset,verified)
    finally:visual.close()
    return prepared


def run_scene(root,cfg,case,folder,plans,device,dataset,verified,do_train=True,do_evaluate=True):
    from .evaluation import calibrate,predict,evaluate_all
    from jointdet.protocol import load_torch
    feature_folder=locations(root,cfg,case['scene'])[0]/'features';data=Data(case,cfg,folder,feature_folder,device)
    heads={};bindings={};initial={}
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            configure_torch(seed,2);head=Head(arm,data.raw_dim,cfg).to(device);key=arm,seed
            init=fp({k:v.detach().cpu().tolist() for k,v in head.relation.reject.state_dict().items()})
            if seed in initial and initial[seed]!=init:raise RuntimeError('O0/O1 rejection initializations differ')
            initial[seed]=init;_,run,report=locations(root,cfg,case['scene'],arm,seed)
            binding=lock(report/'LOCK.json',dict(case=case['identity'],source=case['source_identity'],arm=arm,seed=seed,
                candidate_manifest_sha256=sha(folder/'MANIFEST.json'),features_sha256=sha(feature_folder/'MANIFEST.json'),
                plan=plans[seed]['identity'],initial_rejection_sha256=init,fresh_initialization=True,frozen_PF=True,frozen_OWL=True))
            bindings[key]=binding
            if do_train:
                info=train_head(head,data,cfg,plans[seed],root,arm,seed,binding);write(report/'TRAINING.json',info)
            else:
                state=load_torch(run/'last.pt')
                if not state['complete'] or state['step']!=500 or state['lock']!=binding['identity']:raise RuntimeError('Need matching final500 OWL head')
                head.load_state_dict(state['head'],strict=True);del state
            heads[key]=head.eval()
    if do_evaluate:
        cal=calibrate(root,cfg,data,heads,bindings)
        records,seals=predict(root,cfg,data,heads,bindings,cal,lambda:load_visual(root,cfg,device),dataset,verified)
        evaluate_all(root,cfg,data,heads,bindings,cal,records,seals,dataset,verified)
    del heads,data;release()


def main(root,argv=None):
    parser=argparse.ArgumentParser(description='160: independent frozen OWL reference-verification O0/O1;18 lightweight fits')
    parser.add_argument('command',choices=['run','prepare','train','evaluate','plan','check','summary','export','stage-upload'])
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--scenes',nargs='+',choices=['Medical','Aquatic','Game'])
    args=parser.parse_args(argv);root=Path(root);cfg=read(root/'configs/owl_reference.json');validate(cfg)
    if args.command=='stage-upload':result=stage_upload(root,cfg)
    elif args.command in ['summary','export']:
        result=summary(root,cfg);dino_comparison(root,cfg)
        if args.command=='export':
            from .export import export_results
            result=export_results(root,cfg)
        else:result=dict(complete=result['complete'],missing_cases=len(result['missing_cases']),summary=str(locations(root,cfg)[2]/'SUMMARY.md'))
    else:
        source=sources(root);cases=make_cases(root,cfg,source)
        if args.scenes:
            if len(set(args.scenes))!=len(args.scenes):parser.error('Duplicate scenes')
            cases=[c for c in cases if c['scene'] in args.scenes]
        result=dict(scenes=[c['scene'] for c in cases],lightweight_fits=len(cases)*6,updates_per_fit=500,
            report_directory=str(locations(root,cfg)[2]),required_weight=str(root/cfg['owl_folder']/'pytorch_model.bin'))
        write(locations(root,cfg)[2]/'PLAN.json',result)
        if args.command!='plan':
            verified={};assets=verify_assets(root,cfg,verified)
            if args.command=='check':result.update(weights_verified=True,assets=assets)
            else:
                device=torch.device(args.device)
                if device.type=='cuda':
                    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run on the training instance')
                    torch.cuda.set_device(device)
                    torch.cuda.set_per_process_memory_fraction(min(1.,cfg['max_gpu_gib']*2**30/torch.cuda.get_device_properties(device).total_memory),device)
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                configure_torch(42,2);guard(root,cfg);dataset=read(root/'assets/rwd/DATASET_MANIFEST.json')
                prepared=prepare(root,cfg,cases,device,dataset,verified)
                if args.command!='prepare':
                    for case,folder,plans in track(prepared,'160 scene-specific O0/O1 reference transfer',unit='scenes'):
                        failure=locations(root,cfg,case['scene'])[2]/'FAILURE.json'
                        try:run_scene(root,cfg,case,folder,plans,device,dataset,verified,args.command!='evaluate',args.command!='train')
                        except Exception as exc:
                            write(failure,dict(scene=case['scene'],resolved=False,type=type(exc).__name__,message=str(exc),source=case['source_identity']))
                            raise
                        if failure.exists():write(failure,dict(**{k:v for k,v in read(failure).items() if k!='resolved'},resolved=True))
                        summary(root,cfg)
                    dino_comparison(root,cfg)
                    if args.command in ['run','evaluate']:
                        from .export import export_results
                        result=export_results(root,cfg)
    print(json.dumps(result,ensure_ascii=False,indent=2))
