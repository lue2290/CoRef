from pathlib import Path
import copy,time,json,math
import numpy as np
import torch
from .protocol import read,write,fp,sha,lock,paths,space,progress,split_spec,recover_best
from .model import ReferenceDetector
from .cache import prepare,load_bank,teacher
from .loss import total_loss


def save(root,cfg,path,value):
    from jointdet.protocol import atomic_save
    size=sum(x.numel()*x.element_size() for x in value['model'].values())
    space(root,cfg,size*(4 if value.get('optimizer') else 1))
    atomic_save(value,path)


def small_spec(s):
    s=copy.deepcopy(s);remaining=set(range(len(s['classes'])));chosen=[]
    while remaining:
        n=max(s['fit_images'],key=lambda image:len(remaining & {o['class_id'] for o in s['fit_objects'] if o['image']==image}))
        covered={o['class_id'] for o in s['fit_objects'] if o['image']==n}
        if not (covered&remaining):raise RuntimeError('Smoke cannot cover legal classes')
        chosen.append(n);remaining-=covered
    s['fit_images']=chosen
    for role in ['cal','monitor']:s[role+'_images']=s[role+'_images'][:2]
    for role in ['fit','cal','monitor']:s[role+'_objects']=[o for o in s[role+'_objects'] if o['image'] in s[role+'_images']]
    s['identity']=fp({k:v for k,v in s.items() if k!='identity'});return s


def run(root,cfg,plan,scene,task,seed,device,smoke=False):
    from scenepilot.protocol import gpu,view
    from cil.common import configure_torch,rng_state,restore_rng
    from jointdet.protocol import load_torch,environment
    from .inference import monitor,calibrate,predict
    root=Path(root);s=split_spec(root,scene,task);runpath,report=paths(root,cfg,scene,task,seed,smoke)
    case=next(x for x in plan['cases'] if x['scene']==scene and x['task']==task)
    if s['identity']!=case['identity']:raise RuntimeError('Metadata changed since audit')
    if smoke:s=small_spec(s)
    runpath.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    record=lock(runpath/'LOCK.json',dict(plan=plan['identity'],source=plan['source'],split=s['identity'],seed=seed,smoke=smoke))
    resultpath=report/'RESULT.json'
    if resultpath.exists():
        result=read(resultpath)
        if result['lock']!=record['identity']:raise RuntimeError('Completed experiment changed')
        for n,h in result.get('weights',{}).items():
            if sha(runpath/n)!=h:raise RuntimeError('Completed checkpoint changed')
        return result
    
    for name,want in plan['base_weights'].items():
        if sha(root/name)!=want:raise RuntimeError('Base weight changed: '+name)
    device=gpu(device,cfg);configure_torch(seed,2);env=environment();write(report/'ENVIRONMENT.json',env)
    c=copy.deepcopy(cfg['model'])
    for k in ['pf_config','pf_weights','dino_folder']:c[k]=str(root/c[k])
    model=ReferenceDetector(root,c,list(range(len(s['classes'])))).to(device)
    parameter_bytes=sum(p.numel()*p.element_size() for n,p in model.named_parameters() if model.group_for(n) is not None)
    write(report/'PARAMETERS.json',dict(groups=model.parameter_report(),checkpoint_parameters_gib=parameter_bytes/2**30,all_30_two_weight_files_gib=60*parameter_bytes/2**30))
    viewroot=view(root,scene)
    cachecfg=copy.deepcopy(cfg);cachecfg['provenance']=dict(source=plan['source'],base=plan['base_weights'])
    if smoke:cachecfg['tag']=cfg['tag']+'_smoke'
    folder=prepare(root,viewroot,model,s,cachecfg,report);bank=load_bank(folder,device)
    if set(bank['labels'].tolist())!=set(range(len(s['classes']))):raise RuntimeError('GT reference class missing')
    groups=[]
    for g,lr in [('new',c['new_lr']),('pf',c['pf_lr'])]:
        params=[p for n,p in model.named_parameters() if model.group_for(n)==g]
        groups.append(dict(params=params,lr=lr,base_lr=lr,name=g))
    optimizer=torch.optim.AdamW(groups,weight_decay=c['weight_decay'])
    rng=np.random.default_rng(seed);target=2 if smoke else cfg['updates'];step=0;curve=[];history=[];best_value=None;best_sha=None
    last=runpath/'last.pt';best=runpath/'best.pt'
    if last.exists():
        old=load_torch(last)
        if old['lock']!=record['identity'] or old['bank_manifest']!=bank['manifest_sha256']:raise RuntimeError('Resume provenance mismatch')
        model.load_adapted(old['model']);step=old['step'];curve=old['curve'];history=old['history'];best_value=old['best_value'];best_sha=old['best_sha'];recover_best(runpath,best_sha)
        if not old['complete']:
            optimizer.load_state_dict(old['optimizer']);restore_rng(old['rng']);rng.bit_generator.state=old['sampler']
    if not last.exists():recover_best(runpath,None)
    class_images={cid:sorted({o['image'] for o in s['fit_objects'] if o['class_id']==cid}) for cid in range(len(s['classes']))}
    by={n:[o for o in s['fit_objects'] if o['image']==n] for n in s['fit_images']}
    start=time.monotonic();log=runpath/('training_'+str(time.time_ns())+'.jsonl')
    gradient_reports=[]
    for current in range(step+1,target+1):
        model.configure_stage(current>(1 if smoke else c['unfreeze_at']));model.train();optimizer.zero_grad(set_to_none=True)
        factor=min(1.,current/cfg['warmup'])*(.05+.95*(1+math.cos(math.pi*(current-1)/max(target-1,1)))/2)
        for g in optimizer.param_groups:g['lr']=g['base_lr']*factor
        stats=[];samples=[];losses=[]
        for _ in range(1 if smoke else c['accumulate']):
            cid=int(rng.integers(len(s['classes'])));name=str(rng.choice(class_images[cid]));flip=bool(rng.integers(2))
            out=model.forward_image(viewroot,name,by[name],device,flip)
            t=teacher(folder,name,device,flip);loss,st=total_loss(out,t,model,c,bank,name,rng)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            (loss/(1 if smoke else c['accumulate'])).backward();stats.append(st);losses.append(float(loss.detach()));samples.append(dict(image=name,flip=flip))
            del out,t,loss
        grads={g:sum(p.grad is not None and bool(p.grad.abs().sum()) for n,p in model.named_parameters() if model.group_for(n)==g) for g in ['new','pf']}
        if not grads['new'] or (model._stage and not grads['pf']):raise RuntimeError('Missing gradient group '+str(grads))
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
        if any(p.grad is not None for p in model.dino.parameters()):raise RuntimeError('Frozen reference encoder received gradients')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['grad_clip']);optimizer.step()
        if smoke:gradient_reports.append(dict(step=current,groups=grads))
        row=dict(step=current,loss=float(np.mean(losses)),stats=stats,samples=samples,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups});history.append(row)
        with log.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        if current%(1 if smoke else cfg['validate_every'])==0 or current==target:
            state=rng_state();metrics=monitor(root,viewroot,model,s,cfg,folder,bank,report);restore_rng(state)
            curve.append(dict(step=current,metrics=metrics))
            if (scene=='Medical' and current==target) or (scene!='Medical' and (best_value is None or metrics['objective']<best_value)):
                best_value=metrics['objective'];save(root,cfg,runpath/'best.pending.pt',dict(model=model.adapted_state(),step=current,lock=record['identity'],bank_manifest=bank['manifest_sha256'],monitor=metrics));best_sha=sha(runpath/'best.pending.pt')
            write(report/'CURVE.json',dict(training=history,monitor=curve,selection=cfg['selection']))
        if current%(1 if smoke else cfg['save_every'])==0 or current==target:
            save(root,cfg,last,dict(model=model.adapted_state(),step=current,complete=current==target,lock=record['identity'],bank_manifest=bank['manifest_sha256'],
                  optimizer=None if current==target else optimizer.state_dict(),rng=rng_state(),sampler=rng.bit_generator.state,curve=curve,history=history,best_value=best_value,best_sha=best_sha))
            recover_best(runpath,best_sha)
        peak=torch.cuda.max_memory_allocated(device)/2**30
        if peak>cfg['max_gpu_gib']:raise RuntimeError('GPU allocation exceeds configured budget')
        space(root,cfg);progress(report,current,target,'network training',start)
    del optimizer
    # Endpoint and monitor-selected checkpoint each get their own matching calibration.
    evaluations={};steps={};weights={}
    for label,path in [('terminal',last),('selected',best)]:
        saved=load_torch(path);model.load_adapted(saved['model']);model.eval();steps[label]=saved['step'];weights[path.name]=sha(path)
        dest=report/label;dest.mkdir(parents=True,exist_ok=True)
        if label=='selected' and steps['terminal']==saved['step']:
            evaluations[label]=dict(evaluations['terminal']);write(dest/'REUSES_TERMINAL.json',dict(step=saved['step'],reason='identical checkpoint step'));continue
        cal=calibrate(root,viewroot,model,s,cfg,folder,bank,report)
        write(dest/'CALIBRATION.json',dict(**cal,checkpoint_sha256=weights[path.name],bank_manifest=bank['manifest_sha256']))
        if smoke:
            # No official-test access in smoke; check finite inference on a cal image.
            from .inference import values
            with torch.inference_mode():
                out=model.forward_image(viewroot,s['cal_images'][0],[],device);_,p,o=values(model,out,bank,cfg,cal['reject_shift'])
            if not torch.isfinite(p).all() or not torch.isfinite(o).all():raise RuntimeError('Nonfinite smoke inference')
            evaluations[label]=dict(smoke_inference_passed=True)
        else:evaluations[label]=predict(root,viewroot,model,s,cfg,bank,cal,dest,weights[path.name])
    for name,want in weights.items():
        if sha(runpath/name)!=want:raise RuntimeError('Checkpoint changed during evaluation')
    result=dict(complete=True,lock=record['identity'],scene=scene,task=task,seed=seed,steps=steps,weights=weights,
                evaluations=evaluations,source=plan['source'],gradient_reports=gradient_reports,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,elapsed_seconds=time.monotonic()-start,
                official_test_access=not smoke,training_updates=target,terminal_optimizer_retained=False,
                reference_manifest=bank['manifest_sha256'],rare_classes=s['rare_classes'])
    write(resultpath,result);return result
