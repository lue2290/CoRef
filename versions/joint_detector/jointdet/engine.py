from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import math
import time
import torch
import numpy as np
from cil.common import configure_torch,install_label_guard,rng_state,restore_rng
from .protocol import (digest,fingerprint,read_json,write_json,atomic_save,load_torch,
                       split,source_hashes,gpu,memory_check,check_disk,Progress,stable_seed,environment)
from .model import JointDetector
from .teacher import prepare_teacher,teacher_for
from .losses import joint_loss


def run_path(root,stage,tag,task,held,method,seed):
    if not tag or any(x in tag for x in ['/','\\','..']):raise ValueError('Tag must be a simple directory name')
    return Path(root)/'runs/joint_detector'/stage/tag/task/(f'held_{held}' if held else 'formal')/method/f'seed_{seed}'


def validate_lock_identity(lock):
    payload={k:v for k,v in lock.items() if k!='identity'}
    if fingerprint(payload)!=lock.get('identity'):
        raise RuntimeError('Saved run lock identity is invalid; preserve the run for inspection')


def compatible_sources(root,lock):
    """Admit only the exact audited bookkeeping patch, never arbitrary code edits."""
    validate_lock_identity(lock)
    current=source_hashes(root);original=lock['source']
    if current==original:return None
    changed={n for n in set(current)|set(original) if current.get(n)!=original.get(n)}
    allowed={'versions/joint_detector/jointdet/engine.py','versions/joint_detector/jointdet/orchestrate.py'}
    registry=Path(root)/'tools/joint_resume_compatibility.json'
    if not changed<=allowed or not registry.is_file():
        raise RuntimeError('Training implementation changed. Preserve this run; use a new tag/version. Differences: '+repr(sorted(changed)))
    spec=read_json(registry)
    if spec.get('patch')!='json_lock_roundtrip_v1' or set(spec.get('files',{}))!=allowed:
        raise RuntimeError('Invalid resume compatibility registry')
    for n in changed:
        pair=spec['files'][n]
        if original.get(n)!=pair['before_sha256'] or current.get(n)!=pair['after_sha256']:
            raise RuntimeError('Source edit is not the exact approved JSON-lock patch: '+n)
    return dict(patch=spec['patch'],original_lock_identity=lock['identity'],
                registry_sha256=digest(registry),changed_files=sorted(changed),
                original_source=original,executed_source=current,
                unchanged='model, losses, data, optimizer, schedule, weights and saved checkpoint identity')


def assert_run_sources(root,run,lock):
    audit=compatible_sources(root,lock)
    if audit is not None:write_json(Path(run)/'SOURCE_COMPATIBILITY.json',audit)


def prepare_run_lock(root,run,candidate):
    # JSON object keys are strings on disk (e.g. class-count keys 1 -> "1").
    # Compare the persisted representation without weakening any value checks.
    candidate=json.loads(json.dumps(candidate,allow_nan=False))
    validate_lock_identity(candidate)
    path=Path(run)/'RUN_LOCK.json'
    if not path.exists():return candidate
    original=read_json(path)
    audit=compatible_sources(root,original)
    comparison={**candidate,'source':original['source'],'identity':original['identity']}
    if comparison!=original:
        fields=[k for k in set(comparison)|set(original) if comparison.get(k)!=original.get(k)]
        raise RuntimeError('Existing run has different inputs/config/environment. Preserve it; use a new tag. Fields: '+repr(sorted(fields)))
    # Keep the original lock/checkpoint ID intact; record actual patch provenance
    # separately. No model, optimizer or RNG object is modified here.
    if audit is not None:write_json(Path(run)/'SOURCE_COMPATIBILITY.json',audit)
    return original


def optimizer_for(model,c):
    groups=[]
    for group,lr in [('new',c['new_lr']),('pf',c['pf_lr']),('backbone',c['backbone_lr'])]:
        for decay in [True,False]:
            params=[p for n,p in model.named_parameters() if model.group_for(n)==group and
                    ((p.ndim>1 and not n.endswith('bias'))==decay)]
            if params:groups.append(dict(params=params,lr=lr,base_lr=lr,group_name=group,
                                         weight_decay=c['weight_decay'] if decay else 0))
    return torch.optim.AdamW(groups)


def learning_rate(opt,c,step):
    warm=max(1,min(50,c['updates']//10))
    factor=min(1,(step+1)/warm)*(.5+.5*math.cos(math.pi*min(step,c['updates'])/max(1,c['updates'])))
    for g in opt.param_groups:
        f=factor
        if g['group_name']=='backbone':
            f*=min(1,max(0,step-c['unfreeze_at']+1)/10)
        g['lr']=g['base_lr']*f


def load_run(root,run,device,verify_source=True):
    run=Path(run);lock=read_json(run/'RUN_LOCK.json')
    validate_lock_identity(lock)
    if environment()!=lock['environment']:raise RuntimeError('Checkpoint runtime environment changed')
    if verify_source:assert_run_sources(root,run,lock)
    model=JointDetector(root,lock['config'],lock['split']['active'],lock['method']).to(device)
    saved=load_torch(run/'last.pt')
    if saved['lock_identity']!=lock['identity']:raise RuntimeError('Checkpoint/run lock mismatch')
    model.load_adapted(saved['model']);model.configure_stage(saved['step']>=lock['config']['unfreeze_at'])
    return model,saved,lock


def train(root,c,task='task1',held='fish',method='J3',seed=42,stage='pilot',tag='v1',
          device_spec='cuda:0',hours=None):
    runtime=environment()
    root=Path(root);d=gpu(device_spec,c);check_disk(root,c)
    if stage=='formal':
        selection=read_json(root/'reports/joint_detector'/tag/'FINAL_SELECTION.json')
        if selection['method']!=method or selection['config']!=c or selection['source']!=source_hashes(root):
            raise RuntimeError('Formal run differs from the frozen method/config/source')
    install_label_guard(root,task)
    configure_torch(seed,c['threads'])
    s=split(root,task,held,stage=='formal',c)
    if stage in ['smoke','overfit']:
        s={**s,'purpose':stage}
        s['fit_images']=s['fit_images'][:2 if stage=='smoke' else 20]
        s['fit_objects']=[o for o in s['fit_objects'] if o['image'] in s['fit_images']]
        s['identity']=fingerprint({k:v for k,v in s.items() if k!='identity'})
    run=run_path(root,stage,tag,task,held,method,seed);run.mkdir(parents=True,exist_ok=True)
    inputs=read_json(root/'configs/joint_assets.json')
    for rel,want in [(inputs['pf_weight']['path'],inputs['pf_weight']['sha256']),
                     ('assets/models/dinov3_large/model.safetensors',next(v['sha256'] for v in read_json(root/'assets/manifest.json')['files'] if v['path']=='assets/models/dinov3_large/model.safetensors'))]:
        if digest(root/rel)!=want:raise RuntimeError('Base model file changed: '+rel)
    lock=dict(config=c,split=s,method=method,seed=seed,stage=stage,tag=tag,source=source_hashes(root),environment=runtime,
              base_weights=dict(pf=inputs['pf_weight']['sha256'],dino=digest(root/'assets/models/dinov3_large/model.safetensors')))
    lock['identity']=fingerprint(lock)
    lp=run/'RUN_LOCK.json'
    lock=prepare_run_lock(root,run,lock)
    if not lp.exists():write_json(lp,lock)
    write_json(run/'status.json',dict(phase='teacher',step=0,target=c['updates'],complete=False))
    teacher=prepare_teacher(root,c,s,d,lambda n,total:write_json(run/'status.json',
        dict(phase='teacher',teacher_images=n,teacher_total=total,step=0,target=c['updates'],complete=False)))
    # Teacher preprocessing must not perturb student initialization RNG.
    configure_torch(seed,c['threads'])
    model=JointDetector(root,c,s['active'],method).to(d)
    opt=optimizer_for(model,c);step=0
    if (run/'last.pt').exists():
        saved=load_torch(run/'last.pt')
        if saved['lock_identity']!=lock['identity']:raise RuntimeError('Checkpoint lock changed')
        model.load_adapted(saved['model']);step=saved['step']
        if step<c['updates']:
            if saved['optimizer'] is None:raise RuntimeError('Incomplete checkpoint has no optimizer state')
            opt.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
    write_json(run/'parameters.json',dict(counts=model.parameter_report(),
        parameters=[dict(name=n,shape=list(p.shape),group=model.group_for(n)) for n,p in model.named_parameters() if model.group_for(n)]))
    original_step=step;start=time.monotonic();bar=Progress(c['updates']-step,'train')
    by=defaultdict(list)
    for o in s['fit_objects']:by[o['image']].append(o)
    class_images={j:sorted({o['image'] for o in s['fit_objects'] if o['class_id']==j}) for j in s['active']}
    available=[j for j,v in class_images.items() if v]
    representatives={}
    for n,p in model.named_parameters():
        group=model.group_for(n)
        if group is not None and group not in representatives:representatives[group]=(n,p.detach().cpu().clone())
    updates_report=[]
    if stage=='overfit' and step==0:
        from .visuals import overfit_images
        overfit_images(root,run,model,s,c,'before')
    def save():
        check_disk(root,c)
        atomic_save(dict(step=step,model=model.adapted_state(),optimizer=opt.state_dict() if step<c['updates'] else None,
                         training_complete=step==c['updates'],rng=rng_state(),
                         lock_identity=lock['identity']),run/'last.pt')
    with (run/'training.jsonl').open('a',encoding='utf8') as log:
        while step<c['updates']:
            model.configure_stage(step>=c['unfreeze_at']);model.train()
            learning_rate(opt,c,step);opt.zero_grad(set_to_none=True)
            losses=[];stat=None;tick=time.monotonic()
            for micro in range(c['accumulate']):
                rng=np.random.default_rng(stable_seed(seed,step,micro,'joint_sample'))
                cls=int(rng.choice(available));name=str(rng.choice(class_images[cls]));flip=bool(rng.integers(2))
                out=model.forward_image(root,name,by[name],d,flip)
                target=teacher_for(teacher,name,d,flip)
                loss,stat=joint_loss(out,target,model,c)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at update {step}, image {name}')
                (loss/c['accumulate']).backward();losses.append(float(loss.detach()))
                del out,loss,target
            grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['grad_clip'])
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient; optimizer step refused')
            gcounts={}
            for g in ['new','pf','backbone']:
                gradients=[p.grad for n,p in model.named_parameters() if model.group_for(n)==g and p.grad is not None]
                gcounts[g]=int(torch.stack([v.ne(0) for v in torch._foreach_norm(gradients)]).sum().item()) if gradients else 0
            if not gcounts['new'] or not gcounts['pf']:
                raise RuntimeError('Expected new/PF trainable paths have no nonzero gradient')
            if model._stage and c['unfreeze_backbones'] and not gcounts['backbone']:
                raise RuntimeError('Unfrozen backbone has no nonzero gradient')
            opt.step();step+=1
            mem=memory_check(d,c)
            row=dict(step=step,loss=float(np.mean(losses)),grad_before_clip=float(grad),
                     clipped=bool(grad>c['grad_clip']),seconds=time.monotonic()-tick,
                     stage='backbone' if model._stage else 'decoder',gradient_tensors=gcounts,**mem,details=stat)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            mean_seconds=(time.monotonic()-start)/max(1,step-original_step)
            write_json(run/'status.json',dict(phase='training',step=step,target=c['updates'],complete=False,
                elapsed_seconds=time.monotonic()-start,eta_train_seconds=mean_seconds*(c['updates']-step),**mem))
            bar.show(step-original_step,f'loss {row["loss"]:.3f} peak {mem["peak_allocated_bytes"]/1024**3:.2f}GiB')
            if step%c['save_every']==0 or step==c['updates']:save()
            if hours and time.monotonic()-start>=hours*3600:
                save();break
    if not (run/'last.pt').exists():save()
    for group,(n,before) in representatives.items():
        after=dict(model.named_parameters())[n].detach().cpu()
        updates_report.append(dict(group=group,parameter=n,max_abs_change=float((after-before).abs().max())))
    complete=step==c['updates']
    result=dict(run=run.relative_to(root).as_posix(),step=step,target=c['updates'],complete=complete,
                started_from=original_step,elapsed_seconds=time.monotonic()-start,
                checkpoint_sha256=digest(run/'last.pt'),representative_updates=updates_report,**memory_check(d,c))
    result['checkpoint_kind']='terminal_model' if complete else 'resumable_model_optimizer_rng'
    write_json(run/'execution.json',result);write_json(run/'status.json',result)
    if complete and stage not in ['smoke','overfit']:
        from .scoring import calibrate
        write_json(run/'status.json',{**result,'phase':'gallery/calibration','complete':False})
        calibrate(root,run,model,s,c,result['checkpoint_sha256'])
    if complete and stage=='overfit':
        from .visuals import overfit_images
        overfit_images(root,run,model,s,c,'after')
    write_json(run/'status.json',{**result,'phase':'complete' if complete else 'budget_pause'})
    del opt,model
    torch.cuda.empty_cache()
    return result
