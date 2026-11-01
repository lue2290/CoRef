import math
import time
import numpy as np
import torch
from cil.common import rng_state, restore_rng
from jointdet.protocol import atomic_save, load_torch
from current.progress import Progress, stage
from .protocol import read, write, guard_space


def learning_rate_factor(step, target, warmup):
    return min(1.,step/warmup)*(.05+.95*(1+math.cos(math.pi*(step-1)/max(target-1,1)))/2)


def optimizer_for(model, c):
    groups=[]
    for group, lr in [('new',c['new_lr']),('pf',c['pf_lr'])]:
        params=[p for name,p in model.named_parameters() if model.group_for(name)==group]
        if not params: raise RuntimeError('Empty training parameter group: '+group)
        groups.append(dict(params=params,lr=lr,base_lr=lr,name=group))
    return torch.optim.AdamW(groups,weight_decay=c['weight_decay'])


def train_parent(model, view, root, cfg, original_cfg, spec, bank, cache, run, report, binding, device,
                 stop_after=None, loss_fn=None, teacher_fn=None, monitor_fn=None):
    """Same objective/sampling/schedule as v1. Only own new-tag checkpoints resume."""
    from refnet.loss import total_loss
    from refnet.cache import teacher
    from refnet.inference import monitor
    loss_fn=loss_fn or total_loss; teacher_fn=teacher_fn or teacher; monitor_fn=monitor_fn or monitor
    c=model.c; optimizer=optimizer_for(model,c); target=cfg['updates']; step=0
    rng=np.random.default_rng(binding['seed']); history=[]; curve=[]; elapsed=0.
    run.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True); checkpoint=run/'last.pt'
    initialized_from='original_pretraining'; start=time.monotonic()
    if checkpoint.exists():
        old=load_torch(checkpoint)
        if old['lock']!=binding['identity'] or old['bank_manifest']!=bank['manifest_sha256']:
            raise RuntimeError('158 resume checkpoint belongs to different sources/data/bank')
        if old['step']>target or old['complete']!=(old['step']==target): raise RuntimeError('Invalid158 endpoint state')
        model.load_adapted(old['model']); step=old['step']; history=old['history'];curve=old['curve'];elapsed=old['training_seconds']
        initialized_from=old['initialization']
        if step<target:
            optimizer.load_state_dict(old['optimizer']);restore_rng(old['rng']);rng.bit_generator.state=old['sampler']
        del old
    else:
        # A step0 checkpoint is the reproducible fresh initialization, not an old
        # adapted parent's warm start. Never open historical last.pt/best.pt here.
        initialized_from='original_pretraining'
    by={n:[o for o in spec['fit_objects'] if o['image']==n] for n in spec['fit_images']}
    class_images={cid:sorted({o['image'] for o in spec['fit_objects'] if o['class_id']==cid}) for cid in range(len(spec['classes']))}
    if any(not names for names in class_images.values()): raise RuntimeError('Missing corrected fitting class')
    def save():
        adapted=model.adapted_state(); weight_bytes=sum(t.numel()*t.element_size() for t in adapted.values())
        guard_space(root,cfg,weight_bytes*(4 if step<target else 1))
        atomic_save(dict(model=adapted,step=step,complete=step==target,lock=binding['identity'],
            bank_manifest=bank['manifest_sha256'],optimizer=None if step==target else optimizer.state_dict(),
            rng=rng_state(),sampler=rng.bit_generator.state,history=history,curve=curve,
            initialization=initialized_from,old_adapted_checkpoint_loaded=False,
            training_seconds=elapsed+time.monotonic()-start),checkpoint)
    if not checkpoint.exists(): save()
    limit=target if stop_after is None else min(target,stop_after)
    with Progress(target,f'Game/{spec["task"]}/seed_{binding["seed"]} corrected v1 training','updates') as bar:
        bar.update(completed=step)
        for current in range(step+1,limit+1):
            model.configure_stage(current>c['unfreeze_at']);model.train();optimizer.zero_grad(set_to_none=True)
            factor=learning_rate_factor(current,target,cfg['warmup'])
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            losses=[];stats=[];samples=[]
            for _ in range(c['accumulate']):
                cid=int(rng.integers(len(spec['classes'])));name=str(rng.choice(class_images[cid]));flip=bool(rng.integers(2))
                out=model.forward_image(view,name,by[name],device,flip)
                response=teacher_fn(cache,name,device,flip)
                loss,details=loss_fn(out,response,model,c,bank,name,rng)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite corrected-parent loss')
                (loss/c['accumulate']).backward();losses.append(float(loss.detach()));stats.append(details);samples.append(dict(image=name,flip=flip))
                del out,response,loss
            grads={group:sum(p.grad is not None and bool(p.grad.abs().sum()) for name,p in model.named_parameters()
                            if model.group_for(name)==group) for group in ['new','pf']}
            if not grads['new'] or (model._stage and not grads['pf']):raise RuntimeError('Missing gradient group: '+str(grads))
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError('Nonfinite training gradient')
            if any(p.grad is not None for name,p in model.named_parameters() if model.group_for(name) is None):
                raise RuntimeError('Frozen DINO/PF backbone received gradients')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['grad_clip'],error_if_nonfinite=True)
            optimizer.step();step=current
            history.append(dict(step=current,loss=float(np.mean(losses)),stats=stats,samples=samples,
                                learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},gradient_groups=grads))
            if current%cfg['validate_every']==0 or current==target:
                state=rng_state()
                with stage(f'Game/{spec["task"]}/seed_{binding["seed"]} training-only monitor'):
                    try:metrics=monitor_fn(root,view,model,spec,original_cfg,cache,bank,report)
                    finally:restore_rng(state)
                curve.append(dict(step=current,metrics=metrics))
                write(report/'CURVE.json',dict(training=history,monitor=curve,selection='fixed500; monitor never selects evaluated checkpoint'))
            if current%cfg['save_every']==0 or current==target:save()
            if device.type=='cuda' and torch.cuda.max_memory_allocated(device)>cfg['max_gpu_gib']*2**30:
                raise RuntimeError('Configured GPU memory budget exceeded; last atomic checkpoint retained')
            if current%25==0:guard_space(root,cfg)
            bar.update(completed=step,message=f'loss {history[-1]["loss"]:.4f}')
    # stop_after is an internal fixture hook, not a production CLI budget override.
    if stop_after is not None and step<target:save()
    return dict(step=step,complete=step==target,initialization=initialized_from,history=history,curve=curve,
                training_seconds=elapsed+time.monotonic()-start)
