import time
import math
import numpy as np
import torch
from cil.common import rng_state,restore_rng
from jointdet.protocol import atomic_save,load_torch
from current.progress import Progress,stage
from refnet.cache import teacher
from refnet.inference import monitor
from .model import total_loss
from .protocol import write,guard_space


def train(model,view,root,cfg,original,spec,bank,cache,run,report,binding,plan,device,
          stop_after=None,loss_fn=None,teacher_fn=None,monitor_fn=None):
    loss_fn=loss_fn or total_loss;teacher_fn=teacher_fn or teacher;monitor_fn=monitor_fn or monitor
    c=model.c;groups=[]
    for kind,lr in [('new',c['new_lr']),('pf',c['pf_lr'])]:
        params=[p for n,p in model.named_parameters() if model.group_for(n)==kind]
        if not params:raise RuntimeError('Empty ablation optimizer group: '+kind)
        groups.append(dict(params=params,lr=lr,base_lr=lr,name=kind))
    optimizer=torch.optim.AdamW(groups,weight_decay=c['weight_decay'])
    target=cfg['updates'];step=0;history=[];curve=[];elapsed=0.;path=run/'last.pt';start=time.monotonic()
    if path.exists():
        state=load_torch(path)
        if state['lock']!=binding['identity'] or state['plan']!=plan['identity'] or state['bank_manifest']!=bank['manifest_sha256']:
            raise RuntimeError('159 resume source/plan/reference mismatch')
        if state['complete']!=(state['step']==target) or state['step']>target:raise RuntimeError('Invalid159 endpoint')
        model.load_adapted(state['model']);step=state['step'];history=state['history'];curve=state['curve'];elapsed=state['training_seconds']
        if step<target:optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        del state
    def save():
        adapted=model.adapted_state();size=sum(v.numel()*v.element_size() for v in adapted.values())
        guard_space(root,cfg,size*(4 if step<target else 1))
        atomic_save(dict(model=adapted,step=step,complete=step==target,lock=binding['identity'],plan=plan['identity'],
            bank_manifest=bank['manifest_sha256'],optimizer=None if step==target else optimizer.state_dict(),rng=rng_state(),
            history=history,curve=curve,training_seconds=elapsed+time.monotonic()-start,
            initialization='original_pretraining',old_adapted_checkpoint_loaded=False,variant=model.variant),path)
    if not path.exists():save()
    by={n:[o for o in spec['fit_objects'] if o['image']==n] for n in spec['fit_images']}
    limit=target if stop_after is None else min(target,stop_after)
    with Progress(target,f'{model.variant}/{spec["scene"]}/seed_{binding["seed"]} training','updates') as bar:
        bar.update(completed=step)
        for current in range(step+1,limit+1):
            row=plan['steps'][current-1]
            if row['step']!=current or len(row['microbatches'])!=c['accumulate']:raise RuntimeError('Locked sampling plan length mismatch')
            model.configure_stage(current>c['unfreeze_at']);model.train();optimizer.zero_grad(set_to_none=True)
            factor=min(1.,current/cfg['warmup'])*(.05+.95*(1+math.cos(math.pi*(current-1)/max(target-1,1)))/2)
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            losses=[];stats=[];samples=[]
            for sample in row['microbatches']:
                name=sample['image'];flip=sample['flip']
                out=model.forward_image(view,name,by[name],device,flip)
                response=teacher_fn(cache,name,device,flip)
                rng=np.random.default_rng(sample['query_sampling_seed'])
                loss,details=loss_fn(out,response,model,c,bank,name,rng,sample['episode'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite ablation loss')
                (loss/c['accumulate']).backward();losses.append(float(loss.detach()));stats.append(details);samples.append(dict(image=name,flip=flip))
                del out,response,loss
            grads={kind:sum(p.grad is not None and bool(p.grad.abs().sum()) for n,p in model.named_parameters() if model.group_for(n)==kind) for kind in ['new','pf']}
            if not grads['new'] or (model._stage and not grads['pf']):raise RuntimeError('Missing ablation gradient group: '+str(grads))
            if any(p.grad is not None for n,p in model.named_parameters() if model.group_for(n) is None):raise RuntimeError('Frozen/disabled module received gradients')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['grad_clip'],error_if_nonfinite=True)
            optimizer.step();step=current
            history.append(dict(step=current,loss=float(np.mean(losses)),stats=stats,samples=samples,
                learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},gradient_groups=grads))
            if current%cfg['validate_every']==0 or current==target:
                state=rng_state()
                with stage('training-only monitor; fixed endpoint selection'):
                    try:metrics=monitor_fn(root,view,model,spec,original,cache,bank,report)
                    finally:restore_rng(state)
                curve.append(dict(step=current,metrics=metrics));write(report/'CURVE.json',dict(training=history,monitor=curve,selection='fixed500 only'))
            if current%cfg['save_every']==0 or current==target:save()
            if device.type=='cuda' and torch.cuda.max_memory_allocated(device)>cfg['max_gpu_gib']*2**30:raise RuntimeError('GPU memory budget exceeded; resume previous atomic checkpoint')
            if current%25==0:guard_space(root,cfg)
            bar.update(completed=step,message=f'loss {history[-1]["loss"]:.4f}')
    if stop_after is not None and step<target:save()
    return dict(step=step,complete=step==target,history=history,curve=curve,training_seconds=elapsed+time.monotonic()-start)
