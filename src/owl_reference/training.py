import math
import time
import numpy as np
import torch
from cil.common import rng_state,restore_rng
from jointdet.protocol import atomic_save,load_torch
from current.progress import Progress,stage
from .model import identity_loss
from .protocol import write,sha,lock,locations,guard,fp


@torch.inference_mode()
def monitor(head,data):
    queries=[];labels=[]
    for name in data.spec['monitor_images']:
        a=data.view(name,False);queries.append(a['gt_raw']);labels.extend(a['labels'].tolist())
    if not labels:return dict(defined=False,reason='no legal known monitor GT')
    q=torch.tensor(np.concatenate(queries),device=data.device);y=torch.tensor(labels,device=data.device,dtype=torch.long)
    active=list(range(len(data.spec['classes'])));head.eval()
    known=torch.nn.functional.cross_entropy(head(q,data.bank,active),y);novel=[]
    for held in sorted(set(labels)):
        support=[c for c in active if c!=held];out=head(q[y==held],data.bank,support)
        novel.append(torch.nn.functional.cross_entropy(out,torch.full((int((y==held).sum()),),len(support),device=y.device,dtype=torch.long)))
    return dict(defined=True,known_ce=float(known),pseudo_unknown_ce=float(torch.stack(novel).mean()),
                scope='legal monitor GT identity diagnostic only; no detection/model selection')


def train_head(head,data,cfg,plan,root,arm,seed,binding,stop_after=None,monitor_fn=None):
    monitor_fn=monitor_fn or monitor;_,run,report=locations(root,cfg,data.case['scene'],arm,seed)
    path=run/'last.pt';opt=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    step=0;curve=[];mon=[];elapsed=0.;started=time.monotonic()
    cache_sha=sha(data.folder/'MANIFEST.json')
    if path.exists():
        old=load_torch(path)
        if old['lock']!=binding['identity'] or old['plan']!=plan['identity'] or old['cache']!=cache_sha:raise RuntimeError('OWL head checkpoint source/plan/cache changed')
        if old['complete']!=(old['step']==cfg['updates']) or old['step']>cfg['updates']:raise RuntimeError('Invalid OWL head endpoint')
        head.load_state_dict(old['head'],strict=True);step=old['step'];curve=old['curve'];mon=old['monitor'];elapsed=old['training_seconds']
        if step<cfg['updates']:opt.load_state_dict(old['optimizer']);restore_rng(old['rng'])
        del old
    def save():
        state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};guard(root,cfg,sum(v.numel()*v.element_size() for v in state.values())*4)
        atomic_save(dict(head=state,optimizer=opt.state_dict() if step<cfg['updates'] else None,step=step,complete=step==cfg['updates'],
            lock=binding['identity'],plan=plan['identity'],cache=cache_sha,rng=rng_state(),curve=curve,monitor=mon,
            training_seconds=elapsed+time.monotonic()-started,arm=arm,seed=seed,frozen_backbones_updated=False),path)
    if not path.exists():save()
    end=cfg['updates'] if stop_after is None else min(stop_after,cfg['updates'])
    with Progress(cfg['updates'],f'{data.case["scene"]}/{arm}/seed_{seed} frozen-feature head training','updates') as bar:
        bar.update(completed=step)
        for current in range(step+1,end+1):
            head.train();opt.zero_grad(set_to_none=True)
            factor=min(1.,current/cfg['warmup'])*(.05+.95*(1+math.cos(math.pi*(current-1)/max(cfg['updates']-1,1)))/2)
            for group in opt.param_groups:group['lr']=cfg['learning_rate']*factor
            rows=plan['steps'][current-1]['microbatches'];losses=[];details=[]
            if len(rows)!=cfg['accumulate']:raise RuntimeError('Shared OWL training plan accumulation mismatch')
            for row in rows:
                q,y,allowed=data.query(row)
                loss,stats=identity_loss(head,q,y,data.bank,allowed,row['episode'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite OWL identity loss')
                (loss/cfg['accumulate']).backward();losses.append(float(loss.detach()));details.append(stats)
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            opt.step();step=current
            curve.append(dict(step=step,loss=float(np.mean(losses)),gradient_norm=float(norm),learning_rate=opt.param_groups[0]['lr'],queries=details))
            if step%cfg['monitor_every']==0 or step==cfg['updates']:
                state=rng_state()
                with stage('legal training-held monitor (fixed500 endpoint)'):
                    try:metrics=monitor_fn(head,data)
                    finally:restore_rng(state)
                mon.append(dict(step=step,metrics=metrics));write(report/'CURVE.json',dict(training=curve,monitor=mon,selection='fixed500 only'))
            if step%cfg['save_every']==0 or step==cfg['updates']:save()
            bar.update(completed=step,message=f'loss {curve[-1]["loss"]:.4f}')
    if stop_after is not None and step<cfg['updates']:save()
    head.eval()
    return dict(step=step,complete=step==cfg['updates'],trainable_parameters=sum(p.numel() for p in head.parameters()),
                checkpoint_sha256=sha(path),training_seconds=elapsed+time.monotonic()-started)
