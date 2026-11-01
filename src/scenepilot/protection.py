"""Two-pass accumulated backward and block-wise known-gradient protection."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from cil.common import rng_state,restore_rng
from jointdet.geometry import cxcywh
from jointdet.losses import assignment,dn_indices
from .response_loss import loss_for as native_loss


def block_for(name):
    if name.startswith('pf.decoder.'):return 'pf_decoder'
    if name.startswith('pf.backbone.stages.3.'):return 'swin_last'
    return None


def known_loss(out,model,c):
    gt=cxcywh(out['gt']);meta=out['head'].get('dn_meta');ndn=meta['num_denoising_queries'] if meta else 0
    positive,owners=assignment(out['boxes'][-1,ndn:],gt,out['cls'][-1,ndn:])
    dp,do=dn_indices(gt,meta,out['boxes'].device)
    positive=torch.cat((positive+ndn,dp));owners=torch.cat((owners,do))
    if len(positive):
        local=torch.tensor([model.active.index(x) for x in out['semantic'][owners].tolist()],device=positive.device)
        return F.cross_entropy(out['known'][positive],local)*c['known_weight']
    return out['known'].sum()*0


def cpu_gradients(model,protected_only=False):
    return {n:p.grad.detach().to(device='cpu',copy=True) for n,p in model.named_parameters()
            if p.requires_grad and p.grad is not None and (not protected_only or block_for(n) is not None)}


def dot(a,b):
    return float(np.einsum('i,i->',a.numpy().reshape(-1),b.numpy().reshape(-1),dtype=np.float64))


def project(total,known,blocks,epsilon=1e-12,norm2_floor=1e-20):
    """Modify total CPU FP32 gradients only in the declared shared blocks.

    k is already included in total. If (total-k).k < 0, remove that opposing
    rest component. No second copy of the known loss is added to the objective.
    """
    stats={}
    for block in blocks:
        names=[n for n in known if block_for(n)==block]
        ek=sum(dot(known[n],known[n]) for n in names)
        et=sum(dot(total[n],total[n]) for n in names if n in total)
        tk=sum(dot(total[n],known[n]) for n in names if n in total)
        rk=tk-ek;alpha=0.
        projected=bool(ek>norm2_floor and rk<0)
        if projected:
            alpha=-rk/(ek+epsilon)
            if not math.isfinite(alpha):raise RuntimeError('Nonfinite gradient-projection coefficient')
            for name in names:
                if name not in total:total[name]=torch.zeros_like(known[name])
                total[name].add_(known[name],alpha=alpha)
        post=sum(dot(total[n],known[n]) for n in names if n in total)
        tolerance=max(1e-10,1e-6*math.sqrt(max(0,ek*et))+abs(alpha)*epsilon)
        if not math.isfinite(post) or post < -tolerance:
            raise RuntimeError('Known-protected total gradient violates numerical bound')
        stats[block]=dict(projected=projected,known_gradient_tensors=len(names),known_norm=math.sqrt(ek),
            dot_total_known_before=tk,dot_other_known_before=rk,projection_alpha=alpha,
            dot_total_known_after=post,dot_other_known_after=post-ek,
            numerical_tolerance=tolerance,reason='projected' if projected else ('zero_known_gradient' if ek<=norm2_floor else 'already_compatible'))
    return stats


def accumulated_update(model,samples,forward,known_forward,c,arm):
    if arm not in ('P0','PR'):raise ValueError('Expected P0 or PR')
    before=rng_state();after=None;losses=[];micro_stats=[];sample_rows=[];known_values=[]
    model.zero_grad(set_to_none=True)
    try:
        for item in samples:
            out,original,target=forward(item)
            loss,stats=native_loss(out,original,target,model,c,'L0' if arm=='P0' else 'R0')
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite primary objective')
            if stats['hard_background_labels']:raise RuntimeError('P0/PR expects the reviewed zero-hard-background protocol')
            (loss/len(samples)).backward()
            losses.append(float(loss.detach()));micro_stats.append(stats);sample_rows.append(item)
            known_values.append(stats['loss_known']*c['known_weight'])
            del out,original,target,loss
        after=rng_state();total=cpu_gradients(model)
        model.zero_grad(set_to_none=True);restore_rng(before)
        for item,expected in zip(samples,known_values):
            out=known_forward(item);value=known_loss(out,model,c)
            if not math.isclose(float(value.detach()),expected,rel_tol=1e-5,abs_tol=1e-5):
                raise RuntimeError('Auxiliary known pass differs from the primary stochastic forward')
            (value/len(samples)).backward()
            del out,value
        known=cpu_gradients(model,protected_only=True)
        blocks=project(total,known,c['protection_blocks'],c['protection_epsilon'],c['protection_known_norm2_floor'])
        model.zero_grad(set_to_none=True)
        for name,parameter in model.named_parameters():
            if name in total:
                gradient=total[name]
                if not bool(torch.isfinite(gradient).all()):raise RuntimeError('Nonfinite projected gradient')
                parameter.grad=gradient.to(device=parameter.device,dtype=parameter.dtype,copy=True)
        return dict(losses=losses,micro_stats=micro_stats,samples=sample_rows,known_gradients=known,
            protection=dict(blocks=blocks,RNG_primary_path_preserved=True,known_is_subset_of_total=True,
                passes=2,accumulate=len(samples),weighted_known_loss=float(np.mean(known_values))))
    finally:
        # The auxiliary pass must not advance dropout/DN RNG for the next update.
        restore_rng(after if after is not None else before)


def should_measure(step,c):
    return step%c['protection_actual_update_every']==0 or step in c['protection_actual_update_extra_steps']


def capture_parameters(model,known):
    return {n:p.detach().to(device='cpu',copy=True) for n,p in model.named_parameters() if n in known}


def actual_alignment(model,before,known):
    stats={}
    for name,param in model.named_parameters():
        if name not in before:continue
        delta=param.detach().cpu()-before[name];k=known[name];block=block_for(name)
        s=stats.setdefault(block,dict(dot_known_parameter_delta=0.,known_norm2=0.,parameter_delta_norm2=0.))
        s['dot_known_parameter_delta']+=dot(k,delta);s['known_norm2']+=dot(k,k);s['parameter_delta_norm2']+=dot(delta,delta)
    for value in stats.values():
        denom=math.sqrt(value['known_norm2']*value['parameter_delta_norm2'])
        value['cosine_known_parameter_delta']=value['dot_known_parameter_delta']/denom if denom else None
    return dict(blocks=stats,note='Actual AdamW parameter delta; negative dot with pre-update known gradient is a local first-order decrease proxy, not an observed post-update loss or AP guarantee.')
