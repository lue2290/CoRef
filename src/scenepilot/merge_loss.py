import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from jointdet.geometry import cxcywh,xyxy,overlaps
from jointdet.losses import joint_loss,assignment
from .protection import known_loss


def soft_object_loss(out,target,c):
    meta=out['head'].get('dn_meta');ndn=meta['num_denoising_queries'] if meta else 0
    boxes=out['boxes'][-1,ndn:];scores=out['open_obj'][ndn:];gt=cxcywh(out['gt'])
    with torch.no_grad():
        pos,_=assignment(boxes,gt,out['cls'][-1,ndn:])
        free=torch.ones(len(boxes),device=boxes.device,dtype=torch.bool);free[pos]=False
        tb=target['boxes'];valid=torch.ones(len(tb),device=boxes.device,dtype=torch.bool)
        if len(gt):
            free &= overlaps(xyxy(boxes.detach()),xyxy(gt)).max(1).values<c['response_gt_exclusion_iou']
            valid &= overlaps(xyxy(tb),xyxy(gt)).max(1).values<c['response_gt_exclusion_iou']
        qi=torch.nonzero(free).flatten();ti=torch.nonzero(valid).flatten()
        if not len(qi) or not len(ti):return scores.sum()*0,dict(anchors=0,low=0)
        ov=overlaps(xyxy(boxes.detach()[qi]),xyxy(tb[ti]))
        a,b=linear_sum_assignment(-ov.cpu().numpy());a=torch.as_tensor(a,device=boxes.device);b=torch.as_tensor(b,device=boxes.device)
        keep=ov[a,b]>=c['merge_teacher_iou'];a=qi[a[keep]];b=ti[b[keep]]
        if not len(a):return scores.sum()*0,dict(anchors=0,low=0)
        order=torch.argsort(overlaps(xyxy(boxes.detach()[a]),xyxy(tb[b])).diag(),descending=True,stable=True)
        low=target['low'][b];limit=c['response_per_gt']*max(1,len(gt))
        first=order[low[order]][:limit//2];second=order[~low[order]][:limit-limit//2];chosen=torch.cat([first,second])
        if len(chosen)<limit:
            used=torch.zeros(len(a),device=boxes.device,dtype=torch.bool);used[chosen]=True
            chosen=torch.cat([chosen,order[~used[order]][:limit-len(chosen)]])
        a=a[chosen];b=b[chosen]
    value=F.smooth_l1_loss(scores[a],target['objectness'][b].detach(),beta=c['response_beta'])
    return value,dict(anchors=len(a),low=int(target['low'][b].sum()))


def loss_for(out,target,model,c):
    empty=dict(boxes=out['boxes'].new_empty(0,4),p=out['boxes'].new_empty(0))
    common,stats=joint_loss(out,empty,model,c)
    distill,counts=soft_object_loss(out,target,c)
    total=common+c['teacher_response_weight']*distill
    hard=sum(v.get('weak_negative',0) for v in stats.values() if isinstance(v,dict))
    if hard:raise RuntimeError('Network merge does not introduce hard background labels')
    stats.update(loss_soft_object=float(distill.detach()),teacher_anchors=counts['anchors'],teacher_low=counts['low'],
        hard_background_labels=hard,geometry_matches=0)
    return total,stats


def ce_route(out,model,c):
    value=known_loss(out,model,c)
    # Stop autograd at the common query: asking for PF parameter gradients with
    # autograd.grad is incompatible with upstream reentrant Swin/FairScale CP.
    # Full backward parameter isolation is additionally covered by CPU tests.
    groups={'shared_query':[out['shared_query']],'known_head':[],'known_tail':[]}
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        if n.startswith('known.'):groups['known_head'].append(p)
        elif n.startswith('dino.'):groups['known_tail'].append(p)
    parameters=[p for ps in groups.values() for p in ps]
    gradients=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
    at=0;norm={}
    for name,ps in groups.items():
        selected=gradients[at:at+len(ps)];at+=len(ps)
        norm[name]=float(sum(float(g.detach().square().sum()) for g in selected if g is not None)**.5)
    if norm['known_head']<=0:raise RuntimeError('Known classifier has no CE gradient')
    if model.block_known_to_pf and norm['shared_query']!=0:raise RuntimeError('M1 known CE leaks into common query')
    if not model.block_known_to_pf and norm['shared_query']<=0:raise RuntimeError('M0 known CE unexpectedly cannot reach common query')
    if model._stage and c['unfreeze_backbones'] and norm['known_tail']<=0:raise RuntimeError('Recognition DINO tail has no CE gradient')
    if any(p.requires_grad for p in model.open_blocks.parameters()) or any(p.requires_grad for p in model.open_norm.parameters()):
        raise RuntimeError('Frozen open DINO tail was unfrozen')
    return dict(norms=norm,blocked=model.block_known_to_pf,open_tail_frozen=True,loss=float(value.detach()))
