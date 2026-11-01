"""Explicit sparse targets: upstream dense negative losses are never called."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
from .geometry import xyxy,cxcywh,overlaps,paired_giou


def assignment(boxes,gt,logits=None):
    if not len(gt):
        z=torch.empty(0,device=boxes.device,dtype=torch.long)
        return z,z
    with torch.no_grad():
        cost=5*torch.cdist(boxes,gt,p=1)-2*generalized_box_iou(xyxy(boxes),xyxy(gt))
        if logits is not None:
            # Match upstream BinaryFocalLossCost, one positive object token.
            p=logits.sigmoid();token=1 if logits.shape[1]>1 else 0
            neg=-(1-p+1e-12).log()*.75*p.pow(2)
            positive=-(p+1e-12).log()*.25*(1-p).pow(2)
            cost+=2*(positive[:,token]+neg.sum(1)-neg[:,token])[:,None]
        a,b=linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(a,device=boxes.device),torch.as_tensor(b,device=boxes.device))


def sparse_weights(boxes,gt,pos,teacher,c):
    """Identical policy in J0-J3, including encoder and DN negative queries."""
    b=xyxy(boxes.detach()).clamp(0,1);weight=b.new_zeros(len(b))
    tb=teacher['boxes'];p=teacher['p']
    if len(tb):
        ov=overlaps(b,xyxy(tb))
        protected=ov[:,p>=teacher['thresholds']['ignore']].max(1).values>=c['teacher_protect_iou'] if (p>=teacher['thresholds']['ignore']).any() else torch.zeros(len(b),device=b.device,dtype=torch.bool)
        best,idx=ov.max(1)
        weak=(best>=c['weak_match_iou']) & (p[idx]<=teacher['thresholds']['low']) & ~protected
    else:
        weak=torch.zeros(len(b),device=b.device,dtype=torch.bool)
    if len(gt):weak&=overlaps(b,xyxy(gt)).max(1).values<.2
    weak[pos]=False
    ids=torch.nonzero(weak).flatten()
    # Stable query order, never resample different negatives by method.
    ids=ids[:c['negative_ratio']*max(1,len(gt))]
    weight[ids]=c['weak_negative_weight'];weight[pos]=1
    return weight,dict(positive=len(pos),weak_negative=len(ids),ignore=int((weight==0).sum()))


def centerness(box,gt):
    center=box[:,:2];g=xyxy(gt)
    ltrb=torch.cat((center-g[:,:2],g[:,2:]-center),1).clamp_min(0)
    l,t,r,b=ltrb.unbind(1)
    return ((torch.minimum(l,r)/(torch.maximum(l,r)+1e-12))*
            (torch.minimum(t,b)/(torch.maximum(t,b)+1e-12))).sqrt()


def layer_loss(logits,boxes,ctr,gt,teacher,c,pos=None,owners=None):
    if pos is None:pos,owners=assignment(boxes,gt,logits)
    weight,counts=sparse_weights(boxes,gt,pos,teacher,c)
    target=torch.zeros_like(logits)
    target[pos,1 if logits.shape[1]>1 else 0]=1
    ce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    prob=logits.sigmoid();pt=prob*target+(1-prob)*(1-target)
    alpha=.25*target+.75*(1-target)
    obj=(ce*(1-pt).pow(2)*alpha*weight[:,None]).sum()/max(1,len(pos))
    zero=boxes.sum()*0
    if len(pos):
        box=F.l1_loss(boxes[pos],gt[owners],reduction='sum')/len(pos)
        giou=(1-paired_giou(xyxy(boxes[pos]),xyxy(gt[owners]))).mean()
        ct=F.l1_loss(ctr[pos],centerness(boxes[pos].detach(),gt[owners]))
    else:box=giou=ct=zero
    return obj+5*box+2*giou+5*ct,counts,pos,owners


def preserve_loss(logits,boxes,ctr,gt,occupied,teacher,c):
    reliable=teacher['reliable']
    if len(gt):reliable=reliable & (overlaps(xyxy(teacher['boxes']),xyxy(gt)).max(1).values<.5)
    ti=torch.nonzero(reliable).flatten()
    free=torch.ones(len(boxes),device=boxes.device,dtype=torch.bool);free[occupied]=False
    qi=torch.nonzero(free).flatten()
    zero=boxes.sum()*0
    if not len(ti) or not len(qi):return zero,0
    a,b=assignment(boxes[qi],teacher['boxes'][ti]);a=qi[a];b=ti[b]
    token=1 if logits.shape[1]>1 else 0
    obj=F.binary_cross_entropy_with_logits(logits[a,token],teacher['p'][b].detach())
    ct=F.l1_loss(ctr[a],teacher['ctr'][b].detach())
    tb=teacher['boxes'][b].detach()
    box=F.l1_loss(boxes[a],tb,reduction='sum')/len(a)
    giou=(1-paired_giou(xyxy(boxes[a]),xyxy(tb))).mean()
    return obj+ct+5*box+2*giou,len(a)


def dn_indices(gt,dn_meta,device):
    if not dn_meta or not dn_meta['num_denoising_queries']:
        empty=torch.empty(0,device=device,dtype=torch.long)
        return empty,empty
    groups=dn_meta['num_denoising_groups']
    per=dn_meta['num_denoising_queries']//groups
    owners=torch.arange(len(gt),device=device).repeat(groups)
    pos=(torch.arange(groups,device=device)[:,None]*per+torch.arange(len(gt),device=device)[None]).flatten()
    return pos,owners


def joint_loss(out,teacher,model,c):
    gt=cxcywh(out['gt']);meta=out['head'].get('dn_meta')
    ndn=meta['num_denoising_queries'] if meta else 0
    dp,do=dn_indices(gt,meta,out['boxes'].device)
    totals=[];stats={};last_pos=last_owners=None
    for i in range(len(out['boxes'])):
        value,counts,pos,owners=layer_loss(out['cls'][i,ndn:],out['boxes'][i,ndn:],out['ctr'][i,ndn:],gt,teacher,c)
        stats[f'layer_{i}']=counts
        if ndn:
            dv,dc,_,_=layer_loss(out['cls'][i,:ndn],out['boxes'][i,:ndn],out['ctr'][i,:ndn],gt,teacher,c,dp,do)
            value=value+dv;stats[f'dn_{i}']=dc
        totals.append(value)
        if i==len(out['boxes'])-1:last_pos,last_owners=pos,owners
    pf_loss=torch.stack(totals).mean()
    h=out['head']
    if 'enc_outputs_class' in h:
        ev,ec,_,_=layer_loss(h['enc_outputs_class'][0],h['enc_outputs_coord'][0],h['enc_outputs_conf'][0,:,0],gt,teacher,c)
        pf_loss=pf_loss+ev;stats['encoder']=ec
    if 'router_weights' in h:
        pf_loss=pf_loss+h['router_weights'].std()
    pos=torch.cat((last_pos+ndn,dp));owners=torch.cat((last_owners,do))
    if len(pos):
        local=torch.tensor([model.active.index(x) for x in out['semantic'][owners].tolist()],device=pos.device)
        known=F.cross_entropy(out['known'][pos],local)
        with torch.no_grad():
            target=overlaps(xyxy(out['boxes'][-1,pos]).clamp(0,1),out['gt'][owners]).diag()
        quality=F.smooth_l1_loss(out['quality'][pos],target)
    else:known=quality=out['known'].sum()*0
    if model.protect:
        pr,n=preserve_loss(out['cls'][-1,ndn:],out['boxes'][-1,ndn:],out['ctr'][-1,ndn:],gt,last_pos,teacher,c)
    else:pr=out['boxes'].sum()*0;n=0
    loss=pf_loss+c['known_weight']*known+c['quality_weight']*quality+(c['preserve_weight']*pr if model.protect else 0)
    stats.update(loss_pf=float(pf_loss.detach()),loss_known=float(known.detach()),loss_quality=float(quality.detach()),
                 loss_preserve=float(pr.detach()),teacher_matches=n,mean_match_iou=float(target.mean()) if len(pos) else None)
    return loss,stats
