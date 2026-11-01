import torch
from torch.nn import functional as F
from jointdet.geometry import xyxy,cxcywh,overlaps,paired_giou
from jointdet.losses import layer_loss,dn_indices,assignment,preserve_loss
from .relation import episode_loss


def perturb(gt):
    """Training-only geometry controls. IoU is measured against ALL legal GT."""
    base=cxcywh(gt);rows=[base]
    for dx,dy,scale in [(0.,0.,.55),(.4,0.,1.),(0.,.4,1.),(0.,0.,1.6)]:
        b=base.clone();b[:,:2]+=b[:,2:]*b.new_tensor([dx,dy]);b[:,2:]*=scale
        rows.append(cxcywh(xyxy(b).clamp(0,1)))
    boxes=torch.cat(rows)
    target=overlaps(xyxy(boxes),gt).max(1).values
    return boxes,target


def detection_loss(out,teacher,model,c):
    gt=cxcywh(out['gt']);dn=out['head'].get('dn_meta');n=dn['num_denoising_queries'] if dn else 0
    dp,do=dn_indices(gt,dn,out['boxes'].device);values=[];last_pos=None
    for i in range(len(out['boxes'])):
        v,_,pos,owners=layer_loss(out['cls'][i,n:],out['boxes'][i,n:],out['ctr'][i,n:],gt,teacher,c)
        if n:
            vv,_,_,_=layer_loss(out['cls'][i,:n],out['boxes'][i,:n],out['ctr'][i,:n],gt,teacher,c,dp,do);v=v+vv
        values.append(v);last_pos=pos
    pf=torch.stack(values).mean()
    h=out['head']
    if 'enc_outputs_class' in h:
        v,_,_,_=layer_loss(h['enc_outputs_class'][0],h['enc_outputs_coord'][0],h['enc_outputs_conf'][0,:,0],gt,teacher,c);pf=pf+v
    if 'router_weights' in h:pf=pf+h['router_weights'].std()
    preserve,matched=preserve_loss(out['cls'][-1,n:],out['boxes'][-1,n:],out['ctr'][-1,n:],gt,last_pos,teacher,c)
    refined_preserve,_=preserve_loss(out['cls'][-1,n:],out['refined'][n:],out['ctr'][-1,n:],gt,last_pos,teacher,c)
    preserve=(preserve+refined_preserve)/2
    pos,owners=assignment(out['refined'][n:],gt,out['cls'][-1,n:]);pos=pos+n
    zero=out['refined'].sum()*0
    refine=5*F.l1_loss(out['refined'][pos],gt[owners])+2*(1-paired_giou(xyxy(out['refined'][pos]),out['gt'][owners])).mean() if len(pos) else zero
    target=overlaps(xyxy(out['refined'].detach()),out['gt']).max(1).values
    # Never label arbitrary unmatched proposals as background from sparse GT alone.
    supervised=target>=.1;supervised[:n]=False
    quality=F.binary_cross_entropy_with_logits(out['quality_logit'][supervised],target[supervised]) if supervised.any() else zero
    boxes,qt=perturb(out['gt']);_,ql=model.region_quality(out,boxes)
    quality=quality+F.binary_cross_entropy_with_logits(ql,qt)
    return pf+c['preserve_weight']*preserve+refine+c['quality_weight']*quality, dict(
        pf=float(pf.detach()),preserve=float(preserve.detach()),refine=float(refine.detach()),quality=float(quality.detach()),teacher_matches=matched)


def total_loss(out,teacher,model,c,bank,image,rng):
    loss,stats=detection_loss(out,teacher,model,c)
    # GT guarantees positive identity supervision even when proposals miss a class.
    raw,_=model.region_quality(out,cxcywh(out['gt']))
    labels=out['semantic'];dn=out['head'].get('dn_meta');n=dn['num_denoising_queries'] if dn else 0
    proposals=overlaps(xyxy(out['refined'][n:].detach()),out['gt'])
    i,j=torch.where(proposals>=.5);i=i+n
    if len(i):
        order=torch.as_tensor(rng.permutation(len(i))[:16],device=i.device);i,j=i[order],j[order]
        raw=torch.cat([raw,out['raw'][i]],0);labels=torch.cat([labels,out['semantic'][j]])
    if len(raw)>32:
        ids=torch.as_tensor(rng.choice(len(raw),32,replace=False),device=raw.device);raw=raw[ids];labels=labels[ids]
    query_group=bank.get('image_to_group',{}).get(image,image)
    allowed=torch.tensor([g!=query_group for g in bank.get('groups',bank['images'])],device=raw.device)[None].expand(len(raw),-1)
    identity,details=episode_loss(model.relation,raw.detach(),labels,bank['raw'],bank['labels'],allowed,rng)
    # raw is detached: identity cannot push the shared PF queries or box coordinates.
    stats.update(identity=float(identity.detach()),episode=details)
    return loss+c['identity_weight']*identity,stats
