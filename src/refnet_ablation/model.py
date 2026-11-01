import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from jointdet.geometry import cxcywh,overlaps,xyxy
from refnet.model import ReferenceDetector
from refnet.loss import detection_loss

VARIANTS=('full','no_feedback','no_preserve','raw_relation','no_episode')


class RawRelation(nn.Module):
    """Raw5120 cosine top3; original learned scale/boundary and five-stat head."""
    def __init__(self,old):
        super().__init__();self.neighbors=old.neighbors
        # Reuse *fresh initialization* parameters; no extra random draws and no
        # adapted-parent parameters. The learned embed/pair modules are removed.
        self.log_scale=old.log_scale;self.boundary=old.boundary;self.reject=old.reject

    def encode(self,raw):return F.normalize(raw,dim=-1)

    def forward(self,query,reference,labels,active,allowed=None):
        if not active:raise ValueError('Empty support class set')
        sim=self.encode(query)@self.encode(reference).T;scores=[];valids=[];disp=[]
        for cid in active:
            indices=torch.nonzero(labels==cid).flatten()
            if not len(indices):raise ValueError('Missing GT support class '+str(cid))
            values=sim[:,indices];mask=torch.ones_like(values,dtype=torch.bool) if allowed is None else allowed[:,indices]
            v,j=values.masked_fill(~mask,-1e4).topk(min(self.neighbors,len(indices)),dim=1)
            good=mask.gather(1,j);count=good.sum(1);den=count.clamp_min(1)
            mean=v.masked_fill(~good,0).sum(1)/den
            score=(self.log_scale.exp().clamp(max=50)*(v-self.boundary)).masked_fill(~good,0).sum(1)/den
            variance=(v-mean[:,None]).square().masked_fill(~good,0).sum(1)/den
            scores.append(score.masked_fill(count==0,-1e4));valids.append(count>0);disp.append(variance)
        known=torch.stack(scores,1);valid=torch.stack(valids,1);top=known.topk(min(2,len(active)),dim=1).values
        first=top[:,0];gap=first-top[:,1] if len(active)>1 else torch.zeros_like(first)
        evidence=torch.stack([first.clamp(-50,50),gap.clamp(0,50),valid.float().mean(1),
            torch.stack(disp,1).mean(1),first.new_full(first.shape,math.log1p(len(active)))],1)
        return torch.cat([known,self.reject(evidence)],1)


class AblationDetector(ReferenceDetector):
    def __init__(self,root,c,active,variant='full',**kwargs):
        if variant not in VARIANTS:raise ValueError('Unknown ablation variant')
        object.__setattr__(self,'variant',variant)
        # Instantiate the complete original network first. Disabling a route
        # must not shift initialization of the shared heads.
        super().__init__(root,c,active,**kwargs)
        if variant=='no_feedback':self.feedback=False
        if variant=='raw_relation':self.relation=RawRelation(self.relation)
        if variant=='no_preserve':self.c={**self.c,'preserve_weight':0.}
        self.configure_stage(False)

    def group_for(self,name):
        if self.variant=='no_feedback' and name.startswith(('projection.','bridges.')):return None
        return super().group_for(name)


def common_initialization(model):
    prefixes=('pf.decoder.','pf.query_embedding.','pf.bbox_head.','projection.','bridges.','refine.','loc_quality.',
              'relation.reject.','relation.log_scale','relation.boundary')
    h=hashlib.sha256();names=[]
    for name,p in model.named_parameters():
        if name.startswith(prefixes):
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes());names.append(name)
    return dict(sha256=h.hexdigest(),names=names)


def planned_identity_loss(module,query,y,reference,labels,allowed,episode,disable=False):
    active=sorted(set(labels.tolist()));full=module(query,reference,labels,active,allowed)
    target=torch.tensor([active.index(int(c)) for c in y],device=y.device)
    own=torch.stack([allowed[:,labels==c].any(1) for c in active],1)
    legal=own[torch.arange(len(y),device=y.device),target]
    known=F.cross_entropy(full[legal],target[legal]) if legal.any() else full.sum()*0
    held=int(episode['held_class']);support=list(map(int,episode['support_classes']))
    if held in support or not support or not set(support)<=set(active):raise RuntimeError('Invalid locked leave-class episode')
    details=dict(known_queries=int(legal.sum()),held_class=held,support_classes=support,
        held_missing_after_candidate_subsample=not bool((y==held).any()),episode_disabled=disable)
    if disable:
        details['episode_queries']=0
        return known,details
    logits=module(query,reference,labels,support,allowed)
    t=torch.tensor([support.index(int(c)) if int(c) in support else len(support) for c in y],device=y.device)
    valid=torch.tensor([int(c) not in support for c in y],device=y.device)
    for cid in support:valid|=(y==cid)&allowed[:,labels==cid].any(1)
    terms=[]
    for condition in [t<len(support),t==len(support)]:
        m=condition&valid
        if m.any():terms.append(F.cross_entropy(logits[m],t[m]))
    episodic=torch.stack(terms).mean() if terms else full.sum()*0
    details['episode_queries']=int(valid.sum())
    return known+episodic,details


def total_loss(out,teacher,model,c,bank,image,rng,episode):
    loss,stats=detection_loss(out,teacher,model,c)
    raw,_=model.region_quality(out,cxcywh(out['gt']));labels=out['semantic']
    dn=out['head'].get('dn_meta');n=dn['num_denoising_queries'] if dn else 0
    proposals=overlaps(xyxy(out['refined'][n:].detach()),out['gt']);i,j=torch.where(proposals>=.5);i=i+n
    if len(i):
        order=torch.as_tensor(rng.permutation(len(i))[:16],device=i.device);i,j=i[order],j[order]
        raw=torch.cat([raw,out['raw'][i]],0);labels=torch.cat([labels,out['semantic'][j]])
    if len(raw)>32:
        ids=torch.as_tensor(rng.choice(len(raw),32,replace=False),device=raw.device);raw=raw[ids];labels=labels[ids]
    group=bank.get('image_to_group',{}).get(image,image)
    allowed=torch.tensor([g!=group for g in bank.get('groups',bank['images'])],device=raw.device)[None].expand(len(raw),-1)
    identity,details=planned_identity_loss(model.relation,raw.detach(),labels,bank['raw'],bank['labels'],allowed,
        episode,disable=model.variant=='no_episode')
    stats.update(identity=float(identity.detach()),episode=details)
    return loss+c['identity_weight']*identity,stats
