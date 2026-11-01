"""One support-conditioned classifier handles known identity and rejection."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ReferenceRelation(nn.Module):
    def __init__(self, raw_dim=5120, dim=128, neighbors=3):
        super().__init__(); self.neighbors=neighbors
        self.embed=nn.Sequential(nn.LayerNorm(raw_dim),nn.Linear(raw_dim,256),nn.GELU(),nn.Linear(256,dim))
        self.pair=nn.Sequential(nn.Linear(dim*2+1,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.pair[-1].weight);nn.init.zeros_(self.pair[-1].bias)
        self.log_scale=nn.Parameter(torch.tensor(math.log(10.)))
        self.boundary=nn.Parameter(torch.tensor(.7))
        self.reject=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.reject[-1].weight);nn.init.zeros_(self.reject[-1].bias)

    def encode(self,raw): return F.normalize(self.embed(raw),dim=-1)

    def forward(self,query,reference,labels,active,allowed=None):
        if not len(active): raise ValueError('Empty support class set')
        q=self.encode(query);r=self.encode(reference);sim=q@r.T
        scores=[];valids=[];disp=[]
        for cid in active:
            indices=torch.nonzero(labels==cid).flatten()
            if not len(indices): raise ValueError('Missing GT support class '+str(cid))
            values=sim[:,indices]
            mask=torch.ones_like(values,dtype=torch.bool) if allowed is None else allowed[:,indices]
            k=min(self.neighbors,len(indices));v,j=values.masked_fill(~mask,-1e4).topk(k,dim=1)
            good=mask.gather(1,j);neighbors=r[indices[j]];qe=q[:,None,:].expand_as(neighbors)
            residual=self.pair(torch.cat([abs(qe-neighbors),qe*neighbors,(qe*neighbors).sum(-1,keepdim=True)],-1)).squeeze(-1)
            count=good.sum(1);den=count.clamp_min(1)
            mean=(v.masked_fill(~good,0)*good).sum(1)/den
            score=((self.log_scale.exp().clamp(max=50)*(v-self.boundary)+residual).masked_fill(~good,0)).sum(1)/den
            variance=((v-mean[:,None]).square().masked_fill(~good,0)).sum(1)/den
            scores.append(score.masked_fill(count==0,-1e4));valids.append(count>0);disp.append(variance)
        known=torch.stack(scores,1);valid=torch.stack(valids,1)
        top=known.topk(min(2,len(active)),dim=1).values
        first=top[:,0];gap=first-top[:,1] if len(active)>1 else torch.zeros_like(first)
        evidence=torch.stack([first.clamp(-50,50),gap.clamp(0,50),
                              valid.float().mean(1),torch.stack(disp,1).mean(1),
                              first.new_full(first.shape,math.log1p(len(active)))],1)
        reject=self.reject(evidence).squeeze(1)
        return torch.cat([known,reject[:,None]],1)


def episode_loss(module,query,y,reference,labels,allowed,rng):
    """Outer query images are excluded; held identities never appear in episode support."""
    all_classes=sorted(set(labels.tolist()))
    if len(all_classes)<2: raise ValueError('Need >=2 legal known classes for leave-class episodes')
    # Full known recognition and rejection episodes share exactly the same scorer.
    full=module(query,reference,labels,all_classes,allowed)
    target=torch.tensor([all_classes.index(int(c)) for c in y],device=y.device)
    own=torch.stack([allowed[:,labels==c].any(1) for c in all_classes],1)
    legal=own[torch.arange(len(y),device=y.device),target]
    loss=F.cross_entropy(full[legal],target[legal]) if legal.any() else full.sum()*0
    seen=sorted(set(y.tolist()));held=int(rng.choice(seen))
    candidates=[c for c in all_classes if c!=held]
    # Vary support cardinality without using a test/development label.
    size=int(rng.integers(1,len(candidates)+1))
    active=sorted(map(int,rng.choice(candidates,size=size,replace=False)))
    logits=module(query,reference,labels,active,allowed)
    t=torch.tensor([active.index(int(c)) if int(c) in active else len(active) for c in y],device=y.device)
    valid=torch.tensor([int(c) not in active for c in y],device=y.device)
    for j,c in enumerate(active): valid|=(y==c)&allowed[:,labels==c].any(1)
    # Equal known/held contributions, not dominated by the numerous held classes.
    terms=[]
    for condition in [t<len(active),t==len(active)]:
        m=condition&valid
        if m.any():terms.append(F.cross_entropy(logits[m],t[m]))
    episodic=torch.stack(terms).mean() if terms else full.sum()*0
    return loss+episodic,dict(known_queries=int(legal.sum()),held_class=held,support_classes=active,
                            episode_queries=int(valid.sum()))
