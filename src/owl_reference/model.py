import math
import torch
from torch import nn
from torch.nn import functional as F
from jointdet.geometry import roi_features,cxcywh,xyxy
from refnet.relation import ReferenceRelation


class RawRelation(nn.Module):
    def __init__(self,initial):
        super().__init__();self.neighbors=initial.neighbors
        self.log_scale=initial.log_scale;self.boundary=initial.boundary;self.reject=initial.reject

    def encode(self,raw):return F.normalize(raw,dim=-1)

    def forward(self,query,reference,labels,active,allowed=None):
        if not active:raise ValueError('Empty support classes')
        sim=self.encode(query)@self.encode(reference).T;scores=[];valids=[];disp=[]
        for cid in active:
            ids=torch.nonzero(labels==cid).flatten()
            if not len(ids):raise ValueError('Missing legal GT reference class '+str(cid))
            values=sim[:,ids];mask=torch.ones_like(values,dtype=torch.bool) if allowed is None else allowed[:,ids]
            v,j=values.masked_fill(~mask,-1e4).topk(min(self.neighbors,len(ids)),dim=1)
            good=mask.gather(1,j);count=good.sum(1);den=count.clamp_min(1)
            mean=v.masked_fill(~good,0).sum(1)/den
            score=(self.log_scale.exp().clamp(max=50)*(v-self.boundary)).masked_fill(~good,0).sum(1)/den
            variance=(v-mean[:,None]).square().masked_fill(~good,0).sum(1)/den
            scores.append(score.masked_fill(count==0,-1e4));valids.append(count>0);disp.append(variance)
        known=torch.stack(scores,1);top=known.topk(min(2,len(active)),dim=1).values;first=top[:,0]
        gap=first-top[:,1] if len(active)>1 else torch.zeros_like(first)
        evidence=torch.stack([first.clamp(-50,50),gap.clamp(0,50),torch.stack(valids,1).float().mean(1),
            torch.stack(disp,1).mean(1),first.new_full(first.shape,math.log1p(len(active)))],1)
        return torch.cat([known,self.reject(evidence)],1)


class Head(nn.Module):
    def __init__(self,arm,raw_dim,cfg):
        super().__init__()
        if arm not in ['O0','O1']:raise ValueError('Only OWL O0/O1 are trained')
        initial=ReferenceRelation(raw_dim,cfg['identity_dim'],cfg['neighbors'])
        # Both constructors consume the same RNG and share fresh rejection
        # initialization. No DINO-trained relation parameters are copied.
        self.arm=arm;self.relation=RawRelation(initial) if arm=='O0' else initial

    def forward(self,raw,bank,active,allowed=None):
        return self.relation(raw,bank['raw'],bank['labels'],active,allowed)


def identity_loss(head,q,y,bank,allowed,episode):
    active=sorted(set(bank['labels'].tolist()));full=head(q,bank,active,allowed)
    target=torch.tensor([active.index(int(v)) for v in y],device=y.device)
    legal=torch.stack([allowed[:,bank['labels']==c].any(1) for c in active],1)[torch.arange(len(y),device=y.device),target]
    known=F.cross_entropy(full[legal],target[legal]) if legal.any() else full.sum()*0
    support=episode['support_classes'];held=episode['held_class']
    if held in support or not support or not set(support)<=set(active):raise RuntimeError('Invalid shared episodic plan')
    logits=head(q,bank,support,allowed)
    t=torch.tensor([support.index(int(v)) if int(v) in support else len(support) for v in y],device=y.device)
    valid=torch.tensor([int(v) not in support for v in y],device=y.device)
    for cid in support:valid|=(y==cid)&allowed[:,bank['labels']==cid].any(1)
    terms=[]
    for condition in [t<len(support),t==len(support)]:
        m=valid&condition
        if m.any():terms.append(F.cross_entropy(logits[m],t[m]))
    episodic=torch.stack(terms).mean() if terms else full.sum()*0
    return known+episodic,dict(known_ce=float(known.detach()),episodic_ce=float(episodic.detach()),
        known_queries=int(legal.sum()),episode_queries=int(valid.sum()))


def read_regions(feature,boxes,context_scale=1.25):
    boxes=boxes.detach();spatial=roi_features(feature,boxes[None],2)[0]
    context=boxes.clone();context[:,2:]*=context_scale;context=cxcywh(xyxy(context).clamp(0,1))
    return torch.cat([spatial,roi_features(feature,context[None],1)[0]],-1)


class FrozenVisual(nn.Module):
    def __init__(self,detector):
        super().__init__();self.vision=detector.owlvit.vision_model;self.norm=detector.layer_norm
        self.channels=detector.config.vision_config.hidden_size;self.patch=detector.config.vision_config.patch_size
        self.requires_grad_(False);self.eval()

    @torch.inference_mode()
    def forward(self,pixels):
        # Same non-projected detection feature map as HF image_embedder.
        out=self.vision(pixel_values=pixels,return_dict=True)
        tokens=self.vision.post_layernorm(out.last_hidden_state)
        patches=self.norm(tokens[:,1:,:]*tokens[:,:1,:])
        h,w=pixels.shape[-2]//self.patch,pixels.shape[-1]//self.patch
        if patches.shape[1]!=h*w or patches.shape[-1]!=self.channels:raise RuntimeError('Unexpected OWL detection token layout')
        return patches.transpose(1,2).reshape(len(pixels),self.channels,h,w)


def load_visual(root,cfg,device):
    from pathlib import Path
    from transformers import OwlViTForObjectDetection,OwlViTImageProcessor
    folder=Path(root)/cfg['owl_folder']
    detector,info=OwlViTForObjectDetection.from_pretrained(str(folder),local_files_only=True,use_safetensors=False,output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError('Official OWL detection checkpoint does not load exactly: '+str(info))
    processor=OwlViTImageProcessor.from_pretrained(str(folder),local_files_only=True)
    if processor.do_center_crop or not processor.do_resize:raise RuntimeError('OWL experiment fixes direct square resize, no crop/padding')
    visual=FrozenVisual(detector);del detector
    # The text encoder/class head/box head never move to GPU or run forward.
    return visual.to(device),processor
