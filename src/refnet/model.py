import torch
from torch import nn
from torch.nn import functional as F
from jointdet.model import JointDetector,pixels,make_sample
from jointdet.geometry import roi_features,xyxy,cxcywh
from .relation import ReferenceRelation


def region(feature,boxes):
    boxes=boxes.detach()
    spatial=roi_features(feature,boxes[None],2)[0]
    context=boxes.clone();context[:,2:]*=1.25
    context=cxcywh(xyxy(context).clamp(0,1))
    return torch.cat([spatial,roi_features(feature,context[None],1)[0]],-1)


class ReferenceDetector(JointDetector):
    """Class-agnostic semantic feedback, geometric refinement, shared reference identity."""
    def __init__(self,root,c,active,pf=None,encoder=None,dino_dim=None):
        super().__init__(root,c,active,'J1',pf,encoder,dino_dim)
        del self.known,self.quality,self.controls
        raw=self.dino_dim*5
        self.relation=ReferenceRelation(raw,c['identity_dim'],c['relation_neighbors'])
        self.refine=nn.Sequential(nn.LayerNorm(raw+256),nn.Linear(raw+256,128),nn.GELU(),nn.Linear(128,4))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)
        self.loc_quality=nn.Sequential(nn.LayerNorm(raw+256),nn.Linear(raw+256,128),nn.GELU(),nn.Linear(128,1))
        self.configure_stage(False)

    def parameter_report(self):
        return {str(g):sum(p.numel() for n,p in self.named_parameters() if self.group_for(n)==g)
                for g in ['new','pf',None]}

    def group_for(self,name):
        if name.startswith(('relation.','refine.','loc_quality.','projection.','bridges.')):return 'new'
        if name.startswith(('pf.decoder.','pf.query_embedding.','pf.bbox_head.')):return 'pf'
        return None

    def configure_stage(self,unfreeze):
        self._stage=unfreeze
        for n,p in self.named_parameters():
            g=self.group_for(n);p.requires_grad_(g=='new' or (g=='pf' and unfreeze))
        self.train(self.training)

    def feature_map(self,d):
        # Frozen DINO guarantees that stored GT reference features never become stale.
        with torch.no_grad():tokens=self.dino.forward_features(d)
        h,w=d.shape[-2]//16,d.shape[-1]//16
        if tokens.shape[1] != h*w+self.dino.num_prefix_tokens:raise RuntimeError('DINO token layout mismatch')
        return tokens[:,-h*w:].transpose(1,2).reshape(1,self.dino_dim,h,w)

    def forward_image(self,root,name,objects,device,flip=False):
        x,d,meta=pixels(root,name,self.c,device,flip)
        sample,gt=make_sample(meta,objects,device);feature=self.feature_map(d)
        self._semantic_map=self.projection(feature)
        try:head=self.pf.forward_transformer(self.pf.extract_feat(x),{},[sample])
        finally:self._semantic_map=None
        cls,boxes,ctr=self.pf.bbox_head(head['hidden_states'],head['references'],head['memory_text'],head['text_token_mask'])
        base=cxcywh(xyxy(boxes[-1,0]).clamp(0,1));raw=region(feature,base)
        hidden=head['hidden_states'][-1,0]
        delta=.1*torch.tanh(self.refine(torch.cat([raw,hidden],-1)))
        refined=torch.cat([(base[:,:2]+delta[:,:2]).clamp(0,1),
                            (base[:,2:]*torch.exp(delta[:,2:])).clamp(1e-4,1)],-1)
        refined=cxcywh(xyxy(refined).clamp(0,1))
        final_raw=region(feature,refined)
        quality=self.loc_quality(torch.cat([final_raw,hidden.detach()],-1)).squeeze(-1)
        return dict(cls=cls[:,0],boxes=boxes[:,0],ctr=ctr[:,0,:,0],head=head,gt=gt,
                    semantic=sample.gt_instances.semantic_labels,meta=meta,hidden=hidden,
                    refined=refined,raw=final_raw,quality_logit=quality,feature=feature)

    def region_quality(self,out,boxes):
        from jointdet.geometry import overlaps
        with torch.no_grad():ids=overlaps(xyxy(boxes),xyxy(out['refined'].detach())).argmax(1)
        raw=region(out['feature'],boxes)
        q=self.loc_quality(torch.cat([raw,out['hidden'][ids].detach()],-1)).squeeze(-1)
        return raw,q
