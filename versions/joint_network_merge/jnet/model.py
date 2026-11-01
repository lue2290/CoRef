from copy import deepcopy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from jointdet.model import JointDetector,pixels,make_sample,pf_score
from jointdet.geometry import xyxy,cxcywh,roi_features


def subset(state,prefix):
    return {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}


def load_exact(module,state):
    module.load_state_dict(state,strict=True)


class NetworkMerge(JointDetector):
    """M0 and M1 differ ONLY in the known CE's connection to the PF query."""
    def __init__(self,root,c,active,arm,initial=None,pf=None,encoder=None,dino_dim=None):
        if arm not in ('M0','M1'):raise ValueError('Expected M0 or M1')
        self._merge_ready=False
        super().__init__(root,c,active,'J0',pf=pf,encoder=encoder,dino_dim=dino_dim)
        # Parent admission is exact before adding any new save-state names.
        if initial is not None:JointDetector.load_adapted(self,initial['P0'])
        self.arm=arm;self.block_known_to_pf=arm=='M1'
        self.tail_count=c['merge_tail_blocks'];self.split_at=len(self.dino.blocks)-self.tail_count
        if self.split_at<1:raise RuntimeError('The shared EVA prefix must be nonempty')
        if not all(hasattr(self.dino,k) for k in ('patch_embed','_pos_embed','norm_pre','num_prefix_tokens')):
            raise RuntimeError('Expected the audited EVA/DINOv3 forward API')
        self.open_blocks=nn.ModuleList([deepcopy(b) for b in self.dino.blocks[self.split_at:]])
        self.open_norm=deepcopy(self.dino.norm)
        if initial is not None:
            for j,block in enumerate(self.open_blocks):load_exact(block,subset(initial['S'],f'dino.blocks.{self.split_at+j}.'))
            load_exact(self.open_norm,subset(initial['S'],'dino.norm.'))
        final=len(self.pf.decoder.layers)-1
        self.open_cls=deepcopy(self.pf.bbox_head.cls_branches[final])
        self.open_ctr=deepcopy(self.pf.bbox_head.conf_branches[final])
        if initial is not None:
            load_exact(self.open_cls,subset(initial['S'],f'pf.bbox_head.cls_branches.{final}.'))
            load_exact(self.open_ctr,subset(initial['S'],f'pf.bbox_head.conf_branches.{final}.'))
            load_exact(self.quality,subset(initial['S'],'quality.'))
        self._merge_ready=True;self.configure_stage(False)

    def group_for(self,name):
        if not getattr(self,'_merge_ready',False):return JointDetector.group_for(self,name)
        if name.startswith(('open_blocks.','open_norm.')):return None
        if name.startswith(('open_cls.','open_ctr.')):return 'new'
        return JointDetector.group_for(self,name)

    def train(self,mode=True):
        if not getattr(self,'_merge_ready',False):return JointDetector.train(self,mode)
        nn.Module.train(self,mode)
        self.pf.eval();self.dino.eval();self.open_blocks.eval();self.open_norm.eval()
        if mode:
            self.pf.decoder.train();self.pf.bbox_head.train()
            self.known.train();self.quality.train();self.open_cls.train();self.open_ctr.train()
            if self._stage and self.c['unfreeze_backbones']:
                if hasattr(self.pf.backbone,'stages'):self.pf.backbone.stages[-1].train()
                for block in self.dino.blocks[self.split_at:]:block.train()
                self.dino.norm.train()
            self.pf.training=True
        return self

    def _tokens(self,image):
        """Mirror timm 1.0.20 Eva.forward_features, including axial/mixed RoPE.

        Frozen embedding/prefix run once. The two tails receive the same tokens
        and position tensors. Only the recognition tail can have a graph.
        """
        enc=self.dino
        with torch.no_grad():
            x=enc.patch_embed(image);x,rope=enc._pos_embed(x);x=enc.norm_pre(x)
            mixed=getattr(enc,'rope_mixed',False) and rope is not None
            for i in range(self.split_at):x=enc.blocks[i](x,rope=rope[i] if mixed else rope)
        prefix=x.detach()
        with torch.no_grad():
            opened=prefix
            for j,block in enumerate(self.open_blocks):
                opened=block(opened,rope=rope[self.split_at+j] if mixed else rope)
            opened=self.open_norm(opened)
        known=prefix
        for j in range(self.split_at,len(enc.blocks)):
            block=enc.blocks[j];position=rope[j] if mixed else rope
            use_cp=(self.c['gradient_checkpointing'] and self.training and torch.is_grad_enabled()
                    and any(p.requires_grad for p in block.parameters()))
            known=checkpoint(block,known,rope=position,use_reentrant=False) if use_cp else block(known,rope=position)
        known=enc.norm(known)
        return opened,known

    def feature_maps(self,image):
        opened,known=self._tokens(image);count=(image.shape[-2]//self.c['patch'])*(image.shape[-1]//self.c['patch'])
        def reshape(tokens):
            if tokens.shape[1]!=count+self.dino.num_prefix_tokens:raise RuntimeError('DINO prefix token layout differs')
            return tokens[:,-count:].transpose(1,2).reshape(1,self.dino_dim,image.shape[-2]//self.c['patch'],image.shape[-1]//self.c['patch'])
        return reshape(opened),reshape(known)

    @torch.no_grad()
    def verify_fork(self,image,atol=1e-5):
        was=self.training;self.eval();old_cp=self.dino.grad_checkpointing;self.dino.grad_checkpointing=False
        try:
            opened,known=self._tokens(image)
            ref_known=self.dino.forward_features(image)
            original=[self.dino.blocks[i] for i in range(self.split_at,len(self.dino.blocks))];norm=self.dino.norm
            try:
                for j,block in enumerate(self.open_blocks):self.dino.blocks[self.split_at+j]=block
                self.dino.norm=self.open_norm
                ref_open=self.dino.forward_features(image)
            finally:
                for j,block in enumerate(original):self.dino.blocks[self.split_at+j]=block
                self.dino.norm=norm
            error=dict(known_max_abs=float((known-ref_known).abs().max()),open_max_abs=float((opened-ref_open).abs().max()))
            if any(v>atol for v in error.values()):raise RuntimeError('Shared-prefix fork changes the audited full EVA forward: '+str(error))
            return dict(**error,shared_blocks=self.split_at,private_blocks_each=self.tail_count,prefix_tokens=self.dino.num_prefix_tokens)
        finally:
            self.dino.grad_checkpointing=old_cp;self.train(was)

    def forward_image(self,root,name,objects,device,flip=False):
        x,d,meta=pixels(root,name,self.c,device,flip);sample,norm_gt=make_sample(meta,objects,device)
        opened,known_map=self.feature_maps(d)
        head=self.pf.forward_transformer(self.pf.extract_feat(x),{},[sample])
        cls,boxes,ctr=self.pf.bbox_head(head['hidden_states'],head['references'],head['memory_text'],head['text_token_mask'])
        q=head['hidden_states'][-1];readout=cxcywh(xyxy(boxes[-1]).clamp(0,1)).detach()
        roi_open=roi_features(opened,readout,2);roi_known=roi_features(known_map,readout,2)
        pool=lambda z:z.reshape(1,boxes.shape[2],self.dino_dim,4).mean(-1)
        known_query=q.detach() if self.block_known_to_pf else q
        joint_known=torch.cat([known_query,pool(roi_known)],-1)
        joint_open=torch.cat([q,pool(roi_open)],-1)
        open_logits=self.open_cls(q,head['memory_text'],head['text_token_mask'])[0]
        open_ctr=self.open_ctr(q)[0,:,0].sigmoid()
        quality=self.quality(joint_open)[0,:,0].sigmoid()
        objectness=pf_score(open_logits,open_ctr)*(1-self.c['quality_alpha']+self.c['quality_alpha']*quality)
        return dict(cls=cls[:,0],boxes=boxes[:,0],ctr=ctr[:,0,:,0],known=self.known(joint_known)[0],quality=quality,
            open_cls=open_logits,open_ctr=open_ctr,open_obj=objectness,z=F.normalize(roi_open[0],dim=-1),
            head=head,gt=norm_gt,semantic=sample.gt_instances.semantic_labels,meta=meta,hidden=head['hidden_states'],shared_query=q)

    def save_names(self):
        return {n for n,_ in self.named_parameters() if self.group_for(n) is not None or n.startswith(('open_blocks.','open_norm.'))}

    def adapted_state(self):
        names=self.save_names()
        return {n:v.detach().cpu().clone() for n,v in self.named_parameters() if n in names}

    def load_adapted(self,state):
        if not getattr(self,'_merge_ready',False):return JointDetector.load_adapted(self,state)
        names=self.save_names()
        if names!=set(state):raise RuntimeError('Network merge checkpoint parameter set differs')
        own=dict(self.named_parameters())
        with torch.no_grad():
            for n,v in state.items():
                if own[n].shape!=v.shape:raise RuntimeError('Checkpoint shape differs: '+n)
                own[n].copy_(v)

    def frozen_open_digest(self):
        h=hashlib.sha256()
        for n,v in self.named_parameters():
            if n.startswith(('open_blocks.','open_norm.')):
                h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def parameter_report(self):
        names=self.save_names()
        return dict(counts={g:sum(p.numel() for n,p in self.named_parameters() if self.group_for(n)==g) for g in ('new','pf','backbone',None)},
            save_parameters=sum(p.numel() for n,p in self.named_parameters() if n in names),
            shared_DINO_blocks=self.split_at,private_DINO_blocks_each=self.tail_count,
            PF_paths=1,PF_candidate_sets=1,block_known_to_pf=self.block_known_to_pf)
