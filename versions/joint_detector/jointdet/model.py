from __future__ import annotations
import math
import types
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import sample_grid,roi_features,xyxy,cxcywh
from .protocol import read_json,load_torch


class SemanticFeedback(nn.Module):
    def __init__(self,dim=256,heads=4,context=1.25,gate=.01):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.position=nn.Linear(2,dim)
        self.kind=nn.Parameter(torch.randn(2,dim)*.01)
        self.norm=nn.LayerNorm(dim)
        self.gate=nn.Parameter(torch.tensor(math.log(gate/(1-gate))))
        self.context=context
        self.bypass=False

    def forward(self,q,feature,boxes):
        if self.bypass:
            return q
        parts,masks=[],[]
        for i,e in enumerate([1.,self.context]):
            x,valid,pos=sample_grid(feature,boxes.detach(),3,e)
            parts.append(x+self.position(pos)[None,None]+self.kind[i])
            masks.append(valid)
        tokens=torch.cat(parts,2);valid=torch.cat(masks,2)
        b,n,p,c=tokens.shape
        flat=tokens.reshape(b*n,p,c);mask=~valid.reshape(b*n,p)
        good=(~mask).any(1)
        # MultiheadAttention(all masked) produces NaN. Supply a dummy token,
        # then zero that entire object's residual, including projection bias.
        mask=mask.clone();mask[~good,0]=False
        update=self.attn(self.norm(q).reshape(b*n,1,c),flat,flat,
                         key_padding_mask=mask,need_weights=False)[0].reshape(b,n,c)
        update=update*good.reshape(b,n,1)
        return q+self.gate.sigmoid()*update


class QueryCapacityControl(nn.Module):
    """Parameter-matched residual that never reads semantic memory."""
    def __init__(self,hidden,gate):
        super().__init__();self.norm=nn.LayerNorm(256)
        self.mlp=nn.Sequential(nn.Linear(256,hidden),nn.GELU(),nn.Linear(hidden,256))
        self.gate=nn.Parameter(torch.tensor(math.log(gate/(1-gate))))
    def forward(self,q):return q+self.gate.sigmoid()*self.mlp(self.norm(q))


def decoder_forward(dec,query,value,key_padding_mask,self_attn_mask,reference_points,
                    spatial_shapes,level_start_index,valid_ratios,reg_branches,**kwargs):
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding,inverse_sigmoid
    states=[];refs=[reference_points]
    for i,layer in enumerate(dec.layers):
        rp=reference_points[:,:,None]*torch.cat([valid_ratios,valid_ratios],-1)[:,None]
        pos=dec.ref_point_head(coordinate_to_encoding(rp[:,:,0,:]))
        query=layer(query,query_pos=pos,value=value,key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,valid_ratios=valid_ratios,
                    reference_points=rp,**kwargs)
        owner=dec._joint_owner()
        if owner.feedback and str(i) in owner.bridges:
            query=owner.bridges[str(i)](query,owner._semantic_map,reference_points)
        elif owner.mlp_control and str(i) in owner.controls:
            query=owner.controls[str(i)](query)
        new=(reg_branches[i](query)+inverse_sigmoid(reference_points,eps=1e-3)).sigmoid()
        reference_points=new.detach()
        states.append(dec.norm(query));refs.append(new)
    return torch.stack(states),torch.stack(refs)


def build_pf(root,c,weights=True):
    from mmengine.config import Config
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS
    import mmdet.models.detectors.pf_rpn  # registry entry from portable vendor
    register_all_modules(init_default_scope=True)
    cfg=Config.fromfile(str(Path(root)/c['pf_config']))
    cfg.model.backbone.pop('init_cfg',None)
    cfg.model.pop('init_cfg',None)
    pf=MODELS.build(cfg.model)
    if weights:
        saved=load_torch(Path(root)/c['pf_weights'])
        state=saved.get('state_dict',saved)
        state={k.removeprefix('module.'):v for k,v in state.items()}
        missing,extra=pf.load_state_dict(state,strict=False)
        if missing or extra:
            raise RuntimeError(f'PF checkpoint mismatch missing={missing[:8]} extra={extra[:8]}')
    return pf


def pixels(root,name,c,device,flip=False):
    import mmcv
    with Image.open(Path(root)/'assets/data/images'/name) as raw:
        im=ImageOps.exif_transpose(raw).convert('RGB')
        if flip:im=ImageOps.mirror(im)
        rgb=np.asarray(im).copy();w,h=im.size
    # Match PF's mmcv/cv2 resize geometry rather than PIL approximations.
    pf_rgb=mmcv.imrescale(rgb,(c['pf_max_side'],c['pf_short_side']),interpolation='bilinear')
    ph,pw=pf_rgb.shape[:2]
    x=torch.from_numpy(pf_rgb.copy()).permute(2,0,1).float().to(device)
    mean=x.new_tensor([123.675,116.28,103.53])[:,None,None]
    std=x.new_tensor([58.395,57.12,57.375])[:,None,None]
    x=(x-mean)/std
    pad_h=math.ceil(ph/32)*32;pad_w=math.ceil(pw/32)*32
    x=F.pad(x,(0,pad_w-pw,0,pad_h-ph)).unsqueeze(0)
    from cil.model import resized_shape
    dw,dh=resized_shape(w,h,c['dino_max_side'],c['patch'])
    d=torch.from_numpy(np.asarray(im.resize((dw,dh),Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float().to(device)/255
    d=(d-d.new_tensor([.485,.456,.406])[:,None,None])/d.new_tensor([.229,.224,.225])[:,None,None]
    return x,d.unsqueeze(0),dict(ori_shape=(h,w),img_shape=(ph,pw),batch_input_shape=(pad_h,pad_w),
                                scale_factor=(pw/w,ph/h),flip=flip,img_path=str(Path(root)/'assets/data/images'/name),
                                dino_shape=(dh,dw))


def make_sample(meta,objects,device):
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    s=DetDataSample(metainfo={k:v for k,v in meta.items() if k!='dino_shape'})
    wh=torch.tensor([meta['ori_shape'][1],meta['ori_shape'][0]]*2,device=device)
    box=torch.tensor([o['box'] for o in objects],device=device,dtype=torch.float32).reshape(-1,4)
    if meta['flip']:
        box=box.clone();box[:,[0,2]]=wh[0]-box[:,[2,0]]
    norm=box/wh
    size=torch.tensor([meta['img_shape'][1],meta['img_shape'][0]]*2,device=device)
    g=InstanceData(bboxes=norm*size,labels=torch.zeros(len(box),device=device,dtype=torch.long))
    g.semantic_labels=torch.tensor([o['class_id'] for o in objects],device=device,dtype=torch.long)
    g.positive_maps=box.new_zeros(len(box),3)
    if len(box):g.positive_maps[:,1]=1
    g.text_token_mask=torch.ones(len(box),3,device=device,dtype=torch.bool)
    s.gt_instances=g
    s.token_positive_map=[1]
    return s,norm


class JointDetector(nn.Module):
    def __init__(self,root,c,active,method,pf=None,encoder=None,dino_dim=None):
        super().__init__()
        import weakref
        self.c=c;self.active=list(active);self.method=method
        self.feedback=method in ['J1','J3'];self.mlp_control=method=='JM'
        self.protect=method in ['J2','J3','JM']
        self.pf=pf if pf is not None else build_pf(root,c)
        if encoder is None:
            import timm
            from safetensors.torch import load_file
            folder=Path(root)/c['dino_folder']
            metadata=read_json(folder/'config.json')
            encoder=timm.create_model(metadata['architecture'],pretrained=False,num_classes=0)
            encoder.load_state_dict(load_file(str(folder/'model.safetensors')),strict=True)
            dino_dim=encoder.num_features
        self.dino=encoder;self.dino_dim=dino_dim
        self.projection=nn.Conv2d(dino_dim,256,1)
        self.bridges=nn.ModuleDict({str(i):SemanticFeedback(256,c['bridge_heads'],c['context_scale'],c['initial_gate'])
                                   for i in c['bridge_layers']})
        self.known=nn.Sequential(nn.Linear(256+dino_dim,256),nn.GELU(),nn.Linear(256,len(active)))
        self.quality=nn.Sequential(nn.Linear(256+dino_dim,128),nn.GELU(),nn.Linear(128,1))
        target=(sum(p.numel() for p in self.bridges.parameters())+sum(p.numel() for p in self.projection.parameters()))/len(self.bridges)
        hidden=max(1,round((target-512-1-256)/513))
        # Instantiate in every arm, after common heads, to keep initialization RNG identical.
        self.controls=nn.ModuleDict({str(i):QueryCapacityControl(hidden,c['initial_gate']) for i in c['bridge_layers']})
        # weakref prevents a parent-child module cycle and duplicated state_dict.
        self.pf.decoder._joint_owner=weakref.ref(self)
        self._original_decoder_forward=self.pf.decoder.forward
        self.pf.decoder.forward=types.MethodType(decoder_forward,self.pf.decoder)
        if c['gradient_checkpointing'] and hasattr(self.dino,'set_grad_checkpointing'):
            from timm.layers import set_reentrant_ckpt
            set_reentrant_ckpt(False);self.dino.set_grad_checkpointing(True)
        self._semantic_map=None
        self._stage=None
        self.configure_stage(False)

    def group_for(self,name):
        if name.startswith(('known.','quality.')):return 'new'
        if self.feedback and name.startswith(('projection.','bridges.')):return 'new'
        if self.mlp_control and name.startswith('controls.'):return 'new'
        if name.startswith(('pf.decoder.','pf.query_embedding.','pf.bbox_head.')):return 'pf'
        if self.c['unfreeze_backbones']:
            if name.startswith('pf.backbone.stages.3.'):return 'backbone'
            last=len(self.dino.blocks)-self.c['dino_last_blocks']
            if name.startswith('dino.norm.') or any(name.startswith(f'dino.blocks.{i}.') for i in range(last,len(self.dino.blocks))):
                return 'backbone'
        return None

    def configure_stage(self,unfreeze):
        self._stage=unfreeze
        for n,p in self.named_parameters():
            group=self.group_for(n)
            p.requires_grad_(group is not None and (group!='backbone' or unfreeze))
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        # Keep all frozen stochastic paths deterministic; do not disable autograd
        # through frozen PF encoder when its input backbone stage is trainable.
        self.pf.eval();self.dino.eval()
        if mode:
            self.pf.decoder.train();self.pf.bbox_head.train()
            if self._stage and self.c['unfreeze_backbones']:
                if hasattr(self.pf.backbone,'stages'):self.pf.backbone.stages[-1].train()
                for block in self.dino.blocks[-self.c['dino_last_blocks']:]:block.train()
            # PF.training enables DN without re-enabling frozen encoder dropout.
            self.pf.training=True
        return self

    def forward_image(self,root,name,objects,device,flip=False):
        x,d,meta=pixels(root,name,self.c,device,flip)
        sample,norm_gt=make_sample(meta,objects,device)
        enabled=any(p.requires_grad for p in self.dino.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and enabled):
            tokens=self.dino.forward_features(d)
        count=d.shape[-2]//16*(d.shape[-1]//16)
        if tokens.ndim!=3 or tokens.shape[1]!=count+self.dino.num_prefix_tokens:
            raise RuntimeError('Unexpected DINO token layout')
        feature=tokens[:,-count:].transpose(1,2).reshape(1,self.dino_dim,d.shape[-2]//16,d.shape[-1]//16)
        self._semantic_map=self.projection(feature) if self.feedback else None
        try:
            head=self.pf.forward_transformer(self.pf.extract_feat(x),{},[sample])
        finally:
            self._semantic_map=None
        cls,boxes,ctr=self.pf.bbox_head(head['hidden_states'],head['references'],head['memory_text'],head['text_token_mask'])
        # Head's real bbox, not decoder final reference, supplies all ROI readouts.
        readout_boxes=cxcywh(xyxy(boxes[-1]).clamp(0,1)).detach()
        roi=roi_features(feature,readout_boxes,2)
        pooled=roi.reshape(1,boxes.shape[2],self.dino_dim,4).mean(-1)
        joint=torch.cat([head['hidden_states'][-1],pooled],-1)
        return dict(cls=cls[:,0],boxes=boxes[:,0],ctr=ctr[:,0,:,0],
                    known=self.known(joint)[0],quality=self.quality(joint)[0,:,0].sigmoid(),
                    z=F.normalize(roi[0],dim=-1),head=head,gt=norm_gt,
                    semantic=sample.gt_instances.semantic_labels,meta=meta,hidden=head['hidden_states'])

    def adapted_state(self):
        # Include future-unfrozen parameters even before their first update.
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if self.group_for(n) is not None}

    def load_adapted(self,state):
        own=dict(self.named_parameters())
        expected={n for n in own if self.group_for(n) is not None}
        if expected!=set(state):raise RuntimeError('Checkpoint parameter set changed')
        with torch.no_grad():
            for n,v in state.items():own[n].copy_(v)

    def parameter_report(self):
        result={str(g):sum(p.numel() for n,p in self.named_parameters() if self.group_for(n)==g)
                for g in ['new','pf','backbone',None]}
        a=sum(p.numel() for p in self.projection.parameters())+sum(p.numel() for p in self.bridges.parameters())
        b=sum(p.numel() for p in self.controls.parameters())
        result.update(semantic_feedback_parameters=a,query_capacity_control_parameters=b,
                      relative_capacity_difference=(b-a)/a)
        return result


def pf_score(cls,ctr,mode='both'):
    # Exact upstream head convention: ctr has ALREADY been sigmoid once.
    a=cls.sigmoid().max(-1).values;b=ctr.sigmoid()
    if mode=='both':return a+b
    if mode=='cls':return a
    if mode=='fuse':return a.pow(.4)*b.pow(.6)
    raise ValueError(mode)
