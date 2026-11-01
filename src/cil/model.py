from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from .common import read_json

class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base=base
        for p in base.parameters():p.requires_grad_(False)
        self.A=nn.Parameter(torch.empty(rank,base.in_features))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        self.scale=alpha/rank
        self.drop=nn.Dropout(dropout)

    @property
    def weight(self):
        return self.base.weight+self.scale*(self.B@self.A)

    @property
    def bias(self):
        return self.base.bias

    def forward(self,x):
        return self.base(x)+self.scale*F.linear(F.linear(self.drop(x),self.A),self.B)

def inject_lora(encoder,cfg):
    names=[]
    for i in range(max(0,len(encoder.blocks)-cfg["lora_last_blocks"]),len(encoder.blocks)):
        block=encoder.blocks[i]
        for name in ["qkv","proj"]:
            module=getattr(block.attn,name)
            if not isinstance(module,nn.Linear):raise TypeError(f"Expected linear blocks.{i}.attn.{name}")
            setattr(block.attn,name,LoRALinear(module,cfg["lora_rank"],cfg["lora_alpha"],cfg["lora_dropout"]))
            names.append(f"blocks.{i}.attn.{name}")
    if not names:raise RuntimeError("No LoRA modules were inserted")
    return names

def resized_shape(width,height,max_side,patch):
    ratio=max_side/max(width,height)
    return max(patch,round(width*ratio/patch)*patch),max(patch,round(height*ratio/patch)*patch)

def preprocess(path,max_side,patch,mean,std):
    with Image.open(path) as raw:
        im=ImageOps.exif_transpose(raw).convert("RGB")
        w,h=im.size
        nw,nh=resized_shape(w,h,max_side,patch)
        im=im.resize((nw,nh),Image.Resampling.BICUBIC)
        x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
    x=(x-torch.tensor(mean).view(3,1,1))/torch.tensor(std).view(3,1,1)
    return x.unsqueeze(0),[nw/w,nh/h,nw/w,nh/h],(nw,nh)

def pool_regions(feature,boxes,patch,width,height):
    boxes=boxes.float()
    feature=feature.float()
    a=roi_align(feature,[boxes],(2,2),1/patch,2,True).flatten(1)
    center=(boxes[:,:2]+boxes[:,2:])/2
    half=(boxes[:,2:]-boxes[:,:2])*0.75
    lo=(center-half).clamp_min(0)
    hi=torch.minimum(center+half,boxes.new_tensor([width,height]))
    b=roi_align(feature,[torch.cat([lo,hi],1)],(1,1),1/patch,2,True).flatten(1)
    return torch.cat([a,b],1)

class CILModel(nn.Module):
    def __init__(self,root,cfg,class_ids,method,encoder=None,dim=None):
        super().__init__()
        self.cfg,self.class_ids,self.method=cfg,list(class_ids),method
        if encoder is None:
            import timm
            from safetensors.torch import load_file
            folder=Path(root)/"assets/models/dinov3_large"
            meta=read_json(folder/"config.json")
            if meta["architecture"]!=cfg["architecture"] or meta["pretrained_cfg"]["tag"]!="lvd1689m":
                raise ValueError("Wrong backbone architecture/pretraining")
            encoder=timm.create_model(cfg["architecture"],pretrained=False,num_classes=0)
            encoder.load_state_dict(load_file(str(folder/"model.safetensors")),strict=True)
            dim=int(encoder.num_features)
            self.mean=meta["pretrained_cfg"]["mean"]
            self.std=meta["pretrained_cfg"]["std"]
        else:
            self.mean,self.std=[0.485,0.456,0.406],[0.229,0.224,0.225]
        self.encoder=encoder
        self.dim=dim
        for p in self.encoder.parameters():p.requires_grad_(False)
        self.lora_names=inject_lora(encoder,cfg) if method in ["A1","A2"] else []
        if cfg.get("gradient_checkpointing") and self.lora_names and hasattr(encoder,"set_grad_checkpointing"):
            # timm1.0.20 defaults to non-reentrant checkpointing, required for frozen prefix inputs.
            from timm.layers import set_reentrant_ckpt
            set_reentrant_ckpt(False)
            encoder.set_grad_checkpointing(True)
        d=4*dim
        self.semantic=nn.Sequential(nn.Linear(d,cfg["head_hidden"]),nn.GELU(),nn.Linear(cfg["head_hidden"],d))
        nn.init.zeros_(self.semantic[-1].weight)
        nn.init.zeros_(self.semantic[-1].bias)
        self.class_weight=nn.Parameter(torch.randn(len(class_ids),d)*0.01)
        self.class_log_scale=nn.Parameter(torch.tensor(math.log(10.0)))
        self.quality=nn.Sequential(nn.Linear(5*dim,cfg["head_hidden"]),nn.GELU(),nn.Linear(cfg["head_hidden"],1))
        self.novel_bias=nn.Parameter(torch.tensor(0.0))
        self.novel_log_temp=nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def forward_image(self,root,image_name,boxes,device):
        pixels,scale,(w,h)=preprocess(Path(root)/"assets/data/images"/image_name,self.cfg["max_side"],self.cfg["patch"],self.mean,self.std)
        pixels=pixels.to(device)
        ctx=torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" and self.cfg["precision"]=="bf16" else nullcontext()
        with ctx:
            if self.lora_names:
                tokens=self.encoder.forward_features(pixels)
            else:
                with torch.no_grad():tokens=self.encoder.forward_features(pixels)
            count=(h//self.cfg["patch"])*(w//self.cfg["patch"])
            if tokens.ndim!=3 or tokens.shape[1]!=count+self.encoder.num_prefix_tokens:
                raise RuntimeError("Unexpected DINOv3 token layout")
            feat=tokens[:,-count:].reshape(1,h//self.cfg["patch"],w//self.cfg["patch"],self.dim).permute(0,3,1,2).contiguous()
            b=torch.as_tensor(np.asarray(boxes).reshape(-1,4),device=device,dtype=torch.float32)*torch.tensor(scale,device=device)
            pooled=pool_regions(feat,b,self.cfg["patch"],w,h)
            x=pooled[:,:4*self.dim]
            z=F.normalize(x+self.semantic(x),dim=1)
            logits=z@F.normalize(self.class_weight,dim=1).T*self.class_log_scale.clamp(max=math.log(100)).exp()
            quality=self.quality(pooled).squeeze(1).sigmoid()
        return z.float(),logits.float(),quality.float()

    def adapted_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_adapted(self,state):
        expected={n for n,p in self.named_parameters() if p.requires_grad}
        if expected!=set(state):raise RuntimeError("Checkpoint trainable parameter set does not match")
        own=dict(self.named_parameters())
        with torch.no_grad():
            for n,v in state.items():own[n].copy_(v)

    def parameter_report(self):
        return {"total":sum(p.numel() for p in self.parameters()),
                "trainable":sum(p.numel() for p in self.parameters() if p.requires_grad),"lora_modules":self.lora_names}
