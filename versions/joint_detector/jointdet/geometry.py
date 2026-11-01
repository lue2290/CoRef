import torch
from torch.nn import functional as F


def xyxy(cx):
    return torch.cat((cx[...,:2]-cx[...,2:]/2,cx[...,:2]+cx[...,2:]/2),-1)


def cxcywh(b):
    return torch.cat(((b[...,:2]+b[...,2:])/2,b[...,2:]-b[...,:2]),-1)


def overlaps(a,b):
    wh=(torch.minimum(a[:,None,2:],b[None,:,2:])-torch.maximum(a[:,None,:2],b[None,:,:2])).clamp_min(0)
    inter=wh.prod(-1)
    area=lambda x:(x[:,2:]-x[:,:2]).clamp_min(0).prod(-1)
    union=area(a)[:,None]+area(b)[None,:]-inter
    return inter/union.clamp_min(1e-8)


def paired_giou(a,b):
    inter=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp_min(0).prod(-1)
    area=lambda x:(x[:,2:]-x[:,:2]).clamp_min(0).prod(-1)
    union=area(a)+area(b)-inter
    outer=(torch.maximum(a[:,2:],b[:,2:])-torch.minimum(a[:,:2],b[:,:2])).clamp_min(0).prod(-1)
    return inter/union.clamp_min(1e-8)-(outer-union)/outer.clamp_min(1e-8)


def sample_grid(feature, boxes, side=3, expansion=1.0, padding_mode='zeros'):
    """Normalized VALID-image cxcywh; maps have no spatial padding. [B,N,P,C]."""
    t=(torch.arange(side,device=boxes.device,dtype=boxes.dtype)+0.5)/side-0.5
    yy,xx=torch.meshgrid(t,t,indexing='ij')
    offsets=torch.stack([xx,yy],-1).reshape(-1,2)
    pts=boxes[...,:2].unsqueeze(-2)
    pts=pts+boxes[...,2:].unsqueeze(-2)*offsets*expansion
    valid=((pts>=0)&(pts<=1)).all(-1)
    b,n,p,_=pts.shape
    sampled=F.grid_sample(feature,(2*pts-1).reshape(b,n*p,1,2),align_corners=False,
                          padding_mode=padding_mode,mode='bilinear')
    sampled=sampled.squeeze(-1).transpose(1,2).reshape(b,n,p,feature.shape[1])
    return sampled,valid,offsets


def roi_features(feature,boxes,side=2):
    # Four bilinear samples per bin, equivalent to ROIAlign sampling_ratio=2,
    # aligned=True on the unpadded feature map. Keep B5's Cx2x2 layout.
    sampled,_,_=sample_grid(feature,boxes.detach(),side*2,padding_mode='border')
    b,n,_,c=sampled.shape
    sampled=sampled.transpose(-1,-2).reshape(b,n,c,side,2,side,2).mean((4,6))
    return sampled.flatten(-3)
