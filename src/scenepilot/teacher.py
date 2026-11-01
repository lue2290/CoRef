from pathlib import Path
import numpy as np,time
from .protocol import read,write,sha,fp,progress,space

def prepare_base(root,view,cfg,s,device,folder,report,smoke=False):
 import torch
 from jointdet.model import build_pf
 from jointdet.teacher import extract
 from .policy import pair_views,fit_cutoffs,materialize
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);dest=folder/'manifest.json'
 identity=fp(dict(split=s['identity'],fit=s['fit_images'],config=cfg['model'],smoke=smoke))
 if dest.exists():
  m=read(dest)
  if m['identity']!=identity or any(sha(folder/n)!=h for n,h in m['files'].items()):raise RuntimeError('Base teacher cache mismatch')
  return folder
 model=build_pf(view,cfg['model']).to(device).eval();pairs={};gt={};start=time.monotonic()
 from PIL import Image
 for j,name in enumerate(s['fit_images']):
  with torch.inference_mode():a=extract(model,view,name,cfg['model'],device,False);b=extract(model,view,name,cfg['model'],device,True)
  raw={k:np.concatenate([a[k],b[k]]) for k in ['boxes','p','ctr','logits']};raw['stable']=np.zeros(len(raw['p']),bool)
  from scipy.optimize import linear_sum_assignment
  from .policy import overlaps,xyxy
  ov=overlaps(xyxy(a['boxes']),xyxy(b['boxes']));ii,jj=linear_sum_assignment(-ov);raw['stable'][ii]=ov[ii,jj]>=cfg['model']['teacher_consistency_iou']
  pairs[name]=pair_views(raw,cfg['model'])
  with Image.open(view/'assets/data/images'/name) as im:w,h=im.size
  gt[name]=np.asarray([o['box'] for o in s['fit_objects'] if o['image']==name],np.float32).reshape(-1,4)/np.array([w,h,w,h])
  progress(report,j+1,len(s['fit_images']),'PF teacher original+flip',start)
 del model;torch.cuda.empty_cache()
 if smoke:cutoffs=dict(high_response=.5,geometry_centerness=.5,training_positive_matches=None,smoke_only_unfitted=True)
 else:cutoffs=fit_cutoffs(pairs,gt,cfg['model'])
 files={}
 for name,pair in pairs.items():
  path=folder/(fp(name)[:24]+'.npz');space(root,cfg,128*1024)
  with path.open('wb') as f:np.savez_compressed(f,**materialize(pair,cutoffs,cfg['model']))
  files[path.name]=sha(path)
 write(dest,dict(identity=identity,files=files,cutoffs=cutoffs,fit_images=s['fit_images'],hard_background_labels=False));return folder

def base_target(folder,name,device,flip):
 import torch
 m=read(Path(folder)/'manifest.json')
 if name not in m['fit_images']:raise PermissionError('Non-fit teacher access')
 path=Path(folder)/(fp(name)[:24]+'.npz')
 if sha(path)!=m['files'][path.name]:raise RuntimeError('Teacher file changed')
 with np.load(path,allow_pickle=False) as z:r={k:torch.from_numpy(z[k].copy()).to(device) for k in z.files}
 if flip:r['boxes'][:,0]=1-r['boxes'][:,0]
 return r

def prepare_s(root,view,cfg,s,device,model,folder,report,parent_sha):
 import torch
 from jointdet.scoring import raw_arrays
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);dest=folder/'manifest.json'
 if dest.exists():
  m=read(dest)
  if m['parent_sha']!=parent_sha or m['fit_images']!=s['fit_images'] or any(sha(folder/n)!=h for n,h in m['files'].items()):raise RuntimeError('S target cache mismatch')
  return folder
 model.eval();files={};values=[];start=time.monotonic()
 for j,name in enumerate(s['fit_images']):
  for flip in [False,True]:
   with torch.inference_mode():
    out=model.forward_image(view,name,[],device,flip);raw=raw_arrays(out,cfg['model'])
   wh=np.array([out['meta']['ori_shape'][1],out['meta']['ori_shape'][0]]*2);bb=raw['boxes']/wh;boxes=np.concatenate([(bb[:,:2]+bb[:,2:])/2,bb[:,2:]-bb[:,:2]],1);obj=raw['pf']*(1-cfg['model']['quality_alpha']+cfg['model']['quality_alpha']*raw['q']);path=folder/(fp([name,flip])[:24]+'.npz')
   with path.open('wb') as f:np.savez_compressed(f,boxes=boxes.astype(np.float32),objectness=obj.astype(np.float32))
   files[path.name]=sha(path);values.extend(obj.tolist());del out
  progress(report,j+1,len(s['fit_images']),'new S teacher actual two views',start)
 write(dest,dict(parent_sha=parent_sha,fit_images=s['fit_images'],files=files,low_threshold=float(np.quantile(values,cfg['model']['teacher_low_quantile'],method='lower')),hard_labels=False));return folder

def s_target(folder,name,device,flip):
 import torch
 m=read(Path(folder)/'manifest.json')
 if name not in m['fit_images']:raise PermissionError('Non-fit S teacher access')
 path=Path(folder)/(fp([name,flip])[:24]+'.npz')
 if sha(path)!=m['files'][path.name]:raise RuntimeError('S teacher changed')
 with np.load(path,allow_pickle=False) as z:r={k:torch.from_numpy(z[k].copy()).to(device) for k in z.files}
 r['low']=r['objectness']<=m['low_threshold'];return r
