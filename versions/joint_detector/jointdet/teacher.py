from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .model import build_pf,pixels,make_sample
from .geometry import xyxy,cxcywh,overlaps
from .protocol import digest,fingerprint,read_json,write_json,Progress


def cache_identity(root,c,s):
    return fingerprint(dict(pf=digest(Path(root)/c['pf_weights']),split=s['identity'],
        recipe={k:c[k] for k in ['pf_short_side','pf_max_side','teacher_consistency_iou']}))


@torch.no_grad()
def extract(pf,root,name,c,device,flip):
    x,_,meta=pixels(root,name,c,device,flip)
    sample,_=make_sample(meta,[],device)
    head=pf.forward_transformer(pf.extract_feat(x),{},[sample])
    cls,boxes,ctr=pf.bbox_head(head['hidden_states'],head['references'],head['memory_text'],head['text_token_mask'])
    box=cxcywh(xyxy(boxes[-1,0].detach()).clamp(0,1))
    if flip:
        box=box.clone();box[:,0]=1-box[:,0]
    token=1 if cls.shape[-1]>1 else 0
    return dict(boxes=box.cpu().numpy(),logits=cls[-1,0,:,token].cpu().numpy(),
                p=cls[-1,0,:,token].sigmoid().cpu().numpy(),ctr=ctr[-1,0,:,0].cpu().numpy(),token=token)


def prepare_teacher(root,c,s,device,progress=None):
    root=Path(root);identity=cache_identity(root,c,s)
    folder=root/'prepared/joint_detector/teacher'/identity
    done=folder/'manifest.json'
    if done.exists():
        m=read_json(done)
        if m['split_identity']!=s['identity']:raise RuntimeError('Teacher split mismatch')
        for n,v in m['files'].items():
            if digest(folder/n)!=v:raise RuntimeError('Teacher cache changed')
        return folder
    folder.mkdir(parents=True,exist_ok=True)
    pf=build_pf(root,c).to(device).eval()
    files={};known_responses=[];bar=Progress(len(s['fit_images']),'teacher')
    by={n:[o for o in s['fit_objects'] if o['image']==n] for n in s['fit_images']}
    for i,name in enumerate(s['fit_images']):
        path=folder/(fingerprint(name)[:20]+'.npz')
        a=extract(pf,root,name,c,device,False);b=extract(pf,root,name,c,device,True)
        ov=overlaps(xyxy(torch.from_numpy(a['boxes'])),xyxy(torch.from_numpy(b['boxes']))).numpy()
        ii,jj=linear_sum_assignment(-ov)
        stable=np.zeros(len(a['boxes']),bool);stable[ii]=ov[ii,jj]>=c['teacher_consistency_iou']
        # Retain view union for ignore; high-quality stable targets use view A.
        from PIL import Image,ImageOps
        with Image.open(root/'assets/data/images'/name) as im:w,h=ImageOps.exif_transpose(im).size
        gt=torch.tensor([o['box'] for o in by[name]],dtype=torch.float32)/torch.tensor([w,h,w,h])
        g=overlaps(xyxy(torch.from_numpy(a['boxes'])),gt).numpy()
        for col in range(g.shape[1]):
            index=int(g[:,col].argmax())
            if g[index,col]>=.5:known_responses.append(float(a['p'][index]))
        tmp=path.with_suffix('.tmp')
        with open(tmp,'wb') as f:
            np.savez_compressed(f,boxes=np.concatenate([a['boxes'],b['boxes']]),
                p=np.concatenate([a['p'],b['p']]),ctr=np.concatenate([a['ctr'],b['ctr']]),
                logits=np.concatenate([a['logits'],b['logits']]),stable=np.r_[stable,np.zeros(len(b['boxes']),bool)])
        tmp.replace(path);files[path.name]=digest(path);bar.show(i+1)
        if progress:progress(i+1,len(s['fit_images']))
    del pf
    if device.type=='cuda':torch.cuda.empty_cache()
    required=1 if s.get('purpose')=='smoke' else 20
    if len(known_responses)<required:raise RuntimeError('Teacher known matches insufficient for threshold fit')
    high=float(np.quantile(known_responses,.2));ignore=float(np.quantile(known_responses,.01))
    m=dict(identity=identity,split_identity=s['identity'],files=files,images=s['fit_images'],
           thresholds=dict(high=high,ignore=ignore,low=.1*ignore),known_matches=len(known_responses),
           scope='inner-fit only; no semantic unknown labels; two-view union for ignore')
    write_json(done,m)
    return folder


def teacher_for(folder,name,device,flip=False):
    m=read_json(Path(folder)/'manifest.json')
    if name not in m['images']:raise PermissionError('Teacher requested outside inner-fit')
    path=Path(folder)/(fingerprint(name)[:20]+'.npz')
    with np.load(path,allow_pickle=False) as f:
        r={k:torch.from_numpy(f[k].copy()).to(device) for k in ['boxes','p','ctr','stable']}
    if flip:r['boxes'][:,0]=1-r['boxes'][:,0]
    r['thresholds']=m['thresholds'];r['reliable']=r['stable'] & (r['p']>=m['thresholds']['high'])
    return r
