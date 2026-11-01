from pathlib import Path
import time,gc
import numpy as np
import torch
from .protocol import read,write,sha,fp,space,progress
from .model import region
from jointdet.model import pixels,make_sample
from jointdet.geometry import xyxy,cxcywh,overlaps
from jointdet.teacher import extract


def folder_for(root,cfg,s):
    return Path(root)/'prepared/reference_net'/cfg['tag']/'cache'/s['scene']/s['task']


def prepare(root,view,model,s,cfg,report):
    folder=folder_for(root,cfg,s);folder.mkdir(parents=True,exist_ok=True)
    ident=fp(dict(split=s['identity'],model=cfg['model'],provenance=cfg['provenance']))
    manifest=folder/'MANIFEST.json'
    if manifest.exists():
        v=read(manifest)
        if v['identity']!=ident or any(sha(folder/n)!=h for n,h in v['files'].items()):raise RuntimeError('Reference/teacher cache changed')
        return folder
    from scipy.optimize import linear_sum_assignment
    device=next(model.parameters()).device;model.eval();model.feedback=False
    rows=[];files={};start=time.monotonic();responses=[]
    try:
        for role in ['fit','cal','monitor']:
            for name in s[role+'_images']:
                objects=[o for o in s[role+'_objects'] if o['image']==name]
                with torch.inference_mode():
                    _,d,meta=pixels(view,name,cfg['model'],device);_,gt=make_sample(meta,objects,device)
                    feature=model.feature_map(d);raw=region(feature,cxcywh(gt)).cpu().numpy()
                    payload=dict(raw=raw,ids=np.array([o['id'] for o in objects]),labels=np.array([o['class_id'] for o in objects]),gt=gt.cpu().numpy())
                    if role=='fit':
                        a=extract(model.pf,view,name,cfg['model'],device,False);b=extract(model.pf,view,name,cfg['model'],device,True)
                        ov=overlaps(xyxy(torch.from_numpy(a['boxes'])),xyxy(torch.from_numpy(b['boxes']))).numpy()
                        ii,jj=linear_sum_assignment(-ov);stable=np.zeros(len(a['p']),bool);stable[ii]=ov[ii,jj]>=.7
                        payload.update(boxes=a['boxes'],p=a['p'],ctr=a['ctr'],stable=stable)
                        g=overlaps(xyxy(torch.from_numpy(a['boxes'])),gt.cpu()).numpy()
                        for j in range(len(gt)):
                            k=int(g[:,j].argmax())
                            if g[k,j]>=.5:responses.append(float(a['p'][k]))
                path=folder/(fp(name)[:24]+'.npz');space(root,cfg,sum(x.nbytes for x in payload.values()))
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,**payload)
                tmp.replace(path);files[path.name]=sha(path);rows.append(dict(image=name,role=role,file=path.name,group=s['groups'][name]))
                progress(report,len(rows),sum(len(s[k+'_images']) for k in ['fit','cal','monitor']),'frozen GT references and teacher',start)
                del feature,d,raw
        # Rare-class absence does not invalidate pooled teacher thresholds.
        high=float(np.quantile(responses,.2)) if responses else .8
        ignore=float(np.quantile(responses,.01)) if responses else .5
        write(manifest,dict(identity=ident,split=s['identity'],files=files,rows=rows,
              thresholds=dict(high=high,ignore=ignore,low=.1*ignore),teacher_matched_count=len(responses),
              fallback_teacher_thresholds=not bool(responses),feature_space='frozen DINO; trainable relation projects query and memory together'))
    finally:model.feedback=True
    return folder


def load_bank(folder,device):
    m=read(Path(folder)/'MANIFEST.json');raw=[];labels=[];images=[];ids=[];groups=[]
    for row in m['rows']:
        if row['role']!='fit':continue
        with np.load(Path(folder)/row['file'],allow_pickle=False) as v:
            raw.extend(v['raw']);labels.extend(v['labels']);ids.extend(v['ids'].tolist());images.extend([row['image']]*len(v['raw']));groups.extend([row['group']]*len(v['raw']))
    return dict(raw=torch.tensor(np.asarray(raw),device=device),labels=torch.tensor(labels,device=device,dtype=torch.long),images=images,ids=ids,groups=groups,image_to_group={row['image']:row['group'] for row in m['rows']},
                manifest_sha256=sha(Path(folder)/'MANIFEST.json'))


def teacher(folder,image,device,flip):
    m=read(Path(folder)/'MANIFEST.json');row=next(x for x in m['rows'] if x['image']==image and x['role']=='fit')
    with np.load(Path(folder)/row['file'],allow_pickle=False) as f:
        result={k:torch.from_numpy(f[k].copy()).to(device) for k in ['boxes','p','ctr','stable']}
    if flip:result['boxes'][:,0]=1-result['boxes'][:,0]
    result['thresholds']=m['thresholds'];result['reliable']=result['stable']&(result['p']>=m['thresholds']['high'])
    return result
