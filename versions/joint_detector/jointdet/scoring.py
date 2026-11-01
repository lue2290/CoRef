from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import batched_nms
from .geometry import xyxy,overlaps
from .model import pf_score
from .protocol import atomic_save,digest,fingerprint,read_json,write_json,Progress


def distances(z,bank,groups=None,k=5):
    z=np.asarray(z,np.float32);g=bank['z']
    d=np.clip(1-z@g.T,0,2)
    if groups is not None:d[np.asarray(groups)[:,None]==np.asarray(bank['groups'])[None,:]]=np.inf
    cols=[]
    for cid in bank['active']:
        values=d[:,np.asarray(bank['labels'])==cid]
        if values.shape[1]<k:raise RuntimeError('Fewer than k references in a class')
        near=np.partition(values,k-1,axis=1)[:,:k]
        if not np.isfinite(near).all():raise RuntimeError('Insufficient cross-group references; k is locked')
        cols.append(near.mean(1))
    return np.stack(cols,1)


def support(bank,c):
    d=distances(bank['z'],bank,bank['groups'],c['k']);r=[];s=[]
    for j,cid in enumerate(bank['active']):
        v=d[np.asarray(bank['labels'])==cid,j]
        r.append(np.quantile(v,c['radius_quantile'],method='higher'))
        s.append(max(np.quantile(v,.75)-np.quantile(v,.25),c['iqr_floor']))
    bank['radius']=np.asarray(r,np.float32);bank['scale']=np.asarray(s,np.float32)
    return bank


def raw_arrays(out,c):
    if out['head'].get('dn_meta') is not None:raise RuntimeError('Inference must disable denoising')
    ps=pf_score(out['cls'][-1],out['ctr'][-1])
    ids=ps.argsort(descending=True,stable=True)[:c['proposal_limit']]
    def cpu(x):return x.detach().cpu().numpy()
    wh=out['boxes'].new_tensor([out['meta']['ori_shape'][1],out['meta']['ori_shape'][0]]*2)
    b=xyxy(out['boxes'][-1,ids]).clamp(0,1)*wh
    return dict(boxes=cpu(b),pf=cpu(ps[ids]),object_logit=cpu(out['cls'][-1,ids,1]),
                query_index=cpu(ids),pf_token_logits=cpu(out['cls'][-1,ids]),
                centerness=cpu(out['ctr'][-1,ids]),q=cpu(out['quality'][ids]),
                known_logits=cpu(out['known'][ids]),z=cpu(out['z'][ids]))


def score_arrays(raw,bank,cal,c):
    d=distances(raw['z'],bank,k=c['k'])
    novelty=((d-bank['radius'])/bank['scale']).min(1)
    u=1/(1+np.exp(-np.clip(novelty,-30,30)))
    obj=raw['pf']*(1-c['quality_alpha']+c['quality_alpha']*raw['q'])
    k=raw['known_logits'];prob=np.exp(k-k.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    label=np.asarray(bank['active'])[prob.argmax(1)].copy();label[u>.5]=-1
    su=obj*u;sk=obj*(1-u)*prob.max(1)
    scores=np.where(label==-1,su,sk)
    return dict(u=u,novelty=novelty,obj=obj,unknown_score=su,known_score=sk,
                labels=label,scores=scores,keep=obj>=cal['object_threshold'])


def detections(raw,values,c):
    ids=np.flatnonzero(values['keep'])
    if len(ids):
        b=torch.from_numpy(raw['boxes'][ids]);sc=torch.from_numpy(values['scores'][ids]);la=torch.from_numpy(values['labels'][ids].astype(np.int64)+1)
        kept=batched_nms(b,sc,la,c['nms_iou']).numpy()[:c['detection_limit']];ids=ids[kept]
    return dict(boxes=raw['boxes'][ids].tolist(),scores=values['scores'][ids].tolist(),labels=values['labels'][ids].tolist())


def export_rows(path,name,group,raw,values=None,role='inference',objects=None):
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as f:
        for i in range(len(raw['boxes'])):
            row=dict(image=name,group=group,role=role,candidate=i,box=raw['boxes'][i].tolist(),
                     query_index=int(raw['query_index'][i]),pf_token_logits=raw['pf_token_logits'][i].tolist(),
                     pf=float(raw['pf'][i]),object_logit=float(raw['object_logit'][i]),
                     centerness=float(raw['centerness'][i]),quality=float(raw['q'][i]),
                     known_logits=raw['known_logits'][i].tolist())
            if objects:
                from cil.common import iou
                ov=iou([raw['boxes'][i]],[o['box'] for o in objects])[0];j=int(ov.argmax())
                row.update(selected_gt_id=objects[j]['id'],selected_class_id=objects[j]['class_id'],
                           selected_iou=float(ov[j]),label_source='legal_selected_known_annotation_only')
            if values:
                for key in ['u','novelty','obj','unknown_score','known_score','labels','scores','keep']:
                    row[key]=values[key][i].item()
            f.write(json.dumps(row,allow_nan=False)+'\n')


@torch.inference_mode()
def calibrate(root,run,model,s,c,checkpoint):
    run=Path(run);out=run/'calibration'/checkpoint[:16];out.mkdir(parents=True,exist_ok=True)
    cached=out/'calibration.json'
    if cached.exists():
        cal=read_json(cached)
        if cal['checkpoint_sha256']!=checkpoint:raise RuntimeError('Stale calibration')
        from .protocol import load_torch
        bank=load_torch(out/'bank.pt')
        if digest(out/'bank.pt')!=cal['bank_sha256']:raise RuntimeError('Changed reference bank')
        return bank,cal
    model.eval();device=next(model.parameters()).device
    rows=[];bar=Progress(len(s['fit_images'])+len(s['cal_images']),'gallery/calibration')
    bank=dict(z=[],labels=[],groups=[],objects=[],active=s['active'],checkpoint_sha256=checkpoint)
    raw_cache=[]
    for i,name in enumerate(s['fit_images']):
        raw=raw_arrays(model.forward_image(root,name,[],device),c)
        objects=[o for o in s['fit_objects'] if o['image']==name]
        ov=overlaps(torch.tensor(raw['boxes']),torch.tensor([o['box'] for o in objects],dtype=torch.float32)).numpy()
        for j,o in enumerate(objects):
            ix=int(ov[:,j].argmax())
            if ov[ix,j]<.5:continue
            bank['z'].append(raw['z'][ix]);bank['labels'].append(o['class_id']);bank['groups'].append(o['group']);bank['objects'].append(o['id'])
        raw_cache.append((name,raw));bar.show(i+1)
    if not bank['z']:raise RuntimeError('No matched training references')
    bank['z']=np.asarray(bank['z'],np.float32)
    bank=support(bank,c);atomic_save(bank,out/'bank.pt')
    scores=[];cal_raw=[]
    for j,name in enumerate(s['cal_images']):
        raw=raw_arrays(model.forward_image(root,name,[],device),c)
        objects=[o for o in s['cal_objects'] if o['image']==name]
        ov=overlaps(torch.tensor(raw['boxes']),torch.tensor([o['box'] for o in objects],dtype=torch.float32)).numpy()
        obj=raw['pf']*(1-c['quality_alpha']+c['quality_alpha']*raw['q'])
        for k in range(len(objects)):
            ix=int(ov[:,k].argmax())
            if ov[ix,k]>=.5:scores.append(float(obj[ix]))
        cal_raw.append((name,raw));bar.show(len(s['fit_images'])+j+1)
    if not scores:raise RuntimeError('No qualified inner-cal known candidate')
    threshold=float(np.quantile(scores,1-c['object_retention'],method='lower'))
    cal=dict(checkpoint_sha256=checkpoint,bank_sha256=digest(out/'bank.pt'),split_identity=s['identity'],
             object_threshold=threshold,qualified_cal_objects=len(scores),total_cal_objects=len(s['cal_objects']),
             quality_alpha=c['quality_alpha'],unknown_threshold=.5,scope='group-held inner-cal; known-only threshold; no true unknown labels')
    for filename,data,role in [('fit_candidates.jsonl',raw_cache,'inner_fit_reference'),('cal_candidates.jsonl',cal_raw,'inner_cal')]:
        p=out/filename
        if p.exists():p.unlink()  # this experiment's unfinished atomic calibration only
        objects=s['fit_objects'] if role=='inner_fit_reference' else s['cal_objects']
        for name,raw in data:export_rows(p,name,s['groups'][name],raw,score_arrays(raw,bank,cal,c),role,
                                        [o for o in objects if o['image']==name])
    write_json(cached,cal);write_json(run/'calibration.json',{**cal,'folder':out.relative_to(run).as_posix()})
    return bank,cal
