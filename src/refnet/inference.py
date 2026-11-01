from pathlib import Path
import time,json
import numpy as np
import torch
from torch.nn import functional as F
from jointdet.geometry import xyxy,overlaps
from jointdet.model import pf_score
from .protocol import read,write,sha,progress


def values(model,out,bank,cfg,shift=0.):
    ids=pf_score(out['cls'][-1],out['ctr'][-1]).argsort(descending=True,stable=True)[:cfg['model']['proposal_limit']]
    logits=model.relation(out['raw'][ids],bank['raw'],bank['labels'],list(range(len(model.active))))
    logits=logits.clone();logits[:,-1]-=shift
    prob=logits.softmax(1)
    obj=pf_score(out['cls'][-1,ids],out['ctr'][-1,ids])/2
    score=obj*out['quality_logit'][ids].sigmoid()
    return ids,prob,score


@torch.no_grad()
def calibrate(root,view,model,s,cfg,folder,bank,report):
    model.eval();device=next(model.parameters()).device;manifest=read(Path(folder)/'MANIFEST.json')
    margins=[];objects=[];per_class={};start=time.monotonic()
    for i,name in enumerate(s['cal_images']):
        out=model.forward_image(view,name,[],device)
        row=next(x for x in manifest['rows'] if x['image']==name and x['role']=='cal')
        with np.load(Path(folder)/row['file'],allow_pickle=False) as f:
            raw=torch.tensor(f['raw'],device=device);gt=torch.tensor(f['gt'],device=device);labels=f['labels'].tolist()
        logits=model.relation(raw,bank['raw'],bank['labels'],list(range(len(s['classes']))))
        margins.extend((logits[:,-1]-logits[:,:-1].max(1).values).cpu().tolist())
        ids,_,obj=values(model,out,bank,cfg)
        ov=overlaps(xyxy(out['refined'][ids]),gt);best,q=ov.max(0)
        for j,cid in enumerate(labels):
            rowstats=per_class.setdefault(str(cid),dict(gt=0,matched=0,images=[]));rowstats['gt']+=1
            if best[j]>=.5:
                objects.append(float(obj[q[j]]));rowstats['matched']+=1;rowstats['images'].append(name)
        progress(report,i+1,len(s['cal_images']),'training-held calibration',start)
    if not margins:raise RuntimeError('Empty pooled calibration; no test fallback')
    result=dict(reject_shift=max(0.,float(np.quantile(margins,.95,method='higher'))),
                object_threshold=float(np.quantile(objects,.05,method='lower')) if objects else 0.,
                object_threshold_fallback=not bool(objects),matched_known=len(objects),
                calibrated_gt=len(margins),per_class=per_class,
                missing_cal_classes=s['cal_missing_classes'],rare_classes=s['rare_classes'],
                scope='pooled known-only calibration; rare classes do not have independent calibration guarantees')
    return result


@torch.no_grad()
def monitor(root,view,model,s,cfg,folder,bank,report):
    model.eval();device=next(model.parameters()).device;manifest=read(Path(folder)/'MANIFEST.json')
    raw=[];labels=[]
    for row in manifest['rows']:
        if row['role']=='monitor':
            with np.load(Path(folder)/row['file'],allow_pickle=False) as f:raw.extend(f['raw']);labels.extend(f['labels'].tolist())
    q=torch.tensor(np.asarray(raw),device=device);y=torch.tensor(labels,device=device)
    active=list(range(len(s['classes'])))
    # Deterministic monitor membership; no training RNG is consumed here.
    full=model.relation(q,bank['raw'],bank['labels'],active)
    known=F.cross_entropy(full,y);terms=[]
    for held in sorted(set(labels)):
        ids=y==held;support=[c for c in active if c!=held]
        out=model.relation(q[ids],bank['raw'],bank['labels'],support)
        terms.append(F.cross_entropy(out,torch.full((int(ids.sum()),),len(support),device=device,dtype=torch.long)))
    novel=torch.stack(terms).mean();ious=[];briers=[]
    names=s['monitor_images'][:cfg['monitor_images']];start=time.monotonic()
    for i,name in enumerate(names):
        out=model.forward_image(view,name,[],device)
        row=next(x for x in manifest['rows'] if x['image']==name)
        with np.load(Path(folder)/row['file'],allow_pickle=False) as f:gt=torch.tensor(f['gt'],device=device)
        ids,_,_=values(model,out,bank,cfg)
        best,qids=overlaps(xyxy(out['refined'][ids]),gt).max(0)
        ious.extend(best.cpu().tolist());briers.extend((out['quality_logit'][ids[qids]].sigmoid()-best).square().cpu().tolist())
        progress(report,i+1,len(names),'held-training monitor (not official test)',start)
    geom=1-float(np.mean(ious));quality=float(np.mean(briers))
    return dict(objective=float(known)+float(novel)+geom+quality,known_ce=float(known),
                pseudo_unknown_ce=float(novel),mean_best_iou=1-geom,quality_brier=quality,
                gt_queries=len(y),geometric_images=len(names),test_labels_used=False)


@torch.no_grad()
def predict(root,view,model,s,cfg,bank,cal,report,weights_sha):
    from torchvision.ops import batched_nms
    from .evaluation import evaluate
    model.eval();report=Path(report);report.mkdir(parents=True,exist_ok=True)
    partial=report/'partial.jsonl';rows=[json.loads(x) for x in partial.read_text(encoding='utf8').splitlines() if x.strip()] if partial.exists() else []
    if [r['image'] for r in rows] != s['test_images'][:len(rows)]:raise RuntimeError('Prediction prefix mismatch')
    from .protocol import lock
    lock(report/'PREDICTION_LOCK.json',dict(weights_sha256=weights_sha,bank_manifest=bank['manifest_sha256'],calibration=cal,split=s['identity']))
    device=next(model.parameters()).device;start=time.monotonic();initial=len(rows)
    for i,name in enumerate(s['test_images'][initial:],initial):
        out=model.forward_image(view,name,[],device)
        ids,prob,obj=values(model,out,bank,cfg,cal['reject_shift']);confidence,label=prob.max(1)
        scores=obj*confidence;boxes=xyxy(out['refined'][ids]).clamp(0,1)
        wh=boxes.new_tensor([out['meta']['ori_shape'][1],out['meta']['ori_shape'][0]]*2);boxes=boxes*wh
        keep=torch.nonzero(obj>=cal['object_threshold']).flatten()
        if len(keep):keep=keep[batched_nms(boxes[keep],scores[keep],label[keep],cfg['model']['nms_iou'])[:cfg['model']['detection_limit']]]
        row=dict(image=name,boxes=boxes[keep].cpu().tolist(),scores=scores[keep].cpu().tolist(),labels=label[keep].cpu().tolist())
        with partial.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        rows.append(row);progress(report.parent,i+1,len(s['test_images']),'official test '+report.name,start)
        if i%32==0:
            from .protocol import space
            space(root,cfg)
    write(report/'predictions.json',rows)
    return evaluate(root,report/'predictions.json',s,report)
