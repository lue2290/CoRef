from pathlib import Path
from collections import Counter,defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image,ImageOps
from cil.common import iou,auc
from jointdet.geometry import cxcywh
from current.progress import track,stage
from refnet_f1.runtime import decode,calibrate_threshold
from .protocol import read,write,sha,fp,lock,locations,verify_file,guard
from .cache import candidates
from .model import read_regions


@torch.inference_mode()
def logits(head,raw,bank,classes,batch=32):
    values=[head(raw[j:j+batch],bank,list(range(classes))) for j in range(0,len(raw),batch)]
    out=torch.cat(values)
    if not torch.isfinite(out).all():raise RuntimeError('Nonfinite OWL relation output')
    return out.cpu()


def matched_calibration(spec,folder):
    by=defaultdict(list)
    for o in spec['cal_objects']:by[o['image']].append(o)
    best=[];rows=[]
    for name in spec['cal_images']:
        a=candidates(folder,name);objects=by[name];matrix=iou(a['boxes'],[o['box'] for o in objects])
        eligible=matrix>=.5
        for j,o in enumerate(objects):
            values=a['g'][eligible[:,j]]
            if len(values):best.append(float(values.max()))
        rows.append(dict(image=name,a=a,eligible=eligible,objects=objects,owner=matrix.argmax(1) if objects else np.zeros(len(a['boxes']),int)))
    if not best:raise RuntimeError('160 calibration failed: no IoU>=0.5 PF proposal for any known cal GT; no threshold fallback')
    return float(np.quantile(best,.05,method='lower')),rows,best


@torch.inference_mode()
def calibrate(root,cfg,data,heads,bindings):
    threshold,rows,best=matched_calibration(data.spec,data.candidates);values={k:[] for k in heads};coverage=Counter();groups=defaultdict(set)
    for row in track(rows,data.case['scene']+' six-head known-only F1 calibration',unit='images'):
        name=row['image'];a=row['a'];eligible=row['eligible']&(a['g'][:,None]>=threshold);keep=eligible.any(1)
        for index in np.flatnonzero(keep):
            cid=row['objects'][row['owner'][index]]['class_id'];coverage[cid]+=1;groups[cid].add(data.spec['groups'][name])
        if not keep.any():continue
        cached=data.view(name);slots=cached['pred_slots']
        if not np.array_equal(slots,np.arange(600)):raise RuntimeError('Cal requires all shared600 candidates')
        raw=torch.tensor(cached['pred_raw'][keep],device=data.device)
        for key,head in heads.items():
            l=logits(head,raw,data.bank,len(data.spec['classes']));values[key].extend((l[:,-1]-l[:,:-1].max(1).values).tolist())
    result={}
    for key,head in heads.items():
        arm,seed=key;report=locations(root,cfg,data.case['scene'],arm,seed)[2];m=values[key]
        cal=dict(lock=bindings[key]['identity'],checkpoint_sha256=sha(locations(root,cfg,data.case['scene'],arm,seed)[1]/'last.pt'),
            object_threshold=threshold,threshold_margin=calibrate_threshold(m,cfg['known_retention_target']),
            object_best_gt_matches=len(best),samples=len(m),per_class={str(c):dict(candidates=coverage[c],source_groups=len(groups[c])) for c in range(len(data.spec['classes']))},
            source_groups=len({data.spec['groups'][r['image']] for r in rows if (r['eligible']&(r['a']['g'][:,None]>=threshold)).any()}),
            empirical_known_retention=float(np.mean(np.asarray(m)<=calibrate_threshold(m,cfg['known_retention_target']))),
            object_quantile='known cal GT: maximum g among IoU>=0.5 predictions; Q05 lower; no fallback',
            rejection_quantile='g-gated known cal predictions with IoU>=0.5; Q95 higher; no zero clamp; ties known; candidates equally weighted',
            test_labels_used=False,guarantee='pooled empirical cal retention only; not per-class/test guarantee')
        result[key]=lock(report/'CALIBRATION.json',cal)
    return result


@torch.inference_mode()
def predict(root,cfg,data,heads,bindings,cal,visual_factory,dataset,verified,encoder=None):
    records={k:[] for k in heads};visual=None;processor=None;s=data.spec
    for name in track(s['test_images'],data.case['scene']+' shared OWL official test (six heads)',unit='images'):
        image=Path(root)/'assets/rwd/JPEGImages'/s['scene']/name
        ih=verify_file(image,dataset['files'][f'JPEGImages/{s["scene"]}/{name}']['sha256'],verified)
        a=candidates(data.candidates,name);candidate_sha=sha(data.candidates/(fp(name)[:24]+'.npz'));pending=[]
        for key in heads:
            report=locations(root,cfg,s['scene'],*key)[2];path=report/'test_images'/(fp(name)[:24]+'.json')
            binding=dict(case=bindings[key]['identity'],calibration=cal[key]['identity'],image_sha256=ih,candidates_sha256=candidate_sha)
            if path.exists():
                record=read(path)
                if record['binding']!=binding or record['image']!=name:raise RuntimeError('OWL test resume binding changed')
                records[key].append(record)
            else:pending.append((key,path,binding))
        if not pending:continue
        guard(root,cfg)
        if encoder is not None:raw=encoder(image,a)
        else:
            if visual is None:
                with stage('Load frozen OWL once for shared test forward'):visual,processor=visual_factory()
            with Image.open(image) as im:im=ImageOps.exif_transpose(im).convert('RGB')
            pixels=processor(images=im,return_tensors='pt')['pixel_values'].to(data.device)
            fmap=visual(pixels);w,h=a['size']
            raw=read_regions(fmap,cxcywh(torch.tensor(a['boxes']/np.array([w,h,w,h]),dtype=torch.float32,device=data.device)),cfg['context_scale'])
            del fmap,pixels
        raw=torch.as_tensor(raw,device=data.device,dtype=torch.float32)
        for key,path,binding in pending:
            l=logits(heads[key],raw,data.bank,len(s['classes']))
            prediction,diag=decode(dict(logits=l,joint=torch.tensor(a['g']),boxes=torch.tensor(a['boxes']),ids=torch.tensor(a['proposal_ids'])),
                'F1',cal[key],dict(nms_iou=cfg['nms_iou'],detection_limit=cfg['detection_limit']))
            record=dict(image=name,binding=binding,prediction=prediction,decoding=diag,
                margin=(l[:,-1]-l[:,:-1].max(1).values).tolist(),known_argmax=l[:,:-1].argmax(1).tolist())
            write(path,record);records[key].append(record)
        del raw
    seals={}
    for key,rows in records.items():
        # A resumed subset can have another traversal append order. Rebind explicitly.
        by={r['image']:r for r in rows};rows=[by[n] for n in s['test_images']];records[key]=rows
        report=locations(root,cfg,s['scene'],*key)[2]
        write(report/'predictions.json',[dict(image=r['image'],**r['prediction']) for r in rows])
        seals[key]=lock(report/'PREDICTIONS_COMPLETE.json',dict(case=bindings[key]['identity'],calibration=cal[key]['identity'],
            images=s['test_images'],prediction_sha256=sha(report/'predictions.json'),test_record_sha256=fp(rows),test_annotations_opened=False))
    del visual,processor
    return records,seals


def annotation(xml,classes):
    result=[]
    for obj in ET.parse(xml).getroot().findall('object'):
        if int(obj.findtext('difficult','0')):continue
        name=obj.findtext('name');result.append(dict(label=classes.index(name) if name in classes else len(classes),
            box=[float(obj.find('bndbox').findtext(k))-1 for k in ['xmin','ymin','xmax','ymax']]))
    return result


def diagnostic(rows,spec,folder,cal,gt):
    populations={k:dict(y=[],m=[],c=[]) for k in ['before_gate','after_gate']};ignored=0
    for r in rows:
        a=candidates(folder,r['image']);objects=gt[r['image']];overlap=iou(a['boxes'],[o['box'] for o in objects])
        if not objects:ignored+=len(a['boxes']);continue
        best=overlap.max(1);owner=overlap.argmax(1);good=best>=.5;ignored+=int((~good).sum())
        for key,keep in [('before_gate',good),('after_gate',good&(a['g']>=cal['object_threshold']))]:
            ids=np.flatnonzero(keep);p=populations[key]
            p['y'].extend(objects[owner[j]]['label'] for j in ids);p['m'].extend(r['margin'][j] for j in ids);p['c'].extend(r['known_argmax'][j] for j in ids)
    result={}
    for key,p in populations.items():
        y=np.array(p['y']);m=np.array(p['m']);c=np.array(p['c']);known=y<len(spec['classes']);novel=~known;reject=m>cal['threshold_margin']
        def rate(v,mask):return float(np.mean(v[mask])) if mask.any() else None
        result[key]=dict(known_candidates=int(known.sum()),unknown_candidates=int(novel.sum()),
            closed_set_known_accuracy=rate(c==y,known),final_known_accuracy=rate((c==y)&~reject,known),
            known_false_reject_rate=rate(reject,known),unknown_false_accept_rate=rate(~reject,novel),margin_AUROC=auc(m[novel],m[known]))
    return dict(populations=result,ignored_background_or_low_IoU=ignored,
        population='all top600, best non-difficult GT IoU>=0.5; ties first GT; continuous XYXY; duplicates retained and candidate-weighted; background excluded',
        metric_scope='candidate identity diagnostics, not end-to-end AP; difficult GT omitted')


def evaluate_all(root,cfg,data,heads,bindings,cal,records,seals,dataset,verified):
    from refnet.evaluation import evaluate
    s=data.spec;gt={}
    # Every head has a complete sealed prediction before any test XML is read.
    if set(seals)!=set(heads):raise RuntimeError('All six prediction seals required before official evaluation')
    for name in track(s['test_images'],s['scene']+' verify official annotations after prediction seal',unit='XML'):
        rel=f'Annotations/{s["scene"]}/{Path(name).stem}.xml';xml=Path(root)/'assets/rwd'/rel
        verify_file(xml,dataset['files'][rel]['sha256'],verified);gt[name]=annotation(xml,s['classes'])
    for key in track(heads,s['scene']+' official unchanged VOC07 AP50',unit='heads'):
        report=locations(root,cfg,s['scene'],*key)[2];run=locations(root,cfg,s['scene'],*key)[1]
        with stage(f'{s["scene"]}/{key[0]}/seed_{key[1]} official AP50'):
            metrics=evaluate(root,report/'predictions.json',s,report)
        diag=diagnostic(records[key],s,data.candidates,cal[key],gt);write(report/'DIAGNOSTICS.json',diag)
        artifacts=['CALIBRATION.json','PREDICTIONS_COMPLETE.json','predictions.json','metrics.json','DIAGNOSTICS.json','pr_curves.json','errors_by_image.json']
        result=dict(complete=True,scene=s['scene'],task='task1',arm=key[0],seed=key[1],lock=bindings[key]['identity'],
            metrics=metrics,calibration=cal[key],diagnostics=diag,checkpoint_sha256=sha(run/'last.pt'),updates=cfg['updates'],
            frozen_PF=True,frozen_OWL=True,DINO_used=False,official_evaluator_unchanged=True,
            artifacts={n:sha(report/n) for n in artifacts})
        write(report/'RESULT.json',result)
