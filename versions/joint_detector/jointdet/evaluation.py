from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import copy
import contextlib
import io
import xml.etree.ElementTree as ET
import numpy as np
import torch
from cil.common import TASK_CLASSES,install_label_guard,iou
from cil.evaluate import official_functions,evaluate
from .engine import load_run
from .scoring import raw_arrays,score_arrays,detections,export_rows,calibrate
from .protocol import digest,read_json,write_json,Progress,gpu


def projected_annotations(root,names,out):
    """Validation-only four-known-class universe; never expose future-unknown boxes."""
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    truth={};paths=[]
    for name in names:
        original=ET.parse(root/'assets/data/annotations'/(Path(name).stem+'.xml')).getroot()
        projected=ET.Element('annotation');objects=[]
        for obj in original.findall('object'):
            label=obj.findtext('name')
            if label not in TASK_CLASSES['task1']:
                continue  # Do not read coordinates of real Task-1 unknown objects.
            projected.append(copy.deepcopy(obj))
            b=[float(obj.find('bndbox').findtext(k))-1 for k in ['xmin','ymin','xmax','ymax']]
            objects.append((label,b))
        p=out/(Path(name).stem+'.xml');ET.ElementTree(projected).write(p,encoding='utf8')
        paths.append(str(p));truth[name]=objects
    return paths,truth


@torch.inference_mode()
def validate(root,run,device_spec):
    root=Path(root);run=Path(run);lock=read_json(run/'RUN_LOCK.json');c=lock['config'];s=lock['split']
    if s['formal'] or s['held'] is None:raise ValueError('Use predict/evaluate for formal runs')
    d=gpu(device_spec,c);model,saved,_=load_run(root,run,d)
    if saved['step']!=c['updates']:raise RuntimeError('Development validation requires completed fixed-budget checkpoint')
    checksum=digest(run/'last.pt');bank,cal=calibrate(root,run,model,s,c,checksum);model.eval()
    out=run/'validation'/checksum[:16];out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():return read_json(out/'metrics.json')
    paths,truth=projected_annotations(root,s['val_images'],out/'known_universe_annotations')
    active=[s['classes'][i] for i in s['active']]
    by=defaultdict(list);predictions=[];recalls=defaultdict(list);unknown_records=[];bar=Progress(len(s['val_images']),'development')
    candidates=out/'candidates.jsonl'
    if candidates.exists():candidates.unlink()
    for i,name in enumerate(s['val_images']):
        raw=raw_arrays(model.forward_image(root,name,[],d),c);values=score_arrays(raw,bank,cal,c)
        det=detections(raw,values,c);predictions.append(dict(image=name,**det))
        export_rows(candidates,name,s['groups'][name],raw,values,'outer_development')
        for box,score,label in zip(det['boxes'],det['scores'],det['labels']):
            target='unknown' if label==-1 else s['classes'][label]
            b=np.asarray(box).copy();b[:2]+=1
            by[target].append(f'{i}-^-{Path(name).stem} {score:.3f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}')
            if label==-1:
                overlap=iou([box],[v for _,v in truth[name]])[0]
                center=(np.asarray(box[:2])+np.asarray(box[2:]))/2
                inside=any(np.all(center>=np.asarray(v[:2])) and np.all(center<=np.asarray(v[2:])) for _,v in truth[name])
                unknown_records.append(dict(score=float(f'{score:.3f}'),
                    centered_low_iou=bool(inside and (overlap.max() if len(overlap) else 0)<.2)))
        if truth[name]:
            ov=iou(raw['boxes'],[b for _,b in truth[name]])
            for j,(label,_) in enumerate(truth[name]):
                recalls['held' if label==s['held'] else 'known'].append(bool(ov[:,j].max()>=.5))
        bar.show(i+1)
    funcs=official_functions(root);aps={};curves={};log=io.StringIO()
    for name in active+['unknown']:
        with contextlib.redirect_stdout(log):
            rec,prec,ap,*_=funcs['voc_eval'](by[name],paths,s['val_images'],name,ovthresh=.5,use_07_metric=True,known_classes=active)
        aps[name]=float(ap*100);curves[name]=dict(recall=rec.tolist(),precision=prec.tolist())
    metrics=dict(scope='Development AP in four-known-class annotation universe; held class is pseudo-unknown. '
                       'Real Task-1 unknown boxes are NOT read. Unmatched predictions are not verified true background.',
                 formal=False,real_unknown_labels_used=False,complete_real_background_evaluation=False,
                 checkpoint_sha256=checksum,K_dev_AP50=float(np.mean([aps[n] for n in active])),
                 held_dev_AP50=aps['unknown'],per_class_AP50=aps,
                 R50_known=float(np.mean(recalls['known'])*100) if recalls['known'] else None,
                 R50_held=float(np.mean(recalls['held'])*100) if recalls['held'] else None,
                 held=s['held'],method=lock['method'],images=len(s['val_images']))
    rec=np.asarray(curves['unknown']['recall']);conf=np.asarray([float(x.split()[1]) for x in by['unknown']])
    if len(rec) and rec.max()>=.7:
        sorted_conf=conf[np.argsort(-conf)];index=int(np.flatnonzero(rec>=.7)[0]);threshold=float(sorted_conf[index])
        last=int(np.flatnonzero(sorted_conf>=threshold)[-1])
        metrics['centered_low_iou_at_70_held_recall']=dict(reached=True,threshold=threshold,
            actual_recall_including_ties=float(rec[last]),
            proxy_fp_per_image=sum(x['centered_low_iou'] and x['score']>=threshold for x in unknown_records)/len(s['val_images']),
            scope='center inside a known-universe GT, IoU<0.2; localization proxy, not verified background')
    else:metrics['centered_low_iou_at_70_held_recall']=dict(reached=False,max_recall=float(rec.max()) if len(rec) else 0.)
    write_json(out/'predictions.json',predictions);write_json(out/'pr_curves.json',curves)
    write_json(out/'metrics.json',metrics);write_json(run/'validation_metrics.json',metrics)
    (out/'official_eval.log').write_text(log.getvalue(),encoding='utf8')
    return metrics


@torch.inference_mode()
def predict(root,run,device_spec):
    root=Path(root);run=Path(run);lock=read_json(run/'RUN_LOCK.json');c=lock['config'];s=lock['split']
    if not s['formal']:raise ValueError('Formal prediction requires formal training')
    install_label_guard(root,s['task'])
    d=gpu(device_spec,c);model,saved,_=load_run(root,run,d)
    if saved['step']!=c['updates']:raise RuntimeError('Training incomplete')
    checksum=digest(run/'last.pt');bank,cal=calibrate(root,run,model,s,c,checksum);model.eval()
    out=run/'predictions'/checksum[:16];out.mkdir(parents=True,exist_ok=True)
    partial=out/'partial.jsonl';rows=[]
    if partial.exists():
        import json
        rows=[json.loads(x) for x in partial.read_text(encoding='utf8').splitlines() if x.strip()]
    names=(root/'assets/inputs/test.txt').read_text(encoding='utf8').splitlines();names=[x.strip() for x in names if x.strip()]
    if [r['image'] for r in rows]!=names[:len(rows)]:raise RuntimeError('Prediction resume order changed')
    import json
    bar=Progress(len(names)-len(rows),'formal predictions');done=len(rows)
    for j,name in enumerate(names[done:]):
        raw=raw_arrays(model.forward_image(root,name,[],d),c);values=score_arrays(raw,bank,cal,c)
        row=dict(image=name,**detections(raw,values,c))
        with partial.open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        export_rows(out/'candidates.jsonl',name,name,raw,values,'formal_prediction_no_labels')
        rows.append(row);bar.show(j+1)
    path=out/'predictions.json';write_json(path,rows)
    write_json(out/'PREDICTIONS_LOCK.json',dict(checkpoint_sha256=checksum,calibration=cal,
               predictions_sha256=digest(path),annotation_access=False,task=s['task']))
    write_json(run/'predictions_index.json',dict(path=path.relative_to(root).as_posix()))
    return dict(predictions=str(path),images=len(rows))
