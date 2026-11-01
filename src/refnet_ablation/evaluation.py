from pathlib import Path
import numpy as np
import torch
from current.progress import track
from jointdet.geometry import xyxy,overlaps
from refnet.inference import values
from refnet_f1.runtime import calibrate as calibrate_f1,predict,official_evaluation
from .protocol import read,write,sha,fp,lock,verify_file


@torch.inference_mode()
def calibrate_parent(root,view,model,spec,original,cache,bank,report,weights,dataset,verified):
    device=next(model.parameters()).device;manifest=read(Path(cache)/'MANIFEST.json')
    rows={r['image']:r for r in manifest['rows'] if r['role']=='cal'};records=[];cal_gt={}
    for name in track(spec['cal_images'],spec['scene']+' calibrate variant quality/GT margins',unit='images'):
        row=rows[name]
        with np.load(Path(cache)/row['file'],allow_pickle=False) as z:
            raw=torch.tensor(z['raw'],device=device);gt=torch.tensor(z['gt'],device=device);labels=z['labels'].tolist();cal_gt[name]=z['raw'].copy()
        image=Path(root)/'assets/rwd/JPEGImages'/spec['scene']/name
        image_sha=verify_file(image,dataset['files'][f'JPEGImages/{spec["scene"]}/{name}']['sha256'],verified)
        path=report/'parent_cal_images'/(fp(name)[:24]+'.json')
        if path.exists():
            record=read(path)
            if record['image']!=name or record['image_sha256']!=image_sha:raise RuntimeError('Calibration image changed')
        else:
            out=model.forward_image(view,name,[],device)
            logits=model.relation(raw,bank['raw'],bank['labels'],list(range(len(spec['classes']))))
            ids,_,joint=values(model,out,bank,original);best,owner=overlaps(xyxy(out['refined'][ids]),gt).max(0)
            record=dict(image=name,image_sha256=image_sha,gt_labels=labels,
                margins=(logits[:,-1]-logits[:,:-1].max(1).values).cpu().tolist(),
                matches=[dict(class_id=int(cid),joint=float(joint[owner[j]])) for j,cid in enumerate(labels) if best[j]>=.5])
            write(path,record);del out
        records.append(record)
    margins=[m for r in records for m in r['margins']];matches=[v['joint'] for r in records for v in r['matches']]
    if not margins:raise RuntimeError('No legal cal GT; no test fallback')
    result=dict(reject_shift=max(0.,float(np.quantile(margins,.95,method='higher'))),
        object_threshold=float(np.quantile(matches,.05,method='lower')) if matches else 0.,
        object_threshold_fallback=not bool(matches),matched_known=len(matches),calibrated_gt=len(margins),
        checkpoint_sha256=weights,bank_manifest=bank['manifest_sha256'],test_labels_used=False)
    lock(report/'PARENT_CALIBRATION.json',result)
    return result,cal_gt


def evaluate(root,cfg,original,model,view,case,bank,run,report,binding,dataset,verified):
    spec=case['spec'];weights=sha(run/'last.pt');model.zero_grad(set_to_none=True);model.requires_grad_(False);model.eval()
    lock(report/'EVALUATION_LOCK.json',dict(lock=binding['identity'],weights=weights,bank_manifest=bank['manifest_sha256'],protocol='F1 as157; fresh variant calibration'))
    parent_cal,cal_gt=calibrate_parent(root,view,model,spec,original,case['cache'],bank,report,weights,dataset,verified)
    cal=calibrate_f1(model,view,root,spec,bank,cal_gt,cfg,original['model'],parent_cal,report,verified,dataset)
    predict(model,view,root,spec,bank,cfg,original['model'],cal,report,dataset,verified)
    metrics=official_evaluation(root,spec,cfg,report,dataset,verified)
    names=['PARENT_CALIBRATION.json','CALIBRATION.json','EVALUATION_LOCK.json','PREDICTIONS_COMPLETE.json',
           'F1/predictions.json','F1/metrics.json','F1/pr_curves.json','F1/errors_by_image.json']
    result=dict(complete=True,variant=model.variant,scene=spec['scene'],task='task1',seed=case['seed'],
        lock=binding['identity'],training_updates=cfg['updates'],old_adapted_checkpoint_loaded=False,
        parent_checkpoint=dict(checkpoint=(run/'last.pt').relative_to(root).as_posix(),sha256=weights,step=cfg['updates']),
        calibration=cal,metrics=metrics,full_control_identity=case['identity'],
        artifacts={name:sha(report/name) for name in names})
    write(report/'RESULT.json',result)
    return result
