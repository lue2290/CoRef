from pathlib import Path
import numpy as np
import torch
from current.progress import track
from jointdet.geometry import xyxy, overlaps
from refnet.inference import values
from refnet_f1.runtime import calibrate as calibrate_f1, predict, official_evaluation
from .protocol import read, write, sha, fp, lock, verify_file


@torch.inference_mode()
def calibrate_parent(root, view, model, spec, original_cfg, cache, bank, report, checkpoint_sha, dataset, verified):
    """Original GT reject shift and best-GT-candidate quality quantile; fresh weights."""
    report=Path(report);device=next(model.parameters()).device
    manifest=read(Path(cache)/'MANIFEST.json'); rows={r['image']:r for r in manifest['rows'] if r['role']=='cal'}
    records=[];cal_gt={}
    for name in track(spec['cal_images'],f'Game/{spec["task"]} recalibrate new parent quality/identity',unit='images'):
        row=rows[name]
        with np.load(Path(cache)/row['file'],allow_pickle=False) as z:
            raw=torch.tensor(z['raw'],device=device);gt=torch.tensor(z['gt'],device=device);labels=z['labels'].tolist()
            cal_gt[name]=z['raw'].copy()
        path=report/'parent_cal_images'/(fp(name)[:24]+'.json')
        image=Path(root)/'assets/rwd/JPEGImages/Game'/name
        image_sha=verify_file(image,dataset['files']['JPEGImages/Game/'+name]['sha256'],verified)
        if path.exists():
            record=read(path)
            if record['image']!=name or record['image_sha256']!=image_sha:raise RuntimeError('Parent calibration image changed')
        else:
            out=model.forward_image(view,name,[],device)
            logits=model.relation(raw,bank['raw'],bank['labels'],list(range(len(spec['classes']))))
            ids,_,joint=values(model,out,bank,original_cfg)
            ov=overlaps(xyxy(out['refined'][ids]),gt);best,owner=ov.max(0)
            record=dict(image=name,image_sha256=image_sha,gt_labels=labels,
                margins=(logits[:,-1]-logits[:,:-1].max(1).values).cpu().tolist(),
                matches=[dict(class_id=int(cid),joint=float(joint[owner[j]])) for j,cid in enumerate(labels) if best[j]>=.5])
            write(path,record);del out
        records.append(record)
    margins=[m for r in records for m in r['margins']];matches=[m['joint'] for r in records for m in r['matches']]
    if not margins:raise RuntimeError('No legal known GT calibration; never use test labels as fallback')
    per_class={}
    for r in records:
        for cid in r['gt_labels']:
            row=per_class.setdefault(str(cid),dict(gt=0,matched=0,images=[]));row['gt']+=1
        for match in r['matches']:
            row=per_class[str(match['class_id'])];row['matched']+=1;row['images'].append(r['image'])
    cal=dict(reject_shift=max(0.,float(np.quantile(margins,.95,method='higher'))),
        object_threshold=float(np.quantile(matches,.05,method='lower')) if matches else 0.,
        object_threshold_fallback=not bool(matches),matched_known=len(matches),calibrated_gt=len(margins),
        per_class=per_class,missing_cal_classes=spec['cal_missing_classes'],rare_classes=spec['rare_classes'],
        checkpoint_sha256=checkpoint_sha,bank_manifest=bank['manifest_sha256'],
        scope='new corrected-parent pooled known-only calibration; original v1 Q95 shift/Q05 joint gate; no classwise guarantee')
    lock(report/'PARENT_CALIBRATION.json',cal)
    return cal,cal_gt


def evaluate_parent(root,cfg,model,view,spec,original_cfg,bank,cache,run,report,binding,dataset,verified):
    report=Path(report); checkpoint=Path(run)/'last.pt';checkpoint_sha=sha(checkpoint)
    model.zero_grad(set_to_none=True);model.requires_grad_(False);model.eval()
    lock(report/'EVALUATION_LOCK.json',dict(training_lock=binding['identity'],checkpoint_sha256=checkpoint_sha,
        bank_manifest=bank['manifest_sha256'],split=spec['identity'],protocol='fixed500 F0/F1 as157, both gates/thresholds recalibrated for new parent'))
    parent_cal,cal_gt=calibrate_parent(root,view,model,spec,original_cfg,cache,bank,report,checkpoint_sha,dataset,verified)
    cal=calibrate_f1(model,view,root,spec,bank,cal_gt,cfg,original_cfg['model'],parent_cal,report,verified,dataset)
    predict(model,view,root,spec,bank,cfg,original_cfg['model'],cal,report,dataset,verified)
    metrics=official_evaluation(root,spec,cfg,report,dataset,verified)
    paths=['CALIBRATION.json','PARENT_CALIBRATION.json','PREDICTIONS_COMPLETE.json','EVALUATION_LOCK.json']
    paths += [arm+'/'+name for arm in cfg['arms'] for name in ['predictions.json','metrics.json','pr_curves.json','errors_by_image.json']]
    entry=dict(id=f'Game/{spec["task"]}/seed_{binding["seed"]}',scene='Game',task=spec['task'],seed=binding['seed'],
        reference_cache=Path(cache).relative_to(root).as_posix(),reference_manifest_sha256=bank['manifest_sha256'],
        split=binding['annotation']['split_path'],split_identity=spec['identity'],
        lock=(Path(run)/'LOCK.json').relative_to(root).as_posix(),lock_sha256=sha(Path(run)/'LOCK.json'),
        result=(report/'RESULT.json').relative_to(root).as_posix(),
        terminal=dict(checkpoint=checkpoint.relative_to(root).as_posix(),sha256=checkpoint_sha,step=cfg['updates'],
            calibration=(report/'PARENT_CALIBRATION.json').relative_to(root).as_posix(),
            calibration_sha256=sha(report/'PARENT_CALIBRATION.json'),calibration_checkpoint_sha256=checkpoint_sha),
        corrected_label_retraining=True,initialization=cfg['initialization'])
    lock(report/'CORRECTED_PARENT.json',entry)
    result=dict(complete=True,scene='Game',task=spec['task'],seed=binding['seed'],lock=binding['identity'],
        scope=cfg['scope'],corrected_label_retraining=True,parent_checkpoint=entry['terminal'],
        training_updates=cfg['updates'],f1_extra_training_updates=0,old_adapted_checkpoint_loaded=False,
        calibration=cal,metrics=metrics,artifacts={name:sha(report/name) for name in paths})
    write(report/'RESULT.json',result)
    return result
