"""Separate adapted identity labels from the frozen M1 score path."""
from pathlib import Path
import numpy as np
from pclean.files import require,write
from identitylab.core import score_values,match_objects
from .controls import matched_threshold,boundary_values
from identitylab.evaluation import distances,identity_summary
from .evaluation import evaluate_predictions,projected_truth
from jointdet.scoring import distances as native_distances,detections


def hybrid_values(raw,adapted,original,active):
    """H changes labels only; both label-specific scores use original novelty."""
    require(np.array_equal(adapted['keep'],original['keep']),'Frozen object gate changed')
    require(np.array_equal(adapted['obj'],original['obj']),'Frozen objectness changed')
    logits=raw['known_logits'];prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    known_labels=np.asarray(active)[prob.argmax(1)]
    labels=np.asarray(adapted['labels']).copy()
    require(np.all((labels==-1)|(labels==known_labels)),'Known argmax must remain frozen')
    u=original['u'];obj=original['obj']
    return dict(keep=original['keep'],labels=labels,a=adapted['a'],u=u,obj=obj,
                scores=np.where(labels==-1,obj*u,obj*(1-u)*prob.max(1)))


def evaluate_readouts(root,cache,adapter,bank,original_bank,cal,device,out):
    out=Path(out);spec=cache.spec;cfg=spec['config'];split=spec['split']
    require(list(bank['active'])==list(original_bank['active']),'Active gallery labels changed')
    cal_rows=[]
    for name in split['cal_images']:
        raw=cache.load(name)
        old=score_values(raw,native_distances(raw['z'],original_bank,k=cfg['k']),original_bank,cal,cfg,'R')
        new=score_values(raw,distances(raw,adapter,bank,device,cfg['k']),bank,cal,cfg,'R')
        objects=[o for o in split['cal_objects'] if o['image']==name]
        require(all(o['class_id'] in bank['active'] for o in objects),'Held label in calibration')
        for j,i,_ in match_objects(raw,objects,old['keep']):
            cal_rows.append(dict(a=float(old['a'][i]),accepted=bool(new['labels'][i]!=-1)))
    threshold=matched_threshold([x['a'] for x in cal_rows],sum(x['accepted'] for x in cal_rows))
    write(out/'MATCHED_CALIBRATION.json',dict(rows=cal_rows,threshold=threshold,scope='Known calibration only, fixed rule, no dev threshold search'))
    predictions={'R':[],'H':[],'C_equal':[]};details={'R':[],'H':[],'C_equal':[]}
    for name in split['val_images']:
        raw=cache.load(name)
        d=distances(raw,adapter,bank,device,cfg['k'])
        old_d=native_distances(raw['z'],original_bank,k=cfg['k'])
        adapted=score_values(raw,d,bank,cal,cfg,'R')
        original=score_values(raw,old_d,original_bank,cal,cfg,'R')
        hybrid=hybrid_values(raw,adapted,original,bank['active'])
        for mode,values in [('R',adapted),('H',hybrid),('C_equal',boundary_values(raw,original,bank['active'],threshold['threshold']))]:
            predictions[mode].append(dict(image=name,**detections(raw,values,cfg)))
            details[mode].append(dict(image=name,query_index=raw['query_index'].tolist(),boxes=raw['boxes'].tolist(),
                a=(original['a'] if mode=='C_equal' else adapted['a']).tolist(),labels=values['labels'].tolist(),keep=values['keep'].tolist()))
        # Raw distances and support statistics distinguish semantic movement from
        # radius/scale drift. No GT labels or new features are stored here.
        write(out/'distance_diagnostics'/f'{Path(name).stem}.json',dict(image=name,query_index=raw['query_index'].tolist(),
            active=list(bank['active']),adapted_distance=d.tolist(),original_distance=old_d.tolist(),
            adapted_a=adapted['a'].tolist(),original_a=original['a'].tolist()))
    write(out/'SUPPORT.json',dict(active=list(bank['active']),
        adapted_radius=np.asarray(bank['radius']).tolist(),adapted_scale=np.asarray(bank['scale']).tolist(),
        original_radius=np.asarray(original_bank['radius']).tolist(),original_scale=np.asarray(original_bank['scale']).tolist(),
        calibration_policy='Original frozen object gate; no new calibration or threshold fit',
        readout_H='Adapted identity labels; original M1 novelty for both known and unknown scores; same class NMS and cap'))
    annotation_paths,truth=projected_truth(root,split,out/'projection')
    results={}
    for mode in ('R','H','C_equal'):
        metrics=evaluate_predictions(root,split,predictions[mode],out/mode,annotation_paths,truth)
        metrics['fixed_candidate_identity']=identity_summary(details[mode],truth,split)
        metrics['readout']=mode
        write(out/mode/'candidate_identity.json',details[mode]);write(out/mode/'metrics.json',metrics)
        results[mode]=metrics
    return results
