"""Frozen157 evidence export. No fitting, recalibration, or changes to157 files."""
import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from cil.common import iou
from current.progress import stage, track
from refnet_f1.protocol import read, write, sha, fp, lock, inputs, locations, verify_file, verify_parent, guard_space

TAG = 'reference_f1_candidates_v1'
REPLAY_MODE = 'strict'
EMPTY_ATTEMPT_SOURCE = '3b2e277997b84aad69423cadf405a17d4fe3e655ce53e588ffd2d24a5aba60f3'
EMPTY_ATTEMPT_SOURCES = {EMPTY_ATTEMPT_SOURCE, 'f35a94b5cb8c7b097820dd1a1124b114503d91de62ad237657d8de731965a466'}


def replay_difference(values, pred, diag, old_record):
    digest=hashlib.sha256()
    for key in ['ids','boxes','joint','logits']:digest.update(values[key].numpy().tobytes())
    previous=old_record['predictions']['F1']; previous_diag=old_record['diagnostics']['F1']
    ids=diag['selected_proposal_ids']; old_ids=previous_diag['selected_proposal_ids']
    new_by={v:i for i,v in enumerate(ids)}; old_by={v:i for i,v in enumerate(old_ids)}
    common=sorted(set(ids)&set(old_ids)); changes=[]
    for proposal in common:
        i,j=new_by[proposal],old_by[proposal]
        changes.append(dict(proposal_id=proposal,old_identity=previous['labels'][j],new_identity=pred['labels'][i],
            max_box_difference=max(abs(a-b) for a,b in zip(previous['boxes'][j],pred['boxes'][i])),
            score_difference=pred['scores'][i]-previous['scores'][j]))
    return dict(evidence_exact=digest.hexdigest()==old_record['shared_candidate_evidence_sha256'],
        predictions_exact=pred==previous,diagnostics_exact=diag==previous_diag,
        replay_evidence_sha256=digest.hexdigest(),original_evidence_sha256=old_record['shared_candidate_evidence_sha256'],
        old_diagnostics=previous_diag,new_diagnostics=diag,
        old_output_count=len(old_ids),new_output_count=len(ids),
        lost_output_ids=sorted(set(old_ids)-set(ids)),new_output_ids=sorted(set(ids)-set(old_ids)),
        common_output_changes=changes,common_identity_changes=sum(c['old_identity']!=c['new_identity'] for c in changes),
        maximum_common_box_difference=max((c['max_box_difference'] for c in changes),default=None),
        maximum_common_absolute_score_difference=max((abs(c['score_difference']) for c in changes),default=None),
        limitation='Original600 arrays were not saved; only their aggregate hash and retained outputs can be compared. No tolerance acceptance is inferred.')


def verify_bank_audit(audit, saved):
    # load_references returns an identity-bearing audit;157 lock() fingerprints
    # that complete value once more before replacing its identity field.
    expected = json.loads(json.dumps(audit))
    expected['identity'] = fp(audit)
    if expected != saved:
        fields = sorted(k for k in set(expected)|set(saved) if expected.get(k)!=saved.get(k))
        raise RuntimeError('157 reference bank differs in fields: '+', '.join(fields))


def diagnostic_lock(folder, value):
    path = folder/'LOCK.json'
    if path.exists():
        old = read(path); old_body = {k:v for k,v in old.items() if k!='identity'}
        migrated = json.loads(json.dumps(old_body))
        if migrated.get('source',{}).get('src/reference_f1_candidates.py') in EMPTY_ATTEMPT_SOURCES:
            migrated['source']['src/reference_f1_candidates.py'] = value['source']['src/reference_f1_candidates.py']
            if migrated == value and fp(old_body)==old['identity'] and old_body!=value:
                if any((folder/'images').glob('*')) or (folder/'SUMMARY.json').exists():
                    raise RuntimeError('Old162 contains candidate outputs; preserve it and use a separate diagnostic tag')
                archive=folder/('EMPTY_ATTEMPT_LOCK_'+old['identity']+'.json')
                if archive.exists():raise RuntimeError('Empty-attempt archive already exists; inspect before changing locks')
                path.rename(archive)
    return lock(path,value)


def home(root):
    return Path(root)/'reports/reference_f1_candidates'/TAG


def annotations(path, classes):
    result = []
    for n, obj in enumerate(ET.parse(path).getroot().findall('object')):
        name = obj.findtext('name')
        result.append(dict(gt_id=n, original_class=name, class_id=classes.index(name) if name in classes else len(classes),
            difficult=bool(int(obj.findtext('difficult', '0'))),
            box=[float(obj.find('bndbox').findtext(k))-1 for k in ['xmin','ymin','xmax','ymax']]))
    return result


def attribution(evidence, gt, known_count):
    """Diagnostic continuous IoU>=.5, not VOC AP matching; duplicate proposals counted explicitly."""
    boxes = evidence['boxes']; n = len(boxes)
    overlap = iou(boxes, [g['box'] for g in gt])
    valid = [i for i,g in enumerate(gt) if not g['difficult']]
    labels = np.asarray(evidence['F1_identity'], int)
    gate = np.asarray(evidence['passed_gate'], bool)
    kept = np.asarray(evidence['final_output'], bool)
    matches = []; counts = Counter(candidates=n)
    for j in range(n):
        own = max(valid, key=lambda k: float(overlap[j,k])) if valid else None
        ov = float(overlap[j,own]) if own is not None else 0.
        g = gt[own] if own is not None else None
        good = ov >= .5
        status = ('known_rejected_as_unknown' if labels[j]==known_count else 'known_accepted_correct' if labels[j]==g['class_id'] else 'known_wrong_class') if good and g['class_id']<known_count else ('unknown_accepted_as_known' if labels[j]<known_count else 'unknown_rejected_correct') if good else 'no_nondifficult_gt_iou50'
        counts[status] += 1
        if good:
            population = 'known' if g['class_id']<known_count else 'unknown'
            counts[population+'_localized_candidates'] += 1
            counts[status+'_passed_gate'] += int(gate[j])
            counts[population+'_localized_candidates_passed_gate'] += int(gate[j])
            counts[status+'_final_output'] += int(kept[j])
        matches.append(dict(candidate_rank=j, proposal_id=evidence['proposal_ids'][j],
            matched_gt_id=g['gt_id'] if g else None, matched_gt_class=g['original_class'] if g else None,
            matched_gt_class_id=g['class_id'] if g else None, matched_gt_iou=ov, localized_iou50=good, attribution=status))
    target_records = []
    for k,g in enumerate(gt):
        if g['difficult']:continue
        covered = overlap[:,k] >= .5
        correct = labels == g['class_id']
        wrong_identity = labels==known_count if g['class_id']<known_count else labels<known_count
        pop = 'known' if g['class_id']<known_count else 'unknown'
        hit = bool(covered.any()); surviving = bool((covered & gate).any())
        any_correct = bool((covered & correct).any())
        all_wrong = hit and bool(wrong_identity[covered].all())
        item = dict(**g, covering_candidates=int(covered.sum()), covering_after_gate=int((covered & gate).sum()),
            correctly_identified_candidates=int((covered & correct).sum()),
            identity_error_candidates=int((covered & wrong_identity).sum()),
            correct_after_gate=int((covered & gate & correct).sum()),
            correct_final_output=int((covered & kept & correct).sum()),
            no_localized_candidate=not hit, all_covered_candidates_identity_error=all_wrong,
            all_localized_candidates_gate_deleted=hit and not surviving,
            all_gate_surviving_candidates_identity_error=surviving and bool(wrong_identity[covered & gate].all()))
        target_records.append(item)
        for key,value in dict(gt=1,covered_gt=hit,covered_gt_with_any_identity_error=bool((covered & wrong_identity).any()),
            covered_gt_all_identity_error=all_wrong,covered_gt_with_correct_identity=any_correct,
            covered_gt_all_gate_deleted=hit and not surviving,
            gate_surviving_gt=surviving,gate_surviving_gt_all_identity_error=item['all_gate_surviving_candidates_identity_error'],
            gt_with_correct_final_output=item['correct_final_output']>0).items():counts[pop+'_'+key] += int(value)
    return dict(all_gt=gt, candidate_matches=matches, gt_records=target_records, counts=dict(counts),
        matching='Continuous XYXY IoU>=0.5; closest non-difficult GT for candidate counts; every GT independently for object coverage; not official AP matching')


def percentages(counts):
    pairs = dict(known_candidate_false_reject_rate=('known_rejected_as_unknown','known_localized_candidates'),
        unknown_candidate_absorption_rate=('unknown_accepted_as_known','unknown_localized_candidates'),
        known_candidate_false_reject_after_gate_rate=('known_rejected_as_unknown_passed_gate','known_localized_candidates_passed_gate'),
        unknown_candidate_absorption_after_gate_rate=('unknown_accepted_as_known_passed_gate','unknown_localized_candidates_passed_gate'),
        known_gt_all_localized_candidates_rejected_rate=('known_covered_gt_all_identity_error','known_covered_gt'),
        unknown_gt_all_localized_candidates_absorbed_rate=('unknown_covered_gt_all_identity_error','unknown_covered_gt'))
    return {k:dict(numerator=counts.get(a,0),denominator=counts.get(b,0),percent=100*counts.get(a,0)/counts[b] if counts.get(b,0) else None) for k,(a,b) in pairs.items()}


def save_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    tmp.replace(path)


def load_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def run_case(root, cfg, scene, task, seed, device, dataset, verified):
    import torch
    from cil.common import configure_torch
    from jointdet.protocol import load_torch
    from jointdet.model import pf_score
    from refnet.model import ReferenceDetector
    from scenepilot.protocol import view
    from refnet_f1.runtime import load_references, candidate_values, decode
    spec,parent,ann = inputs(root,scene,task,seed)
    old = locations(root,cfg,scene,task,seed); result = read(old/'RESULT.json'); old_lock = read(old/'LOCK.json')
    if not result['complete'] or result['parent_checkpoint']!=parent['terminal']:raise RuntimeError('Require completed157 result with original terminal parent')
    if old_lock['config']!=cfg or old_lock['annotation']!=ann:raise RuntimeError('157 config/dataset changed')
    for name,want in track(sorted(old_lock['sources'].items()),f'162 {scene}/{task}/s{seed} verify157 sources',unit='files'):verify_file(root/name,want,verified)
    for name in ['CALIBRATION.json','REFERENCE_BANK.json','PREDICTIONS_COMPLETE.json']:
        verify_file(old/name,result['artifacts'][name],verified)
    calibration = read(old/'CALIBRATION.json')
    cal_numbers={k:v for k,v in calibration.items() if k!='identity'}
    if cal_numbers!=result['calibration']:raise RuntimeError('157 calibration summary mismatch')
    verify_parent(root,parent,verified)
    original=read(root/'configs/reference_net.json'); c=original['model']
    if (c['proposal_limit'],c['detection_limit'],c['nms_iou'])!=(600,100,.5):raise RuntimeError('Require unchanged157 detection budget')
    outdir=home(root)/scene/task/f'seed_{seed}'
    own_source={n:sha(root/n) for n in ['run_reference_f1_candidates.py','src/reference_f1_candidates.py']}
    replay_env=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
    binding=diagnostic_lock(outdir,dict(format=TAG,source=own_source,original_result=sha(old/'RESULT.json'),
        original_case=old_lock['identity'],calibration=sha(old/'CALIBRATION.json'),annotation=ann,
        candidate_limit=600,iou=.5,difficult='excluded from rates',training_updates=0,
        **(dict(replay_mode=REPLAY_MODE,replay_environment=replay_env) if REPLAY_MODE=='diagnostic' else {})))
    records=[]; total=Counter(); pending=[]
    for name in track(spec['test_images'],f'162 {scene}/{task}/s{seed} check resume',unit='images'):
        image=root/'assets/rwd/JPEGImages'/scene/name; xml=root/'assets/rwd/Annotations'/scene/(Path(name).stem+'.xml')
        ih=verify_file(image,dataset['files'][f'JPEGImages/{scene}/{name}']['sha256'],verified)
        xh=verify_file(xml,dataset['files'][f'Annotations/{scene}/{Path(name).stem}.xml']['sha256'],verified)
        dest=outdir/'images'/(fp(name)[:24]+'.json.gz'); commit=dest.with_suffix('.commit.json')
        identity=fp(dict(case=binding['identity'],image=name,image_sha256=ih,xml_sha256=xh))
        if dest.exists() and commit.exists():
            receipt=read(commit)
            if receipt['identity']!=identity:raise RuntimeError('162 resume binding changed')
            verify_file(dest,receipt['sha256'],verified); record=load_gzip(dest)
            if record['identity']!=identity or fp(record['evidence'])!=record['evidence_fingerprint']:raise RuntimeError('162 evidence commit mismatch')
            total.update(record['analysis']['counts']);records.append(dict(path=dest.relative_to(root).as_posix(),sha256=receipt['sha256']))
        else:pending.append((name,ih,xh,dest,commit,identity))
    if pending:
        guard_space(root,cfg);configure_torch(seed,2)
        with stage(f'162 {scene}/{task}/s{seed} load frozen157 parent'):
            model=ReferenceDetector(root,c,list(range(len(spec['classes'])))).to(device)
            state=load_torch(root/parent['terminal']['checkpoint'])
            if state['step']!=500 or not state['complete']:raise RuntimeError('Incomplete terminal weight')
            model.load_adapted(state['model']);del state;model.requires_grad_(False);model.eval()
        try:
            bank,_,audit=load_references(root,spec,parent,device,dataset,verified)
            verify_bank_audit(audit,read(old/'REFERENCE_BANK.json'))
            v=view(root,scene)
            for name,ih,xh,dest,commit,identity in track(pending,f'162 {scene}/{task}/s{seed} frozen600 export',unit='images'):
                guard_space(root,cfg)
                with torch.inference_mode():
                    output=model.forward_image(v,name,[],device);values=candidate_values(model,output,bank,cfg,c)
                    pred,diag=decode(values,'F1',calibration,c)
                    old_record=read(old/'test_images'/(fp(name)[:24]+'.json'))
                    difference=replay_difference(values,pred,diag,old_record)
                    replay_exact=all(difference[k] for k in ['evidence_exact','predictions_exact','diagnostics_exact'])
                    if not replay_exact and REPLAY_MODE=='strict':
                        report=dict(image=name,case=binding['identity'],difference=difference,
                            replay_environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=str(device),
                                gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32),
                            original_environment=read(locations(root,cfg)/'ENVIRONMENT.json') if (locations(root,cfg)/'ENVIRONMENT.json').exists() else None,
                            calibration=calibration,original_image_sha256=old_record['image_sha256'],current_image_sha256=ih,
                            accepted_as_original157=False)
                        write(outdir/'REPLAY_MISMATCH.json',report)
                        save_gzip(outdir/'REPLAY_MISMATCH_EVIDENCE.json.gz',dict(image=name,accepted_as_original157=False,
                            proposal_ids=values['ids'].tolist(),boxes=values['boxes'].tolist(),gate_score=values['joint'].tolist(),
                            logits=values['logits'].tolist(),F1_prediction=pred,diagnostics=diag))
                        raise RuntimeError('Frozen replay differs from157; details saved to '+str(outdir/'REPLAY_MISMATCH.json'))
                    logits=values['logits'];margin=logits[:,-1]-logits[:,:-1].max(1).values
                    labels=logits[:,:-1].argmax(1);labels=torch.where(margin>calibration['threshold_margin'],len(spec['classes']),labels)
                    ids=values['ids'].tolist();selected=set(diag['selected_proposal_ids']);passed=(values['joint']>=calibration['object_threshold']).tolist()
                    # Separate NMS survivors from the final top100 budget using the same CPU operation.
                    from torchvision.ops import batched_nms
                    u=margin.sigmoid();pk=logits[:,:-1].softmax(1).max(1).values
                    score=values['joint']*torch.where(labels==len(spec['classes']),u,(1-u)*pk)
                    eligible=torch.nonzero(torch.tensor(passed)).flatten()
                    survivors=eligible[batched_nms(values['boxes'][eligible],score[eligible],labels[eligible],c['nms_iou'])] if len(eligible) else eligible
                    nms_ids={ids[i] for i in survivors.tolist()}
                    objectness=(pf_score(output['cls'][-1],output['ctr'][-1])[values['ids'].to(device)]/2).cpu()
                    quality=output['quality_logit'][values['ids'].to(device)].sigmoid().cpu()
                    evidence=dict(proposal_ids=ids,boxes=values['boxes'].tolist(),logits=logits.tolist(),margin=margin.tolist(),
                        best_known_identity=logits[:,:-1].argmax(1).tolist(),F1_identity=labels.tolist(),
                        objectness=objectness.tolist(),quality=quality.tolist(),gate_score=values['joint'].tolist(),
                        margin_threshold=calibration['threshold_margin'],gate_threshold=calibration['object_threshold'],
                        passed_gate=passed,final_score=score.tolist(),final_output=[i in selected for i in ids],
                        suppression_reason=['gate' if not g else 'nms' if i not in nms_ids else 'top100_budget' if i not in selected else 'output' for i,g in zip(ids,passed)],
                        original157_evidence_sha256=difference['original_evidence_sha256'],replay_evidence_sha256=difference['replay_evidence_sha256'],
                        original157_predictions_exactly_reproduced=difference['predictions_exact'],
                        original157_all_evidence_exact=replay_exact,F1_prediction=pred,replay_difference=difference)
                    evidence_fingerprint=fp(evidence)
                    del output,values
                # Official XML is read only after frozen evidence and predictions have been sealed.
                gt=annotations(root/'assets/rwd/Annotations'/scene/(Path(name).stem+'.xml'),spec['classes'])
                analysis=attribution(evidence,gt,len(spec['classes']))
                record=dict(identity=identity,image=name,image_sha256=ih,xml_sha256=xh,classes=spec['classes'],PK=spec['PK'],CK=spec['CK'],
                    evidence=evidence,evidence_fingerprint=evidence_fingerprint,analysis=analysis)
                save_gzip(dest,record);receipt=dict(identity=identity,sha256=sha(dest));write(commit,receipt)
                records.append(dict(path=dest.relative_to(root).as_posix(),sha256=receipt['sha256']));total.update(analysis['counts'])
        finally:
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    summary=dict(complete=len(records)==len(spec['test_images']),scene=scene,task=task,seed=seed,images=len(records),
        expected_images=len(spec['test_images']),counts=dict(total),rates=percentages(total),files=sorted(records,key=lambda r:r['path']),
        training_updates=0,calibration_recomputed=False,source_parent=parent['terminal'],lock=binding['identity'])
    if REPLAY_MODE=='diagnostic':
        from refnet.evaluation import evaluate
        predictions=[];differences=[]
        for item in track(summary['files'],f'162 {scene}/{task}/s{seed} seal replay predictions',unit='images'):
            row=load_gzip(root/item['path']);predictions.append(dict(image=row['image'],**row['evidence']['F1_prediction']))
            differences.append(dict(image=row['image'],**row['evidence']['replay_difference']))
        order={name:i for i,name in enumerate(spec['test_images'])};predictions.sort(key=lambda r:order[r['image']])
        path=outdir/'REPLAY_F1/predictions.json';write(path,predictions)
        lock(outdir/'REPLAY_PREDICTIONS_COMPLETE.json',dict(case=binding['identity'],prediction_sha256=sha(path),images=len(predictions)))
        with stage(f'162 {scene}/{task}/s{seed} evaluate current-device frozen replay'):
            metrics=evaluate(root,path,spec,outdir/'REPLAY_F1')
        summary.update(replay_mode='diagnostic',interpretation='Current-device frozen replay; NOT original157 candidate records',
            replay_metrics=metrics,original157_metrics=result['metrics']['F1'],
            metric_delta={k:metrics[k]-result['metrics']['F1'][k] if metrics.get(k) is not None and result['metrics']['F1'].get(k) is not None else None for k in ['K_AP50','U_AP50','PK_AP50','CK_AP50']},
            mismatch_images={key:sum(not r[key] for r in differences) for key in ['evidence_exact','predictions_exact','diagnostics_exact']})
        write(outdir/'REPLAY_DIFFERENCES.json',differences)
    write(outdir/'SUMMARY.json',summary)
    return summary


def summarize(root):
    cases=[read(p) for p in sorted(home(root).glob('*/*/seed_*/SUMMARY.json'))]
    result=dict(cases=len(cases),expected_cases=30,complete=len(cases)==30 and all(c['complete'] for c in cases),
        by_case=[{k:v for k,v in c.items() if k!='files'} for c in cases],
        note='No pooled cross-scene/seed rate presented as independent evidence; duplicate candidate counts and distinct GT counts reported separately.')
    write(home(root)/'SUMMARY.json',result)
    lines=['# Frozen157 top600 candidate diagnosis','',f'Replay mode: {REPLAY_MODE}. Diagnostic mode is CURRENT-DEVICE replay, not original157 candidate records.','',
        'Diagnostic: continuous XYXY IoU>=0.5, non-difficult GT. Identity is recorded before gate/NMS; final_output is after top100. No training or calibration.','',
        '| Case | Localized known candidates | Known→unknown | Localized unknown candidates | Unknown→known | Covered known GT: all rejected | Covered unknown GT: all absorbed |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in cases:
        c=r['counts'];fmt=lambda a,b:f"{c.get(a,0)}/{c.get(b,0)}"
        lines.append('| '+f"{r['scene']}/{r['task']}/s{r['seed']}"+' | '+' | '.join([str(c.get('known_localized_candidates',0)),str(c.get('known_rejected_as_unknown',0)),str(c.get('unknown_localized_candidates',0)),str(c.get('unknown_accepted_as_known',0)),fmt('known_covered_gt_all_identity_error','known_covered_gt'),fmt('unknown_covered_gt_all_identity_error','unknown_covered_gt')])+' |')
    (home(root)/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result


def export(root,cfg):
    import zipfile
    summary=summarize(root);paths=[home(root)/'SUMMARY.json',home(root)/'SUMMARY.md']
    for p in sorted(home(root).glob('*/*/seed_*/SUMMARY.json')):
        r=read(p);paths.extend([p,p.parent/'LOCK.json'])
        if r.get('replay_mode')=='diagnostic':
            paths.extend([p.parent/'REPLAY_DIFFERENCES.json',p.parent/'REPLAY_PREDICTIONS_COMPLETE.json'])
            paths.extend(sorted((p.parent/'REPLAY_F1').glob('*.json')))
        for item in r['files']:
            path=root/item['path'];verify_file(path,item['sha256']);paths.append(path)
    if not summary['cases']:raise RuntimeError('No completed diagnostic case; run first')
    guard_space(root,cfg,sum(p.stat().st_size for p in paths))
    dest=home(root)/'REFERENCE_F1_CANDIDATES.zip';tmp=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in track(paths,'162 pack all candidate evidence',unit='files'):z.write(p,p.relative_to(root).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.gz' else zipfile.ZIP_DEFLATED)
    with stage('162 verify diagnostic ZIP CRC'):
        with zipfile.ZipFile(tmp) as z:
            if z.testzip():raise RuntimeError('ZIP CRC failed')
    tmp.replace(dest);receipt=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),cases=summary['cases'],complete=summary['complete']);write(home(root)/'DOWNLOAD_RECEIPT.json',receipt);return receipt


def export_candidates_only(root):
    """CPU-only packaging; read-only with respect to committed inference files."""
    import zipfile
    root=Path(root);base=home(root);members=[];cases=0
    for p in sorted(base.glob('*/*/seed_*/SUMMARY.json')):
        record=read(p)
        if not record.get('complete'):raise RuntimeError('Incomplete diagnostic case: '+str(p))
        for item in record['files']:
            path=root/item['path']
            if path.parent!=p.parent/'images' or not path.name.endswith('.json.gz'):
                raise RuntimeError('Unexpected candidate record path: '+str(path))
            members.append((path,item['sha256']))
        cases+=1
    if not members:raise RuntimeError('No completed candidate JSON records found')
    if len({str(p) for p,_ in members})!=len(members):raise RuntimeError('Duplicate candidate record paths')
    for p,h in track(members,'162 verify candidate JSON files',unit='files'):verify_file(p,h)
    # Files are already gzip-compressed. Stored ZIP needs approximately their
    # total size, plus ZIP headers; inference's8GiB reserve is irrelevant here.
    required=sum(p.stat().st_size+2*len(p.relative_to(root).as_posix().encode('utf-8'))+256 for p,_ in members)+256*2**20
    free=shutil.disk_usage(base).free
    if free<required:raise RuntimeError(f'Candidate-only ZIP needs {required/2**30:.2f} GiB including256MiB reserve; available {free/2**30:.2f} GiB. Download images folders directly or free the difference.')
    dest=base/'REFERENCE_F1_CANDIDATE_JSON_ONLY.zip';tmp=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for p,_ in track(members,'162 pack candidate JSON only',unit='files'):z.write(p,p.relative_to(root).as_posix())
    with stage('162 verify candidate-only ZIP CRC and SHA256'):
        with zipfile.ZipFile(tmp) as z:
            if z.testzip():raise RuntimeError('Candidate-only ZIP CRC failed')
        digest=sha(tmp)
    tmp.replace(dest)
    receipt=dict(path=str(dest),bytes=dest.stat().st_size,sha256=digest,cases=cases,complete=cases==30,
        candidate_json_files=len(members),contains_images=False,contains_weights=False,contains_only_original_candidate_json=True)
    write(base/'CANDIDATE_JSON_DOWNLOAD_RECEIPT.json',receipt)
    return receipt


def main(root,argv=None):
    global TAG, REPLAY_MODE
    parser=argparse.ArgumentParser(description='162: frozen157 top600 evidence export, no retraining or threshold tuning')
    parser.add_argument('command',choices=['run','summary','export','pack-json','stage-upload'])
    parser.add_argument('--scenes',nargs='+');parser.add_argument('--tasks',nargs='+');parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--replay-mode',choices=['strict','diagnostic'],default='strict')
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args(argv);root=Path(root);cfg=read(root/'configs/reference_f1.json')
    REPLAY_MODE=args.replay_mode
    TAG='reference_f1_candidates_replay_v1' if REPLAY_MODE=='diagnostic' else 'reference_f1_candidates_v1'
    if args.command=='stage-upload':
        files=['run_reference_f1_candidates.py','src/reference_f1_candidates.py','tests/test_reference_f1_candidates.py','docs/REFERENCE_F1_CANDIDATES.md','.run/162_Reference_F1_Candidates.run.xml']
        out=home(root)/'UPLOAD/OpenDet_Current'
        for n in files:
            dest=out/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/n,dest)
        write(out/'REFERENCE_F1_CANDIDATES_UPLOAD.json',dict(files={n:sha(root/n) for n in files},original157_files_modified=False))
        result=dict(upload_directory=str(out),files=len(files))
    elif args.command=='summary':
        s=summarize(root);result=dict(cases=s['cases'],complete=s['complete'],path=str(home(root)/'SUMMARY.md'))
    elif args.command=='export':result=export(root,cfg)
    elif args.command=='pack-json':result=export_candidates_only(root)
    else:
        import torch
        scenes=args.scenes or cfg['scenes'];tasks=args.tasks or cfg['tasks'];seeds=args.seeds or cfg['seeds']
        for chosen,allowed in [(scenes,cfg['scenes']),(tasks,cfg['tasks']),(seeds,cfg['seeds'])]:
            if not set(chosen)<=set(allowed) or len(set(chosen))!=len(chosen):parser.error('Choose unique configured scenes/tasks/seeds')
        device=torch.device(args.device)
        if device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run on remote GPU')
            torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        dataset=read(root/'assets/rwd/DATASET_MANIFEST.json');verified={};index=read(root/'MAINLINE_PARENT_INDEX.json')
        for n,h in track(sorted(index['base_weights'].items()),'162 verify original PF/DINO weights',unit='files'):verify_file(root/n,h,verified)
        cases=[(s,t,z) for s in scenes for t in tasks for z in seeds]
        for s,t,z in track(cases,'162 frozen157 candidate diagnosis',unit='cases'):
            run_case(root,cfg,s,t,z,device,dataset,verified);summarize(root)
        result=export(root,cfg)
    print(json.dumps(result,ensure_ascii=False,indent=2))
