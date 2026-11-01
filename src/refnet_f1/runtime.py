from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps
from cil.common import iou
from current.progress import track, stage
from .protocol import read, write, sha, fp, lock, verify_file, remap_cached, guard_space


def load_references(root, spec, parent, device, dataset, verified):
    root = Path(root); folder = root/parent['reference_cache']
    verify_file(folder/'MANIFEST.json', parent['reference_manifest_sha256'], verified)
    manifest = read(folder/'MANIFEST.json'); original = read(root/parent['split'])
    if manifest['split'] != original['identity']:
        raise RuntimeError('Original frozen GT cache has another split')
    row_index = {(r['role'], r['image']): r for r in manifest['rows']}
    if len(row_index) != len(manifest['rows']):
        raise RuntimeError('Duplicate GT cache rows')
    bank_raw, labels, reference_rows, calibration_gt = [], [], [], {}
    old_by = {}; new_by = {}
    for obj in original['objects']: old_by.setdefault(obj['image'], []).append(obj)
    for obj in spec['objects']: new_by.setdefault(obj['image'], []).append(obj)
    for role in ['fit', 'cal']:
        for name in track(spec[role+'_images'], f'{spec["scene"]}/{spec["task"]} verify {role} GT references', unit='images'):
            row = row_index[role, name]; path = folder/row['file']
            verify_file(path, manifest['files'][row['file']], verified)
            image = root/'assets/rwd/JPEGImages'/spec['scene']/name
            verify_file(image, dataset['files'][f'JPEGImages/{spec["scene"]}/{name}']['sha256'], verified)
            with Image.open(image) as im:
                w, h = ImageOps.exif_transpose(im).size
            with np.load(path, allow_pickle=False) as z:
                raw, ys = remap_cached(z['raw'], z['ids'].tolist(), z['gt'], z['labels'],
                                      old_by.get(name, []), new_by.get(name, []), np.array([w, h, w, h]))
            if role == 'fit':
                bank_raw.append(raw); labels.extend(ys)
                reference_rows.extend(dict(id=o['id'], original_id=o.get('original_instance_id', o['id']),
                    image=name, class_id=y, group=spec['groups'][name], cache_sha256=manifest['files'][row['file']])
                    for o, y in zip(new_by[name], ys))
            else:
                calibration_gt[name] = raw
    if set(labels) != set(range(len(spec['classes']))):
        raise RuntimeError('Full known bank has missing class')
    audit = dict(original_manifest_sha256=parent['reference_manifest_sha256'], split_identity=spec['identity'],
                 rows=reference_rows, class_counts=dict(Counter(map(str, labels))), corrected_labels=spec['scene']=='Game',
                 feature_policy='same frozen DINO descriptor for the same original GT identity and box; corrected class labels applied explicitly')
    audit['identity'] = fp(audit)
    return dict(raw=torch.tensor(np.concatenate(bank_raw), device=device),
                labels=torch.tensor(labels, device=device, dtype=torch.long)), calibration_gt, audit


@torch.inference_mode()
def relation_logits(model, raw, bank, cfg):
    active = list(range(len(model.active)))
    values = [model.relation(raw[i:i+cfg['relation_batch']], bank['raw'], bank['labels'], active)
              for i in range(0, len(raw), cfg['relation_batch'])]
    return torch.cat(values) if values else raw.new_empty((0, len(active)+1))


@torch.inference_mode()
def candidate_values(model, out, bank, cfg, parent_cfg):
    from jointdet.model import pf_score
    from jointdet.geometry import xyxy
    pf = pf_score(out['cls'][-1], out['ctr'][-1])
    ids = pf.argsort(descending=True, stable=True)[:parent_cfg['proposal_limit']]
    logits = relation_logits(model, out['raw'][ids], bank, cfg)
    joint = pf[ids]/2*out['quality_logit'][ids].sigmoid()
    boxes = xyxy(out['refined'][ids]).clamp(0, 1)
    if not all(torch.isfinite(v).all() for v in [logits, joint, boxes]):
        raise RuntimeError('Nonfinite frozen parent candidate values')
    wh = boxes.new_tensor([out['meta']['ori_shape'][1], out['meta']['ori_shape'][0]]*2)
    return dict(logits=logits.cpu(), joint=joint.cpu(), boxes=(boxes*wh).cpu(), ids=ids.cpu())


def calibrate_threshold(margins, target=.95):
    values = np.asarray(margins, np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError('Empty/nonfinite known calibration; no test or GT fallback for F1')
    return float(np.quantile(values, target, method='higher'))


def decode(values, arm, calibration, parent_cfg):
    from torchvision.ops import batched_nms
    logits, joint, boxes = values['logits'], values['joint'], values['boxes']
    if arm == 'F0':
        shifted = logits.clone(); shifted[:, -1] -= calibration['f0_reject_shift']
        confidence, label = shifted.softmax(1).max(1)
    elif arm == 'F1':
        margin = logits[:, -1]-logits[:, :-1].max(1).values
        u = margin.sigmoid(); pk, label = logits[:, :-1].softmax(1).max(1)
        unknown = margin > calibration['threshold_margin']
        label = torch.where(unknown, torch.full_like(label, logits.shape[1]-1), label)
        confidence = torch.where(unknown, u, (1-u)*pk)
    else:
        raise ValueError('Unknown frozen rejection control')
    scores = joint*confidence
    keep = torch.nonzero(joint >= calibration['object_threshold']).flatten()
    if len(keep):
        keep = keep[batched_nms(boxes[keep], scores[keep], label[keep], parent_cfg['nms_iou'])[:parent_cfg['detection_limit']]]
    return dict(boxes=boxes[keep].tolist(), scores=scores[keep].tolist(), labels=label[keep].tolist()), dict(
        selected_proposal_ids=values['ids'][keep].tolist(), passed_gate=int((joint >= calibration['object_threshold']).sum()),
        identity_unknown=int((label == logits.shape[1]-1).sum()), output_unknown=int((label[keep] == logits.shape[1]-1).sum()))


@torch.inference_mode()
def calibrate(model, view, root, spec, bank, cal_gt, cfg, parent_cfg, parent_cal, report, verified, dataset):
    report = Path(report); commits = report/'cal_images'; device = next(model.parameters()).device
    gt_by = {}
    for obj in spec['cal_objects']: gt_by.setdefault(obj['image'], []).append(obj)
    records = []
    for name in track(spec['cal_images'], f'{spec["scene"]}/{spec["task"]} F1 known-only calibration', unit='images'):
        path = commits/(fp(name)[:24]+'.json')
        image = Path(root)/'assets/rwd/JPEGImages'/spec['scene']/name
        image_sha = verify_file(image, dataset['files'][f'JPEGImages/{spec["scene"]}/{name}']['sha256'], verified)
        if path.exists():
            record = read(path)
            if record['image'] != name or record['image_sha256'] != image_sha:
                raise RuntimeError('Calibration resume image mismatch')
        else:
            out = model.forward_image(view, name, [], device)
            v = candidate_values(model, out, bank, cfg, parent_cfg)
            obs = gt_by.get(name, [])
            overlaps = iou(v['boxes'].numpy(), [o['box'] for o in obs])
            best = overlaps.max(1) if obs else np.zeros(len(v['boxes']))
            owner = overlaps.argmax(1) if obs else np.zeros(len(v['boxes']), int)
            eligible = (best >= cfg['calibration_min_iou']) & (v['joint'].numpy() >= parent_cal['object_threshold'])
            margin = (v['logits'][:, -1]-v['logits'][:, :-1].max(1).values).numpy()
            gt_margins = []
            if spec['scene'] == 'Game' and len(cal_gt[name]):
                gt_logits = relation_logits(model, torch.tensor(cal_gt[name], device=device), bank, cfg)
                gt_margins = (gt_logits[:, -1]-gt_logits[:, :-1].max(1).values).cpu().tolist()
            record = dict(image=name, image_sha256=image_sha, group=spec['groups'][name],
                gt_count=len(obs), candidates=len(margin), passed_gate=int((v['joint'] >= parent_cal['object_threshold']).sum()),
                samples=[dict(proposal_id=int(v['ids'][j]), margin=float(margin[j]), joint=float(v['joint'][j]),
                              iou=float(best[j]), class_id=int(obs[owner[j]]['class_id'])) for j in np.flatnonzero(eligible)],
                corrected_bank_gt_margins=gt_margins)
            write(path, record)
            del out, v
        records.append(record)
    samples = [s for r in records for s in r['samples']]
    threshold = calibrate_threshold([s['margin'] for s in samples], cfg['known_retention_target'])
    per_class = {}
    for cid, name in enumerate(spec['classes']):
        selected = [(r, s) for r in records for s in r['samples'] if s['class_id']==cid]
        per_class[str(cid)] = dict(name=name, cal_gt=sum(o['class_id']==cid for o in spec['cal_objects']),
            candidates=len(selected), source_groups=len({r['group'] for r, _ in selected}),
            retained_fraction=float(np.mean([s['margin'] <= threshold for _, s in selected])) if selected else None)
    f0_shift = parent_cal['reject_shift']
    if spec['scene']=='Game':
        f0_shift = max(0., calibrate_threshold([m for r in records for m in r['corrected_bank_gt_margins']]))
    cal = dict(threshold_margin=threshold, threshold_probability=float(torch.tensor(threshold, dtype=torch.float64).sigmoid()),
        target=cfg['known_retention_target'], empirical_known_retention=float(np.mean([s['margin'] <= threshold for s in samples])),
        calibrated_candidates=len(samples), calibrated_source_groups=len({r['group'] for r in records if r['samples']}),
        per_class=per_class, missing_predicted_calibration_classes=[int(c) for c, v in per_class.items() if v['candidates']==0],
        thin_predicted_calibration_classes=[int(c) for c, v in per_class.items() if v['source_groups']<2],
        rare_fit_classes=spec['rare_classes'], object_threshold=parent_cal['object_threshold'],
        object_threshold_fallback=parent_cal.get('object_threshold_fallback', False),
        f0_reject_shift=f0_shift, f0_shift_policy='corrected-bank known GT Q95 clamped to zero' if spec['scene']=='Game' else 'unchanged indexed parent calibration',
        quantile='numpy higher; F1 margin threshold is NOT clamped to zero; ties accepted known',
        population='all original top600 cal predictions with known-shot IoU>=0.5 and unchanged parent quality gate; pooled candidate weighting',
        limitations='95% is an empirical pooled known-calibration target, not a per-class or test guarantee; no source repair or reference-bank changes',
        test_labels_used=False, new_training_updates=0)
    lock(report/'CALIBRATION.json', cal)
    return cal


@torch.inference_mode()
def predict(model, view, root, spec, bank, cfg, parent_cfg, cal, report, dataset, verified):
    report = Path(report); device = next(model.parameters()).device
    records = []; commits = report/'test_images'
    for index, name in enumerate(track(spec['test_images'], f'{spec["scene"]}/{spec["task"]} F0+F1 official test', unit='images')):
        if index % 32 == 0: guard_space(root, cfg)
        image = Path(root)/'assets/rwd/JPEGImages'/spec['scene']/name
        image_sha = verify_file(image, dataset['files'][f'JPEGImages/{spec["scene"]}/{name}']['sha256'], verified)
        path = commits/(fp(name)[:24]+'.json')
        if path.exists():
            record = read(path)
            if record['image'] != name or record['image_sha256'] != image_sha:
                raise RuntimeError('Prediction resume image mismatch')
        else:
            out = model.forward_image(view, name, [], device)
            v = candidate_values(model, out, bank, cfg, parent_cfg)
            predictions = {}; diagnostics = {}
            for arm in cfg['arms']:
                predictions[arm], diagnostics[arm] = decode(v, arm, cal, parent_cfg)
            trace = hashlib.sha256()
            for key in ['ids', 'boxes', 'joint', 'logits']:
                trace.update(v[key].numpy().tobytes())
            record = dict(image=name, image_sha256=image_sha, candidates=len(v['ids']),
                shared_candidate_evidence_sha256=trace.hexdigest(), predictions=predictions, diagnostics=diagnostics)
            write(path, record)
            del out, v
        records.append(record)
    for arm in cfg['arms']:
        write(report/arm/'predictions.json', [dict(image=r['image'], **r['predictions'][arm]) for r in records])
    # Seal every prediction before reading official test XML labels.
    seal = lock(report/'PREDICTIONS_COMPLETE.json', dict(images=spec['test_images'],
        calibration_sha256=sha(report/'CALIBRATION.json'),
        predictions={arm: sha(report/arm/'predictions.json') for arm in cfg['arms']}))
    return seal


def official_evaluation(root, spec, cfg, report, dataset, verified):
    from refnet.evaluation import evaluate
    report = Path(report)
    for name in track(spec['test_images'], spec['scene']+' verify official test annotations', unit='XML'):
        rel = f'Annotations/{spec["scene"]}/{Path(name).stem}.xml'
        verify_file(Path(root)/'assets/rwd'/rel, dataset['files'][rel]['sha256'], verified)
    metrics = {}
    for arm in cfg['arms']:
        with stage(f'{spec["scene"]}/{spec["task"]} {arm} official VOC AP50'):
            metrics[arm] = evaluate(root, report/arm/'predictions.json', spec, report/arm)
    return metrics
