"""Bounded teacher-response anchoring, not hard background pseudo-labeling."""
import torch
from torch.nn import functional as F
from jointdet.geometry import xyxy, cxcywh, overlaps, paired_giou
from jointdet.losses import joint_loss as legacy_loss, assignment


def select_response(boxes, gt, occupied, target, c):
    """Discrete routing is stop-gradient; the loss still trains selected logits."""
    with torch.no_grad():
        valid = torch.nonzero(target['response_valid'].detach()).flatten()
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        if not len(valid) or not len(boxes): return empty, empty
        ov = overlaps(xyxy(boxes.detach()).clamp(0, 1), xyxy(target['boxes'].detach()[valid]))
        best, nearest = ov.max(1); matched = valid[nearest]
        eligible = best >= c['response_student_match_iou']
        if len(gt): eligible &= overlaps(xyxy(boxes.detach()).clamp(0, 1), xyxy(gt.detach())).max(1).values < c['response_gt_exclusion_iou']
        eligible[occupied] = False
        ids = torch.nonzero(eligible).flatten()
        order = ids[torch.argsort(best[ids], descending=True, stable=True)]
        low = target['response_low'].detach()[matched]
        limit = c['response_per_gt'] * max(1, len(gt))
        a = order[low[order]][:limit // 2]
        b = order[~low[order]][:limit - limit // 2]
        chosen = torch.cat([a, b])
        if len(chosen) < limit:
            used = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device); used[chosen] = True
            chosen = torch.cat([chosen, order[~used[order]][:limit - len(chosen)]])
        return chosen, matched[chosen]


def response_loss(logits, boxes, gt, occupied, target, c):
    qi, ti = select_response(boxes, gt, occupied, target, c)
    if not len(qi):
        return logits.sum() * 0, dict(anchors=0, low_response=0, other_response=0, mean_abs_logit_drift=None)
    token = 1 if logits.shape[-1] > 1 else 0
    teacher = target['logits'].detach()[ti]
    student = logits[qi, token]
    # No sigmoid calibration assumption; Huber bounds a wrong teacher target's influence.
    loss = F.smooth_l1_loss(student, teacher, beta=c['response_beta'])
    low = int(target['response_low'].detach()[ti].sum())
    return loss, dict(anchors=len(qi), low_response=low, other_response=len(qi)-low,
                      mean_abs_logit_drift=float((student.detach()-teacher).abs().mean()))


def geometry_loss(boxes, ctr, gt, occupied, target, c):
    zero = boxes.sum() * 0
    with torch.no_grad():
        teacher_boxes = target['boxes'].detach(); teacher_ctr = target['ctr'].detach()
        reliable = target['geometry_valid'].detach().clone()
        if len(gt): reliable &= overlaps(xyxy(teacher_boxes), xyxy(gt.detach())).max(1).values < c['geometry_gt_exclusion_iou']
        ti = torch.nonzero(reliable).flatten()
        free = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device); free[occupied] = False
        if len(gt): free &= overlaps(xyxy(boxes.detach()), xyxy(gt.detach())).max(1).values < c['response_gt_exclusion_iou']
        qi = torch.nonzero(free).flatten()
        if not len(ti) or not len(qi): return zero, 0
        a, b = assignment(boxes[qi], teacher_boxes[ti]); a = qi[a]; b = ti[b]
        iou = overlaps(xyxy(boxes.detach()[a]), xyxy(teacher_boxes[b])).diag()
        keep = iou >= c['geometry_min_student_iou']; a = a[keep]; b = b[keep]
    if not len(a): return zero, 0
    value = (5 * F.l1_loss(boxes[a], teacher_boxes[b], reduction='sum') / len(a)
             + 2 * (1-paired_giou(xyxy(boxes[a]), xyxy(teacher_boxes[b]))).mean()
             + F.l1_loss(ctr[a], teacher_ctr[b]))
    return value, len(a)


def loss_for(out, original_teacher, target, model, c, arm):
    if model.feedback or model.protect or model.mlp_control:
        raise RuntimeError('L0/R0 require the identical J0 architecture with semantic feedback off')
    if arm == 'L0':
        value, stats = legacy_loss(out, original_teacher, model, c)
        hard = sum(v.get('weak_negative', 0) for v in stats.values() if isinstance(v, dict))
        stats.update(loss_response=0., loss_teacher_geometry=0., response_anchors=0,
                     low_response_anchors=0, geometry_matches=0, hard_background_labels=hard)
        return value, stats
    if arm != 'R0': raise ValueError('Unsupported arm')
    # Explicitly remove the old hard-negative sieve; positives/classification/quality stay exact.
    empty = dict(boxes=out['boxes'].new_empty(0, 4), p=out['boxes'].new_empty(0))
    common, stats = legacy_loss(out, empty, model, c)
    gt = cxcywh(out['gt']); meta = out['head'].get('dn_meta')
    ndn = meta['num_denoising_queries'] if meta else 0
    values = []; counts = {}; last_pos = None
    for i in range(len(out['boxes'])):
        boxes = out['boxes'][i, ndn:]; logits = out['cls'][i, ndn:]
        pos, _ = assignment(boxes, gt, logits)
        value, count = response_loss(logits, boxes, gt, pos, target, c)
        values.append(value); counts['layer_' + str(i)] = count; last_pos = pos
    response = torch.stack(values).mean()
    h = out['head']
    if 'enc_outputs_class' in h:
        boxes = h['enc_outputs_coord'][0]; logits = h['enc_outputs_class'][0]
        pos, _ = assignment(boxes, gt, logits)
        value, count = response_loss(logits, boxes, gt, pos, target, c)
        response = response + value; counts['encoder'] = count
    geom, matched = geometry_loss(out['boxes'][-1, ndn:], out['ctr'][-1, ndn:], gt, last_pos, target, c)
    total = common + c['teacher_response_weight'] * response + c['teacher_geometry_weight'] * geom
    stats.update(loss_response=float(response.detach()), loss_teacher_geometry=float(geom.detach()),
                 response_anchors=sum(v['anchors'] for v in counts.values()),
                 low_response_anchors=sum(v['low_response'] for v in counts.values()),
                 geometry_matches=matched, hard_background_labels=0, response_by_layer=counts,
                 dn_response_policy='DN positives keep original supervision; all DN unmatched queries ignored')
    return total, stats
