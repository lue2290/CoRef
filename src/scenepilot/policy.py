import numpy as np

def xyxy(b):
    b = np.asarray(b, np.float32).reshape(-1, 4)
    return np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)

def overlaps(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 4); b = np.asarray(b, np.float32).reshape(-1, 4)
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh.prod(-1)
    area = lambda v: np.maximum(0, v[:, 2:] - v[:, :2]).prod(-1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, np.float32(1e-8))

def nms(boxes, scores, eligible, threshold):
    order = np.flatnonzero(eligible)
    order = order[np.argsort(-np.asarray(scores)[order], kind='stable')]
    kept = []
    while len(order):
        i = int(order[0]); kept.append(i)
        order = order[1:]
        if len(order): order = order[overlaps(boxes[i:i+1], boxes[order])[0] <= threshold]
    mask = np.zeros(len(boxes), bool); mask[kept] = True
    return mask

def pair_views(raw, c):
    from scipy.optimize import linear_sum_assignment
    n = len(raw['p']) // 2
    ov = overlaps(xyxy(raw['boxes'][:n]), xyxy(raw['boxes'][n:]))
    ii, jj = linear_sum_assignment(-ov)
    partner = np.empty(n, np.int64); partner[ii] = jj + n
    consistent = ov[np.arange(n), partner - n] >= c['teacher_consistency_iou']
    # Cached stable is independently reproduced, not trusted as a replacement for matching.
    if not np.array_equal(consistent, raw['stable'][:n]):
        raise RuntimeError('Reconstructed teacher view correspondence differs from the original cache')
    good = consistent & (np.abs(raw['logits'][:n] - raw['logits'][partner]) <= c['response_pair_logit_tolerance'])
    return dict(partner=partner, consistent=consistent, response_valid=good,
                boxes=raw['boxes'][:n].astype(np.float32),
                logits=((raw['logits'][:n] + raw['logits'][partner]) / 2).astype(np.float32),
                ctr=((raw['ctr'][:n] + raw['ctr'][partner]) / 2).astype(np.float32),
                min_p=np.minimum(raw['p'][:n], raw['p'][partner]),
                min_ctr=np.minimum(raw['ctr'][:n], raw['ctr'][partner]))

def fit_cutoffs(pairs, ground_truth, c):
    responses, centers = [], []
    for name, pair in pairs.items():
        gt = ground_truth[name]
        if not len(gt): continue
        ov = overlaps(xyxy(pair['boxes']), gt); ids = ov.argmax(0)
        ok = ov[ids, np.arange(len(gt))] >= .5
        # Only spatially qualified selected inner-fit known objects determine cutoffs.
        responses.extend(pair['min_p'][ids[ok]].tolist())
        centers.extend(pair['min_ctr'][ids[ok]].tolist())
    if len(responses) < 20:
        raise RuntimeError('Fewer than 20 matched training-known objects; do not loosen thresholds automatically')
    return dict(high_response=float(np.quantile(responses, c['response_high_quantile'])),
                geometry_centerness=float(np.quantile(centers, c['geometry_centerness_quantile'])),
                training_positive_matches=len(responses))

def materialize(pair, cutoffs, c):
    high = (pair['min_p'] >= cutoffs['high_response']) & (pair['min_ctr'] >= cutoffs['geometry_centerness'])
    reliable = pair['response_valid'] & high
    reliable = nms(xyxy(pair['boxes']), pair['min_p'] * pair['min_ctr'], reliable, c['geometry_nms_iou'])
    # Scores retain their original meaning. They are not declared calibrated foreground probabilities.
    return dict(boxes=pair['boxes'], logits=pair['logits'], ctr=pair['ctr'],
                response_valid=pair['response_valid'], geometry_valid=reliable,
                response_low=pair['min_p'] < cutoffs['high_response'])
