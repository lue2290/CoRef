"""Enable exactly one existing teacher constraint; common supervision is exact."""
import torch
from .response_loss import legacy_loss, assignment, response_loss, geometry_loss, cxcywh


def loss_for(out, original_teacher, target, model, c, arm):
    if arm not in ('S','G') or model.feedback or model.protect or model.mlp_control:
        raise RuntimeError('S/G require the same J0 architecture without semantic feedback')
    if (arm == 'S' and c['teacher_geometry_weight'] != 0) or (arm == 'G' and c['teacher_response_weight'] != 0):
        raise RuntimeError('Inactive component must have zero configured weight')
    empty = dict(boxes=out['boxes'].new_empty(0,4),p=out['boxes'].new_empty(0))
    common,stats = legacy_loss(out,empty,model,c)
    gt = cxcywh(out['gt']); meta = out['head'].get('dn_meta')
    ndn = meta['num_denoising_queries'] if meta else 0
    response = common.new_zeros(()); geom = common.new_zeros(()); counts = {}; matched = 0
    if arm == 'S':
        values = []
        for i in range(len(out['boxes'])):
            boxes = out['boxes'][i,ndn:]; logits = out['cls'][i,ndn:]
            pos,_ = assignment(boxes,gt,logits)
            value,count = response_loss(logits,boxes,gt,pos,target,c)
            values.append(value); counts['layer_'+str(i)] = count
        response = torch.stack(values).mean(); h = out['head']
        if 'enc_outputs_class' in h:
            boxes = h['enc_outputs_coord'][0]; logits = h['enc_outputs_class'][0]
            pos,_ = assignment(boxes,gt,logits)
            value,count = response_loss(logits,boxes,gt,pos,target,c)
            response = response+value; counts['encoder'] = count
    else:
        boxes = out['boxes'][-1,ndn:]; logits = out['cls'][-1,ndn:]
        pos,_ = assignment(boxes,gt,logits)
        geom,matched = geometry_loss(boxes,out['ctr'][-1,ndn:],gt,pos,target,c)
    total = common+c['teacher_response_weight']*response+c['teacher_geometry_weight']*geom
    stats.update(loss_response=float(response.detach()),loss_teacher_geometry=float(geom.detach()),
                 response_anchors=sum(v['anchors'] for v in counts.values()),
                 low_response_anchors=sum(v['low_response'] for v in counts.values()),
                 geometry_matches=matched,hard_background_labels=0,response_by_layer=counts,
                 active_component='soft_response' if arm == 'S' else 'geometry',
                 dn_response_policy='Original DN positives; DN unmatched queries ignored')
    return total,stats
