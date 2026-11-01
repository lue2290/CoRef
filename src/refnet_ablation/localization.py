from pathlib import Path
import gc
import xml.etree.ElementTree as ET
import numpy as np
import torch
from cil.common import iou,configure_torch
from jointdet.geometry import xyxy
from jointdet.model import pf_score
from current.progress import track,stage
from .protocol import read,write,sha,fp,lock,verify_file,home,guard_space


def read_targets(path,classes):
    targets=[];ignored=0
    for obj in ET.parse(path).getroot().findall('object'):
        if int(obj.findtext('difficult','0')):
            ignored+=1;continue
        box=[float(obj.find('bndbox').findtext(k))-1 for k in ['xmin','ymin','xmax','ymax']]
        name=obj.findtext('name');targets.append(dict(name=name,kind='known' if name in classes else 'unknown',box=box))
    return targets,ignored


def coverage(boxes,targets,joint=None,threshold=None,budgets=(100,300,600)):
    boxes=np.asarray(boxes,np.float32).reshape(-1,4)
    if len(boxes)<max(budgets):raise RuntimeError('Need at least600 real ranked proposals; never pad or infer600 from300')
    gt=[t['box'] for t in targets];before={};after={};kept={}
    for n in budgets:
        ov=iou(boxes[:n],gt);before[str(n)]=ov.max(0).tolist() if len(gt) else []
        if joint is not None:
            mask=np.asarray(joint[:n])>=threshold;selected=boxes[:n][mask]
            after[str(n)]=iou(selected,gt).max(0).tolist() if len(selected) and len(gt) else [0.]*len(gt)
            kept[str(n)]=int(mask.sum())
    return dict(before=before,after=after,post_gate_counts=kept)


def aggregate(records,budgets=(100,300,600),has_gate=False):
    counts={kind:sum(t['kind']==kind for r in records for t in r['targets']) for kind in ['known','unknown']}
    panels={}
    for panel in ['before']+(['after'] if has_gate else []):
        result={}
        for kind in ['known','unknown']:
            for n,threshold in [(n,.5) for n in budgets]+[(600,.75)]:
                hits=sum(value>=threshold for r in records for t,value in zip(r['targets'],r[panel][str(n)]) if t['kind']==kind)
                key=f'{kind}/R{int(threshold*100)}@{n}'
                result[key]=dict(hits=int(hits),gt=counts[kind],recall=100.*hits/counts[kind] if counts[kind] else None)
        panels[panel]=result
    return dict(images=len(records),gt_counts=counts,metrics=panels,
        ignored_difficult=sum(r['ignored_difficult'] for r in records),
        mean_post_gate_candidates={str(n):float(np.mean([r['post_gate_counts'][str(n)] for r in records])) for n in budgets} if has_gate and records else {},
        convention='continuous XYXY, no +1, IoU>=threshold; targetwise maximum IoU; difficult excluded; localization recall, not official AP',
        post_gate_policy='select PF-ranked topN first, then apply joint gate; no refill, NMS or identity filtering')


@torch.inference_mode()
def b0_values(pf,view,name,c,device):
    from jointdet.model import pixels,make_sample
    x,d,meta=pixels(view,name,c,device);del d
    sample,_=make_sample(meta,[],device)
    head=pf.forward_transformer(pf.extract_feat(x),{},[sample])
    cls,boxes,ctr=pf.bbox_head(head['hidden_states'],head['references'],head['memory_text'],head['text_token_mask'])
    score=pf_score(cls[-1,0],ctr[-1,0,:,0]);ids=score.argsort(descending=True,stable=True)[:600]
    boxes=xyxy(boxes[-1,0,ids]).clamp(0,1);wh=boxes.new_tensor([meta['ori_shape'][1],meta['ori_shape'][0]]*2)
    return dict(boxes=(boxes*wh).cpu().numpy(),joint=None,proposal_ids=ids.cpu().tolist(),pf_scores=score[ids].cpu().tolist())


@torch.inference_mode()
def full_values(model,view,name,c,device):
    out=model.forward_image(view,name,[],device)
    score=pf_score(out['cls'][-1],out['ctr'][-1]);ids=score.argsort(descending=True,stable=True)[:600]
    boxes=xyxy(out['refined'][ids]).clamp(0,1);wh=boxes.new_tensor([out['meta']['ori_shape'][1],out['meta']['ori_shape'][0]]*2)
    return dict(boxes=(boxes*wh).cpu().numpy(),joint=(score[ids]/2*out['quality_logit'][ids].sigmoid()).cpu().numpy(),
        proposal_ids=ids.cpu().tolist(),pf_scores=score[ids].cpu().tolist())


def diagnose_case(root,cfg,case,kind,model,view,c,device,dataset,verified):
    spec=case['spec'];scene=spec['scene'];seed=case['seed'] if kind=='full' else None
    report=home(root,cfg)/'localization'/kind/scene/(f'seed_{seed}' if kind=='full' else 'fixed_pf')
    parent=case['parent'];threshold=case['result']['calibration']['object_threshold'] if kind=='full' else None
    binding=dict(scene=scene,kind=kind,seed=seed,split=spec['identity'],dataset_sha256=sha(Path(root)/'assets/rwd/DATASET_MANIFEST.json'),
        checkpoint_sha256=parent['terminal']['sha256'] if kind=='full' else read(Path(root)/'MAINLINE_PARENT_INDEX.json')['base_weights'][c['pf_weights']],
        quality_threshold=threshold,budgets=cfg['localization_budgets'],sources=cfg['_source_identity'])
    record=lock(report/'LOCK.json',binding);records=[]
    for index,name in enumerate(track(spec['test_images'],f'{kind}/{scene}/'+(f'seed_{seed}' if seed is not None else 'one frozen PF')+' localization',unit='images')):
        if index%32==0:guard_space(root,cfg)
        image=Path(root)/'assets/rwd/JPEGImages'/scene/name
        xml=Path(root)/'assets/rwd/Annotations'/scene/(Path(name).stem+'.xml')
        image_sha=verify_file(image,dataset['files'][f'JPEGImages/{scene}/{name}']['sha256'],verified)
        xml_sha=verify_file(xml,dataset['files'][f'Annotations/{scene}/{Path(name).stem}.xml']['sha256'],verified)
        path=report/'images'/(fp(name)[:24]+'.json')
        if path.exists():
            row=read(path)
            if row['image']!=name or row['image_sha256']!=image_sha or row['annotation_sha256']!=xml_sha:raise RuntimeError('Localization resume inputs changed')
        else:
            # Only boxes/scores are produced before reading test targets.
            values=b0_values(model,view,name,c,device) if kind=='B0' else full_values(model,view,name,c,device)
            targets,ignored=read_targets(xml,spec['classes'])
            measured=coverage(values['boxes'],targets,values['joint'],threshold,cfg['localization_budgets'])
            row=dict(image=name,image_sha256=image_sha,annotation_sha256=xml_sha,targets=targets,
                ignored_difficult=ignored,proposals=len(values['boxes']),proposal_ids=values['proposal_ids'],**measured)
            write(path,row)
        records.append(row)
    result=aggregate(records,cfg['localization_budgets'],kind=='full')
    result.update(complete=True,kind=kind,scene=scene,seed=seed,lock=record['identity'],quality_threshold=threshold,
        image_records_sha256=fp({r['image']:sha(report/'images'/(fp(r['image'])[:24]+'.json')) for r in records}))
    write(report/'RESULT.json',result)
    return result


def run(root,cfg,cases,original,device,dataset,verified):
    from jointdet.model import build_pf
    from jointdet.protocol import load_torch
    from scenepilot.protocol import view
    from .model import AblationDetector
    c=original['model'];configure_torch(0,2)
    with stage('159 load one frozen PF-RPN for B0; no DINO encoder'):
        pf=build_pf(root,c).to(device).eval();pf.requires_grad_(False)
    try:
        for scene in cfg['scenes']:
            case=next(x for x in cases if x['scene']==scene)
            diagnose_case(root,cfg,case,'B0',pf,view(root,scene),c,device,dataset,verified)
    finally:
        del pf;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    for case in track(cases,'159 full-model localization: three seeds',unit='parents'):
        configure_torch(case['seed'],2)
        with stage(f'{case["scene"]}/seed_{case["seed"]} load full parent for localization'):
            model=AblationDetector(root,c,list(range(len(case['spec']['classes']))),'full').to(device)
            state=load_torch(Path(root)/case['parent']['terminal']['checkpoint']);model.load_adapted(state['model']);del state
            model.requires_grad_(False);model.eval()
        try:diagnose_case(root,cfg,case,'full',model,view(root,case['scene']),c,device,dataset,verified)
        finally:
            del model;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    return summarize(root,cfg)


def summarize(root,cfg):
    table={};complete=True;macros={};denominators={}
    def stats(values):return dict(n=len(values),mean=float(np.mean(values)) if values else None,
        sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None)
    cases={}
    for scene in cfg['scenes']:
        b=home(root,cfg)/'localization/B0'/scene/'fixed_pf/RESULT.json'
        if b.exists():
            row=read(b);cases['B0',scene,None]=row;denominators[scene]=row['gt_counts']
            for key,r in row['metrics']['before'].items():table[f'B0/{scene}/before/{key}']=dict(n=1,mean=r['recall'],sample_std=None)
        else:complete=False
        for seed in cfg['seeds']:
            p=home(root,cfg)/'localization/full'/scene/f'seed_{seed}/RESULT.json'
            if p.exists():
                row=read(p)
                if scene in denominators and row['gt_counts']!=denominators[scene]:raise RuntimeError('B0/full localization target populations differ')
                cases['full',scene,seed]=row
            else:complete=False
        full=[cases['full',scene,seed] for seed in cfg['seeds'] if ('full',scene,seed) in cases]
        if full:
            for panel in ['before','after']:
                for key in full[0]['metrics'][panel]:
                    values=[r['metrics'][panel][key]['recall'] for r in full if r['metrics'][panel][key]['recall'] is not None]
                    table[f'full/{scene}/{panel}/{key}']=stats(values)
    keys=[f'{kind}/R50@{n}' for kind in ['known','unknown'] for n in cfg['localization_budgets']]+[f'{kind}/R75@600' for kind in ['known','unknown']]
    for panel in ['before','after']:
        for key in keys:
            per_seed={}
            for seed in cfg['seeds']:
                rows=[cases.get(('full',s,seed)) for s in cfg['scenes']]
                if all(r and r['metrics'][panel][key]['recall'] is not None for r in rows):
                    per_seed[str(seed)]=float(np.mean([r['metrics'][panel][key]['recall'] for r in rows]))
            macros[f'full/{panel}/{key}']=dict(**stats(list(per_seed.values())),by_seed=per_seed)
    result=dict(complete=complete,table=table,three_scene_full_macro=macros,gt_counts=denominators,
        expected_cases=12,completed_cases=len(cases),B0_repeated_per_seed=False,
        note='B0 is one deterministic proposal source per scene; full uses3 seeds. Post-gate uses same topN without refill. R values are percentages, not AP.')
    write(home(root,cfg)/'LOCALIZATION_SUMMARY.json',result)
    lines=['# B0 versus full-model localization','',f'Complete: {complete}; {len(cases)}/12 cases','',
        '| Model/scene/gate/metric | Mean | Sample SD | n |','|---|---:|---:|---:|']
    def val(x):return '--' if x is None else f'{x:.2f}'
    for key,r in table.items():lines.append(f'| {key} | {val(r["mean"])} | {val(r["sample_std"])} | {r["n"]} |')
    lines+=['','B0 has no artificial three-seed SD. Full localization includes learned box refinement; this does not isolate feedback alone.',
        'Continuous XYXY IoU>=.50/.75; difficult ignored. Gate losses measured after fixed PF topN selection; no refill or NMS.']
    (home(root,cfg)/'LOCALIZATION_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
