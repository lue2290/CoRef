from pathlib import Path
import importlib.metadata
import inspect
import shutil
import zipfile
import numpy as np
from refnet.protocol import read,write,sha,fp,lock
from refnet_f1.protocol import inputs,verify_file
from current.progress import track,stage


def locations(root,cfg,scene=None,arm=None,seed=None):
    tail=Path('owl_reference')/cfg['tag']
    if scene:tail=tail/scene
    if arm:tail=tail/arm/f'seed_{seed}'
    return tuple(Path(root)/b/tail for b in ['prepared','runs','reports'])


def guard(root,cfg,extra=0):
    if shutil.disk_usage(root).free<cfg['min_free_gib']*2**30+extra:
        raise RuntimeError('160 disk reserve insufficient; completed caches/checkpoints preserved; free space and resume')


def sources(root):
    root=Path(root);original=read(root/'MAINLINE_MANIFEST.json')['files']
    prefixes=('versions/joint_detector/jointdet/','vendor/pfrpn/mmdet/','vendor/pfrpn/configs/','src/cil/','src/fivescene/','src/scenepilot/')
    names=[n for n in original if n.endswith('.py') and n.startswith(prefixes)]
    names += ['MAINLINE_PARENT_INDEX.json','configs/reference_net.json','run.py','run_owl_reference.py','configs/owl_reference.json',
        'src/refnet/protocol.py','src/refnet/relation.py','src/refnet/evaluation.py','src/current/progress.py','src/current/progress_io.py',
        'src/refnet_f1/protocol.py','src/refnet_f1/runtime.py','vendor/fomo_open_world_eval.py']
    names += [p.relative_to(root).as_posix() for p in (root/'src/owl_reference').glob('*.py')]
    out={}
    for name in track(sorted(set(names)),'160 verify independent numerical sources',unit='files'):
        out[name]=sha(root/name)
        if name in original and out[name]!=original[name]:raise RuntimeError('Frozen mainline changed: '+name)
    receipt=root/'OWL_REFERENCE_UPLOAD_MANIFEST.json'
    if receipt.exists():
        for name,h in read(receipt)['files'].items():verify_file(root/name,h)
    return out


def environment():
    import torch,transformers
    from transformers.models.owlvit import modeling_owlvit,image_processing_owlvit
    return dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
        modeling_owlvit_sha256=sha(inspect.getfile(modeling_owlvit)),
        processor_owlvit_sha256=sha(inspect.getfile(image_processing_owlvit)),
        precision='fp32; no autocast or TF32; lossless float32 feature caches')


def verify_assets(root,cfg,verified):
    root=Path(root);folder=root/cfg['owl_folder']
    for name,h in track(cfg['owl_files'].items(),'160 verify official OWL files',unit='files'):
        path=folder/name
        if not path.is_file():raise FileNotFoundError('Place official '+cfg['owl_repository']+' file at '+str(path))
        verify_file(path,h,verified)
    index=read(root/'MAINLINE_PARENT_INDEX.json');c=read(root/'configs/reference_net.json')['model']
    verify_file(root/c['pf_weights'],index['base_weights'][c['pf_weights']],verified)
    owl=read(folder/'config.json');pre=read(folder/'preprocessor_config.json')
    if owl['model_type']!='owlvit' or (owl['vision_config']['patch_size'],owl['vision_config']['image_size'],owl['vision_config']['hidden_size'])!=(16,768,768):
        raise RuntimeError('160 fixes official detection OWL-ViT B/16,768 input,768 channels')
    if pre['do_center_crop'] or not pre['do_resize']:raise RuntimeError('OWL preprocessing must resize square without crop')
    return dict(owl=cfg['owl_files'],pf={c['pf_weights']:index['base_weights'][c['pf_weights']]},
        revision=cfg['owl_revision'],raw_dim=5*owl['vision_config']['hidden_size'])


def make_cases(root,cfg,source):
    root=Path(root);cases=[];env=environment();folder=locations(root,cfg)[2]
    record=lock(folder/'SOURCE/LOCK.json',dict(config=cfg,source=source,owl_implementation=env))
    for name,h in source.items():
        if name.startswith('src/owl_reference/') or name in ['run_owl_reference.py','configs/owl_reference.json']:
            path=folder/'SOURCE'/name;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():shutil.copy2(root/name,path)
            verify_file(path,h)
    for scene in cfg['scenes']:
        spec,_,annotation=inputs(root,scene,'task1',42)
        # Parent metadata supplies only the legal split, never model weights.
        item=lock(locations(root,cfg,scene)[0]/'INPUT_LOCK.json',dict(spec_identity=spec['identity'],annotation=annotation,
            owl_files=cfg['owl_files'],repository=cfg['owl_repository'],revision=cfg['owl_revision'],source=record['identity']))
        cases.append(dict(scene=scene,spec=spec,identity=item['identity'],source_identity=record['identity']))
    write(folder/'ENVIRONMENT.json',env)
    return cases


def summary(root,cfg):
    folder=locations(root,cfg)[2];lookup={}
    for scene in cfg['scenes']:
        for arm in cfg['arms']:
            for seed in cfg['seeds']:
                path=locations(root,cfg,scene,arm,seed)[2]/'RESULT.json'
                if path.exists():
                    item=read(path)
                    if item.get('complete'):lookup[arm,scene,seed]=item
    def stat(v):return dict(n=len(v),mean=float(np.mean(v)) if v else None,sample_std=float(np.std(v,ddof=1)) if len(v)>1 else None)
    by_scene={};macro={};paired={};complete={}
    for arm in cfg['arms']:
        n=sum((arm,s,z) in lookup for s in cfg['scenes'] for z in cfg['seeds']);complete[arm]=dict(cases=n,expected=9,complete=n==9)
        for metric in ['K_AP50','U_AP50']:
            for scene in cfg['scenes']:
                data={str(z):lookup[arm,scene,z]['metrics'][metric] for z in cfg['seeds'] if (arm,scene,z) in lookup}
                by_scene[f'{arm}/{scene}/{metric}']=dict(**stat([v for v in data.values() if v is not None]),by_seed=data)
            data={}
            for seed in cfg['seeds']:
                if all((arm,s,seed) in lookup and lookup[arm,s,seed]['metrics'][metric] is not None for s in cfg['scenes']):
                    data[str(seed)]=float(np.mean([lookup[arm,s,seed]['metrics'][metric] for s in cfg['scenes']]))
            macro[f'{arm}/{metric}']=dict(**stat(list(data.values())),by_seed=data)
    for metric in ['K_AP50','U_AP50']:
        for scene in cfg['scenes']:
            data={str(z):lookup['O1',scene,z]['metrics'][metric]-lookup['O0',scene,z]['metrics'][metric] for z in cfg['seeds']
                if all((a,scene,z) in lookup and lookup[a,scene,z]['metrics'][metric] is not None for a in cfg['arms'])}
            paired[f'{scene}/{metric}']=dict(**stat(list(data.values())),by_seed=data)
        a=macro['O0/'+metric]['by_seed'];b=macro['O1/'+metric]['by_seed'];data={z:b[z]-a[z] for z in a if z in b}
        paired['macro/'+metric]=dict(**stat(list(data.values())),by_seed=data)
    result=dict(complete=all(v['complete'] for v in complete.values()),completion=complete,by_scene=by_scene,macro=macro,paired_O1_minus_O0=paired,
        scope=cfg['scope'],statistic='scene macro within each seed, then mean +/- sample SD(ddof=1); paired differences within seed',
        candidate_protocol='frozen native PF top600, shared across all6 scene heads; g=pf_score/2; no DINO quality/refinement/feedback',
        missing_cases=[f'{a}/{s}/seed_{z}' for a in cfg['arms'] for s in cfg['scenes'] for z in cfg['seeds'] if (a,s,z) not in lookup])
    write(folder/'SUMMARY.json',result)
    lines=['# OWL O0/O1 reference verification transfer','',f'Complete: {result["complete"]}', '',
        '| Configuration/scene/metric | Mean | Sample SD | n |','|---|---:|---:|---:|']
    def fmt(x):return '--' if x is None else f'{x:.2f}'
    for key,row in by_scene.items():lines.append(f'| {key} | {fmt(row["mean"])} | {fmt(row["sample_std"])} | {row["n"]} |')
    lines += ['','| Macro | Mean | Sample SD | n |','|---|---:|---:|---:|']
    for key,row in macro.items():lines.append(f'| {key} | {fmt(row["mean"])} | {fmt(row["sample_std"])} | {row["n"]} |')
    lines += ['','| Paired O1-O0 | Mean | Sample SD | n |','|---|---:|---:|---:|']
    for key,row in paired.items():lines.append(f'| {key} | {fmt(row["mean"])} | {fmt(row["sample_std"])} | {row["n"]} |')
    lines += ['','Not a strict DINO/OWL2x2 factorial experiment; candidates/training differ between the representation families.',
        'Frozen OWL detection spatial features, no category text/embeddings. No Task2/other scenes or SOTA claim.']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def dino_comparison(root,cfg):
    path=Path(root)/'reports/reference_ablation'/cfg['dino_ablation_tag']/'CORE_SUMMARY.json';folder=locations(root,cfg)[2]
    ready=False;data={};paired={}
    if path.exists():
        d=read(path);ready=all(d['completion'].get(a,{}).get('complete',False) for a in ['full','raw_relation'])
        if ready:
            data={k:v for k,v in d['delta_from_full'].items() if k.startswith('raw_relation/')}
            for key,row in data.items():
                paired[key.removeprefix('raw_relation/')]=dict(n=row['n'],mean=-row['mean'] if row['mean'] is not None else None,
                    sample_std=row['sample_std'],by_seed={z:-v for z,v in row['by_seed'].items()})
    result=dict(dino_ready=ready,dino_source=str(path),dino_raw_minus_full=data,
        paired_macro_D1_minus_D0=paired,dino_effect_direction='D1(full)-D0(raw_relation), paired by seed',
        note='Read-only reuse159 D0/D1; no additional DINO training. Absolute OWL/DINO scores do not isolate encoder quality.')
    write(folder/'DINO_COMPARISON.json',result)
    return result


def stage_upload(root,cfg):
    root=Path(root);out=locations(root,cfg)[2]/'UPLOAD/OpenDet_Current'
    names=['run_owl_reference.py','configs/owl_reference.json','.run/160_OWL_Reference_Transfer.run.xml',
        'docs/OWL_REFERENCE_TRANSFER.md','tests/test_owl_reference.py',
        cfg['owl_folder']+'/config.json',cfg['owl_folder']+'/preprocessor_config.json']
    names += [p.relative_to(root).as_posix() for p in (root/'src/owl_reference').glob('*.py')]
    files={}
    for name in track(names,'160 stage independent upload tree',unit='files'):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,path);files[name]=sha(path)
    write(out/'OWL_REFERENCE_UPLOAD_MANIFEST.json',dict(tag=cfg['tag'],files=files,large_weights_included=False,
        required_owl_weight=cfg['owl_folder']+'/pytorch_model.bin',required_owl_sha256=cfg['owl_files']['pytorch_model.bin']))
    return dict(upload_directory=str(out),files=len(files),large_weights_included=False,
        required_weight=str(root/cfg['owl_folder']/'pytorch_model.bin'))
