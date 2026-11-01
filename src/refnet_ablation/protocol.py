from pathlib import Path
import shutil
import zipfile
import numpy as np
from refnet.protocol import read,write,sha,fp,lock
from refnet_f1.protocol import inputs,verify_file,guard_space
from current.progress import track,stage


def home(root,cfg):return Path(root)/'reports/reference_ablation'/cfg['tag']


def case_paths(root,cfg,variant,scene,seed):
    tail=Path('reference_ablation')/cfg['tag']/variant/scene/'task1'/f'seed_{seed}'
    return Path(root)/'runs'/tail,Path(root)/'reports'/tail


def sources(root):
    root=Path(root);original=read(root/'MAINLINE_MANIFEST.json')['files']
    prefixes=('src/refnet/','versions/joint_detector/jointdet/','vendor/pfrpn/mmdet/',
              'vendor/pfrpn/configs/','src/cil/','src/scenepilot/','src/fivescene/')
    names=[n for n in original if n.endswith('.py') and n.startswith(prefixes)]
    names += ['MAINLINE_PARENT_INDEX.json','configs/reference_net.json','run.py','vendor/fomo_open_world_eval.py',
        'assets/models/dinov3_large/config.json','src/current/progress.py','src/current/progress_io.py',
        'run_reference_ablation.py','configs/reference_ablation.json',
        'src/refnet_f1/__init__.py','src/refnet_f1/protocol.py','src/refnet_f1/runtime.py']
    names += [p.relative_to(root).as_posix() for p in (root/'src/refnet_ablation').glob('*.py')]
    result={}
    for name in track(sorted(set(names)),'159 verify numerical sources',unit='files'):
        result[name]=sha(root/name)
        if name in original and result[name]!=original[name]:raise RuntimeError('Frozen mainline changed: '+name)
    receipt=root/'REFERENCE_ABLATION_UPLOAD_MANIFEST.json'
    if receipt.exists():
        for name,h in read(receipt)['files'].items():verify_file(root/name,h)
    return result


def snapshot(root,cfg,source):
    folder=home(root,cfg)/'SOURCE';lock(folder/'LOCK.json',dict(config=cfg,source=source))
    for name,h in source.items():
        if name.startswith(('src/refnet_ablation/','src/refnet_f1/')) or name in ['run_reference_ablation.py','configs/reference_ablation.json']:
            dest=folder/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(Path(root)/name,dest)
            verify_file(dest,h)


def baseline_location(root,cfg,scene,seed):
    if scene=='Game':return Path(root)/'reports/reference_game'/cfg['corrected_game_tag']/'Game/task1'/f'seed_{seed}'
    return Path(root)/'reports/reference_f1'/cfg['full_f1_tag']/scene/'task1'/f'seed_{seed}'


def baseline_metadata(root,cfg,scene,seed):
    spec,historical,annotation=inputs(root,scene,'task1',seed)
    report=baseline_location(root,cfg,scene,seed)
    return dict(spec=spec,historical=historical,annotation=annotation,report=report,scene=scene,seed=seed)


def load_baselines(root,cfg,source,verified,with_plans=True):
    root=Path(root);cases=[baseline_metadata(root,cfg,s,seed) for s in cfg['scenes'] for seed in cfg['seeds']]
    missing=[str(c['report']/'RESULT.json') for c in cases if not (c['report']/'RESULT.json').is_file()]
    if missing:raise RuntimeError('Finish157 Medical/Aquatic Task1 and158 corrected Game Task1 before159; missing full-model results:\n'+'\n'.join(missing))
    dataset_sha=sha(root/'assets/rwd/DATASET_MANIFEST.json')
    for case in track(cases,'159 verify nine full-model controls',unit='parents'):
        spec,report=case['spec'],case['report'];result=read(report/'RESULT.json')
        if not result['complete'] or (result['scene'],result['task'],result['seed'])!=(case['scene'],'task1',case['seed']):
            raise RuntimeError('Incomplete/mismatched full-model control')
        if case['scene']=='Game':
            if not result.get('corrected_label_retraining'):raise RuntimeError('Game control must be158 corrected-label retraining')
            parent=read(report/'CORRECTED_PARENT.json');binding=read(root/parent['lock'])
            case['annotation']={**case['annotation'],'parent_retrained_on_corrections':True}
        else:
            parent=case['historical'];binding=read(report/'LOCK.json')
        if binding['annotation']['dataset_manifest_sha256']!=dataset_sha:raise RuntimeError('Full control uses another dataset revision')
        if parent['split_identity']!=spec['identity'] or result['parent_checkpoint']['sha256']!=parent['terminal']['sha256']:
            raise RuntimeError('Full control checkpoint/split mismatch')
        if parent['terminal']['step']!=500 or result['calibration']['target']!=.95:raise RuntimeError('Require fixed500 full model with95% F1 calibration')
        for name,h in result['artifacts'].items():verify_file(report/name,h,verified)
        verify_file(root/parent['terminal']['checkpoint'],parent['terminal']['sha256'],verified)
        verify_file(root/parent['reference_cache']/'MANIFEST.json',parent['reference_manifest_sha256'],verified)
        if result['metrics']['F1']['official_evaluator_sha256']!=sha(root/'vendor/fomo_open_world_eval.py'):
            raise RuntimeError('Full control official evaluator changed')
        case.update(parent=parent,result=result,cache=root/parent['reference_cache'],
            identity=fp(dict(parent=parent,result_sha256=sha(report/'RESULT.json'),annotation=case['annotation'])))
        if with_plans:case['plan']=training_plan(root,cfg,case)
    write(home(root,cfg)/'FULL_BASELINES.json',dict(cases=[dict(scene=c['scene'],seed=c['seed'],parent=c['parent'],
        result_sha256=sha(c['report']/'RESULT.json'),metrics=c['result']['metrics']['F1'],identity=c['identity']) for c in cases]))
    return cases


def training_plan(root,cfg,case):
    from jointdet.protocol import load_torch
    path=Path(root)/'prepared/reference_ablation'/cfg['tag']/'plans'/case['scene']/f'seed_{case["seed"]}.json'
    if path.exists():
        result=read(path)
        if result['full_control_identity']!=case['identity'] or result['identity']!=fp({k:v for k,v in result.items() if k!='identity'}):
            raise RuntimeError('Locked ablation sampling plan changed')
        return result
    with stage(f'{case["scene"]}/seed_{case["seed"]} read full-model training trace (no warm start)'):
        state=load_torch(Path(root)/case['parent']['terminal']['checkpoint'])
        if state['step']!=cfg['updates'] or not state['complete']:raise RuntimeError('Full control training incomplete')
        history=state['history'];del state
    if [r['step'] for r in history]!=list(range(1,cfg['updates']+1)):raise RuntimeError('Need all500 steps of original full-model training history')
    rows=[];fit=set(case['spec']['fit_images']);classes=set(range(len(case['spec']['classes'])))
    accumulate=read(Path(root)/'configs/reference_net.json')['model']['accumulate']
    for row in history:
        if len(row['samples'])!=accumulate or len(row['stats'])!=accumulate:raise RuntimeError('Full-model accumulation trace mismatch')
        micro=[]
        for j,(sample,stats) in enumerate(zip(row['samples'],row['stats'])):
            episode=stats['episode'];held=int(episode['held_class']);support=list(map(int,episode['support_classes']))
            if sample['image'] not in fit or held not in classes or held in support or not support or not set(support)<=classes:
                raise RuntimeError('Illegal full-model sampling/episode trace')
            micro.append(dict(image=sample['image'],flip=bool(sample['flip']),episode=dict(held_class=held,support_classes=support),
                query_sampling_seed=int(fp([case['scene'],case['seed'],row['step'],j,'query_sampling'])[:16],16)))
        rows.append(dict(step=row['step'],microbatches=micro))
    result=dict(full_control_identity=case['identity'],steps=rows,
        policy='reuse full-control image/flip and held/support-class trace; independent per-step/query RNG for16/32 candidate sampling; no adapted-parent warm start',
        limitation='proposal pools differ when modules change; uniform query subsampling follows the original16/32 rule, not guaranteed bitwise full-parent replay')
    return lock(path,result)


def validate_cache(root,case,verified):
    cache=case['cache'];m=read(cache/'MANIFEST.json');s=case['spec']
    if m['split']!=s['identity']:raise RuntimeError('Full-control GT cache split mismatch')
    dataset=read(Path(root)/'assets/rwd/DATASET_MANIFEST.json');by={}
    for obj in s['objects']:by.setdefault(obj['image'],[]).append(obj)
    expected={(role,name) for role in ['fit','cal','monitor'] for name in s[role+'_images']}
    rows={(r['role'],r['image']):r for r in m['rows']}
    if not expected<=set(rows):raise RuntimeError('Missing fit/cal/monitor GT cache rows')
    for role,name in track(sorted(expected),case['scene']+' verify read-only full reference/teacher cache',unit='images'):
        row=rows[role,name];path=cache/row['file'];verify_file(path,m['files'][row['file']],verified)
        verify_file(Path(root)/'assets/rwd/JPEGImages'/case['scene']/name,dataset['files'][f'JPEGImages/{case["scene"]}/{name}']['sha256'],verified)
        with np.load(path,allow_pickle=False) as z:
            if z['ids'].tolist()!=[o['id'] for o in by[name]] or z['labels'].tolist()!=[o['class_id'] for o in by[name]]:
                raise RuntimeError('Reference/teacher cache labels do not match active split: '+name)
            if role=='fit' and not {'boxes','p','ctr','stable'}<=set(z.files):raise RuntimeError('Missing pretrained PF teacher arrays')


def summarize(root,cfg):
    lookup={}
    p=home(root,cfg)/'FULL_BASELINES.json'
    if p.exists():
        for r in read(p)['cases']:lookup['full',r['scene'],r['seed']]=r['metrics']
    for variant in cfg['variants']:
        for p in (home(root,cfg)/variant).glob('*/task1/seed_*/RESULT.json'):
            r=read(p);lookup[variant,r['scene'],r['seed']]=r['metrics']['F1']
    def stats(values):return dict(n=len(values),mean=float(np.mean(values)) if values else None,
        sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None)
    by_scene={};macro={};completion={};deltas={}
    for variant in ['full']+cfg['variants']:
        count=sum((variant,s,seed) in lookup for s in cfg['scenes'] for seed in cfg['seeds'])
        completion[variant]=dict(cases=count,expected=9,complete=count==9)
        for metric in ['K_AP50','U_AP50']:
            for scene in cfg['scenes']:
                values={str(seed):lookup[variant,scene,seed][metric] for seed in cfg['seeds'] if (variant,scene,seed) in lookup}
                by_scene[f'{variant}/{scene}/{metric}']=dict(**stats([v for v in values.values() if v is not None]),by_seed=values)
            per_seed={}
            for seed in cfg['seeds']:
                if all((variant,s,seed) in lookup and lookup[variant,s,seed][metric] is not None for s in cfg['scenes']):
                    per_seed[str(seed)]=float(np.mean([lookup[variant,s,seed][metric] for s in cfg['scenes']]))
            macro[f'{variant}/{metric}']=dict(**stats(list(per_seed.values())),by_seed=per_seed)
            if variant!='full':
                a=macro[f'full/{metric}']['by_seed'];d={k:v-a[k] for k,v in per_seed.items() if k in a}
                deltas[f'{variant}/{metric}']=dict(**stats(list(d.values())),by_seed=d)
    result=dict(complete=all(v['complete'] for v in completion.values()),completion=completion,
        by_scene=by_scene,macro=macro,delta_from_full=deltas,
        statistic='Task1 only; each variant9 cases. Three-scene macro within each seed, then mean +/- sample SD across42/43/44',
        scope=cfg['scope'],new_training_cases=sum(completion[v]['cases'] for v in cfg['variants']),
        trained_cases=len(list(home(root,cfg).glob('*/*/task1/seed_*/TRAINING_COMPLETE.json'))),
        failed_cases=[read(p) for p in sorted(home(root,cfg).glob('*/*/task1/seed_*/FAILURE.json')) if not read(p).get('resolved')])
    write(home(root,cfg)/'CORE_SUMMARY.json',result)
    lines=['# Three-scene core ablations','',f'Complete: {result["complete"]}', '',
        '| Variant/scene/metric | Mean | Sample SD | n |','|---|---:|---:|---:|']
    def val(x):return '--' if x is None else f'{x:.2f}'
    for key,r in by_scene.items():lines.append(f'| {key} | {val(r["mean"])} | {val(r["sample_std"])} | {r["n"]} |')
    lines+=['','| Three-scene macro | Mean | Sample SD | n |','|---|---:|---:|---:|']
    for key,r in macro.items():lines.append(f'| {key} | {val(r["mean"])} | {val(r["sample_std"])} | {r["n"]} |')
    lines+=['','Full results are reused157/158; corrected Game is required. No best-seed/model selection.']
    (home(root,cfg)/'CORE_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def stage_upload(root,cfg):
    root=Path(root);out=home(root,cfg)/'UPLOAD/OpenDet_Current'
    names=['run_reference_ablation.py','configs/reference_ablation.json','.run/159_Reference_Ablation.run.xml',
        'docs/REFERENCE_ABLATION.md','tests/test_reference_ablation.py',
        'src/refnet_f1/__init__.py','src/refnet_f1/protocol.py','src/refnet_f1/runtime.py']
    names += [p.relative_to(root).as_posix() for p in (root/'src/refnet_ablation').glob('*.py')]
    hashes={}
    for name in track(names,'159 stage upload tree',unit='files'):
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest);hashes[name]=sha(dest)
    write(out/'REFERENCE_ABLATION_UPLOAD_MANIFEST.json',dict(tag=cfg['tag'],files=hashes))
    return dict(upload_directory=str(out),files=len(hashes),weights_included=False)


def export_results(root,cfg):
    root=Path(root);folder=home(root,cfg)
    if not (list(folder.glob('*/*/task1/seed_*/RESULT.json')) or list(folder.glob('*/*/task1/seed_*/FAILURE.json')) or (folder/'LOCALIZATION_SUMMARY.json').exists()):
        raise RuntimeError('No completed core case or localization summary to export')
    for p in track(sorted(folder.glob('*/*/task1/seed_*/RESULT.json')),'159 verify completed core outputs',unit='cases'):
        for name,h in read(p)['artifacts'].items():verify_file(p.parent/name,h)
    names=[p for base in [folder,Path(root)/'runs/reference_ablation'/cfg['tag']] for p in base.rglob('*') if p.is_file()
        and 'UPLOAD' not in p.parts and p.suffix in ['.json','.jsonl','.md','.py','.log'] and p.name!='DOWNLOAD_RECEIPT.json']
    guard_space(root,cfg,sum(p.stat().st_size for p in names));path=folder/'REFERENCE_ABLATION_RESULTS.zip';tmp=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in track(names,'159 pack core and localization results',unit='files'):z.write(p,p.relative_to(root).as_posix())
    with stage('159 verify ZIP CRC/SHA256'):
        with zipfile.ZipFile(tmp) as z:
            if z.testzip():raise RuntimeError('ZIP CRC failed')
        h=sha(tmp)
    tmp.replace(path);result=dict(path=str(path),bytes=path.stat().st_size,sha256=h,crc_verified=True,weights_included=False)
    write(folder/'DOWNLOAD_RECEIPT.json',result)
    return result
