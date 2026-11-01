from pathlib import Path
from collections import Counter
import hashlib, json, shutil, time
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_files(root):
    root = Path(root)
    names = ['run.py', 'run_reference_net.py', 'configs/reference_net.json',
             'configs/scene_pilot.json', 'configs/b5_suite.json', 'tests/test_reference_net.py', 'src/b5suite/protocol.py',
             'src/b5suite/evaluate.py', 'src/scenepilot/protocol.py', 'configs/five_scene.json', 'src/cil/common.py', 'src/cil/evaluate.py', 'src/cil/model.py',
             'vendor/fomo_open_world_eval.py', 'assets/models/dinov3_large/config.json',
             'vendor/pfrpn/configs/pf-rpn/pf-rpn_coco-imagenet.py']
    for folder in ['src/refnet', 'versions/joint_detector/jointdet', 'src/fivescene']:
        names.extend(p.relative_to(root).as_posix() for p in (root/folder).glob('*.py'))
    # Vendor implementation participates in source identity without copying it.
    names.extend(p.relative_to(root).as_posix() for p in (root/'vendor/pfrpn/mmdet').rglob('*.py'))
    names.extend(p.relative_to(root).as_posix() for p in (root/'vendor/pfrpn/configs').rglob('*.py'))
    b5=read(root/'configs/b5_suite.json')
    names.extend((root/'prepared/b5_suite'/b5['tag']/(scene+'.json')).relative_to(root).as_posix() for scene in read(root/'configs/reference_net.json')['scenes'])
    names.extend(p.relative_to(root).as_posix() for p in (root/'assets/rwd/ImageSets').rglob('*') if p.is_file())
    return {n: sha(root/n) for n in sorted(set(names))}


def lock(path, value):
    value = json.loads(json.dumps(value)); value['identity'] = fp(value)
    if Path(path).exists() and read(path) != value:
        raise RuntimeError('Locked configuration/input/source changed: ' + str(path))
    if not Path(path).exists(): write(path, value)
    return value


def split_spec(root, scene, task):
    from b5suite.protocol import task as original_task
    cfg = read(Path(root)/'configs/reference_net.json')
    spec = original_task(root, scene, task)
    by = {n: [o for o in spec['objects'] if o['image'] == n] for n in spec['train_images']}
    # Merge declared source/video groups and exact-byte duplicates BEFORE splitting.
    declared=read(Path(root)/'assets/rwd/ImageSets'/scene/'groups.json')['groups']
    dataset=read(Path(root)/'assets/rwd/DATASET_MANIFEST.json')['files']
    parent={n:n for n in by}
    def find(n):
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    seen={}
    for n in by:
        for key in [('source',declared[n]),('sha',dataset[f'JPEGImages/{scene}/{n}']['sha256'])]:
            if key in seen:parent[find(n)]=find(seen[key])
            else:seen[key]=n
    units={}
    for n in by:units.setdefault(find(n),[]).append(n)
    group_of={n:fp(sorted(unit)) for unit in units.values() for n in unit}
    remaining=list(units.values());counts=Counter(o['class_id'] for o in spec['objects'])
    roles={'fit':set(by),'cal':set(),'monitor':set()}
    target=max(2,round(len(by)*cfg['holdout_fraction']))
    for role in ['cal','monitor']:
        while remaining:
            candidates=[]
            for unit in remaining:
                taking=Counter(o['class_id'] for n in unit for o in by[n])
                if any(counts[c]<=v for c,v in taking.items()):continue
                candidates.append((abs(target-len(roles[role])-len(unit)),fp([cfg['split_seed'],scene,task,sorted(unit)]),unit,taking))
            if not candidates:break
            _,_,unit,taking=min(candidates,key=lambda x:x[:2])
            if len(roles[role])>=2 and abs(target-len(roles[role])-len(unit))>=abs(target-len(roles[role])):break
            remaining.remove(unit);roles['fit'].difference_update(unit);roles[role].update(unit);counts.subtract(taking)
            if len(roles[role])>=target:break
    if min(len(roles['cal']),len(roles['monitor']))<2:
        raise RuntimeError('No source-disjoint fit/cal/monitor split retaining all classes; no image-level fallback')
    result = {**spec, 'formal': True, 'groups':group_of, 'split_policy':'metadata-only source-group/exact-hash-disjoint fit/cal/monitor; patient identity not independently certified',
              'test_use': 'terminal confirmation only; these benchmarks have already been inspected during research'}
    for role, names in roles.items():
        result[role+'_images'] = [n for n in spec['train_images'] if n in names]
        result[role+'_objects'] = [o for o in spec['objects'] if o['image'] in names]
    result['rare_classes'] = [c for c in range(len(spec['classes']))
                              if len({group_of[o['image']] for o in result['fit_objects'] if o['class_id'] == c}) < 2]
    result['cal_missing_classes'] = [c for c in range(len(spec['classes']))
                                     if not any(o['class_id'] == c for o in result['cal_objects'])]
    result['identity'] = fp({k:v for k,v in result.items() if k != 'identity'})
    return result


def paths(root, cfg, scene, task, seed, smoke=False):
    tag = cfg['tag'] + ('_smoke' if smoke else '')
    return tuple(Path(root)/base/'reference_net'/tag/scene/task/f'seed_{seed}' for base in ['runs','reports'])


def space(root, cfg, extra=0):
    root=Path(root); free=shutil.disk_usage(root).free
    if free < cfg['min_free_gib']*2**30 + extra:
        raise RuntimeError('Insufficient free disk reserve; current checkpoints preserved')
    used=sum(p.stat().st_size for base in ['runs','prepared','reports']
             for p in (root/base/'reference_net').rglob('*') if p.is_file())
    if used+extra > cfg['output_budget_gib']*2**30:
        raise RuntimeError('Reference-net output budget reached; archive completed runs before continuing')
    return dict(free_gib=free/2**30, new_output_gib=used/2**30)


def progress(report, done, total, label, start):
    elapsed=time.monotonic()-start
    write(Path(report)/'PROGRESS.json', dict(done=done,total=total,label=label,elapsed=elapsed,
                                           eta=elapsed/max(done,1)*(total-done)))


def audit(root):
    root=Path(root);cfg=read(root/'configs/reference_net.json');sources=source_files(root)
    manifest=read(root/'PROJECT_MANIFEST.json')['files'];base={}
    for name in [cfg['model']['pf_weights'], cfg['model']['dino_folder']+'/model.safetensors']:
        base[name]=sha(root/name)
        if base[name] != manifest[name]['sha256']: raise RuntimeError('Base weights changed: '+name)
    dataset=read(root/'assets/rwd/DATASET_MANIFEST.json');cases=[];inputs={}
    for scene in cfg['scenes']:
        for task in ['task1','task2']:
            s=split_spec(root,scene,task)
            test_hashes={dataset['files'][f'JPEGImages/{scene}/{n}']['sha256'] for n in s['test_images']}
            for n in s['train_images']:
                rel=f'JPEGImages/{scene}/{n}'
                if dataset['files'][rel]['sha256'] in test_hashes: raise RuntimeError('Exact image overlap with test')
            for n in s['train_images']+s['test_images']:
                for rel in [f'JPEGImages/{scene}/{n}',f'Annotations/{scene}/{Path(n).stem}.xml']:
                    inputs[rel]=dataset['files'][rel]['sha256']
            print(f'Audit {scene}/{task}: fit={len(s["fit_images"])} cal={len(s["cal_images"])} monitor={len(s["monitor_images"])}',flush=True)
            write(root/'prepared/reference_net'/cfg['tag']/'splits'/scene/(task+'.json'),s)
            cases.append(dict(scene=scene,task=task,identity=s['identity'],classes=len(s['classes']),
                              images={k:len(s[k+'_images']) for k in ['fit','cal','monitor']},
                              rare_classes=[s['classes'][c] for c in s['rare_classes']],
                              missing_cal_classes=[s['classes'][c] for c in s['cal_missing_classes']]))
    from concurrent.futures import ThreadPoolExecutor
    def verify(item):
        rel,want=item
        if sha(root/'assets/rwd'/rel)!=want:raise RuntimeError('Dataset bytes changed: '+rel)
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i,_ in enumerate(executor.map(verify,inputs.items()),1):
            if i%128==0 or i==len(inputs):
                print(f'\rDataset SHA {i}/{len(inputs)} | elapsed {time.monotonic()-start:.0f}s',end='',flush=True)
    print()
    plan=lock(root/'reports/reference_net'/cfg['tag']/'PLAN.json',dict(config=cfg,source=fp(sources),
              source_files=sources,base_weights=base,cases=cases,dataset_identity=dataset['identity']))
    write(root/'reports/reference_net'/cfg['tag']/'AUDIT.json',dict(passed=True,plan=plan['identity'],
              training_updates=0,disk=space(root,cfg)))
    return plan


def get_plan(root):
    cfg=read(Path(root)/'configs/reference_net.json')
    p=read(Path(root)/'reports/reference_net'/cfg['tag']/'PLAN.json')
    if p['config'] != cfg or p['source'] != fp(source_files(root)):
        raise RuntimeError('Audit required for current source/config')
    return cfg,p


def recover_best(runpath, committed_sha):
    """Publish a prepared best only after the corresponding last checkpoint committed."""
    runpath=Path(runpath);pending=runpath/'best.pending.pt';best=runpath/'best.pt'
    if pending.exists():
        if committed_sha and sha(pending)==committed_sha:pending.replace(best)
        else:pending.unlink()
    if committed_sha and (not best.exists() or sha(best)!=committed_sha):
        raise RuntimeError('Committed best checkpoint missing/changed; stop before training')
