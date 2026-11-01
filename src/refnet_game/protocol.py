from pathlib import Path
from collections import Counter
import copy
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageOps
from refnet.protocol import read, write, sha, fp, lock
from refnet_f1.protocol import inputs, verify_file, guard_space, remap_cached, summary as f1_summary
from current.progress import track, stage


def locations(root, cfg, task=None, seed=None):
    tail = Path('reference_game')/cfg['tag']
    if task: tail = tail/'Game'/task/f'seed_{seed}'
    return tuple(Path(root)/base/tail for base in ['prepared', 'runs', 'reports'])


def sources(root):
    root = Path(root); original = read(root/'MAINLINE_MANIFEST.json')['files']
    prefixes = ('src/refnet/', 'versions/joint_detector/jointdet/', 'vendor/pfrpn/mmdet/',
                'vendor/pfrpn/configs/', 'src/cil/', 'src/scenepilot/', 'src/fivescene/')
    names = [n for n in original if n.endswith('.py') and n.startswith(prefixes)]
    names += ['MAINLINE_PARENT_INDEX.json', 'configs/reference_net.json', 'run.py',
        'vendor/fomo_open_world_eval.py', 'assets/models/dinov3_large/config.json',
        'src/current/progress.py', 'src/current/progress_io.py', 'run_reference_game.py', 'configs/reference_game.json',
        'src/refnet_f1/__init__.py', 'src/refnet_f1/protocol.py', 'src/refnet_f1/runtime.py']
    names += [p.relative_to(root).as_posix() for p in (root/'src/refnet_game').glob('*.py')]
    result = {}
    for name in track(sorted(set(names)), '158 verify numerical sources', unit='files'):
        result[name] = sha(root/name)
        if name in original and result[name] != original[name]: raise RuntimeError('Frozen mainline changed: '+name)
    receipt = root/'REFERENCE_GAME_UPLOAD_MANIFEST.json'
    if receipt.exists():
        for name, expected in read(receipt)['files'].items(): verify_file(root/name, expected)
    return result


def make_plans(root, cfg, source, tasks, seeds):
    cases = []; original_cfg = read(Path(root)/'configs/reference_net.json')
    _, _, home = locations(root, cfg)
    lock(home/'SOURCE/LOCK.json', dict(config=cfg, source=source, original_training_config=original_cfg))
    for name, h in source.items():
        if name.startswith(('src/refnet_game/', 'src/refnet_f1/')) or name in ['run_reference_game.py', 'configs/reference_game.json']:
            dest = home/'SOURCE'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists(): shutil.copy2(Path(root)/name, dest)
            verify_file(dest, h)
    for task in tasks:
        for seed in seeds:
            spec, historical, annotation = inputs(root, 'Game', task, seed)
            annotation = {**annotation, 'parent_retrained_on_corrections': True}
            cache, run, report = locations(root, cfg, task, seed)
            record = lock(run/'LOCK.json', dict(config=cfg, original_training_config=original_cfg,
                source=source, annotation=annotation, seed=seed, task=task,
                original_cache=historical['reference_cache'], original_cache_sha256=historical['reference_manifest_sha256'],
                initialization=cfg['initialization'], old_adapted_checkpoint_loaded=False))
            write(report/'PLAN.json', dict(lock=record['identity'], seed=seed, task=task, classes=spec['classes'],
                fit_images=len(spec['fit_images']), fit_objects=len(spec['fit_objects']),
                cal_images=len(spec['cal_images']), monitor_images=len(spec['monitor_images']),
                test_images=len(spec['test_images']), updates=cfg['updates'], initialization=cfg['initialization'],
                old_adapted_checkpoint_loaded=False, training_split=annotation['split_path']))
            cases.append(dict(spec=spec, historical=historical, annotation=annotation, record=record,
                              run=run, report=report, cache_root=cache))
    return cases


def prepare_cache(root, cfg, spec, historical, verified):
    """GT boxes/pixels/teacher are unchanged: rebuild labels, not the visual encoder."""
    root = Path(root); original_folder = root/historical['reference_cache']
    verify_file(original_folder/'MANIFEST.json', historical['reference_manifest_sha256'], verified)
    original = read(root/historical['split']); manifest = read(original_folder/'MANIFEST.json')
    if manifest['split'] != original['identity']: raise RuntimeError('Historical GT cache has another split')
    dataset = read(root/'assets/rwd/DATASET_MANIFEST.json')
    cache = locations(root, cfg)[0]/'cache/Game'/spec['task']; cache.mkdir(parents=True, exist_ok=True)
    binding = lock(cache/'LOCK.json', dict(split=spec['identity'],
        original_manifest_sha256=historical['reference_manifest_sha256'],
        base_weights=read(root/'MAINLINE_PARENT_INDEX.json')['base_weights'],
        policy='same GT identity/box/pixels and original frozen PF teacher; copy unchanged arrays, rebuild corrected labels/IDs'))
    old_by = {}; new_by = {}
    for obj in original['objects']: old_by.setdefault(obj['image'], []).append(obj)
    for obj in spec['objects']: new_by.setdefault(obj['image'], []).append(obj)
    rows = []; files = {}; index = {(r['role'], r['image']): r for r in manifest['rows']}
    for role in ['fit', 'cal', 'monitor']:
        for name in track(spec[role+'_images'], f'Game/{spec["task"]} rebuild corrected {role} cache', unit='images'):
            row = index[role, name]; source = original_folder/row['file']
            verify_file(source, manifest['files'][row['file']], verified)
            image = root/'assets/rwd/JPEGImages/Game'/name
            verify_file(image, dataset['files']['JPEGImages/Game/'+name]['sha256'], verified)
            with Image.open(image) as im: w, h = ImageOps.exif_transpose(im).size
            with np.load(source, allow_pickle=False) as z:
                raw, labels = remap_cached(z['raw'], z['ids'].tolist(), z['gt'], z['labels'],
                    old_by[name], new_by[name], np.array([w, h, w, h]))
                payload = {key: z[key].copy() for key in z.files}
                payload.update(raw=raw, labels=np.array(labels, dtype=np.int64), ids=np.array([o['id'] for o in new_by[name]]),
                               gt=np.array([o['box'] for o in new_by[name]], dtype=np.float32)/np.array([w,h,w,h],dtype=np.float32))
            if role=='fit' and not {'boxes','p','ctr','stable'} <= set(payload):
                raise RuntimeError('Original pretrained PF teacher payload missing: '+name)
            dest = cache/(fp(name)[:24]+'.npz')
            if dest.exists():
                with np.load(dest, allow_pickle=False) as z:
                    if set(z.files)!=set(payload) or any(not np.array_equal(z[k], v) for k, v in payload.items()):
                        raise RuntimeError('Corrected cache payload changed: '+name)
            else:
                guard_space(root, cfg, sum(a.nbytes for a in payload.values()))
                tmp = dest.with_suffix('.npz.tmp')
                with tmp.open('wb') as stream: np.savez_compressed(stream, **payload)
                tmp.replace(dest)
            files[dest.name] = sha(dest)
            rows.append(dict(image=name, role=role, group=spec['groups'][name], file=dest.name))
    result = dict(identity=binding['identity'], split=spec['identity'], rows=rows, files=files,
        thresholds=manifest['thresholds'], teacher_matched_count=manifest['teacher_matched_count'],
        fallback_teacher_thresholds=manifest['fallback_teacher_thresholds'],
        feature_space='unchanged frozen DINO GT features; corrected labels/instance IDs',
        historical_manifest_sha256=historical['reference_manifest_sha256'], corrected_labels=True)
    path = cache/'MANIFEST.json'
    if path.exists() and read(path) != result: raise RuntimeError('Corrected cache manifest changed')
    write(path, result)
    return cache


def table_text(result, title):
    lines = ['# '+title, '', f'Complete: {result["complete"]}; {result["cases"]}/{result["expected_cases"]} cases.', '',
        '| Scene/task/arm/metric | Mean | Sample SD | n |', '|---|---:|---:|---:|']
    def val(x): return '--' if x is None else f'{x:.2f}'
    for key, v in result['by_scene'].items():
        lines.append(f'| {key} | {val(v["mean"])} | {val(v["sample_std"])} | {v["n"]} |')
    lines += ['', '| Across-scene macro | Mean | Sample SD | n |', '|---|---:|---:|---:|']
    for key, v in result['macro'].items():
        lines.append(f'| {key} | {val(v["mean"])} | {val(v["sample_std"])} | {v["n"]} |')
    lines += ['', 'Fixed500 endpoints, seeds42/43/44; mean +/- sample SD (ddof=1).',
        'Game parents were retrained on corrected labels from original pretraining. Old parents/index are preserved.',
        'No test threshold/model selection and no blind-test or SOTA claim.']
    return '\n'.join(lines)+'\n'


def summary_config(root, cfg, combined=False):
    scenes = read(Path(root)/'configs/reference_net.json')['scenes'] if combined else ['Game']
    return dict(scenes=scenes, tasks=cfg['tasks'], seeds=cfg['seeds'], arms=cfg['arms'], scope=cfg['scope'],
        historical_macro_floors={'task1/K_AP50':54.56,'task2/PK_AP50':56.16,'task2/CK_AP50':53.24})


def save_summary(root, cfg):
    home = locations(root, cfg)[2]
    rows = [read(p) for p in sorted(home.glob('Game/*/seed_*/RESULT.json'))]
    result = f1_summary(rows, summary_config(root,cfg))
    # Five-scene historical floors do not apply to the Game-only report.
    result['constraints'] = {k: {'paired_mean_drop_at_most_5pp':v['paired_mean_drop_at_most_5pp']} for k,v in result['constraints'].items()}
    result['statistic'] = 'Game only: mean +/- sample SD across three fixed seeds; not a five-scene macro'
    result['corrected_label_retraining'] = True
    result['note'] = 'All Game parents start from original pretraining; F1 has no extra trained head'
    write(home/'SUMMARY.json', result)
    (home/'SUMMARY.md').write_text(table_text(result,'Corrected Game Reference Net v1 F0/F1'),encoding='utf-8')
    entries = [read(p) for p in sorted(home.glob('Game/*/seed_*/CORRECTED_PARENT.json'))]
    write(home/'CORRECTED_PARENT_INDEX.json',dict(format='corrected-game-parents-v1', models=entries,
        base_weights=read(Path(root)/'MAINLINE_PARENT_INDEX.json')['base_weights'],
        note='Separate six-parent index. Does not replace MAINLINE_PARENT_INDEX.json.'))
    return result


def combine(root, cfg):
    root=Path(root); home=locations(root,cfg)[2]; selection=summary_config(root,cfg,True)
    original_index=read(root/'MAINLINE_PARENT_INDEX.json')['models']
    rows=[]; composition=[]; missing=[]; dataset_sha=sha(root/'assets/rwd/DATASET_MANIFEST.json')
    for scene in selection['scenes']:
        for task in cfg['tasks']:
            for seed in cfg['seeds']:
                report = home/'Game'/task/f'seed_{seed}' if scene=='Game' else root/'reports/reference_f1'/cfg['comparison_tag']/scene/task/f'seed_{seed}'
                path=report/'RESULT.json'
                if not path.exists(): missing.append(f'{scene}/{task}/seed_{seed}');continue
                row=read(path)
                if not row['complete'] or (row['scene'],row['task'],row['seed'])!=(scene,task,seed): raise RuntimeError('Combined result identity mismatch')
                for name,h in row['artifacts'].items(): verify_file(report/name,h)
                if scene!='Game':
                    parent=next(p for p in original_index if (p['scene'],p['task'],p['seed'])==(scene,task,seed))
                    if row['parent_checkpoint']['sha256']!=parent['terminal']['sha256']: raise RuntimeError('Other-scene parent changed')
                    binding=read(report/'LOCK.json')
                    if binding['annotation']['dataset_manifest_sha256']!=dataset_sha: raise RuntimeError('Cannot combine different data revisions')
                elif not row.get('corrected_label_retraining'): raise RuntimeError('Never substitute historical Game into corrected summary')
                if row['calibration']['target']!=cfg['known_retention_target']: raise RuntimeError('F1 calibration target differs')
                if any(m['official_evaluator_sha256']!=sha(root/'vendor/fomo_open_world_eval.py') for m in row['metrics'].values()): raise RuntimeError('Evaluator changed')
                rows.append(row);composition.append(dict(scene=scene,task=task,seed=seed,path=path.relative_to(root).as_posix(),sha256=sha(path),corrected_parent=scene=='Game'))
    result=f1_summary(rows,selection)
    result.update(composition=composition,missing_cases=missing,corrected_game_replaces_all_six=True,
        note='Fixed composition: corrected Game six parents plus unchanged157 other24; never select the better old/new Game result')
    write(home/'COMBINED_FIVE_SCENE.json',result)
    (home/'COMBINED_FIVE_SCENE.md').write_text(table_text(result,'Five scenes with corrected Game parents'),encoding='utf-8')
    return dict(complete=result['complete'],cases=result['cases'],missing=missing,path=str(home/'COMBINED_FIVE_SCENE.json'))


def stage_upload(root,cfg):
    root=Path(root); out=locations(root,cfg)[2]/'UPLOAD/OpenDet_Current'
    names=['run_reference_game.py','configs/reference_game.json','.run/158_Reference_Game_Retrain.run.xml',
        'docs/REFERENCE_GAME_RETRAIN.md','tests/test_reference_game.py',
        'src/refnet_f1/__init__.py','src/refnet_f1/protocol.py','src/refnet_f1/runtime.py']
    names += [p.relative_to(root).as_posix() for p in (root/'src/refnet_game').glob('*.py')]
    hashes={}
    for name in track(names,'158 stage upload tree',unit='files'):
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest);hashes[name]=sha(dest)
    write(out/'REFERENCE_GAME_UPLOAD_MANIFEST.json',dict(tag=cfg['tag'],files=hashes))
    return dict(upload_directory=str(out),files=len(hashes),weights_included=False)


def export_results(root,cfg):
    root=Path(root);home=locations(root,cfg)[2]
    completed=sorted(home.glob('Game/*/seed_*/RESULT.json'))
    if not completed: raise RuntimeError('No evaluated158 case to export; finish run/evaluate first')
    for path in track(completed,'158 verify evaluated output bindings',unit='cases'):
        for name,h in read(path)['artifacts'].items():verify_file(path.parent/name,h)
    names=[p for base in [home,locations(root,cfg)[1]] for p in base.rglob('*') if p.is_file()
        and 'UPLOAD' not in p.parts and p.suffix in ['.json','.jsonl','.md','.py','.log'] and p.name!='DOWNLOAD_RECEIPT.json']
    path=home/'REFERENCE_GAME_RESULTS.zip';tmp=path.with_suffix('.zip.tmp')
    guard_space(root,cfg,sum(p.stat().st_size for p in names))
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in track(names,'158 pack results without model weights',unit='files'):z.write(p,p.relative_to(root).as_posix())
    with stage('158 verify ZIP CRC/SHA256'):
        with zipfile.ZipFile(tmp) as z:
            if z.testzip(): raise RuntimeError('ZIP CRC failed')
        h=sha(tmp)
    tmp.replace(path);receipt=dict(path=str(path),bytes=path.stat().st_size,sha256=h,crc_verified=True,weights_included=False)
    write(home/'DOWNLOAD_RECEIPT.json',receipt)
    return receipt
