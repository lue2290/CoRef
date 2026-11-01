from pathlib import Path
from collections import Counter
import shutil
import zipfile
import numpy as np
from refnet.protocol import read, write, sha, fp, lock
from current.progress import track, stage


def locations(root, cfg, scene=None, task=None, seed=None):
    path = Path(root)/'reports/reference_f1'/cfg['tag']
    return path/scene/task/f'seed_{seed}' if scene else path


def sources(root):
    root = Path(root); original = read(root/'MAINLINE_MANIFEST.json')['files']
    prefixes = ('src/refnet/', 'versions/joint_detector/jointdet/', 'vendor/pfrpn/mmdet/',
                'vendor/pfrpn/configs/', 'src/cil/', 'src/scenepilot/', 'src/fivescene/')
    names = [n for n in original if n.endswith('.py') and n.startswith(prefixes)]
    names += ['MAINLINE_PARENT_INDEX.json', 'configs/reference_net.json', 'run.py',
              'vendor/fomo_open_world_eval.py', 'assets/models/dinov3_large/config.json',
              'src/current/progress.py', 'src/current/progress_io.py',
              'run_reference_f1.py', 'configs/reference_f1.json']
    names += [p.relative_to(root).as_posix() for p in (root/'src/refnet_f1').glob('*.py')]
    result = {}
    for name in track(sorted(set(names)), '157 verify runtime sources', unit='files'):
        result[name] = sha(root/name)
        if name in original and result[name] != original[name]:
            raise RuntimeError('Frozen mainline changed: '+name)
    receipt = root/'REFERENCE_F1_UPLOAD_MANIFEST.json'
    if receipt.exists():
        for name, expected in read(receipt)['files'].items(): verify_file(root/name, expected)
    return result


def inputs(root, scene, task, seed):
    root = Path(root); revision = read(root/'assets/rwd/ACTIVE_ANNOTATION_REVISION.json')
    index = read(root/'MAINLINE_PARENT_INDEX.json')
    parent = next((p for p in index['models'] if (p['scene'], p['task'], p['seed']) == (scene, task, seed)), None)
    if parent is None:
        raise ValueError(f'No indexed parent: {scene}/{task}/seed_{seed}')
    original = read(root/parent['split'])
    if original['identity'] != parent['split_identity'] or original['identity'] != fp({k: v for k, v in original.items() if k != 'identity'}):
        raise RuntimeError('Original parent split mismatch')
    path = root/parent['split']
    if scene == 'Game':
        row = revision['training_inputs'][task]; path = root/row['path']
        if sha(path) != row['sha256']:
            raise RuntimeError('Corrected Game split changed')
        dataset = read(root/'assets/rwd/DATASET_MANIFEST.json')['files']
        for correction in revision['corrections']:
            xml = correction['xml']
            verify_file(root/xml, dataset[xml.removeprefix('assets/rwd/')]['sha256'])
    spec = read(path)
    if spec['identity'] != fp({k: v for k, v in spec.items() if k != 'identity'}):
        raise RuntimeError('Split content identity mismatch')
    if spec['classes'] != original['classes'] or spec['test_images'] != original['test_images']:
        raise RuntimeError('Formal class order or test population changed')
    roles = []
    for role in ['fit', 'cal', 'monitor']:
        if spec[role+'_images'] != original[role+'_images']:
            raise RuntimeError('Formal experiment preserves parent fit/cal/monitor image roles')
        roles.append({spec['groups'][n] for n in spec[role+'_images']})
    if any(a & b for i, a in enumerate(roles) for b in roles[i+1:]):
        raise RuntimeError('Source groups overlap across fit/cal/monitor')
    if set(spec['train_images']) & set(spec['test_images']):
        raise RuntimeError('Train/test image overlap')
    if set(o['class_id'] for o in spec['fit_objects']) != set(range(len(spec['classes']))):
        raise RuntimeError('Full formal bank must contain every known class')
    if task == 'task2' and spec['classes'] != spec['PK']+spec['CK']:
        raise RuntimeError('Task2 requires the indexed Task2 parent and PK+CK bank')
    return spec, parent, dict(split_path=path.relative_to(root).as_posix(), split_sha256=sha(path),
        revision_sha256=sha(root/'assets/rwd/ACTIVE_ANNOTATION_REVISION.json'),
        dataset_manifest_sha256=sha(root/'assets/rwd/DATASET_MANIFEST.json'),
        corrected_reference_labels=scene == 'Game', parent_retrained_on_corrections=False)


def guard_space(root, cfg, extra=0):
    free = shutil.disk_usage(root).free
    if free < cfg['min_free_gib']*2**30+extra:
        raise RuntimeError('Insufficient disk reserve; predictions are committed, free space then resume157')


def verify_file(path, expected, verified=None):
    path = Path(path)
    stat = path.stat() if path.is_file() else None
    key = (str(path), stat.st_size, stat.st_mtime_ns) if stat else None
    actual = verified.get(key) if verified is not None and key else None
    if actual is None:
        if not stat:
            raise FileNotFoundError('Required retained asset missing: '+str(path))
        actual = sha(path)
        if verified is not None:
            verified[key] = actual
    if actual != expected:
        raise RuntimeError('Asset SHA256 mismatch: '+str(path))
    return actual


def verify_parent(root, parent, verified):
    root = Path(root); terminal = parent['terminal']
    if terminal['step'] != 500 or terminal['calibration_checkpoint_sha256'] != terminal['sha256']:
        raise RuntimeError('Require matching fixed500 terminal parent/calibration')
    for path, h in [(terminal['checkpoint'], terminal['sha256']),
                    (terminal['calibration'], terminal['calibration_sha256']),
                    (parent['lock'], parent['lock_sha256']),
                    (parent['reference_cache']+'/MANIFEST.json', parent['reference_manifest_sha256'])]:
        verify_file(root/path, h, verified)
    cal = read(root/terminal['calibration'])
    if cal['checkpoint_sha256'] != terminal['sha256']:
        raise RuntimeError('Calibration bound to another checkpoint')
    return cal


def remap_cached(raw, ids, gt, cached_labels, original_objects, corrected_objects, wh):
    """Relabel identical GT instances; never reuse descriptors from a different box."""
    old = {str(o['id']): o for o in original_objects}
    pos = {str(k): i for i, k in enumerate(ids)}
    if len(pos) != len(ids) or set(pos) != set(old):
        raise RuntimeError('Cached GT instance identities differ from indexed parent split')
    if raw.ndim != 2 or raw.shape != (len(ids), 5120) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid frozen DINO GT descriptors')
    for key, obj in old.items():
        j = pos[key]
        if int(cached_labels[j]) != obj['class_id'] or not np.allclose(gt[j], np.asarray(obj['box'])/wh, atol=2e-6, rtol=0):
            raise RuntimeError('Cached GT label/box differs from original parent input')
    indices = []
    for obj in corrected_objects:
        key = str(obj.get('original_instance_id', obj['id']))
        if key not in old or obj['box'] != old[key]['box']:
            raise RuntimeError('Corrected reference is not an unchanged cached GT instance')
        indices.append(pos[key])
    if len(indices) != len(old) or len(set(indices)) != len(indices):
        raise RuntimeError('Correction unexpectedly added/removed/duplicated GT instances')
    return raw[indices].copy(), [int(o['class_id']) for o in corrected_objects]


def summary(results, cfg):
    """Aggregate fixed seeds before averaging scenes; never pick scene-specific seeds."""
    lookup = {(r['scene'], r['task'], r['seed']): r for r in results}
    expected = {(s, t, seed) for s in cfg['scenes'] for t in cfg['tasks'] for seed in cfg['seeds']}
    if len(lookup) != len(results) or set(lookup)-expected:
        raise RuntimeError('Duplicate or unexpected formal cases')
    def stats(values):
        return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                    sample_std=float(np.std(values, ddof=1)) if len(values)>1 else None)
    by_scene = {}; macros = {}; deltas = {}; constraints = {}
    for task in cfg['tasks']:
        metrics = ['K_AP50', 'U_AP50']+(['PK_AP50', 'CK_AP50'] if task == 'task2' else [])
        for arm in cfg['arms']:
            for metric in metrics:
                key = f'{task}/{arm}/{metric}'; per_seed = {}
                for scene in cfg['scenes']:
                    values = {str(seed): lookup[scene, task, seed]['metrics'][arm][metric]
                              for seed in cfg['seeds'] if (scene, task, seed) in lookup}
                    valid = [v for v in values.values() if v is not None]
                    by_scene[f'{scene}/{key}'] = dict(**stats(valid), by_seed=values)
                for seed in cfg['seeds']:
                    rows = [lookup.get((scene, task, seed)) for scene in cfg['scenes']]
                    if all(r is not None and r['metrics'][arm][metric] is not None for r in rows):
                        per_seed[str(seed)] = float(np.mean([r['metrics'][arm][metric] for r in rows]))
                macros[key] = dict(**stats(list(per_seed.values())), by_seed=per_seed,
                                   complete=len(per_seed)==len(cfg['seeds']))
        for metric in metrics:
            a = macros[f'{task}/F0/{metric}']['by_seed']; b = macros[f'{task}/F1/{metric}']['by_seed']
            ds = {k: b[k]-a[k] for k in b if k in a}
            deltas[f'{task}/{metric}'] = dict(**stats(list(ds.values())), by_seed=ds)
        for metric in (['K_AP50'] if task == 'task1' else ['PK_AP50', 'CK_AP50']):
            v = deltas[f'{task}/{metric}']; m = macros[f'{task}/F1/{metric}']
            floor = cfg['historical_macro_floors'][f'{task}/{metric}']
            constraints[f'{task}/{metric}'] = dict(paired_mean_drop_at_most_5pp=v['mean'] >= -5 if m['complete'] else None,
                historical_floor=floor, above_historical_floor=m['mean'] >= floor if m['complete'] else None,
                historical_comparability='Game uses corrected reference labels; use paired F0 as the primary control')
    return dict(scope=cfg['scope'], complete=set(lookup)==expected, cases=len(results), expected_cases=len(expected),
                by_scene=by_scene, macro=macros, paired_macro_delta= deltas, constraints=constraints,
                statistic='mean +/- sample SD across fixed seeds; macro is mean across five scenes within each seed',
                sotA_claim=False, note='F1 selected from development only; no new head training; previously inspected benchmark, not blind test')


def save_summary(root, cfg):
    folder = locations(root, cfg)
    results = [read(p) for p in sorted(folder.glob('*/*/seed_*/RESULT.json'))]
    result = summary(results, cfg); write(folder/'SUMMARY.json', result)
    lines = ['# Reference Net v1 F1 formal benchmark', '', f'Complete: {result["complete"]}; {len(results)}/{result["expected_cases"]} cases.', '',
             'Five-scene macros, mean ± sample SD across seeds42/43/44. Partial rows are not final results.', '',
             '| Task / arm / metric | Mean | Sample SD | n |', '|---|---:|---:|---:|']
    def val(x): return '--' if x is None else f'{x:.2f}'
    for key, v in result['macro'].items():
        lines.append(f'| {key} | {val(v["mean"])} | {val(v["sample_std"])} | {v["n"]} |')
    lines += ['', '## Per scene', '', '| Scene / task / arm / metric | Mean | Sample SD | n |', '|---|---:|---:|---:|']
    for key, v in result['by_scene'].items():
        lines.append(f'| {key} | {val(v["mean"])} | {val(v["sample_std"])} | {v["n"]} |')
    lines += ['', 'F0 and F1 share candidates, complete fit reference bank, gate, NMS and100 output limit.',
              'Game uses corrected reference labels with historical frozen parents; F0 old-head GT shift is recalibrated there.',
              'Missing/rare calibration classes are reported per case; pooled95% calibration is not a per-class or test-set guarantee.',
              'No comparable strongest-baseline rerun is included; these results alone do not establish SOTA.']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def export_results(root, cfg):
    root = Path(root); folder = locations(root, cfg)
    for path in track(sorted(folder.glob('*/*/seed_*/RESULT.json')), '157 verify completed output bindings', unit='cases'):
        result = read(path)
        for name, h in result.get('artifacts', {}).items(): verify_file(path.parent/name, h)
    paths = [p for p in folder.rglob('*') if p.is_file() and 'UPLOAD' not in p.parts
             and p.suffix in ['.json', '.jsonl', '.gz', '.md', '.log', '.py'] and p.name != 'DOWNLOAD_RECEIPT.json']
    if not any(p.name == 'RESULT.json' for p in paths):
        raise RuntimeError('No completed157 case; finish at least one before exporting')
    guard_space(root, cfg, sum(p.stat().st_size for p in paths))
    path = folder/'REFERENCE_F1_RESULTS.zip'; tmp = path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in track(paths, '157 pack results', unit='files'):
            z.write(p, p.relative_to(root).as_posix())
    with stage('157 verify ZIP CRC and SHA256'):
        with zipfile.ZipFile(tmp) as z:
            if z.testzip(): raise RuntimeError('Result ZIP CRC failed')
        digest = sha(tmp)
    tmp.replace(path)
    receipt = dict(path=str(path), bytes=path.stat().st_size, sha256=digest, crc_verified=True, files=len(paths))
    write(folder/'DOWNLOAD_RECEIPT.json', receipt)
    return receipt


def stage_upload(root, cfg):
    root = Path(root); out = locations(root, cfg)/'UPLOAD/OpenDet_Current'
    files = ['run_reference_f1.py', 'configs/reference_f1.json', '.run/157_Reference_F1_Formal.run.xml',
             'docs/REFERENCE_F1_FORMAL.md', 'tests/test_reference_f1.py']
    files += [p.relative_to(root).as_posix() for p in (root/'src/refnet_f1').glob('*.py')]
    hashes = {}
    for name in track(files, '157 stage upload tree', unit='files'):
        path = out/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, path); hashes[name] = sha(path)
    write(out/'REFERENCE_F1_UPLOAD_MANIFEST.json', dict(tag=cfg['tag'], files=hashes))
    return dict(upload_directory=str(out), files=len(hashes), weights_included=False)
