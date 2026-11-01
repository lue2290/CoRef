import argparse
import gc
import json
import shutil
import platform
from pathlib import Path
import torch
from current.progress import stage, track
from .protocol import (read, write, sha, lock, locations, sources, inputs, verify_file,
                       verify_parent, guard_space, save_summary, stage_upload, export_results)


def validate_config(cfg):
    if cfg['arms'] != ['F0', 'F1'] or cfg['parent_checkpoint'] != 'terminal':
        raise ValueError('157 fixes original F0 and calibrated F1 at the terminal parent')
    if cfg['known_retention_target'] != .95 or cfg['calibration_min_iou'] != .5:
        raise ValueError('157 locks the156 F1 calibration target and known candidate population')
    if cfg['relation_batch'] < 1 or cfg['min_free_gib'] < 0:
        raise ValueError('Invalid inference batch/disk reserve')
    if cfg['seeds'] != [42, 43, 44] or len(set(cfg['scenes'])) != 5 or cfg['tasks'] != ['task1', 'task2']:
        raise ValueError('Formal benchmark locks five scenes, two tasks and three seeds')


def make_plans(root, cfg, source, scenes, tasks, seeds):
    cases = []
    for scene in scenes:
        for task in tasks:
            for seed in seeds:
                spec, parent, annotation = inputs(root, scene, task, seed)
                report = locations(root, cfg, scene, task, seed)
                record = lock(report/'LOCK.json', dict(config=cfg, sources=source, parent=parent, annotation=annotation,
                    complete_known_class_count=len(spec['classes']), reference_role='original fit GT only; no class held out',
                    training_updates=0, official_test_images=len(spec['test_images'])))
                write(report/'PLAN.json', dict(lock=record['identity'], scene=scene, task=task, seed=seed,
                    classes=spec['classes'], PK=spec['PK'], CK=spec['CK'],
                    fit_images=len(spec['fit_images']), cal_images=len(spec['cal_images']), test_images=len(spec['test_images']),
                    cal_missing_gt_classes=spec['cal_missing_classes'], rare_fit_classes=spec['rare_classes'],
                    threshold_selection='pooled original cal known predicted candidates, no official test selection'))
                cases.append(dict(spec=spec, parent=parent, report=report, record=record))
    return cases


def run_case(root, cfg, case, dataset, verified, device):
    from cil.common import configure_torch
    from jointdet.protocol import load_torch
    from refnet.model import ReferenceDetector
    from scenepilot.protocol import view
    from .runtime import load_references, calibrate, predict, official_evaluation
    spec, parent, report = case['spec'], case['parent'], case['report']
    if (report/'RESULT.json').exists():
        result = read(report/'RESULT.json')
        if result['lock'] != case['record']['identity']:
            raise RuntimeError('Completed157 result lock changed')
        for name, h in result['artifacts'].items(): verify_file(report/name, h, verified)
        return result
    guard_space(root, cfg); configure_torch(parent['seed'], 2)
    original = read(Path(root)/'configs/reference_net.json'); c = original['model']
    if (c['proposal_limit'], c['detection_limit'], c['nms_iou'], c['relation_neighbors']) != (600, 100, .5, 3):
        raise RuntimeError('157 preserves600 proposals, top3 neighbors, NMS.5 and100 detections')
    parent_cal = verify_parent(root, parent, verified)
    with stage(parent['id']+' load frozen terminal parent'):
        model = ReferenceDetector(root, c, list(range(len(spec['classes'])))).to(device)
        state = load_torch(Path(root)/parent['terminal']['checkpoint'])
        if state['step'] != 500 or not state['complete']:
            raise RuntimeError('Incomplete parent endpoint')
        model.load_adapted(state['model']); del state
        model.requires_grad_(False); model.eval()
        if any(p.requires_grad for p in model.parameters()): raise RuntimeError('Parent must be entirely frozen')
    try:
        bank, cal_gt, audit = load_references(root, spec, parent, device, dataset, verified)
        lock(report/'REFERENCE_BANK.json', audit)
        v = view(root, spec['scene'])
        with stage(parent['id']+' bind inference/calibration inputs'):
            lock(report/'RUNTIME_LOCK.json', dict(case=case['record']['identity'], bank=audit['identity'],
                original_parent_calibration=parent_cal, dtype='fp32', autocast=False, relation_batch=cfg['relation_batch']))
        cal = calibrate(model, v, root, spec, bank, cal_gt, cfg, c, parent_cal, report, verified, dataset)
        predict(model, v, root, spec, bank, cfg, c, cal, report, dataset, verified)
        metrics = official_evaluation(root, spec, cfg, report, dataset, verified)
        paths = ['CALIBRATION.json', 'REFERENCE_BANK.json', 'PREDICTIONS_COMPLETE.json', 'RUNTIME_LOCK.json']
        paths += [arm+'/'+name for arm in cfg['arms'] for name in ['predictions.json', 'metrics.json', 'pr_curves.json', 'errors_by_image.json']]
        result = dict(complete=True, scene=spec['scene'], task=spec['task'], seed=parent['seed'],
            lock=case['record']['identity'], scope=cfg['scope'], parent_checkpoint=parent['terminal'], metrics=metrics,
            new_training_updates=0, trainable_parameters=0, official_evaluator_unchanged=True,
            calibration=cal, artifacts={name: sha(report/name) for name in paths})
        write(report/'RESULT.json', result)
        return result
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main(root, argv=None):
    parser = argparse.ArgumentParser(description='157: formal frozen-parent F0/F1 comparison; no new training')
    parser.add_argument('command', choices=['formal', 'plan', 'summary', 'export', 'stage-upload'])
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv); root = Path(root)
    cfg = read(root/'configs/reference_f1.json'); validate_config(cfg)
    if args.command=='stage-upload': result = stage_upload(root, cfg)
    elif args.command=='summary': result = save_summary(root, cfg)
    elif args.command=='export':
        save_summary(root, cfg); result = export_results(root, cfg)
    else:
        scenes, tasks, seeds = args.scenes or cfg['scenes'], args.tasks or cfg['tasks'], args.seeds or cfg['seeds']
        for selected, allowed in [(scenes, cfg['scenes']), (tasks, cfg['tasks']), (seeds, cfg['seeds'])]:
            if not set(selected) <= set(allowed) or len(set(selected)) != len(selected):
                parser.error('Select unique configured scenes/tasks/seeds only')
        source = sources(root)
        folder = locations(root, cfg)
        lock(folder/'SOURCE/LOCK.json', dict(config=cfg, sources=source))
        for name in source:
            if name.startswith('src/refnet_f1/') or name in ['run_reference_f1.py', 'configs/reference_f1.json']:
                dest = folder/'SOURCE'/name; dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): shutil.copy2(root/name, dest)
                if sha(dest) != source[name]: raise RuntimeError('157 source snapshot changed')
        cases = make_plans(root, cfg, source, scenes, tasks, seeds)
        result = dict(cases=len(cases), parents=[c['parent']['id'] for c in cases], report_directory=str(folder),
                      new_training_updates=0, weights_checked=False)
        write(folder/'PLAN.json', result)
        if args.command=='formal':
            device = torch.device(args.device)
            if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; use the remote training instance')
            if device.type=='cuda': torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            dataset = read(root/'assets/rwd/DATASET_MANIFEST.json'); index = read(root/'MAINLINE_PARENT_INDEX.json'); verified = {}
            guard_space(root, cfg)
            with stage('157 verify retained PF/DINO base weights'):
                for name, h in index['base_weights'].items(): verify_file(root/name, h, verified)
            for case in track(cases, '157 verify all selected terminal parents', unit='parents'):
                verify_parent(root, case['parent'], verified)
            environment = dict(python=platform.python_version(), torch=torch.__version__, device=str(device),
                cuda=torch.version.cuda, new_training_updates=0, allow_tf32=False, autocast=False,
                nms_device='cpu, identical for F0/F1')
            if device.type=='cuda': environment.update(gpu=torch.cuda.get_device_name(device), memory_bytes=torch.cuda.get_device_properties(device).total_memory)
            write(folder/'ENVIRONMENT.json', environment)
            for case in track(cases, '157 full official F0/F1 benchmark', unit='parents'):
                with stage(case['parent']['id']):
                    run_case(root, cfg, case, dataset, verified, device)
                save_summary(root, cfg)
            result = export_results(root, cfg)
    # Large summaries belong in files, not on the progress console.
    if args.command=='summary': result = dict(complete=result['complete'], cases=result['cases'], path=str(locations(root,cfg)/'SUMMARY.md'))
    if args.command=='plan': result = {k: v for k, v in result.items() if k != 'parents'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
