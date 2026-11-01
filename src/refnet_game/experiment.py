import argparse
import copy
import gc
import json
import platform
from pathlib import Path
import torch
from current.progress import track,stage
from .protocol import (read,write,sha,locations,sources,make_plans,prepare_cache,verify_file,
                       guard_space,save_summary,combine,stage_upload,export_results)


def validate_config(cfg,original):
    if cfg['scene']!='Game' or cfg['tasks']!=['task1','task2'] or cfg['seeds']!=[42,43,44]:
        raise ValueError('158 fixes Game, two tasks and seeds42/43/44')
    for key in ['updates','save_every','validate_every','warmup']:
        if cfg[key]!=original[key]:raise ValueError('Keep original Reference Net training budget: '+key)
    if cfg['updates']!=500 or cfg['arms']!=['F0','F1'] or cfg['known_retention_target']!=.95:
        raise ValueError('Fixed500 and predeclared F0/F1 comparison required')
    if original['model']['unfreeze_backbones'] or original['model']['unfreeze_at']!=100:
        raise ValueError('Keep v1 frozen backbones and stage101 PF update policy')
    if cfg['calibration_min_iou']!=.5 or cfg['relation_batch']!=600:raise ValueError('Keep157 full600 candidate protocol')


def run_case(root,cfg,original,case,dataset,verified,device,command):
    from cil.common import configure_torch
    from jointdet.protocol import load_torch
    from scenepilot.protocol import view
    from refnet.model import ReferenceDetector
    from refnet.cache import load_bank
    from .training import train_parent
    from .evaluation import evaluate_parent
    report,run,spec=case['report'],case['run'],case['spec']
    if (report/'RESULT.json').exists():
        result=read(report/'RESULT.json')
        if result['lock']!=case['record']['identity']:raise RuntimeError('Completed158 result lock mismatch')
        verify_file(run/'last.pt',result['parent_checkpoint']['sha256'],verified)
        for name,h in result['artifacts'].items():verify_file(report/name,h,verified)
        return result
    cache=prepare_cache(root,cfg,spec,case['historical'],verified)
    configure_torch(case['record']['seed'],2)
    c=copy.deepcopy(original['model'])
    for key in ['pf_config','pf_weights','dino_folder']:c[key]=str(Path(root)/c[key])
    with stage(f'Game/{spec["task"]}/seed_{case["record"]["seed"]} initialize from original pretraining'):
        model=ReferenceDetector(root,c,list(range(len(spec['classes'])))).to(device)
        # No historical adapted checkpoint is loaded. Only train_parent's own
        # new-tag last.pt may resume after initialization.
        bank=load_bank(cache,device);v=view(root,'Game')
    try:
        if command!='evaluate':
            training=train_parent(model,v,root,cfg,original,spec,bank,cache,run,report,case['record'],device)
            if not training['complete']:raise RuntimeError('Incomplete fixed500 training')
            write(report/'TRAINING_COMPLETE.json',dict(step=training['step'],initialization=training['initialization'],
                old_adapted_checkpoint_loaded=False,training_seconds=training['training_seconds'],checkpoint_sha256=sha(run/'last.pt')))
        else:
            state=load_torch(run/'last.pt')
            if state['lock']!=case['record']['identity'] or state['bank_manifest']!=bank['manifest_sha256'] or not state['complete'] or state['step']!=500:
                raise RuntimeError('Run158 corrected training to500 before evaluation')
            model.load_adapted(state['model']);del state
        if command=='train':return dict(complete=True,task=spec['task'],seed=case['record']['seed'],training_updates=500)
        return evaluate_parent(root,cfg,model,v,spec,original,bank,cache,run,report,case['record'],dataset,verified)
    finally:
        del model,bank
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def main(root,argv=None):
    root=Path(root);parser=argparse.ArgumentParser(description='158 corrected Game retraining and F0/F1 official evaluation')
    parser.add_argument('command',choices=['run','train','evaluate','plan','combine','summary','export','stage-upload'])
    parser.add_argument('--tasks',nargs='+',choices=['task1','task2']);parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args(argv)
    cfg=read(root/'configs/reference_game.json');original=read(root/'configs/reference_net.json');validate_config(cfg,original)
    if args.command=='stage-upload':result=stage_upload(root,cfg)
    elif args.command=='combine':result=combine(root,cfg)
    elif args.command=='summary':
        result=save_summary(root,cfg);result=dict(complete=result['complete'],cases=result['cases'],path=str(locations(root,cfg)[2]/'SUMMARY.md'))
    elif args.command=='export':
        save_summary(root,cfg);combine(root,cfg);result=export_results(root,cfg)
    else:
        tasks=args.tasks or cfg['tasks'];seeds=args.seeds or cfg['seeds']
        if len(set(tasks))!=len(tasks) or len(set(seeds))!=len(seeds) or not set(seeds)<=set(cfg['seeds']):parser.error('Select distinct configured tasks/seeds')
        source=sources(root);cases=make_plans(root,cfg,source,tasks,seeds)
        result=dict(cases=len(cases),report_directory=str(locations(root,cfg)[2]),training_updates_per_parent=500,
                    expected_new_parents=len(cases),old_parents_overwritten=False,weights_checked=False)
        write(locations(root,cfg)[2]/'PLAN.json',result)
        if args.command!='plan':
            device=torch.device(args.device)
            if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Run real158 training/evaluation on the remote CUDA instance')
            torch.cuda.set_device(device);torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            guard_space(root,cfg);verified={};index=read(root/'MAINLINE_PARENT_INDEX.json')
            with stage('158 verify original pretrained PF/DINO weights'):
                for name,h in index['base_weights'].items():verify_file(root/name,h,verified)
            dataset=read(root/'assets/rwd/DATASET_MANIFEST.json')
            write(locations(root,cfg)[2]/'ENVIRONMENT.json',dict(python=platform.python_version(),torch=torch.__version__,
                gpu=torch.cuda.get_device_name(device),memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                cuda=torch.version.cuda,precision='fp32',allow_tf32=False,autocast=False))
            for case in track(cases,'158 corrected Game two tasks / three seeds',unit='parents'):
                torch.cuda.reset_peak_memory_stats(device)
                with stage(f'Game/{case["spec"]["task"]}/seed_{case["record"]["seed"]}'):
                    run_case(root,cfg,original,case,dataset,verified,device,args.command)
                save_summary(root,cfg)
            if args.command=='train':result=dict(training_complete=True,parents=len(cases),next_command='evaluate')
            else:
                composition=combine(root,cfg);receipt=export_results(root,cfg)
                result=dict(download=receipt,five_scene_composition=composition)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
