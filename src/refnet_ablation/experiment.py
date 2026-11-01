import argparse
import copy
import gc
import json
import platform
import traceback
from pathlib import Path
import torch
from current.progress import track,stage
from .protocol import (read,write,sha,fp,lock,home,case_paths,sources,snapshot,baseline_metadata,
    load_baselines,validate_cache,summarize,stage_upload,export_results,verify_file,guard_space)


def validate_config(cfg,original):
    if cfg['scenes']!=['Medical','Aquatic','Game'] or cfg['task']!='task1' or cfg['seeds']!=[42,43,44]:
        raise ValueError('159 fixes three scenes, Task1 and seeds42/43/44')
    if cfg['variants']!=['no_feedback','no_preserve','raw_relation','no_episode']:raise ValueError('Keep all four controlled variants')
    for key in ['updates','save_every','validate_every','warmup']:
        if cfg[key]!=original[key]:raise ValueError('Keep original training budget: '+key)
    if cfg['updates']!=500 or cfg['arms']!=['F1'] or cfg['known_retention_target']!=.95 or cfg['calibration_min_iou']!=.5:
        raise ValueError('Fixed500 F1 ablation protocol required')
    if cfg['relation_batch']!=600 or cfg['localization_budgets']!=[100,300,600]:raise ValueError('Keep600 candidate localization budget')
    if original['model']['unfreeze_at']!=100 or original['model']['unfreeze_backbones']:raise ValueError('Keep original frozen-backbone/stage101 schedule')


def run_variant(root,cfg,original,source,case,variant,device,dataset,verified):
    from cil.common import configure_torch
    from scenepilot.protocol import view
    from refnet.cache import load_bank
    from .model import AblationDetector,common_initialization
    from .training import train
    from .evaluation import evaluate
    run,report=case_paths(root,cfg,variant,case['scene'],case['seed'])
    binding=lock(run/'LOCK.json',dict(config=cfg,source=source,variant=variant,scene=case['scene'],seed=case['seed'],
        split=case['spec']['identity'],full_control_identity=case['identity'],plan_identity=case['plan']['identity'],
        initialization='original_pretraining; common modules instantiated before ablation',old_adapted_checkpoint_loaded=False))
    if (report/'RESULT.json').exists():
        result=read(report/'RESULT.json')
        if result['lock']!=binding['identity']:raise RuntimeError('Completed ablation lock changed')
        verify_file(run/'last.pt',result['parent_checkpoint']['sha256'],verified)
        for name,h in result['artifacts'].items():verify_file(report/name,h,verified)
        return result
    guard_space(root,cfg);configure_torch(case['seed'],2)
    c=copy.deepcopy(original['model'])
    with stage(f'{variant}/{case["scene"]}/seed_{case["seed"]} fresh original pretraining'):
        model=AblationDetector(root,c,list(range(len(case['spec']['classes']))),variant).to(device)
        common=common_initialization(model)
        common_path=Path(root)/'prepared/reference_ablation'/cfg['tag']/'initialization'/case['scene']/f'seed_{case["seed"]}.json'
        lock(common_path,common)
        counts={}
        for state in [False,True]:
            model.configure_stage(state);counts['stage2' if state else 'stage1']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.configure_stage(False)
        write(report/'PARAMETERS.json',dict(variant=variant,total=sum(p.numel() for p in model.parameters()),trainable=counts,
            common_initialization=common,feedback=model.feedback,preserve_weight=model.c['preserve_weight'],
            episode_enabled=variant!='no_episode',relation='raw5120 cosine' if variant=='raw_relation' else 'learned128 pair relation'))
        bank=load_bank(case['cache'],device);v=view(root,case['scene'])
        if set(bank['labels'].tolist())!=set(range(len(case['spec']['classes']))):raise RuntimeError('Missing full known reference class')
    try:
        result=train(model,v,root,cfg,original,case['spec'],bank,case['cache'],run,report,binding,case['plan'],device)
        if not result['complete']:raise RuntimeError('Incomplete500-step ablation endpoint')
        write(report/'TRAINING_COMPLETE.json',dict(step=result['step'],training_seconds=result['training_seconds'],
            old_adapted_checkpoint_loaded=False,plan_identity=case['plan']['identity'],checkpoint_sha256=sha(run/'last.pt')))
        return evaluate(root,cfg,original,model,v,case,bank,run,report,binding,dataset,verified)
    finally:
        del model,bank;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def run_job(root,cfg,original,source,case,variant,device,dataset,verified):
    try:
        result=run_variant(root,cfg,original,source,case,variant,device,dataset,verified)
    except RuntimeError as exc:
        # A scientific failure to calibrate is retained as undefined, never an
        # invented threshold/AP0. Programming, asset and CUDA errors fail fast.
        if not str(exc).startswith(('Empty/nonfinite known calibration','No legal cal GT')):raise
        run,report=case_paths(root,cfg,variant,case['scene'],case['seed'])
        write(report/'FAILURE.json',dict(variant=variant,scene=case['scene'],seed=case['seed'],
            stage='known-only calibration',error=str(exc),ap_defined=False,test_fallback_used=False,
            checkpoint_sha256=sha(run/'last.pt') if (run/'last.pt').exists() else None))
        (report/'FAILURE.log').write_text(traceback.format_exc(),encoding='utf-8')
        return None
    failure=case_paths(root,cfg,variant,case['scene'],case['seed'])[1]/'FAILURE.json'
    if failure.exists():
        # Do not erase a prior failed attempt; mark its resolution explicitly.
        record=read(failure);record['resolved']=True;write(failure,record)
    return result


def main(root,argv=None):
    root=Path(root);parser=argparse.ArgumentParser(description='159 all core ablations and B0/full localization, one entry')
    parser.add_argument('command',choices=['run','core','diagnose','plan','summary','export','stage-upload'])
    parser.add_argument('--variants',nargs='+');parser.add_argument('--scenes',nargs='+');parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args(argv)
    cfg=read(root/'configs/reference_ablation.json');original=read(root/'configs/reference_net.json');validate_config(cfg,original)
    if args.command=='stage-upload':result=stage_upload(root,cfg)
    elif args.command=='summary':
        from .localization import summarize as summarize_localization
        core=summarize(root,cfg);diag=summarize_localization(root,cfg)
        result=dict(core_complete=core['complete'],localization_complete=diag['complete'],report_directory=str(home(root,cfg)))
    elif args.command=='export':
        summarize(root,cfg);result=export_results(root,cfg)
    else:
        source=sources(root);cfg={**cfg,'_source_identity':fp(source)};snapshot(root,cfg,source)
        full=[baseline_metadata(root,cfg,s,seed) for s in cfg['scenes'] for seed in cfg['seeds']]
        missing=[str(c['report']/'RESULT.json') for c in full if not (c['report']/'RESULT.json').is_file()]
        result=dict(core_training_cases=36,full_controls=9,B0_fixed_sources=3,full_localization_cases=9,
            updates_per_training=500,missing_full_results=missing,report_directory=str(home(root,cfg)),weights_checked=False)
        write(home(root,cfg)/'PLAN.json',result)
        if args.command!='plan':
            variants=args.variants or cfg['variants'];scenes=args.scenes or cfg['scenes'];seeds=args.seeds or cfg['seeds']
            for selected,allowed in [(variants,cfg['variants']),(scenes,cfg['scenes']),(seeds,cfg['seeds'])]:
                if len(set(selected))!=len(selected) or not set(selected)<=set(allowed):parser.error('Select distinct configured variants/scenes/seeds only')
            device=torch.device(args.device)
            if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Real159 training/inference belongs on remote CUDA; local only runs fixtures')
            torch.cuda.set_device(device);torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            write(home(root,cfg)/'ENVIRONMENT.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                device=str(device),gpu=torch.cuda.get_device_name(device),memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                precision='fp32',allow_tf32=False,autocast=False))
            guard_space(root,cfg);verified={};index=read(root/'MAINLINE_PARENT_INDEX.json')
            with stage('159 verify original pretrained PF/DINO weights'):
                for name,h in index['base_weights'].items():verify_file(root/name,h,verified)
            cases=load_baselines(root,cfg,source,verified,with_plans=args.command!='diagnose')
            dataset=read(root/'assets/rwd/DATASET_MANIFEST.json')
            if args.command in ['run','core']:
                checked=set()
                for case in cases:
                    key=(str(case['cache']),case['spec']['identity'])
                    if key not in checked:validate_cache(root,case,verified);checked.add(key)
                jobs=[(variant,case) for variant in variants for case in cases if case['scene'] in scenes and case['seed'] in seeds]
                for variant,case in track(jobs,'159 controlled core ablations',unit='parents'):
                    torch.cuda.reset_peak_memory_stats(device)
                    with stage(f'{variant}/{case["scene"]}/seed_{case["seed"]}'):
                        run_job(root,cfg,original,source,case,variant,device,dataset,verified)
                    summarize(root,cfg)
            if args.command in ['run','diagnose']:
                from .localization import run as diagnose
                diagnose(root,cfg,cases,original,device,dataset,verified)
            summarize(root,cfg);result=export_results(root,cfg)
    if args.command=='plan':result={**{k:v for k,v in result.items() if k!='missing_full_results'},'missing_full_result_count':len(result['missing_full_results'])}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
