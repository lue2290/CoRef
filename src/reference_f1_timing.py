"""Actual CUDA measurements, without prediction export, GT matching or AP evaluation."""
import argparse
import gc
import json
from pathlib import Path
import platform
import shutil
import time
from contextlib import contextmanager
import numpy as np
from current.progress import track,stage
from refnet_f1.protocol import read,write,sha,fp,lock,inputs,locations,verify_parent,verify_file,guard_space

TAG='reference_f1_timing_v1'


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('Empty/nonfinite timings')
    return dict(n=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),p90_ms=float(np.quantile(a,.9)),min_ms=float(a.min()),max_ms=float(a.max()))


def select_images(names,count):
    return sorted(names,key=lambda n:(fp(n),n))[:count]


@contextmanager
def instrumentation(model):
    """Temporary process-local wrappers; no original source or parameters changed."""
    import torch
    import refnet.model as module
    records=[];restore=[]
    def wrap(obj,name,label):
        original=getattr(obj,name)
        def measured(*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();start.record()
            value=original(*args,**kwargs)
            end.record();records.append((label,start,end,(time.perf_counter()-wall)*1000))
            return value
        setattr(obj,name,measured);restore.append((obj,name,original))
    for obj,name,label in [(model,'feature_map','DINO_feature_map'),(model.projection,'forward','semantic_projection'),
        (model.pf,'extract_feat','PF_backbone'),(model.pf,'forward_transformer','PF_transformer_with_feedback'),
        (model.pf.bbox_head,'forward','PF_bbox_head'),(module,'region','ROI_spatial_context'),
        (model.refine,'forward','box_refinement_MLP'),(model.loc_quality,'forward','quality_MLP'),
        (model.relation,'forward','reference_retrieval_relation_rejection')]:wrap(obj,name,label)
    try:yield records
    finally:
        for obj,name,original in reversed(restore):setattr(obj,name,original)


def measure_case(root,cfg,scene,task,seed,args,verified,dataset):
    import torch
    import refnet.model as module
    from refnet.model import ReferenceDetector
    from jointdet.protocol import load_torch
    from cil.common import configure_torch
    from scenepilot.protocol import view
    from refnet_f1.runtime import load_references,candidate_values,decode
    spec,parent,annotation=inputs(root,scene,task,seed);old=locations(root,cfg,scene,task,seed)
    original_result=read(old/'RESULT.json');old_lock=read(old/'LOCK.json')
    if not original_result['complete'] or original_result['parent_checkpoint']!=parent['terminal']:raise RuntimeError('Require completed157 endpoint')
    if old_lock['config']!=cfg or old_lock['annotation']!=annotation:raise RuntimeError('157 inputs/config changed')
    for n,h in track(sorted(old_lock['sources'].items()),f'163 {scene}/{task}/s{seed} verify157 source',unit='files'):verify_file(root/n,h,verified)
    for n in ['CALIBRATION.json','REFERENCE_BANK.json']:verify_file(old/n,original_result['artifacts'][n],verified)
    verify_parent(root,parent,verified);cal=read(old/'CALIBRATION.json');c=read(root/'configs/reference_net.json')['model']
    folder=root/'reports/reference_f1_timing'/TAG/scene/task/f'seed_{seed}'
    names=select_images(spec['test_images'],args.images)
    env=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(args.device),device=args.device,tf32=False,precision='fp32',autocast=False,
        batch_size=1,threads=2)
    binding=lock(folder/'LOCK.json',dict(source={n:sha(root/n) for n in ['run_reference_f1_timing.py','src/reference_f1_timing.py']},
        original_result=sha(old/'RESULT.json'),annotation=annotation,environment=env,images=names,
        repeats=args.repeats,warmup=args.warmup,arms=args.arms,proposal_limit=c['proposal_limit'],detection_limit=c['detection_limit']))
    if (folder/'RESULT.json').exists():
        r=read(folder/'RESULT.json')
        if r['lock']!=binding['identity']:raise RuntimeError('Timing resume changed')
        verify_file(folder/'SAMPLES.json',r['samples_sha256']);return r
    for name in track(names,'163 verify timing images',unit='images'):
        verify_file(root/'assets/rwd/JPEGImages'/scene/name,dataset['files'][f'JPEGImages/{scene}/{name}']['sha256'],verified)
    configure_torch(seed,2)
    if shutil.disk_usage(root).free<512*2**20:raise RuntimeError('Timing logs require at least512MiB free disk space')
    torch.cuda.synchronize();start=time.perf_counter()
    with stage(f'163 {scene}/{task}/s{seed} load model'):
        model=ReferenceDetector(root,c,list(range(len(spec['classes'])))).to(args.device)
        state=load_torch(root/parent['terminal']['checkpoint']);model.load_adapted(state['model']);del state
        model.requires_grad_(False);model.eval()
    torch.cuda.synchronize();load_ms=(time.perf_counter()-start)*1000
    try:
        start=time.perf_counter();bank,_,audit=load_references(root,spec,parent,args.device,dataset,verified);torch.cuda.synchronize()
        bank_ms=(time.perf_counter()-start)*1000
        expected=json.loads(json.dumps(audit));expected['identity']=fp(audit)
        if expected!=read(old/'REFERENCE_BANK.json'):raise RuntimeError('157 reference audit differs')
        v=view(root,scene);real_pixels=module.pixels;samples=[]
        def pipeline(name,arm,wall_stages=None):
            output=model.forward_image(v,name,[],args.device)
            if wall_stages is not None:torch.cuda.synchronize();started=time.perf_counter()
            values=candidate_values(model,output,bank,cfg,c)
            if wall_stages is not None:
                torch.cuda.synchronize();wall_stages['top600_relation_gate_D2H_total']=(time.perf_counter()-started)*1000
                started=time.perf_counter()
            decoded=decode(values,arm,cal,c)
            if wall_stages is not None:wall_stages['CPU_identity_decode_NMS_top100']=(time.perf_counter()-started)*1000
            return decoded,output
        with torch.inference_mode():
            # Warmup covers the selected dynamic shapes; discarded and not saved as measurements.
            for arm in args.arms:
                for i in track(range(args.warmup),f'163 {scene}/{task}/s{seed}/{arm} warmup',unit='images'):
                    pipeline(names[i%len(names)],arm)
                torch.cuda.synchronize()
                for mode in ['online','end_to_end','stages']:
                    for name in track(names,f'163 {scene}/{task}/s{seed}/{arm} {mode}',unit='images'):
                        prepared=real_pixels(v,name,c,args.device) if mode!='end_to_end' else None
                        # Verify prepared-input substitution against the untouched native path outside timing.
                        if mode=='online':
                            native,_=pipeline(name,arm)
                            module.pixels=lambda *a,**k:prepared
                            try:cached,_=pipeline(name,arm)
                            finally:module.pixels=real_pixels
                            if native!=cached:raise RuntimeError('Prepared input path changes prediction; refuse timing substitution')
                            del native,cached,_
                        for repeat in range(args.repeats):
                            if mode!='end_to_end':module.pixels=lambda *a,**k:prepared
                            try:
                                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                                start=time.perf_counter()
                                if mode=='stages':
                                    wall_stages={}
                                    with instrumentation(model) as events:
                                        prediction,output=pipeline(name,arm,wall_stages)
                                    torch.cuda.synchronize()
                                    elapsed=(time.perf_counter()-start)*1000
                                    stage_rows={}
                                    for label,a,b,host_ms in events:
                                        row=stage_rows.setdefault(label,dict(calls=0,cuda_span_ms=0.,host_dispatch_ms=0.))
                                        row['calls']+=1;row['cuda_span_ms']+=a.elapsed_time(b);row['host_dispatch_ms']+=host_ms
                                    for label,ms in wall_stages.items():stage_rows[label]=dict(calls=1,sync_wall_ms=ms)
                                else:
                                    prediction,output=pipeline(name,arm);torch.cuda.synchronize()
                                    elapsed=(time.perf_counter()-start)*1000;stage_rows=None
                                sample=dict(arm=arm,mode=mode,image=name,repeat=repeat,wall_ms=elapsed,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                    pf_input_hw=list(output['meta']['batch_input_shape']),dino_input_hw=list(output['meta']['dino_shape']),
                                    output_detections=len(prediction[0]['boxes']),stages=stage_rows)
                                del output,prediction
                            finally:module.pixels=real_pixels
                            # Timers have ended before any file/log writes.
                            samples.append(sample)
                        del prepared
        write(folder/'SAMPLES.json',samples)
        summary={}
        for arm in args.arms:
            summary[arm]={mode:stats([r['wall_ms'] for r in samples if r['arm']==arm and r['mode']==mode]) for mode in ['online','end_to_end','stages']}
            stage_labels=sorted({k for r in samples if r['arm']==arm and r['stages'] for k,v in r['stages'].items() if 'cuda_span_ms' in v})
            summary[arm]['stage_cuda_spans']={k:stats([r['stages'].get(k,{}).get('cuda_span_ms',0.) for r in samples if r['arm']==arm and r['mode']=='stages']) for k in stage_labels}
            wall_labels=sorted({k for r in samples if r['arm']==arm and r['stages'] for k,v in r['stages'].items() if 'sync_wall_ms' in v})
            summary[arm]['stage_sync_wall']={k:stats([r['stages'][k]['sync_wall_ms'] for r in samples if r['arm']==arm and r['mode']=='stages']) for k in wall_labels}
        result=dict(complete=True,scene=scene,task=task,seed=seed,lock=binding['identity'],environment=env,
            selected_images=len(names),summary=summary,setup_model_load_ms=load_ms,setup_reference_verify_load_ms=bank_ms,
            samples_sha256=sha(folder/'SAMPLES.json'),trainable_parameters=0,
            model_parameters=sum(p.numel() for p in model.parameters()),
            notes='online: device-resident input through final CPU decode/NMS/list outputs, no instrumentation. end_to_end: native read/preprocess/H2D plus same pipeline, OS page cache uncontrolled. stages: separate instrumented pass, CUDA spans can include host-induced idle; not pure kernel time, do not sum them into total. Setup excludes weight hashing before loading.')
        write(folder/'RESULT.json',result);return result
    finally:
        module.pixels=real_pixels if 'real_pixels' in locals() else module.pixels
        del model;gc.collect();torch.cuda.empty_cache()


def export(root):
    import zipfile
    folder=root/'reports/reference_f1_timing'/TAG
    results=[read(p) for p in sorted(folder.glob('*/*/seed_*/RESULT.json'))]
    write(folder/'SUMMARY.json',dict(cases=len(results),results=results,protocol='Actual current GPU measurements; three timing passes; no4080 emulation or conversion'))
    lines=['# Current-GPU inference timing','','| Case | Arm | Online median ms | Online P90 ms | End-to-end median ms | End-to-end P90 ms |','|---|---|---:|---:|---:|---:|']
    for r in results:
        for arm,s in r['summary'].items():lines.append(f"| {r['scene']}/{r['task']}/s{r['seed']} | {arm} | {s['online']['median_ms']:.3f} | {s['online']['p90_ms']:.3f} | {s['end_to_end']['median_ms']:.3f} | {s['end_to_end']['p90_ms']:.3f} |")
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=folder/'REFERENCE_F1_TIMING.zip'
    with zipfile.ZipFile(dest.with_suffix('.zip.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
        files=sorted(folder.glob('*/*/seed_*/*.json'))+[folder/'SUMMARY.json',folder/'SUMMARY.md']
        for p in track(files,'163 package timing logs',unit='files'):z.write(p,p.relative_to(root).as_posix())
    with zipfile.ZipFile(dest.with_suffix('.zip.tmp')) as z:
        if z.testzip():raise RuntimeError('Timing ZIP CRC failed')
    dest.with_suffix('.zip.tmp').replace(dest)
    return dict(path=str(dest),cases=len(results),bytes=dest.stat().st_size,sha256=sha(dest))


def main(root,argv=None):
    p=argparse.ArgumentParser(description='163: actual current-GPU inference efficiency; no training')
    p.add_argument('command',choices=['run','export','stage-upload']);p.add_argument('--scenes',nargs='+');p.add_argument('--tasks',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--arms',nargs='+',choices=['F0','F1'],default=['F1']);p.add_argument('--images',type=int,default=30);p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmup',type=int,default=10);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(argv);root=Path(root);cfg=read(root/'configs/reference_f1.json')
    if a.command=='stage-upload':
        out=root/'reports/reference_f1_timing'/TAG/'UPLOAD/OpenDet_Current'
        names=['run_reference_f1_timing.py','src/reference_f1_timing.py','tests/test_reference_f1_timing.py','docs/REFERENCE_F1_TIMING.md','.run/163_Reference_F1_Timing.run.xml']
        for n in names:
            dest=out/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/n,dest)
        write(out/'TIMING_UPLOAD_MANIFEST.json',{n:sha(root/n) for n in names});result=dict(upload_directory=str(out))
    elif a.command=='export':result=export(root)
    else:
        import torch
        if not a.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Formal CUDA timing requires your actual GPU instance')
        if min(a.images,a.repeats,a.warmup)<1:p.error('images/repeats/warmup must be positive')
        scenes=a.scenes or cfg['scenes'];tasks=a.tasks or cfg['tasks'];seeds=a.seeds or cfg['seeds']
        for chosen,allowed in [(scenes,cfg['scenes']),(tasks,cfg['tasks']),(seeds,cfg['seeds'])]:
            if not set(chosen)<=set(allowed) or len(set(chosen))!=len(chosen):p.error('Choose unique configured scenes/tasks/seeds')
        torch.cuda.set_device(a.device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        dataset=read(root/'assets/rwd/DATASET_MANIFEST.json');verified={}
        for n,h in track(sorted(read(root/'MAINLINE_PARENT_INDEX.json')['base_weights'].items()),'163 verify PF/DINO assets',unit='files'):verify_file(root/n,h,verified)
        for s,t,z in track([(s,t,z) for s in scenes for t in tasks for z in seeds],'163 formal current-GPU timings',unit='cases'):measure_case(root,cfg,s,t,z,a,verified,dataset)
        result=export(root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
