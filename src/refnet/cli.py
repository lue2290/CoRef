from pathlib import Path
import argparse,subprocess,sys,time,json,unittest,zipfile,hashlib
from datetime import datetime,timezone
from .protocol import read,write,audit,get_plan,paths,sha,fp,space


def tests(root):
    suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_reference_net.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    cfg=read(root/'configs/reference_net.json')
    from .protocol import source_files
    write(root/'reports/reference_net'/cfg['tag']/'TESTS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,source=fp(source_files(root))))
    if not result.wasSuccessful():raise SystemExit(1)


def summary(root,cfg):
    import numpy as np
    rows=[]
    for scene in cfg['scenes']:
        for task in ['task1','task2']:
            for seed in cfg['seeds']:
                _,report=paths(root,cfg,scene,task,seed)
                if (report/'RESULT.json').exists():rows.append(read(report/'RESULT.json'))
    aggregates=[]
    for task in ['task1','task2']:
        for scene in cfg['scenes']:
            selected=[r for r in rows if r['scene']==scene and r['task']==task]
            for point in ['terminal','selected']:
                metrics={}
                for key in ['K_AP50','U_AP50','PK_AP50','CK_AP50','unknown_background_fp_per_image']:
                    vals=[r['evaluations'][point].get(key) for r in selected]
                    vals=[v for v in vals if v is not None]
                    if vals:metrics[key]=dict(mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)) if len(vals)>1 else None,n=len(vals))
                aggregates.append(dict(scene=scene,task=task,checkpoint=point,complete=len(selected)==len(cfg['seeds']),metrics=metrics))
    write(root/'reports/reference_net'/cfg['tag']/'SUMMARY.json',dict(completed=len(rows),expected=30,rows=rows,aggregates=aggregates,selection='terminal primary; selected is training-held proxy best, never chosen using test AP'))


def bundle(root,cfg):
    summary(root,cfg)
    base=root/'reports/reference_net';out=base/('reference_net_diagnostics_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
    names=[]
    for tag in [cfg['tag'],cfg['tag']+'_smoke']:
        names += [p for p in (base/tag).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log','.txt']]
        names += [p for p in (root/'runs/reference_net'/tag).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl']]
    names += [root/'run_reference_net.py',root/'configs/reference_net.json',root/'REFERENCE_NET_README.md']+list((root/'src/refnet').glob('*.py'))
    names=sorted(set(names));manifest={p.relative_to(root).as_posix():sha(p) for p in names}
    space(root,cfg,sum(p.stat().st_size for p in names))
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in names:z.write(p,p.relative_to(root).as_posix())
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip():raise RuntimeError('Diagnostic archive CRC failure')
        for n,h in manifest.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Diagnostic archive member mismatch')
    print('Diagnostics: '+str(out))


def main(root):
    root=Path(root);p=argparse.ArgumentParser(description='Reference-conditioned open-world detector; independent version')
    p.add_argument('command',choices=['test','audit','smoke','train','bundle'])
    p.add_argument('--device',default='cuda:0');p.add_argument('--task',choices=['task1','task2'],default='task1')
    p.add_argument('--scene',choices=['Aquatic','Aerial','Game','Medical','Surgical'])
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.command=='test':tests(root);return
    if a.command=='audit':
        result=audit(root);print('Audit passed; 10 task splits locked, no training. Plan: '+result['identity']);return
    cfg,plan=get_plan(root)
    if a.command=='bundle':bundle(root,cfg);return
    smoke=a.command=='smoke';task='task2' if smoke else a.task
    scenes=[a.scene] if a.scene else cfg['scenes'];seeds=a.seeds or ([42] if smoke else cfg['seeds'])
    if any(seed not in cfg['seeds'] for seed in seeds):raise RuntimeError('Seed outside locked plan')
    if a.worker:
        if not a.scene or len(seeds)!=1:raise RuntimeError('Worker needs one scene/seed')
        from .engine import run
        run(root,cfg,plan,a.scene,task,seeds[0],a.device,smoke);return
    t=read(root/'reports/reference_net'/cfg['tag']/'TESTS.json')
    if not t['passed'] or t['source']!=plan['source']:raise RuntimeError('Run current CPU tests first')
    if not smoke:
        for scene in cfg['scenes']:
            _,report=paths(root,cfg,scene,'task2',42,True)
            r=read(report/'RESULT.json')
            if not r['complete'] or r['source']!=plan['source']:raise RuntimeError('All five-scene smoke checks must pass first')
    jobs=[(scene,seed) for scene in scenes for seed in seeds];start=time.monotonic()
    try:
        for i,(scene,seed) in enumerate(jobs):
            _,report=paths(root,cfg,scene,task,seed,smoke);report.mkdir(parents=True,exist_ok=True)
            log=report/('console_'+str(time.time_ns())+'.log')
            args=[sys.executable,str(root/'run_reference_net.py'),a.command,'--worker','--scene',scene,'--task',task,'--seeds',str(seed),'--device',a.device]
            with log.open('w',encoding='utf8') as f:
                proc=subprocess.Popen(args,cwd=root,stdout=f,stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        message='initializing'
                        try:
                            v=read(report/'PROGRESS.json');message=f"{v['label']} {v['done']}/{v['total']} stage ETA {v['eta']:.0f}s"
                        except (OSError,ValueError):pass
                        print(f'\rjobs {i+1}/{len(jobs)} | {scene}/{task}/s{seed} | elapsed {time.monotonic()-start:.0f}s | {message}'.ljust(190),end='',flush=True)
                        time.sleep(1)
                except KeyboardInterrupt:
                    proc.terminate();proc.wait();raise
            print()
            if proc.returncode:raise RuntimeError('Worker failed. Read '+str(log))
            summary(root,cfg)
    finally:
        summary(root,cfg)
    print('Completed. Run 86 to export diagnostics (no weights).')
