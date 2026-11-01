from pathlib import Path
import argparse,sys,subprocess,time,json,zipfile,hashlib
from .protocol import read,write,sha,source,audit,task

def summary(root):
 cfg=read(root/'configs/b5_suite.json');rows=[];historical=read(root/'assets/b5/results.json')
 for row in historical['seeds']:
  for t in ['task1','task2']:rows.append(dict(method='B5_strict',scene='Aquatic',task=t,seed=row['seed'],status='historical_completed',metrics=row[t],source_sha256=sha(root/'assets/b5/results.json')))
 for p in (root/'reports/b5_m1_suite'/cfg['tag']/'B5').glob('*/RESULT.json'):
  v=read(p)
  for x in v['results']:rows.append(dict(method='B5_'+x['variant'],scene=v['scene'],**x))
 for p in (root/'reports/b5_m1_suite'/cfg['tag']/'M1').glob('*/*/seed_*/RESULT.json'):
  v=read(p);rows.append(dict(method='M1',scene=p.parents[2].name,task=p.parents[1].name,seed=int(p.parent.name.split('_')[1]),**{k:x for k,x in v.items() if k not in ['scene','task','seed']}))
 result=dict(rows=rows,scope='Official per-scene test results; historical Aquatic B5 retained; B5 variants separate; blocked and missing are not zero AP; do not macro-average only completed scenes as five-scene performance. Same total shots, different method calibration/training cost.');write(root/'reports/b5_m1_suite'/cfg['tag']/'SUMMARY.json',result);return result

def bundle(root):
 cfg=read(root/'configs/b5_suite.json');out=root/'reports/b5_m1_suite'/cfg['tag'];out.mkdir(parents=True,exist_ok=True);summary(root);files={}
 for directory in [out,root/'runs/b5_m1_suite'/cfg['tag']]:
  for p in directory.rglob('*'):
   if p.is_file() and p.suffix in ['.json','.jsonl','.log']:files[p.relative_to(root).as_posix()]=p
 for p in list((root/'src/b5suite').glob('*.py'))+[root/'configs/b5_suite.json',root/'B5_M1_SUITE_README.md']:files[p.relative_to(root).as_posix()]=p
 hashes={n:sha(p) for n,p in files.items()};dest=out/('b5_m1_suite_diagnostics_'+time.strftime('%Y%m%dT%H%M%S')+'.zip')
 with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
  for n,p in files.items():z.write(p,n)
  z.writestr('MANIFEST.json',json.dumps(dict(files=hashes,weights_included=False),indent=2))
 with zipfile.ZipFile(dest) as z:
  if z.testzip() is not None or any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in hashes.items()):raise RuntimeError('ZIP integrity failure')
 return dest

def main(root):
 root=Path(root);p=argparse.ArgumentParser(description='Original B5, explicitly separate adaptive variant, and formal fixed-budget M1; H paused')
 p.add_argument('command',choices=['audit','smoke','b5','m1','bundle','test']);p.add_argument('--device',default='cuda:0');p.add_argument('--seeds',nargs='+',type=int,choices=[42,43,44],default=[42]);p.add_argument('--scene',choices=['Aquatic','Aerial','Game','Medical','Surgical']);p.add_argument('--task',choices=['task1','task2']);p.add_argument('--seed',type=int);p.add_argument('--worker',action='store_true');a=p.parse_args();cfg=read(root/'configs/b5_suite.json')
 if a.command=='test':
  import unittest
  result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_b5_suite.py'))
  if not result.wasSuccessful():raise SystemExit(1)
  return
 if a.command=='audit':
  v=audit(root);summary(root);print('Audit passed; cases:',len(v['cases']));print('Raw strict-5NN warnings:',[(r['scene'],r['task'],r['strict_B5_raw_reference_warning']) for r in v['cases'] if r['strict_B5_raw_reference_warning']]);return
 if a.command=='bundle':print(bundle(root));return
 plan=read(root/'reports/b5_m1_suite'/cfg['tag']/'PLAN.json')
 if plan['source']!=source(root) or plan['config']!=cfg:raise RuntimeError('Run audit for exactly this source/config')
 if a.worker:
  if a.command in ['b5','smoke']:
   from .engine import run_scene
   run_scene(root,a.scene,a.seeds,a.device,a.command=='smoke')
  else:
   from .m1 import run
   run(root,a.scene,a.task,a.seed,a.device)
  return
 if a.command=='smoke':jobs=[('Aerial',None,None)]
 elif a.command=='b5':jobs=[(scene,None,None) for scene in ([a.scene] if a.scene else cfg['scenes'])]
 else:jobs=[(scene,t,seed) for scene in ([a.scene] if a.scene else cfg['m1_scenes']) for t in ([a.task] if a.task else ['task1','task2']) for seed in a.seeds]
 if a.command=='b5' and any(j[0]=='Aquatic' for j in jobs):raise RuntimeError('Aquatic strict B5 already complete; use archived results instead of retraining')
 if a.command in ['b5','m1']:
  smoke=read(root/'reports/b5_m1_suite'/cfg['tag']/'B5/Aerial/SMOKE.json')
  if not smoke.get('passed') or smoke['source']!=source(root):raise RuntimeError('Run new frozen-B5 smoke before full suite')
 start=time.monotonic()
 for i,(scene,t,seed) in enumerate(jobs,1):
  report=root/'reports/b5_m1_suite'/cfg['tag']/('B5' if a.command in ['b5','smoke'] else 'M1')/scene
  if t:report=report/t/f'seed_{seed}'
  report.mkdir(parents=True,exist_ok=True);status=report/'PROGRESS.json';log=report/('console_'+str(time.time_ns())+'.log');cmd=[sys.executable,'-B',str(root/'run_b5_m1_suite.py'),a.command,'--worker','--device',a.device,'--scene',scene,'--seeds']+list(map(str,a.seeds))
  if t:cmd+=['--task',t,'--seed',str(seed)]
  stamp=time.time()
  with log.open('w',encoding='utf8') as f:
   proc=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
   try:
    while proc.poll() is None:
     line=f'jobs {i}/{len(jobs)} {scene} {t or ""} {seed or ""} | elapsed {time.monotonic()-start:.0f}s'
     if status.exists() and status.stat().st_mtime>=stamp:
      try:
       v=read(status);line+=f' | {v["label"]} {v["done"]}/{v["total"]} stage ETA {v["eta"]:.0f}s'
      except (ValueError,OSError,KeyError):pass
     print('\r'+line.ljust(210),end='',flush=True);time.sleep(1)
   except BaseException:
    proc.terminate()
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    raise
  print()
  if proc.returncode:raise RuntimeError('Worker failed; read '+str(log))
 summary(root);print('Finished requested jobs; export diagnostics with78.')
