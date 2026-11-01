from pathlib import Path
import argparse,subprocess,sys,time,json,zipfile,hashlib
from .protocol import read,write,source,audit,paths,space,sha

def bundle(root,cfg):
 root=Path(root);dest=root/'reports/scene_pilot'/cfg['tag'];dest.mkdir(parents=True,exist_ok=True);files={}
 for tag in [cfg['tag'],cfg['tag']+'_smoke']:
  for base in ['reports','runs']:
   for p in (root/base/'scene_pilot'/tag).rglob('*'):
    if p.is_file() and p.suffix in ['.json','.jsonl','.log']:files[p.relative_to(root).as_posix()]=p
 for p in list((root/'src/scenepilot').glob('*.py'))+[root/'run_scene_pilot.py',root/'configs/scene_pilot.json',root/'SCENE_PILOT_README.md']:files[p.relative_to(root).as_posix()]=p
 hashes={n:sha(p) for n,p in files.items()};path=dest/('scene_pilot_diagnostics_'+time.strftime('%Y%m%dT%H%M%S')+'.zip')
 with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
  for n,p in files.items():z.write(p,n)
  z.writestr('MANIFEST.json',json.dumps(dict(files=hashes,weights_included=False,candidate_feature_caches_included=False),indent=2))
 with zipfile.ZipFile(path) as z:
  if z.testzip() is not None or any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in hashes.items()):raise RuntimeError('Archive validation failed')
 return path

def main(root):
 root=Path(root);p=argparse.ArgumentParser(description='Four-scene actual P0/S/M1/H pilot; no official-test evaluation')
 p.add_argument('command',choices=['audit','smoke','pilot','bundle','test']);p.add_argument('--device',default='cuda:0');p.add_argument('--scene',choices=['Aquatic','Aerial','Game','Surgical']);p.add_argument('--worker',action='store_true');a=p.parse_args();cfg=read(root/'configs/scene_pilot.json')
 if a.command=='test':
  import unittest
  result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_scene_pilot.py'))
  if not result.wasSuccessful():raise SystemExit(1)
  return
 if a.command=='audit':print(json.dumps(audit(root,cfg),indent=2));return
 if a.command=='bundle':print('Diagnostics:',bundle(root,cfg));return
 plan=read(root/'reports/scene_pilot'/cfg['tag']/'PLAN.json')
 if plan['source_identity']!=source(root) or plan['config']!=cfg:raise RuntimeError('Run59 audit after source/config changes')
 if a.worker:
  if a.scene is None:raise RuntimeError('Worker scene required')
  from .engine import run_scene
  print(json.dumps(run_scene(root,cfg,a.scene,a.device,a.command=='smoke'),indent=2));return
 scenes=[a.scene] if a.scene else cfg['scenes'][:1] if a.command=='smoke' else cfg['scenes'];start=time.monotonic()
 for i,scene in enumerate(scenes,1):
  if a.command=='pilot':
   smoke=read(root/'reports/scene_pilot'/(cfg['tag']+'_smoke')/cfg['scenes'][0]/'RESULT.json')
   if smoke.get('source_identity')!=source(root) or not smoke.get('full_recipe_smoke_passed'):raise RuntimeError('Run60 full recipe smoke before pilot')
  childcfg={**cfg,'tag':cfg['tag']+('_smoke' if a.command=='smoke' else '')};_,report,_=paths(root,childcfg,scene)
  completed=report/'RESULT.json'
  if completed.exists():
   old=read(completed)
   if old.get('complete') and old.get('source_identity')==source(root):
    print(scene,'already complete; no training repeated');continue
  report.mkdir(parents=True,exist_ok=True);log=report/('console_'+str(time.time_ns())+'.log');status=report/'PROGRESS.json'
  # Preserve an old progress record while avoiding a stale ETA in a new attempt.
  stamp=time.time()
  with log.open('w',encoding='utf8') as stream:
   proc=subprocess.Popen([sys.executable,'-B',str(root/'run_scene_pilot.py'),a.command,'--worker','--scene',scene,'--device',a.device],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
   try:
    while proc.poll() is None:
     line=f'scenes {i}/{len(scenes)} | {scene} | total elapsed {time.monotonic()-start:.0f}s'
     if status.exists() and status.stat().st_mtime>=stamp:
      try:
       v=read(status);fraction=v['done']/max(1,v['total']);line='['+'#'*int(20*fraction)+'-'*(20-int(20*fraction))+'] '+line+f' | {v["label"]} {v["done"]}/{v["total"]} stage ETA {v["eta"]:.0f}s'
      except (ValueError,OSError,KeyError):pass
     print('\r'+line.ljust(200),end='',flush=True);time.sleep(1)
   except BaseException:
    proc.terminate()
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    raise
  print()
  if proc.returncode:raise RuntimeError('Worker failed; log: '+str(log))
 rows=[]
 for scene in cfg['scenes']:
  path=root/'reports/scene_pilot'/cfg['tag']/scene/'RESULT.json'
  if path.exists():rows.append(read(path))
 write(root/'reports/scene_pilot'/cfg['tag']/'SUMMARY.json',dict(cases=rows,complete=len(rows)==len(cfg['scenes']),scope='Internal development, not five-scene formal results; H skips reported',source_identity=source(root)))
 print('Diagnostics:',bundle(root,cfg))
