from pathlib import Path
import argparse,subprocess,sys,time,zipfile,json
from .data import read,write,sha,source,audit

def bundle(root,cfg):
 root=Path(root);dest=root/'reports/five_scene'/cfg['tag'];dest.mkdir(parents=True,exist_ok=True);files=[p for p in dest.rglob('*') if p.is_file() and p.suffix in ('.json','.log')]+list((root/'prepared/five_scene'/cfg['tag']).rglob('*.json'))+[root/'configs/five_scene.json']+list((root/'src/fivescene').glob('*.py'))
 hashes={p.relative_to(root).as_posix():sha(p) for p in files};path=dest/('five_scene_diagnostics_'+time.strftime('%Y%m%dT%H%M%S')+'.zip')
 with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.relative_to(root).as_posix())
  z.writestr('MANIFEST.json',json.dumps(dict(files=hashes,weights_included=False),indent=2))
 with zipfile.ZipFile(path) as z:
  import hashlib
  assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
 return str(path)

def main(root):
 root=Path(root);p=argparse.ArgumentParser(description='Five-scene admission and real architecture smoke. No formal parent or H training command is exposed.')
 p.add_argument('command',choices=['audit','smoke','bundle','test']);p.add_argument('--device',default='cuda:0');p.add_argument('--scene');p.add_argument('--worker',action='store_true');a=p.parse_args();cfg=read(root/'configs/five_scene.json')
 if a.command=='test':
  import unittest
  result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_five_scene.py'))
  if not result.wasSuccessful():raise SystemExit(1)
  return
 if a.command=='audit':
  v=audit(root);print('Admitted',v['admitted'],'/',len(v['cases']));print('Diagnostics:',bundle(root,cfg));return
 if a.command=='bundle':print(bundle(root,cfg));return
 if a.worker:
  if a.scene not in cfg['scene_order']:raise ValueError('Unknown scene')
  from .smoke import smoke_one
  print(json.dumps(smoke_one(root,a.scene,a.device),indent=2));return
 auditpath=root/'reports/five_scene'/cfg['tag']/'AUDIT.json'
 if not auditpath.exists():raise RuntimeError('Run51 CPU audit first')
 rows=[];start=time.monotonic()
 for i,scene in enumerate(cfg['scene_order'],1):
  out=root/'reports/five_scene'/cfg['tag']/'smoke'/scene;out.mkdir(parents=True,exist_ok=True);log=out/'console.log'
  with log.open('a',encoding='utf8') as stream:
   process=subprocess.Popen([sys.executable,'-B',str(root/'run_five_scene.py'),'smoke','--worker','--scene',scene,'--device',a.device],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
   try:
    while process.poll() is None:
     elapsed=time.monotonic()-start;eta=f'{elapsed/(i-1)*(6-i):.0f}s' if i>1 else 'estimating';bar='#'*(4*(i-1))+'-'*(20-4*(i-1));print(f'\r[{bar}] jobs {i}/5 | {scene} architecture smoke | elapsed {elapsed:.0f}s | ETA {eta}'.ljust(170),end='',flush=True);time.sleep(1)
   except BaseException:
    process.terminate();process.wait(timeout=30);raise
  print()
  if process.returncode:raise RuntimeError('Worker failed; read '+str(log))
  rows.append(read(out/'RESULT.json'))
 write(root/'reports/five_scene'/cfg['tag']/'SMOKE_SUMMARY.json',dict(complete=True,cases=rows,source_identity=source(root),new_checkpoints=0));print('Diagnostics:',bundle(root,cfg))
