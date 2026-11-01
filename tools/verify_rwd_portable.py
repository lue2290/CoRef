"""Verify uploaded portable RWD data. CPU only; no training or label-based selection."""
from pathlib import Path
import argparse,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify_clean(root,dataset_identity,required=False):
 root=Path(root).resolve();base=root/'prepared/rwd_task1/clean_shots_v1';path=base/'MANIFEST.json'
 if not path.is_file():
  if required:raise RuntimeError('Missing clean training metadata: '+str(path))
  return None
 m=json.loads(path.read_text(encoding='utf8'));want=m.pop('identity')
 if hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=want:raise RuntimeError('Clean manifest identity differs')
 if m['dataset_identity']!=dataset_identity:raise RuntimeError('Clean shots reference a different dataset')
 for name,row in m['files'].items():
  p=(root/name).resolve()
  if base.resolve() not in p.parents:raise RuntimeError('Clean metadata path escapes root')
  if not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise RuntimeError('Clean metadata upload differs: '+name)
 return dict(passed=True,identity=want,files=len(m['files']),scenes=m['scenes'])

def verify(root=ROOT,decode=False):
 root=Path(root).resolve();base=root/'assets/rwd';m=json.loads((base/'DATASET_MANIFEST.json').read_text(encoding='utf8'));identity=m.pop('identity')
 actual=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if actual!=identity:raise RuntimeError('Dataset manifest identity differs')
 started=time.monotonic();bad=[];size=0;decoded=0
 if decode:from PIL import Image
 for i,(name,row) in enumerate(m['files'].items(),1):
  p=base/name;resolved=p.resolve()
  if base.resolve() not in resolved.parents:raise RuntimeError('Dataset path escapes root: '+name)
  if not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:bad.append(name);continue
  size+=row['bytes']
  if decode and p.suffix.lower() in ('.jpg','.png','.jpeg'):
   try:
    with Image.open(p) as im:im.load()
    decoded+=1
   except Exception as e:bad.append(name+': '+str(e))
  if i%100==0 or i==len(m['files']):
   elapsed=time.monotonic()-started;eta=elapsed/max(1,i)*(len(m['files'])-i);print(f'\rDataset verify {i}/{len(m["files"])} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s'.ljust(130),end='',flush=True)
 clean=verify_clean(root,identity,required=m.get("version")=="rwd_portable_task1_visible_roi_v2")
 print();result=dict(clean_training_metadata=clean,passed=not bad,identity=identity,files=len(m['files']),logical_gib=size/2**30,decoded_images=decoded,bad=bad,training_ready=False,training_updates=0,scope='Uploaded file bytes and SHA verification. Training is not started. Internal group-aware development splits and per-scene parent models remain separate steps.')
 out=root/'reports/h_five_scene/upload_verify';out.mkdir(parents=True,exist_ok=True);p=out/('RESULT_'+time.strftime('%Y%m%dT%H%M%S')+'.json');p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
 if bad:raise RuntimeError('Dataset upload check failed; details: '+str(p))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--decode-images',action='store_true');args=p.parse_args();verify(decode=args.decode_images)
