from pathlib import Path
import json,hashlib,shutil,time
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def source(root):
 root=Path(root);files=[root/'configs/b5_suite.json',root/'configs/scene_pilot.json',root/'vendor/pfrpn/configs/pf-rpn/pf-rpn_coco-imagenet.py',root/'assets/models/dinov3_large/config.json',root/'run_b5_m1_suite.py']+sorted((root/'src/b5suite').rglob('*.py'))
 for folder in ['src/scenepilot','versions/joint_detector/jointdet','versions/joint_network_merge/jnet']:files+=sorted((root/folder).glob('*.py'))
 files+=[root/'src/cil/model.py',root/'vendor/fomo_open_world_eval.py']
 return fp({p.relative_to(root).as_posix():sha(p) for p in files})
def space(root,cfg):
 if shutil.disk_usage(root).free<cfg['min_free_gib']*2**30:raise RuntimeError('Less than8GiB free data disk; no automatic old-file cleanup')
 used=sum(p.stat().st_size for base in ['runs','prepared'] for p in (Path(root)/base/'b5_m1_suite'/cfg['tag']).rglob('*') if p.is_file())
 if used>cfg.get('output_budget_gib',8)*2**30:raise RuntimeError('New experiment exceeds8GiB retained-output budget; pause and export models')
def progress(p,done,total,label,start):write(Path(p)/'PROGRESS.json',dict(done=done,total=total,label=label,eta=(time.monotonic()-start)/max(done,1)*(total-done)))
def lock(path,v):
 v=json.loads(json.dumps(v));v['identity']=fp(v)
 if Path(path).exists():
  if read(path)!=v:raise RuntimeError('Locked inputs/config/source changed; use a new tag')
 else:write(path,v)
 return v

def task(root,scene,name):
 root=Path(root);cfg=read(root/'configs/b5_suite.json');m=read(root/'prepared/b5_suite'/cfg['tag']/(scene+'.json'));classes=m['PK'] if name=='task1' else m['PK']+m['CK'];meta=root/'assets/rwd/ImageSets'/scene;train=(meta/'train.txt').read_text().splitlines();test=(meta/'test.txt').read_text().splitlines();dims=read(meta/'groups.json')['dimensions'];objects=[]
 for cid,c in enumerate(classes):
  for i,(image,b) in enumerate(m['shots'][c]):
   if image not in train or image in test:raise RuntimeError('Training shot outside official train')
   w,h=dims[image];objects.append(dict(id=f'{cid}_{i}',image=image,class_id=cid,group=image,box=[x*y for x,y in zip(b,[w,h,w,h])]))
 images=[n for n in train if any(o['image']==n for o in objects)];labels=[o['class_id'] for o in objects];names=[o['image'] for o in objects];folds={}
 split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
 for fold,(_,va) in enumerate(split.split(np.zeros(len(labels)),labels,names)):
  for i in va:folds[names[i]]=fold
 s=dict(scene=scene,task=name,classes=classes,PK=m['PK'],CK=m['CK'] if name=='task2' else [],objects=objects,train_images=images,test_images=test,image_folds=folds,counts={c:sum(o['class_id']==cid for o in objects) for cid,c in enumerate(classes)})
 s['identity']=fp(s);return s

def m1_split(s):
 import itertools
 chosen=None
 for size in range(1,5):
  for folds in itertools.combinations(range(5),size):
   fit=[o for o in s['objects'] if s['image_folds'][o['image']] not in folds];cal=[o for o in s['objects'] if s['image_folds'][o['image']] in folds];valid=True
   for cid in range(len(s['classes'])):
    f=[o for o in fit if o['class_id']==cid];a=[o for o in cal if o['class_id']==cid];counts={n:sum(o['image']==n for o in f) for n in {o['image'] for o in f}}
    if len(a)<5 or len({o['image'] for o in a})<2 or len(f)-max(counts.values(),default=0)<5:valid=False;break
   if valid:chosen=(fit,cal,folds);break
  if chosen:break
 if chosen is None:raise ValueError('No metadata-only fit/cal partition meets original five references, five calibration objects and two calibration images per class')
 fit,cal,folds=chosen
 v=dict(scene=s['scene'],classes=s['classes'],active=list(range(len(s['classes']))),held=None,fit_objects=fit,cal_objects=cal,fit_images=[n for n in s['train_images'] if s['image_folds'][n] not in folds],cal_images=[n for n in s['train_images'] if s['image_folds'][n] in folds],val_images=[],groups={n:n for n in s['train_images']},formal=True,calibration_folds=list(folds))
 v['identity']=fp(v);return v

def audit(root):
 root=Path(root);cfg=read(root/'configs/b5_suite.json');space(root,cfg);rows=[]
 base=read(root/'PROJECT_MANIFEST.json')['files']
 for n in ['assets/models/dinov3_large/model.safetensors','assets/models/pfrpn/pf_rpn_swinb_5p_coco_imagenet.pth']:
  if sha(root/n)!=base[n]['sha256']:raise RuntimeError('Base asset differs: '+n)
 dataset=read(root/'assets/rwd/DATASET_MANIFEST.json')
 for scene in cfg['classes']:
  for name in ['task1','task2']:
   s=task(root,scene,name)
   testsha={dataset['files'][f'JPEGImages/{scene}/{n}']['sha256'] for n in s['test_images']}
   for image in s['train_images']:
    if dataset['files'][f'JPEGImages/{scene}/{image}']['sha256'] in testsha:raise RuntimeError('Selected training image duplicates official test image')
   small={c:n for c,n in s['counts'].items() if n<6};rows.append(dict(scene=scene,task=name,shots=s['counts'],strict_B5_raw_reference_warning=small,split_identity=s['identity'],M1_fit_cal_policy='first metadata-admissible predefined fold combination; no test model selection'))
 v=lock(root/'reports/b5_m1_suite'/cfg['tag']/'PLAN.json',dict(config=cfg,source=source(root),metadata={sc:sha(root/'prepared/b5_suite'/cfg['tag']/(sc+'.json')) for sc in cfg['classes']},cases=rows,scope='Original B5 adapted only in category count and paths; M1 separate calibration policy, fixed endpoint, no H; same formal test images and annotation budget.'))
 return v
