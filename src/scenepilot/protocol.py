from pathlib import Path
import hashlib,json,os,shutil,time
from fivescene.data import read,write,fp

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def source(root):
 root=Path(root);files=[root/'run_scene_pilot.py',root/'configs/scene_pilot.json',root/'configs/five_scene.json']+sorted((root/'src/scenepilot').glob('*.py'))
 for folder in ['src/fivescene','src/identitylab','src/identityepisode','src/hrepeat','versions/joint_detector/jointdet','versions/joint_network_merge/jnet']:
  files+=sorted((root/folder).glob('*.py'))
 files += [root/'vendor/pfrpn/configs/pf-rpn/pf-rpn_coco-imagenet.py',root/'assets/models/dinov3_large/config.json',root/'src/cil/common.py',root/'src/cil/model.py',root/'src/cil/evaluate.py',root/'vendor/fomo_open_world_eval.py']
 return fp({x.relative_to(root).as_posix():sha(x) for x in files})

def space(root,cfg,extra=0):
 free=shutil.disk_usage(root).free
 if free<cfg['min_free_gib']*2**30+extra:raise RuntimeError('Data disk reserve insufficient; preserve logs and current checkpoint')
 total=0
 for base in ['runs','prepared']:
  d=Path(root)/base/'scene_pilot'/cfg['tag']
  total+=sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
 if total+extra>cfg['max_output_gib']*2**30:raise RuntimeError('Pilot output budget reached')
 return dict(free_gib=free/2**30,output_gib=total/2**30)

def paths(root,cfg,scene):
 return tuple(Path(root)/x/'scene_pilot'/cfg['tag']/scene for x in ['runs','reports','prepared'])

def progress(report,done,total,label,start):
 elapsed=time.monotonic()-start;write(Path(report)/'PROGRESS.json',dict(done=done,total=total,label=label,elapsed=elapsed,eta=elapsed/max(done,1)*(total-done)))

def lock(path,value):
 value=json.loads(json.dumps(value));value['identity']=fp(value)
 if Path(path).exists():
  if read(path)!=value:raise RuntimeError('Run source/config changed; new tag required')
 else:write(path,value)
 return value

def case(root,cfg,scene,smoke=False):
 from fivescene.data import source as split_source
 base=read(Path(root)/'configs/five_scene.json');a=read(Path(root)/'reports/five_scene'/base['tag']/'AUDIT.json')
 if a['source_identity']!=split_source(root):raise RuntimeError('51 source differs from current split implementation')
 record=next(x for x in a['cases'] if x['scene']==scene and x['admitted']);s=read(Path(root)/record['split'])
 if fp({k:v for k,v in s.items() if k!='identity'})!=s['identity']:raise RuntimeError('Split fingerprint differs')
 if smoke:
  s={**s,'fit_images':s['fit_images'][:2],'cal_images':s['cal_images'][:2],'dev_images':s['dev_images'][:2]}
  for role in ['fit','cal','dev']:s[role+'_objects']=[o for o in s[role+'_objects'] if o['image'] in s[role+'_images']]
 s={**s,'val_images':s['dev_images'],'groups':{o['image']:o['group'] for role in ['fit','cal','dev'] for o in s[role+'_objects']},'formal':False}
 return s

def audit(root,cfg):
 from fivescene.smoke import view_root
 # Dataset bytes were verified by50; metadata and base weights are checked again here.
 manifest=read(Path(root)/'PROJECT_MANIFEST.json')['files']
 for n in ['assets/models/dinov3_large/model.safetensors','assets/models/pfrpn/pf_rpn_swinb_5p_coco_imagenet.pth']:
  if sha(Path(root)/n)!=manifest[n]['sha256']:raise RuntimeError('Base weight changed: '+n)
 clean=read(Path(root)/'prepared/rwd_task1/clean_shots_v1/MANIFEST.json')
 for n,row in clean['files'].items():
  if sha(Path(root)/n)!=row['sha256']:raise RuntimeError('Clean metadata changed')
 dataset=read(Path(root)/'assets/rwd/DATASET_MANIFEST.json')
 rows=[]
 for scene in cfg['scenes']:
  s=case(root,cfg,scene)
  if s['clean_identity']!=clean['identity'] or s['dataset_identity']!=dataset['identity']:raise RuntimeError('Split belongs to different data')
  for role in ['fit','cal','val']:
   for name in s[role+'_images']:
    for kind,filename in [('JPEGImages',name),('Annotations',Path(name).stem+'.xml')]:
     rel=f'{kind}/{scene}/{filename}'
     if sha(Path(root)/'assets/rwd'/rel)!=dataset['files'][rel]['sha256']:raise RuntimeError('Selected input changed: '+rel)
  rows.append(dict(scene=scene,held=s['held'],split_identity=s['identity'],fit=len(s['fit_images']),cal=len(s['cal_images']),dev=len(s['val_images'])))
 out=Path(root)/'reports/scene_pilot'/cfg['tag'];value=lock(out/'PLAN.json',dict(config=cfg,source_identity=source(root),cases=rows,clean_identity=clean['identity'],formal_test_access=False,medical_included=False))
 write(out/'AUDIT.json',dict(passed=True,plan_identity=value['identity'],training_updates=0,space=space(root,cfg)));return value

def gpu(device,cfg):
 import torch
 if not device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Remote CUDA GPU required')
 if torch.cuda.get_device_properties(device).total_memory<cfg['min_gpu_gib']*2**30:raise RuntimeError('At least24GiB GPU required; local full model execution blocked')
 limit=Path('/sys/fs/cgroup/memory.max')
 if limit.exists() and limit.read_text().strip().isdigit() and int(limit.read_text())<8*2**30:raise RuntimeError('At least8GiB CPU cgroup RAM required')
 torch.cuda.set_device(device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats(device)
 return torch.device(device)

def view(root,scene):
 from fivescene.smoke import view_root
 return view_root(root,scene)
