from pathlib import Path
import json,hashlib,collections
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def source(root):
 r=Path(root);files=[r/'run_five_scene.py',r/'configs/five_scene.json']+list((r/'src/fivescene').glob('*.py'))
 files += [r/n for n in ['versions/joint_network_merge/jnet/model.py','versions/joint_detector/jointdet/model.py','versions/joint_detector/jointdet/geometry.py','versions/joint_detector/jointdet/losses.py','versions/joint_detector/jointdet/engine.py','src/cil/common.py','src/cil/model.py']]
 return fp({p.relative_to(r).as_posix():sha(p) for p in files})
def counts(objects,active):
 return {str(c):dict(objects=sum(o['class_id']==c for o in objects),groups=len({o['group'] for o in objects if o['class_id']==c})) for c in active}
def solve_partition(matrix,objects,groups,held,cfg):
 from scipy.optimize import milp,LinearConstraint,Bounds
 from scipy.sparse import lil_matrix
 n=len(groups);rows=[];lower=[];upper=[]
 def add(values,lo,hi=float('inf')):rows.append(values);lower.append(lo);upper.append(hi)
 for i in range(n):add({i*3+j:1 for j in range(3)},1,1)
 nf=max(1,round(n*cfg['fit_fraction']));nc=max(1,round(n*cfg['cal_fraction']))
 add({i*3:1 for i in range(n)},nf,nf);add({i*3+1:1 for i in range(n)},nc,nc)
 for col in range(matrix.shape[1]):
  cnt=matrix[:,col]
  for role,key,minimum in [(0,'min_fit_groups',0),(1,'min_cal_groups',cfg['min_cal_objects']),(2,'min_dev_groups',cfg['min_dev_objects'])]:
   add({i*3+role:1 for i in range(n) if cnt[i]>0},cfg[key])
   if minimum:add({i*3+role:float(cnt[i]) for i in range(n) if cnt[i]},minimum)
  for excluded in range(n):
   if cnt[excluded]>0:add({i*3:float(cnt[i]) for i in range(n) if i!=excluded and cnt[i]},cfg['k'])
 heldcounts={g:sum(o['group']==g and o['class_id']==held for o in objects) for g in groups}
 add({i*3+2:float(heldcounts[g]) for i,g in enumerate(groups) if heldcounts[g]},cfg['min_dev_objects'])
 A=lil_matrix((len(rows),n*3),dtype=float)
 for i,row in enumerate(rows):
  for j,value in row.items():A[i,j]=value
 rng=np.random.default_rng(cfg['split_seed']);result=milp(rng.uniform(0,1,n*3),integrality=np.ones(n*3),bounds=Bounds(0,1),constraints=LinearConstraint(A.tocsr(),lower,upper),options={'time_limit':cfg.get('solver_seconds',20),'mip_rel_gap':.05})
 if result.x is None:return None
 assignment=result.x.reshape(n,3).argmax(1);parts=[np.flatnonzero(assignment==j) for j in range(3)]
 f,a,v=[matrix[idx] for idx in parts]
 if len(parts[0])!=nf or len(parts[1])!=nc or any(not len(x) for x in parts):return None
 if np.any((f>0).sum(0)<cfg['min_fit_groups']) or np.any((a>0).sum(0)<cfg['min_cal_groups']) or np.any((v>0).sum(0)<cfg['min_dev_groups']):return None
 if np.any(f.sum(0)-f.max(0)<cfg['k']) or np.any(a.sum(0)<cfg['min_cal_objects']) or np.any(v.sum(0)<cfg['min_dev_objects']):return None
 if sum(heldcounts[groups[i]] for i in parts[2])<cfg['min_dev_objects']:return None
 return (0.,*[{groups[i] for i in idx} for idx in parts])

def choose_split(objects,active,held,cfg):
 groups=sorted({o['group'] for o in objects});matrix=np.array([[sum(o['group']==g and o['class_id']==c for o in objects) for c in active] for g in groups]);required=cfg['min_fit_groups']+cfg['min_cal_groups']+cfg['min_dev_groups'];support=(matrix>0).sum(0);bad=[dict(class_id=c,available_groups=int(n),required_groups=required) for c,n in zip(active,support) if n<required]
 if bad:return dict(admitted=False,reason='insufficient_group_support',coverage=bad)
 n=len(groups);nf=max(1,round(n*cfg['fit_fraction']));nc=max(1,round(n*cfg['cal_fraction']));rng=np.random.default_rng(cfg['split_seed']);best=None
 for _ in range(cfg['search_trials']):
  order=rng.permutation(n);fi=order[:nf];ca=order[nf:nf+nc];dv=order[nf+nc:];f=matrix[fi];a=matrix[ca];v=matrix[dv]
  if not len(dv):continue
  if np.any((f>0).sum(0)<cfg['min_fit_groups']) or np.any((a>0).sum(0)<cfg['min_cal_groups']) or np.any((v>0).sum(0)<cfg['min_dev_groups']):continue
  if np.any(f.sum(0)-f.max(0)<cfg['k']) or np.any(a.sum(0)<cfg['min_cal_objects']) or np.any(v.sum(0)<cfg['min_dev_objects']):continue
  devgroups={groups[i] for i in dv}
  if sum(o['class_id']==held and o['group'] in devgroups for o in objects)<cfg['min_dev_objects']:continue
  score=float(np.abs(f.sum(0)/matrix.sum(0)-cfg['fit_fraction']).sum()+np.abs(a.sum(0)/matrix.sum(0)-cfg['cal_fraction']).sum())
  if best is None or score<best[0]:best=(score,{groups[i] for i in fi},{groups[i] for i in ca},devgroups)
 if best is None:best=solve_partition(matrix,objects,groups,held,cfg)
 if best is None:return dict(admitted=False,reason='search_solver_did_not_find_legal_partition',coverage=counts(objects,active))
 _,fit,cal,dev=best
 result=dict(admitted=True,fit_objects=[o for o in objects if o['class_id'] in active and o['group'] in fit],cal_objects=[o for o in objects if o['class_id'] in active and o['group'] in cal],dev_objects=[o for o in objects if o['group'] in dev])
 for role in ['fit','cal','dev']:
  result[role+'_images']=sorted({o['image'] for o in result[role+'_objects']});result[role+'_groups']=sorted({o['group'] for o in result[role+'_objects']})
 for a,b in [('fit','cal'),('fit','dev'),('cal','dev')]:assert not set(result[a+'_groups'])&set(result[b+'_groups'])
 assert all(o['class_id'] in active for role in ['fit','cal'] for o in result[role+'_objects'])
 result['counts']={role:counts(result[role+'_objects'],active+[held] if role=='dev' else active) for role in ['fit','cal','dev']};return result

def audit(root):
 root=Path(root);cfg=read(root/'configs/five_scene.json');dataset=read(root/'assets/rwd/DATASET_MANIFEST.json');clean=read(root/'prepared/rwd_task1/clean_shots_v1/MANIFEST.json');assert clean['dataset_identity']==dataset['identity']
 # This small metadata check does not repeat the already-completed 4GiB upload scan.
 for n,row in clean['files'].items():assert sha(root/n)==row['sha256'],n
 for n,row in dataset['files'].items():
  if n.startswith('ImageSets/'):assert sha(root/'assets/rwd'/n)==row['sha256'],n
 out=root/'reports/five_scene'/cfg['tag'];rows=[];sid=source(root)
 for scene in cfg['scene_order']:
  meta=root/'assets/rwd/ImageSets'/scene;known=(meta/'known_classnames.txt').read_text().splitlines();shots=read(root/'prepared/rwd_task1/clean_shots_v1'/scene/'task1_shots_100.json');group=read(meta/'groups.json');train=set((meta/'train.txt').read_text().splitlines());test=set((meta/'test.txt').read_text().splitlines());testsha={dataset['files'][f'JPEGImages/{scene}/{n}']['sha256'] for n in test};objects=[]
  for cid,name in enumerate(known):
   for j,(image,b) in enumerate(shots[name]):
    assert image in train and image not in test and dataset['files'][f'JPEGImages/{scene}/{image}']['sha256'] not in testsha
    w,h=group['dimensions'][image];objects.append(dict(id=f'{name}_{j:03d}',image=image,class_id=cid,group=group['groups'][image],box=[v*s for v,s in zip(b,[w,h,w,h])]))
  folds=['fish','jellyfish'] if scene=='Aquatic' else known[:2]
  for heldname in folds:
   held=known.index(heldname);active=[i for i in range(len(known)) if i!=held];path=root/'prepared/five_scene'/cfg['tag']/scene/(heldname.replace(' ','_')+'.json')
   if path.exists():
    split=read(path)
    if split['identity']!=fp({k:v for k,v in split.items() if k!='identity'}) or split['source_identity']!=sid or split['dataset_identity']!=dataset['identity'] or split['clean_identity']!=clean['identity'] or split['active']!=active or split['held']!=heldname:raise RuntimeError('Locked split changed; use a new tag')
   else:
    split=choose_split(objects,active,held,cfg);split.update(scene=scene,held=heldname,active=active,classes=known,source_identity=sid,dataset_identity=dataset['identity'],clean_identity=clean['identity'],scope='Known-only selected training images partitioned for development. Official test annotations not accessed. Filename/SHA group heuristic does not certify patient/video independence.')
    split['identity']=fp(split);write(path,split)
   rows.append(dict(scene=scene,held=heldname,admitted=split['admitted'],reason=split.get('reason'),coverage=split.get('coverage'),split=path.relative_to(root).as_posix(),identity=split['identity']));print(scene,heldname,'ADMITTED' if split['admitted'] else 'BLOCKED '+split['reason'],flush=True)
 result=dict(complete=True,source_identity=sid,dataset_identity=dataset['identity'],clean_identity=clean['identity'],cases=rows,admitted=sum(x['admitted'] for x in rows),training_updates=0,new_checkpoints=0,training_ready=False,scope='Split admission only. No scene parents have been trained; cannot claim M1/H five-scene results. Blocked cases must not weaken group rules silently.')
 write(out/'AUDIT.json',result);return result
