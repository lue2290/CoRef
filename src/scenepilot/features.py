from pathlib import Path
import numpy as np,time
from .protocol import write,sha,fp,progress,space

def raw_arrays(out,c):
 import torch
 from jointdet.scoring import raw_arrays as public
 from jointdet.model import pf_score
 raw=public(out,c)
 if 'open_cls' in out:
  ids=torch.as_tensor(raw['query_index'],device=out['open_cls'].device);cpu=lambda x:x.detach().cpu().numpy()
  raw.update(proposal_pf=raw['pf'].copy(),pf=cpu(pf_score(out['open_cls'][ids],out['open_ctr'][ids])),object_logit=cpu(out['open_cls'][ids,1]),pf_token_logits=cpu(out['open_cls'][ids]),centerness=cpu(out['open_ctr'][ids]))
 return raw

def collect(root,view,model,s,cfg,checkpoint,report,cache_folder=None):
 import torch
 from identitylab.core import match_objects
 from jointdet.scoring import support
 c=cfg['model'];device=next(model.parameters()).device;model.eval();start=time.monotonic();total=len(s['fit_images'])+len(s['cal_images']);done=0
 bank=dict(z=[],labels=[],groups=[],objects=[],active=s['active'],checkpoint_sha256=checkpoint,split_identity=s['identity']);matched=[]
 if cache_folder is not None:Path(cache_folder).mkdir(parents=True,exist_ok=True)
 for role in ['fit','cal']:
  for name in s[role+'_images']:
   with torch.inference_mode():raw=raw_arrays(model.forward_image(view,name,[],device),c)
   objects=[o for o in s[role+'_objects'] if o['image']==name]
   pairs=match_objects(raw,objects)
   if role=='fit':
    for j,i,_ in pairs:
     o=objects[j];bank['z'].append(raw['z'][i]);bank['labels'].append(o['class_id']);bank['groups'].append(o['group']);bank['objects'].append(o['id'])
   else:
    obj=raw['pf']*(1-c['quality_alpha']+c['quality_alpha']*raw['q'])
    for j,i,_ in pairs:matched.append(dict(score=float(obj[i]),class_id=objects[j]['class_id'],group=objects[j]['group']))
    if cache_folder is not None:save_raw(root,cfg,cache_folder,name,raw)
   done+=1;progress(report,done,total,'matching gallery and known calibration',start)
 if not bank['z']:raise RuntimeError('No matched fit references; do not lower IoU automatically')
 bank['z']=np.asarray(bank['z'],np.float32);bank=support(bank,c)
 for cid in s['active']:
  rows=[x for x in matched if x['class_id']==cid]
  if len(rows)<5 or len({x['group'] for x in rows})<2:raise RuntimeError('Insufficient matched known calibration coverage for class '+str(cid))
 cal=dict(object_threshold=float(np.quantile([x['score'] for x in matched],1-c['object_retention'],method='lower')),split_identity=s['identity'],checkpoint_sha256=checkpoint,matched_known=len(matched),records=matched)
 return bank,cal

def save_raw(root,cfg,folder,name,raw):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/(fp(name)[:24]+'.npz');space(root,cfg,sum(x.nbytes for x in raw.values())+1024)
 tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**raw)
 tmp.replace(path)
 return path

class Cache:
 def __init__(self,root,cfg,s,folder):self.spec=dict(split=s,config=cfg['model']);self.folder=Path(folder)
 def load(self,name):
  if name not in set(self.spec['split']['cal_images']+self.spec['split']['val_images']):raise PermissionError('Cache request outside cal/dev')
  from .protocol import read
  m=read(self.folder/'MANIFEST.json');path=self.folder/(fp(name)[:24]+'.npz')
  if sha(path)!=m['files'][path.name]:raise RuntimeError('Candidate cache changed')
  with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def validate(root,view,model,s,cfg,checkpoint,out,report,cache_folder=None):
 import torch
 from jointdet.scoring import score_arrays,detections
 from .evaluation import evaluate_predictions
 out=Path(out);bank,cal=collect(root,view,model,s,cfg,checkpoint,report,cache_folder);predictions=[];start=time.monotonic()
 for i,name in enumerate(s['val_images']):
  with torch.inference_mode():raw=raw_arrays(model.forward_image(view,name,[],next(model.parameters()).device),cfg['model'])
  values=score_arrays(raw,bank,cal,cfg['model']);predictions.append(dict(image=name,**detections(raw,values,cfg['model'])))
  if cache_folder is not None:save_raw(root,cfg,cache_folder,name,raw)
  progress(report,i+1,len(s['val_images']),'development evaluation',start)
 metrics=evaluate_predictions(root,s,predictions,out)
 if cache_folder is not None:
  files={p.name:sha(p) for p in Path(cache_folder).glob('*.npz')};write(Path(cache_folder)/'MANIFEST.json',dict(checkpoint_sha256=checkpoint,split_identity=s['identity'],files=files))
 write(out/'CALIBRATION.json',cal);return metrics,bank,cal
