from collections import Counter
import numpy as np
from .scoring import representation

class Samples:
 def __init__(self,s,seeds):
  self.s=s;self.seeds=seeds;self.positive=[];self.labels=[];self.negative={seed:[] for seed in seeds};self.rng={seed:np.random.default_rng(seed) for seed in seeds};self.refs=[];self.quality=[];self.missed=Counter()
 def add(self,image,boxes,pf,features):
  from cil.common import iou
  objects=[o for o in self.s['objects'] if o['image']==image]
  if not objects:return
  gt=np.asarray([o['box'] for o in objects],np.float32);ov=iou(boxes,gt);z,struct=representation(features);assign={};centers=(boxes[:,:2]+boxes[:,2:])/2;contains=((centers[:,None]>=gt[None,:,:2]).all(2)&(centers[:,None]<=gt[None,:,2:]).all(2));owner=np.where(contains,ov,-1).argmax(1);has=contains.any(1)
  for j,o in enumerate(objects):
   ids=np.flatnonzero(ov[:,j]>=.5)
   for i in ids[np.argsort(-ov[ids,j])[:3]]:
    if i not in assign or ov[i,j]>assign[i][1]:assign[int(i)]=(o['class_id'],ov[i,j])
   best=int(ov[:,j].argmax())
   if ov[best,j]>=.5:self.refs.append(dict(z=z[best].copy(),structure=struct[best].copy(),label=o['class_id'],image=image,id=o['id'],pf=float(pf[best])))
   else:self.missed[self.s['classes'][o['class_id']]]+=1
   owned=np.flatnonzero(has&(owner==j))
   for good in [1,0]:
    q=owned[ov[owned,j]>=.5] if good else owned[ov[owned,j]<=.2];q=q[np.argsort(-ov[q,j],kind='stable')[:3]]
    for i in q:self.quality.append(dict(x=struct[i].copy(),y=good,c=o['class_id'],image=image,id=o['id'],pf=float(pf[i])))
  for i,(cid,_) in assign.items():self.positive.append(features[i].copy());self.labels.append(cid)
  candidates=np.flatnonzero((pf<=np.quantile(pf,.2))&(ov.max(1)<.05))
  if len(candidates):
   for seed in self.seeds:
    ids=self.rng[seed].choice(candidates,size=min(10,len(candidates)),replace=False);self.negative[seed].extend(features[ids].copy())
 def calibration(self):
  refs=self.refs;q=self.quality;fold=self.s['image_folds'];n=len(self.s['classes'])
  if set(r['label'] for r in refs)!=set(range(n)):raise ValueError('Missing a reference class')
  return dict(z=np.asarray([r['z'] for r in refs],np.float32),ref_struct=np.asarray([r['structure'] for r in refs],np.float32),labels=np.asarray([r['label'] for r in refs]),images=np.asarray([r['image'] for r in refs]),object_ids=np.asarray([r['id'] for r in refs]),pf=np.asarray([r['pf'] for r in refs],np.float32),qx=np.asarray([r['x'] for r in q],np.float32),qy=np.asarray([r['y'] for r in q]),qc=np.asarray([r['c'] for r in q]),qi=np.asarray([r['image'] for r in q]),q_object_ids=np.asarray([r['id'] for r in q]),qpf=np.asarray([r['pf'] for r in q],np.float32),fold=np.asarray([fold[r['image']] for r in refs]),qfold=np.asarray([fold[r['image']] for r in q]))
 def head_data(self,seed):
  if set(self.labels)!=set(range(len(self.s['classes']))) or not self.negative[seed]:raise ValueError('Missing positive class or proxy negatives')
  x=np.concatenate([np.asarray(self.positive,np.float32),np.asarray(self.negative[seed],np.float32)]);mean=x.mean(0);std=x.std(0).clip(min=1e-4);y=np.r_[self.labels,[-1]*len(self.negative[seed])];fg=np.r_[np.ones(len(self.positive),np.float32),np.zeros(len(self.negative[seed]),np.float32)]
  return ((x-mean)/std).astype(np.float32),y.astype(np.int64),fg,mean,std
