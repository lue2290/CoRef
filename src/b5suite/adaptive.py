import numpy as np,joblib
from .scoring import class_distances,fit_quality
from .calibration import calibrate as strict_calibrate
from .protocol import write

def distances(z,gallery,labels,k_by_class):
 # CPU FP32 cosine and partition match original B5 arithmetic; only class k differs.
 import torch
 with torch.inference_mode():d=(1-torch.as_tensor(z,dtype=torch.float32)@torch.as_tensor(gallery,dtype=torch.float32).T).clamp(0,2).numpy()
 return np.stack([np.partition(d[:,labels==cid],k-1,axis=1)[:,:k].mean(1) for cid,k in enumerate(k_by_class)],1)

def eligible_classes(data):
 result=[]
 for cid in sorted(set(data['labels'].tolist())):
  valid=True
  for fold in range(5):
   mask=(data['labels']==cid)&(data['fold']!=fold);ims=data['images'][mask]
   if len(ims)<5 or len(ims)-max((sum(ims==n) for n in set(ims)),default=0)<5:valid=False;break
  if valid:result.append(cid)
 return result

def calibrate(data,folder,n_classes,progress=None):
 common=eligible_classes(data)
 if len(common)<2:raise ValueError('Adaptive variant needs at least two five-fold-capable classes')
 mapping={c:i for i,c in enumerate(common)};rm=np.isin(data['labels'],common);qm=np.isin(data['qc'],common);sub={}
 for k,v in data.items():sub[k]=v[qm] if k in ['qx','qy','qc','qi','q_object_ids','qpf','qfold'] else v[rm]
 sub['labels']=np.array([mapping[x] for x in sub['labels']]);sub['qc']=np.array([mapping[x] for x in sub['qc']]);params=strict_calibrate(sub,folder,len(common),progress)
 with np.load(folder/'gallery.npz',allow_pickle=False) as z:common_radius=z['radius'].copy();common_scale=z['scale'].copy()
 radius=[];scale=[];ks=[];policy=[]
 # Common-class statistics exactly use the strict image-excluded fit-support calculation.
 for cid in range(n_classes):
  count=int((data['labels']==cid).sum())
  if count==0:raise ValueError('No reference for class '+str(cid))
  ks.append(min(5,count))
  if cid in mapping:radius.append(float(common_radius[mapping[cid]]));scale.append(float(common_scale[mapping[cid]]));policy.append('class_specific_strict_5NN_cross_image')
  else:radius.append(float(np.median(common_radius)));scale.append(float(np.median(common_scale)));policy.append('borrow_median_radius_scale_from_common_classes_no_self_fit')
 np.savez_compressed(folder/'gallery.npz',z=data['z'],labels=data['labels'],images=data['images'],object_ids=data['object_ids'],radius=np.asarray(radius,np.float32),scale=np.asarray(scale,np.float32),k_by_class=np.asarray(ks))
 joblib.dump(fit_quality(data['qx'],data['qy']),folder/'quality.joblib');params.update(variant='B5_adaptive_k_and_rare_class_support_v1',k_by_class=ks,common_calibration_classes=common,uncalibrated_rare_classes=[c for c in range(n_classes) if c not in common],radius=radius,scale=scale,support_policy=policy,limitation='Rare classes participate in head/reference fitting but do not have class-specific held-out calibration; pooled common-class calibration is an explicit extrapolation, not original B5')
 write(folder/'parameters.json',params);return params
