from pathlib import Path
import os,time,gc
from .data import read,write,sha,source

def view_root(root,scene):
 root=Path(root).resolve();view=root/'prepared/five_scene/views'/scene;target=root/'assets/rwd/JPEGImages'/scene;link=view/'assets/data/images';link.parent.mkdir(parents=True,exist_ok=True)
 if link.exists():
  if link.resolve()!=target.resolve():raise RuntimeError('Scene view points at different images')
 else:os.symlink(target,link,target_is_directory=True)
 return view

def smoke_one(root,scene,device):
 import torch
 from cil.common import configure_torch
 from jointdet.engine import optimizer_for
 from jointdet.losses import joint_loss
 from jnet.model import NetworkMerge
 root=Path(root);cfg=read(root/'configs/five_scene.json');audit=read(root/'reports/five_scene'/cfg['tag']/'AUDIT.json')
 if audit['source_identity']!=source(root):raise RuntimeError('Sources changed after split audit')
 cases=[x for x in audit['cases'] if x['scene']==scene and x['admitted']]
 report=root/'reports/five_scene'/cfg['tag']/'smoke'/scene
 if not cases:
  result=dict(complete=True,status='blocked_no_admitted_split',training_updates=0,new_checkpoints=0);write(report/'RESULT.json',result);return result
 if not device.startswith('cuda') or not torch.cuda.is_available() or torch.cuda.get_device_properties(device).total_memory<cfg['min_gpu_gib']*2**30:raise RuntimeError('Run real-model smoke on the remote GPU, at least24GiB; local full inference blocked')
 limit=Path('/sys/fs/cgroup/memory.max')
 if limit.is_file() and limit.read_text().strip().isdigit() and int(limit.read_text())<6*2**30:raise RuntimeError('CPU memory cgroup below6GiB; enable instance RAM before model smoke')
 torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device);configure_torch(145,2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 manifest=read(root/'PROJECT_MANIFEST.json')['files']
 for n in ['versions/joint_network_merge/jnet/model.py','versions/joint_detector/jointdet/model.py','versions/joint_detector/jointdet/geometry.py','versions/joint_detector/jointdet/losses.py','assets/models/dinov3_large/model.safetensors','assets/models/pfrpn/pf_rpn_swinb_5p_coco_imagenet.pth']:
  if sha(root/n)!=manifest[n]['sha256']:raise RuntimeError('Base weight differs: '+n)
 split=read(root/cases[0]['split']);view=view_root(root,scene);c=read(root/'configs/joint_detector.json');c.update(merge_tail_blocks=2,unfreeze_backbones=True,dino_last_blocks=2,updates=100,unfreeze_at=50)
 for key in ['pf_config','pf_weights','dino_folder']:c[key]=str(root/c[key])
 model=NetworkMerge(view,c,split['active'],'M1',initial=None).to(device);opt=optimizer_for(model,c);frozen=model.frozen_open_digest();image=split['fit_images'][0];objects=[o for o in split['fit_objects'] if o['image']==image];updates=[];start=time.monotonic()
 from jointdet.model import pixels
 _,dino_image,_=pixels(view,image,c,device)
 fork=model.verify_fork(dino_image);del dino_image
 for step in [1,2]:
  model.configure_stage(step==2);model.train();opt.zero_grad(set_to_none=True);out=model.forward_image(view,image,objects,device)
  # This smoke checks only audited sparse-positive plumbing. It is NOT the parent S/P0/M1 training objective.
  empty=dict(boxes=out['boxes'].new_empty(0,4),p=out['boxes'].new_empty(0));loss,stats=joint_loss(out,empty,model,c)
  if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite smoke loss')
  if step==2:
   from fivescene.smoke import known_ce
   ce=known_ce(out,model,c);qgrad=torch.autograd.grad(ce,[out['shared_query']],retain_graph=True,allow_unused=True)[0]
   if qgrad is not None and bool(qgrad.abs().sum()>0):raise RuntimeError('M1 known CE leaked into shared PF query')
  loss.backward();nonzero={g:sum(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters() if model.group_for(n)==g) for g in ['new','pf','backbone']}
  if not nonzero['new'] or not nonzero['pf'] or (step==2 and not nonzero['backbone']):raise RuntimeError('Required gradient group absent: '+str(nonzero))
  if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError('Nonfinite smoke gradients')
  if any(p.grad is not None for n,p in model.named_parameters() if n.startswith(('open_blocks.','open_norm.'))):raise RuntimeError('Open tail unexpectedly trainable')
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['grad_clip']);opt.step();updates.append(dict(step=step,loss=float(loss.detach()),gradient_tensors=nonzero));write(report/'PROGRESS.json',dict(done=step,total=2,label='two-stage architecture smoke',eta=0));del out,loss
 if model.frozen_open_digest()!=frozen:raise RuntimeError('Frozen open tail changed')
 peak=torch.cuda.max_memory_allocated(device)/2**30
 if peak>cfg['max_gpu_gib']:raise RuntimeError('Smoke GPU budget exceeded')
 result=dict(complete=True,status='architecture_smoke_passed',scene=scene,held=split['held'],split_identity=split['identity'],source_identity=source(root),steps=updates,peak_allocated_gib=peak,elapsed_seconds=time.monotonic()-start,new_checkpoints=0,base_weights_only=True,trained_parent_loaded=False,feature_fork=fork,known_ce_shared_query_gradient_blocked=True,scope='Real shared-prefix M1 graph two-stage forward/backward; fresh base initialization, not historical trained M1 and not a performance experiment. No teacher constraint or H training was run.')
 write(report/'RESULT.json',result);del model,opt;gc.collect();torch.cuda.empty_cache();return result

def known_ce(out,model,c):
 import torch
 from torch.nn import functional as F
 from jointdet.geometry import cxcywh
 from jointdet.losses import assignment,dn_indices
 meta=out['head'].get('dn_meta');ndn=meta['num_denoising_queries'] if meta else 0
 pos,owners=assignment(out['boxes'][-1,ndn:],cxcywh(out['gt']),out['cls'][-1,ndn:]);dp,do=dn_indices(cxcywh(out['gt']),meta,out['boxes'].device);pos=torch.cat([pos+ndn,dp]);owners=torch.cat([owners,do]);labels=torch.tensor([model.active.index(i) for i in out['semantic'][owners].tolist()],device=pos.device)
 return F.cross_entropy(out['known'][pos],labels)*c['known_weight'] if len(pos) else out['known'].sum()*0
