from pathlib import Path
import numpy as np,time,json
from .protocol import source,lock,write,space,progress,sha
from scenepilot.engine import state_digest,better
def save(root,cfg,path,value):
    from jointdet.protocol import atomic_save
    import shutil
    estimate=sum(v.numel()*v.element_size() for v in value.get('model',{}).values())*(3 if value.get('optimizer') is not None else 1)
    space(root,cfg)
    if shutil.disk_usage(root).free<8*2**30+estimate:raise RuntimeError('Insufficient reserve for atomic checkpoint replacement')
    atomic_save(value,path)

def train_phase(root,viewroot,cfg,s,device,run,report,arm,teacher,initial=None,smoke=False):
 import torch
 from cil.common import configure_torch,rng_state,restore_rng
 from jointdet.model import JointDetector
 from jnet.model import NetworkMerge
 from jointdet.engine import optimizer_for,learning_rate
 from jointdet.protocol import load_torch,environment
 from scenepilot.teacher import base_target,s_target
 from scenepilot.features import validate
 from scenepilot.protection import accumulated_update
 from scenepilot.s_loss import loss_for as s_loss
 from scenepilot.merge_loss import loss_for as m_loss,ce_route
 c=cfg['model'];run=Path(run)/arm;run.mkdir(parents=True,exist_ok=True);target=2 if smoke else cfg['merge_updates'] if arm=='M1' else cfg['parent_updates'];every=1 if smoke else cfg['validate_every']
 record=lock(run/'LOCK.json',dict(environment=environment(),source=source(root),split_identity=s['identity'],fit_images=s['fit_images'],arm=arm,target=target,model_config=c,seed=cfg['seed'],initial={k:state_digest(v) for k,v in initial.items()} if initial else None,smoke=smoke))
 configure_torch(cfg['seed'],c['threads']);model=(NetworkMerge(viewroot,c,s['active'],'M1',initial=initial) if arm=='M1' else JointDetector(viewroot,c,s['active'],'J0')).to(device);optimizer=optimizer_for(model,c)
 frozen=model.frozen_open_digest() if arm=='M1' else None
 rng=np.random.default_rng(cfg['seed']);history=[];curve=[];step=0;baseline=None;best=None;last=run/'last.pt';bestpath=run/'best.pt'
 if last.exists():
  old=load_torch(last)
  if old['lock_identity']!=record['identity']:raise RuntimeError('Checkpoint identity differs')
  model.load_adapted(old['model']);step=old['step'];history=old['history'];curve=old['curve'];baseline=old.get('baseline')
  if bestpath.exists():
   best=load_torch(bestpath)
   if best['lock_identity']!=record['identity']:raise RuntimeError('Best identity differs')
   if best['step']>step:best=None
  if old['complete']:return model,old
  optimizer.load_state_dict(old['optimizer']);restore_rng(old['rng']);rng.bit_generator.state=old['sampler']
 if False: # Formal fixed-endpoint run has no development-set model selection.
  states=rng_state();baseline,_,_=validate(root,viewroot,model,s,cfg,state_digest(model.adapted_state()),run/'evaluations/step_0000',report);restore_rng(states);curve.append(dict(step=0,metrics=baseline))
 by={n:[o for o in s['fit_objects'] if o['image']==n] for n in s['fit_images']};start=time.monotonic();attempt=str(time.time_ns());log=run/('training_'+attempt+'.jsonl')
 for current in range(step+1,target+1):
  space(root,cfg);model.configure_stage(current>(1 if smoke else c['unfreeze_at']));model.train();learning_rate(optimizer,c,current-1);optimizer.zero_grad(set_to_none=True)
  batch=[dict(image=str(rng.choice(s['fit_images'])),flip=bool(rng.integers(2))) for _ in range(1 if smoke else c['accumulate'])]
  if arm=='P0':
   def forward(item):
    out=model.forward_image(viewroot,item['image'],by[item['image']],device,item['flip']);empty=dict(boxes=out['boxes'].new_empty(0,4),p=out['boxes'].new_empty(0));return out,empty,{}
   def known(item):return model.forward_image(viewroot,item['image'],by[item['image']],device,item['flip'])
   details=accumulated_update(model,batch,forward,known,c,'P0');losses=details['losses'];stats=details['micro_stats'];route=details['protection']
  else:
   losses=[];stats=[];route={}
   for item in batch:
    out=model.forward_image(viewroot,item['image'],by[item['image']],device,item['flip'])
    if arm=='S':
     targetrow=base_target(teacher,item['image'],device,item['flip']);empty=dict(boxes=out['boxes'].new_empty(0,4),p=out['boxes'].new_empty(0));loss,row=s_loss(out,empty,targetrow,model,c,'S')
    else:
     targetrow=s_target(teacher,item['image'],device,item['flip']);loss,row=m_loss(out,targetrow,model,c)
     if current in [1,(1 if smoke else c['unfreeze_at'])+1,target]:route=ce_route(out,model,c)
    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
    (loss/len(batch)).backward();losses.append(float(loss.detach()));stats.append(row);del out,loss,targetrow
  if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError('Nonfinite gradients')
  if not any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters()):raise RuntimeError('No parameter gradient')
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['grad_clip']);optimizer.step()
  row=dict(step=current,mean_loss=float(np.mean(losses)),stats=stats,route=route,samples=batch);history.append(row)
  with log.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
  if False: # Official test is accessed only after training and calibration are locked.
   states=rng_state();metrics,bank,cal=validate(root,viewroot,model,s,cfg,state_digest(model.adapted_state()),run/f'evaluations/step_{current:04d}',report);restore_rng(states);curve.append(dict(step=current,metrics=metrics))
   if better(best,metrics,baseline,cfg['known_floor_drop']):
    best=dict(model=model.adapted_state(),bank=bank,calibration=cal,metrics=metrics,step=current,lock_identity=record['identity']);save(root,cfg,bestpath,best)
   write(run/'CURVE.json',dict(training=history,validation=curve,selected_step=best['step'] if best else 0))
  if frozen is not None and current in [1,target] and model.frozen_open_digest()!=frozen:raise RuntimeError('Frozen open tail changed')
  peak=torch.cuda.max_memory_allocated(device)/2**30
  if peak>cfg['max_gpu_gib']:raise RuntimeError('GPU memory budget exceeded')
  if current%every==0 or current==target:
   saved=dict(model=model.adapted_state(),step=current,history=history,curve=curve,baseline=baseline,lock_identity=record['identity'],complete=current==target,optimizer=None if current==target else optimizer.state_dict(),rng=rng_state(),sampler=rng.bit_generator.state,terminal_exact_resume=current<target)
   save(root,cfg,last,saved)
  progress(report,current,target,arm+' training',start)
 # Fresh-run completion and rerun completion share the same output schema.
 write(run/'RESULT.json',dict(complete=True,step=target,arm=arm,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,baseline=baseline,terminal_metrics=curve[-1]['metrics'] if curve else None,best_step=best['step'] if best else None,weights_sha256=sha(last),terminal_optimizer_retained=False))
 del optimizer;return model,saved
