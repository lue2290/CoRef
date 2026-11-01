from pathlib import Path
import copy,json,time,gc
import numpy as np
from .protocol import read,write,sha,fp,lock,paths,case,view,source,space,progress,gpu

def state_digest(state):
 import hashlib
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
 return h.hexdigest()

def eligible(metrics,baseline,drop):return metrics['K_dev_AP50']>=baseline['K_dev_AP50']-drop-1e-10

def better(previous,metrics,baseline,drop):
 return eligible(metrics,baseline,drop) and metrics['held_dev_AP50']>baseline['held_dev_AP50']+1e-12 and (previous is None or metrics['held_dev_AP50']>previous['metrics']['held_dev_AP50']+1e-12)

def save(root,cfg,path,value):
 from jointdet.protocol import atomic_save
 estimate=sum(v.numel()*v.element_size() for v in value.get('model',{}).values())
 # Reserve space for an atomic replacement and optimizer moments.
 space(root,cfg,max(estimate*(3 if value.get('optimizer') is not None else 1),1024**2));atomic_save(value,path)

def train_phase(root,viewroot,cfg,s,device,run,report,arm,teacher,initial=None,smoke=False):
 import torch
 from cil.common import configure_torch,rng_state,restore_rng
 from jointdet.model import JointDetector
 from jnet.model import NetworkMerge
 from jointdet.engine import optimizer_for,learning_rate
 from jointdet.protocol import load_torch,environment
 from .teacher import base_target,s_target
 from .features import validate
 from .protection import accumulated_update
 from .s_loss import loss_for as s_loss
 from .merge_loss import loss_for as m_loss,ce_route
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
 if arm=='M1' and baseline is None and not smoke:
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
  if arm=='M1' and current%every==0 and not smoke:
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

def run_scene(root,cfg,scene,device,smoke=False):
 import torch
 from jointdet.protocol import load_torch
 from .teacher import prepare_base,prepare_s
 from .features import validate
 from .h_engine import train_h
 cfg=copy.deepcopy(cfg);cfg['tag']+=('_smoke' if smoke else '')
 if smoke:cfg['model'].update(updates=2,unfreeze_at=1)
 for key in ['pf_config','pf_weights','dino_folder']:cfg['model'][key]=str(Path(root)/cfg['model'][key])
 s=case(root,cfg,scene,smoke);run,report,prepared=paths(root,cfg,scene);report.mkdir(parents=True,exist_ok=True);viewroot=view(root,scene);device=gpu(device,cfg)
 teacher=prepare_base(root,viewroot,cfg,s,device,prepared/'base_teacher',report,smoke)
 p0,ps=train_phase(root,viewroot,cfg,s,device,run,report,'P0',teacher,smoke=smoke);del p0,ps;gc.collect();torch.cuda.empty_cache()
 model,ss=train_phase(root,viewroot,cfg,s,device,run,report,'S',teacher,smoke=smoke)
 st=prepare_s(root,viewroot,cfg,s,device,model,prepared/'s_teacher',report,sha(run/'S/last.pt'));del model,ss;gc.collect();torch.cuda.empty_cache()
 initial={arm:load_torch(run/arm/'last.pt')['model'] for arm in ['P0','S']}
 model,ms=train_phase(root,viewroot,cfg,s,device,run,report,'M1',st,initial=initial,smoke=smoke);del initial,ms
 if smoke:
  result=dict(complete=True,scene=scene,full_recipe_smoke_passed=True,scope='P0 projection, S paired soft responses and S-initialized M1 soft objectness each two steps; H GPU path requires pilot gallery coverage.',H_trained=False)
 else:
  bestpath=run/'M1/best.pt';selected=load_torch(bestpath) if bestpath.exists() else None
  # If no update beats the zero M1 incumbent under the K floor, reconstruct that exact parent initialization.
  if selected is None:
   from jnet.model import NetworkMerge
   del model;gc.collect();torch.cuda.empty_cache();initial={arm:load_torch(run/arm/'last.pt')['model'] for arm in ['P0','S']};model=NetworkMerge(viewroot,cfg['model'],s['active'],'M1',initial=initial).to(device);selected_step=0;del initial
  else:
   from jnet.model import NetworkMerge
   del model;gc.collect();torch.cuda.empty_cache();model=NetworkMerge(viewroot,cfg['model'],s['active'],'M1',initial=None).to(device);model.load_adapted(selected['model']);selected_step=selected['step']
  digest=state_digest(model.adapted_state());m1,bank,cal=validate(root,viewroot,model,s,cfg,digest,run/'selected_M1',report,prepared/'candidates')
  from identitylab.evaluation import serialize
  original_eval=run/'M1/evaluations'/f'step_{selected_step:04d}'/'predictions.json'
  if serialize(read(original_eval))!=serialize(read(run/'selected_M1/predictions.json')):raise RuntimeError('Selected M1 failed serialized prediction replay')
  write(report/'M1_REPLAY.json',dict(passed=True,selected_step=selected_step,model_digest=digest,training_updates=0))
  # Small selected gallery/calibration artifact, not a duplicate network snapshot.
  save(root,cfg,run/'M1/gallery.pt',dict(bank=bank,calibration=cal,model_digest=digest,selected_step=selected_step))
  del model,selected;gc.collect();torch.cuda.empty_cache()
  h=train_h(root,cfg,s,device,run,report,prepared,bank,cal,m1,digest)
  result=dict(complete=True,scene=scene,held=s['held'],selected_M1_step=selected_step,M1=m1,H=h,formal=False,official_test_access=False)
 result['source_identity']=source(root)
 write(report/'RESULT.json',result);return result
