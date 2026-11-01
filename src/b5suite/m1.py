from pathlib import Path
import copy,time,json,gc
import numpy as np,torch
from .protocol import read,write,sha,source,task,m1_split,lock,space,progress

def run(root,scene,name,seed,device):
 from scenepilot.protocol import gpu,view
 from scenepilot.teacher import prepare_base,prepare_s
 from scenepilot.features import collect,raw_arrays
 from scenepilot.engine import state_digest
 from jointdet.protocol import load_torch,atomic_save
 from jointdet.scoring import score_arrays,detections
 from jnet.model import NetworkMerge
 from .m1_training import train_phase
 from .evaluate import evaluate
 root=Path(root);cfg=read(root/'configs/b5_suite.json');spec=task(root,scene,name);report=root/'reports/b5_m1_suite'/cfg['tag']/'M1'/scene/name/f'seed_{seed}';report.mkdir(parents=True,exist_ok=True);runroot=root/'runs/b5_m1_suite'/cfg['tag']/'M1'/scene/name/f'seed_{seed}';prepared=root/'prepared/b5_m1_suite'/cfg['tag']/'M1'/scene/name;space(root,cfg)
 if (report/'RESULT.json').exists():
  old=read(report/'RESULT.json')
  if old['source']!=source(root):raise RuntimeError('Completed M1 belongs to different source')
  if old['status']=='completed' and sha(runroot/'M1/last.pt')!=old['model_sha256']:raise RuntimeError('Retained M1 terminal changed')
  return old
 try:s=m1_split(spec)
 except ValueError as e:
  result=dict(complete=True,status='blocked_metadata_support',reason=str(e),source=source(root),counts=spec['counts']);write(report/'RESULT.json',result);return result
 c=read(root/'configs/scene_pilot.json')['model'];c=copy.deepcopy(c)
 for key in ['pf_config','pf_weights','dino_folder']:c[key]=str(root/c[key])
 rcfg=dict(tag=cfg['tag'],seed=seed,model=c,parent_updates=100,merge_updates=100,validate_every=25,min_free_gib=8,max_output_gib=cfg['output_budget_gib'],max_gpu_gib=24,min_gpu_gib=24,known_floor_drop=.33,output_budget_gib=cfg['output_budget_gib'])
 lock(report/'PROTOCOL.json',dict(source=source(root),split=s,task_identity=spec['identity'],config=rcfg,selection='fixed_step100_no_test_model_selection',same_total_shot_budget_as_B5=True,calibration_policy_differs_from_B5=True))
 d=gpu(device,rcfg);viewroot=view(root,scene);terminal=runroot/'M1/last.pt';saved=load_torch(terminal) if terminal.exists() else None
 if saved is not None and saved['complete']:
  original=read(runroot/'M1/LOCK.json')
  if original['source']!=source(root) or saved['lock_identity']!=original['identity']:raise RuntimeError('Completed terminal provenance changed')
  m=NetworkMerge(viewroot,c,s['active'],'M1',initial=None).to(d);m.load_adapted(saved['model']);del saved
 else:
  del saved
  teacher=prepare_base(root,viewroot,rcfg,s,d,prepared/'base_teacher',report)
  m,_=train_phase(root,viewroot,rcfg,s,d,runroot,report,'P0',teacher);del m,_;gc.collect();torch.cuda.empty_cache()
  m,_=train_phase(root,viewroot,rcfg,s,d,runroot,report,'S',teacher);st=prepare_s(root,viewroot,rcfg,s,d,m,prepared/f's_teacher_{seed}',report,sha(runroot/'S/last.pt'));del m,_;gc.collect();torch.cuda.empty_cache()
  initial={a:load_torch(runroot/a/'last.pt')['model'] for a in ['P0','S']};m,saved=train_phase(root,viewroot,rcfg,s,d,runroot,report,'M1',st,initial=initial);del initial,saved;gc.collect();torch.cuda.empty_cache()
 m.eval();modelsha=sha(terminal)
 try:bank,cal=collect(root,viewroot,m,s,rcfg,modelsha,report)
 except (RuntimeError,ValueError) as e:
  result=dict(complete=True,status='blocked_matched_calibration',reason=str(e),source=source(root),model_sha256=modelsha,training_complete=True);write(report/'RESULT.json',result);return result
 atomic_save(bank,runroot/'M1/gallery.pt');write(runroot/'M1/calibration.json',cal)
 # Freeze fit/cal artifacts before any official-test inference or labels.
 lock(report/'INFERENCE_LOCK.json',dict(source=source(root),model_sha256=modelsha,bank_sha256=sha(runroot/'M1/gallery.pt'),calibration=cal,task_identity=spec['identity'],test_access_before_lock=False))
 partial=report/'partial.jsonl';preds=[json.loads(x) for x in partial.read_text().splitlines() if x.strip()] if partial.exists() else []
 if [r['image'] for r in preds]!=spec['test_images'][:len(preds)]:raise RuntimeError('M1 test prediction prefix differs')
 start=time.monotonic()
 for i,image in enumerate(spec['test_images'][len(preds):],len(preds)):
  with torch.inference_mode():raw=raw_arrays(m.forward_image(viewroot,image,[],d),c)
  values=score_arrays(raw,bank,cal,c);row=dict(image=image,**detections(raw,values,c));preds.append(row)
  with partial.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
  progress(report,i+1,len(spec['test_images']),'M1 official test inference, no labels',start)
  if torch.cuda.max_memory_allocated(d)>24*2**30:raise RuntimeError('M1 GPU budget exceeded')
  if i%32==0:space(root,cfg)
 write(report/'predictions.json',preds);metrics=evaluate(root,report/'predictions.json',spec,report)
 if sha(runroot/'M1/last.pt')!=modelsha:raise RuntimeError('Inference changed saved model')
 saved=load_torch(runroot/'M1/last.pt')
 if set(saved['model'])!=m.save_names() or state_digest(saved['model'])!=state_digest(m.adapted_state()):raise RuntimeError('M1 terminal lacks self-contained adapted/frozen-tail parameters')
 # Remove only this new experiment's redundant parent checkpoint files after successful evaluation.
 removals=[]
 for arm in ['P0','S']:
  path=runroot/arm/'last.pt'
  if not path.exists():
   if not (report/'AUXILIARY_CHECKPOINT_RETIREMENT.json').exists():raise RuntimeError('Missing parent without retirement record')
   continue
  expected=read(runroot/arm/'RESULT.json')['weights_sha256'];resolved=path.resolve()
  if path.is_symlink() or any(a.is_symlink() for a in path.parents if a!=root and root in a.parents) or not resolved.is_relative_to((root/'runs/b5_m1_suite').resolve()) or sha(path)!=expected:raise RuntimeError('Auxiliary cleanup verification failed')
  removals.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=expected))
 write(report/'AUXILIARY_CHECKPOINT_RETIREMENT.json',dict(files=removals,final_model_sha256=modelsha,logs_retained=True))
 for row in removals:(root/row['path']).unlink()
 result=dict(complete=True,status='completed',source=source(root),metrics=metrics,model_sha256=modelsha,seed=seed,task=name,scene=scene,retained_weights='M1/last.pt only; terminal optimizer removed',selection='fixed100_endpoint',calibration_policy='metadata-only image-held fit/cal; differs from B5 five-fold OOF')
 write(report/'RESULT.json',result);return result
