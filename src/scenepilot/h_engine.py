from pathlib import Path
import time
import numpy as np
from .protocol import write,read,lock,source,progress,space,sha

def train_h(root,cfg,s,device,run,report,prepared,bank,cal,baseline,digest):
 import torch
 from cil.common import configure_torch,rng_state,restore_rng
 from jointdet.protocol import load_torch
 from identitylab.core import IdentityAdapter,torch_distances
 from identitylab.evaluation import map_bank,serialize
 from identityepisode.core import difficulty_report
 from identityepisode.engine import update
 from .features import Cache
 from .h_readout import evaluate_readouts
 from .engine import save,better
 configure_torch(145,cfg['model']['threads']);h={**cfg['h'],'probe_updates':cfg['h_updates'],'validate_every':cfg['h_validate_every']};base=cfg['identity'];folder=run/'H';folder.mkdir(parents=True,exist_ok=True);cache=Cache(root,cfg,s,prepared/'candidates');adapter=IdentityAdapter(bank['z'].shape[1]//4,base['bottleneck']).to(device)
 data=dict(z=np.asarray(bank['z']),labels=np.asarray(bank['labels']),groups=np.asarray(bank['groups']),ids=np.asarray(bank['objects']))
 record=lock(folder/'LOCK.json',dict(source=source(root),split_identity=s['identity'],parent_digest=digest,H=h,identity_config=base,training_seed=145))
 with torch.no_grad():
  dist=torch_distances(torch.tensor(data['z'],device=device),torch.tensor(bank['z'],device=device),torch.tensor(bank['labels'],device=device),bank['active'],cfg['model']['k'],data['groups'],bank['groups'])
  difficulty=difficulty_report(dist,data['labels'],data['groups'],torch.tensor(bank['radius'],device=device),torch.tensor(bank['scale'],device=device),bank['active'],s['classes'],h)
 write(folder/'DIFFICULTY.json',difficulty)
 zero=evaluate_readouts(root,cache,adapter,bank,bank,cal,device,folder/'zero')
 if serialize(read(folder/'zero/H/predictions.json'))!=serialize(read(run/'selected_M1/predictions.json')):raise RuntimeError('Zero H changed selected M1 serialized predictions')
 if not difficulty['probe_admissible']:
  result=dict(complete=True,status='skipped_uninformative_episodes',reasons=difficulty['reasons'],baseline_M1=baseline,zero_readouts=zero,training_updates=0);write(folder/'RESULT.json',result);return result
 optimizer=torch.optim.AdamW(adapter.parameters(),lr=base['lr'],weight_decay=base['weight_decay']);rng=np.random.default_rng(145);history=[];curve=[];step=0;best=None;last=folder/'last.pt';bestpath=folder/'best.pt';spec=cache.spec
 if last.exists():
  saved=load_torch(last)
  if saved['lock_identity']!=record['identity']:raise RuntimeError('H checkpoint identity mismatch')
  if (folder/'RESULT.json').exists() and saved['complete']:return read(folder/'RESULT.json')
  adapter.load_state_dict(saved['adapter']);step=saved['step'];history=saved['history'];curve=saved['curve'];rng.bit_generator.state=saved['sampler'];restore_rng(saved['rng'])
  if saved['optimizer'] is not None:optimizer.load_state_dict(saved['optimizer'])
  if bestpath.exists():best=load_torch(bestpath)
  if best is not None and best['step']>step:best=None
 started=time.monotonic()
 # A completed terminal checkpoint without RESULT resumes evaluation only.
 for current in range(step+1,cfg['h_updates']+1):
  space(root,cfg);adapter.train();row=update(adapter,optimizer,data,bank,spec,base,h,rng,device,current);history.append(row)
  if current%cfg['h_validate_every']==0:
   adapter.eval();mapped=map_bank(adapter,bank,device,cfg['model']);readouts=evaluate_readouts(root,cache,adapter,mapped,bank,cal,device,folder/f'evaluations/step_{current:04d}');metrics=readouts['H'];curve.append(dict(step=current,readouts=readouts))
   if better(best,metrics,baseline,cfg['known_floor_drop']):
    best=dict(adapter={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},bank=mapped,original_bank=bank,calibration=cal,metrics=metrics,step=current,lock_identity=record['identity'],parent_digest=digest,readouts=readouts);save(root,cfg,bestpath,best)
   write(folder/'CURVE.json',dict(training=history,validation=curve,selected_step=best['step'] if best else 0))
   saved=dict(adapter={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},step=current,history=history,curve=curve,rng=rng_state(),sampler=rng.bit_generator.state,lock_identity=record['identity'],complete=current==cfg['h_updates'],optimizer=None if current==cfg['h_updates'] else optimizer.state_dict(),bank=mapped,calibration=cal,parent_digest=digest)
   save(root,cfg,last,saved)
  if torch.cuda.max_memory_allocated(device)>cfg['max_gpu_gib']*2**30:raise RuntimeError('H memory limit exceeded')
  progress(report,current,cfg['h_updates'],'H identity training',started)
 if bestpath.exists():
  best=load_torch(bestpath);fresh=IdentityAdapter(adapter.channels,base['bottleneck']).to(device);fresh.load_state_dict(best['adapter']);fresh.eval();replayed=evaluate_readouts(root,cache,fresh,best['bank'],bank,cal,device,folder/'best_replay')
  if serialize(read(folder/'best_replay/H/predictions.json'))!=serialize(read(folder/f'evaluations/step_{best["step"]:04d}/H/predictions.json')):raise RuntimeError('Saved H best failed cold replay')
  replay=dict(passed=True,step=best['step'],sha256=sha(bestpath));write(folder/'BEST_REPLAY.json',replay)
 else:replay=None
 result=dict(complete=True,status='trained',baseline_M1=baseline,terminal_readouts=curve[-1]['readouts'],selected_step=best['step'] if best else 0,selected_readouts=best['readouts'] if best else zero,best_replay=replay,curve=curve,training_updates=cfg['h_updates'],official_test_access=False)
 write(folder/'RESULT.json',result);return result
