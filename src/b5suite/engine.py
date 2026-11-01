from pathlib import Path
import time,random,json,gc
import numpy as np,torch,joblib
from .protocol import read,write,sha,source,lock,task,progress,space
from .head import RegionHead
from .samples import Samples
from .scoring import representation,class_distances,novelty,sigmoid,objectness

def train_head(builder,seed,folder,cfg,report,device):
 path=folder/f'head_{seed}.pt'
 if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 x,y,fg,mean,std=builder.head_data(seed);model=RegionHead(x.shape[1],len(builder.s['classes'])).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);features=torch.from_numpy(x).to(device);known=torch.from_numpy(y).to(device);foreground=torch.from_numpy(fg).to(device);history=[];start=time.monotonic()
 for epoch in range(40):
  model.train();total=0.
  for ids in torch.randperm(len(features),device=device).split(128):
   logits,f=model(features[ids]);pos=known[ids]>=0;cls=torch.nn.functional.cross_entropy(logits[pos],known[ids][pos]) if pos.any() else logits.sum()*0;fg_loss=torch.nn.functional.binary_cross_entropy_with_logits(f,foreground[ids],reduction='none');weights=torch.where(pos,1.,.15);loss=cls+.5*(fg_loss*weights).sum()/weights.sum();opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss.detach())*len(ids)
  history.append(total/len(features));progress(report,epoch+1,40,f'B5 head seed{seed}',start)
 saved=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},dimension=x.shape[1],n_classes=len(builder.s['classes']),mean=mean,std=std,seed=seed,epochs=40,losses=history)
 from jointdet.protocol import atomic_save
 atomic_save(saved,path);write(folder/f'head_{seed}_TRAIN.json',dict(seed=seed,epochs=40,losses=history,sha256=sha(path),positive_proposals=len(builder.positive),weak_negatives=len(builder.negative[seed])));del model,opt,features;torch.cuda.empty_cache();return saved

def decide(boxes,pf,features,head,saved,gallery,quality,params,device):
 from torchvision.ops import batched_nms
 from .adaptive import distances as adaptive_distances
 z,structure=representation(features)
 if 'k_by_class' in params:d=adaptive_distances(z,gallery['z'],gallery['labels'],params['k_by_class'])
 else:d=class_distances(z,gallery['z'],gallery['labels'])
 u=sigmoid(novelty(d,gallery['radius'],gallery['scale'])-params['known_shift']);q=quality.predict_proba(structure)[:,1].astype(np.float32);obj=objectness(pf,q,params['chosen']['weight'])
 with torch.inference_mode():logits,_=head(torch.from_numpy(((features-saved['mean'])/saved['std']).astype(np.float32)).to(device));probs=logits.softmax(1).cpu().numpy()
 labels=np.where(u>.5,saved['n_classes'],probs.argmax(1)).astype(np.int64);scores=np.where(labels==saved['n_classes'],obj*u,obj*(1-u)*probs.max(1)).astype(np.float32);ids=np.flatnonzero(obj>=params['chosen']['threshold'])
 if len(ids):ids=ids[batched_nms(torch.from_numpy(boxes[ids]),torch.from_numpy(scores[ids]),torch.from_numpy(labels[ids]),.5).numpy()[:100]]
 return dict(boxes=boxes[ids].tolist(),scores=scores[ids].tolist(),labels=labels[ids].tolist())

def run_scene(root,scene,seeds,device,smoke=False):
 from scenepilot.protocol import gpu
 from .extract import Extractor
 from .calibration import calibrate as strict_calibrate
 from .adaptive import calibrate as adaptive_calibrate
 from .evaluate import evaluate
 root=Path(root);cfg=read(root/'configs/b5_suite.json');gpu(device,cfg);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;space(root,cfg);report=root/'reports/b5_m1_suite'/cfg['tag']/'B5'/scene;report.mkdir(parents=True,exist_ok=True);work=root/'runs/b5_m1_suite'/cfg['tag']/'B5'/scene
 specs={t:task(root,scene,t) for t in ['task1','task2']};record=lock(work/'LOCK.json',dict(source=source(root),config=cfg,scene=scene,splits={t:s['identity'] for t,s in specs.items()},all_declared_seeds=cfg['seeds'],policies=['strict','adaptive']))
 if smoke:
  extractor=Extractor(root,device);name=specs['task1']['train_images'][0];b,p,f=extractor(scene,name)
  if f.shape!=(len(b),5120) or not np.isfinite(f).all():raise RuntimeError('Unexpected frozen B5 feature output')
  result=dict(passed=True,proposals=len(b),feature_dimension=f.shape[1],training_updates=0,source=source(root));write(report/'SMOKE.json',result);return result
 jobs={};builders={};extractor=None
 for t,s in specs.items():
  for variant in ['strict','adaptive']:
   folder=work/t/variant;folder.mkdir(parents=True,exist_ok=True)
   if (folder/'CAL_LOCK.json').exists():
    saved=read(folder/'CAL_LOCK.json')
    if saved['parent_lock']!=record['identity'] or any(sha(folder/n)!=h for n,h in saved['files'].items()):raise RuntimeError('B5 calibration artifacts changed')
    jobs[(t,variant)]='ready'
   elif (folder/'BLOCKED.json').exists():jobs[(t,variant)]='blocked'
   else:jobs[(t,variant)]='pending'
  if any(jobs[(t,v)]=='pending' for v in ['strict','adaptive']) or any(not (work/t/f'head_{seed}.pt').exists() for seed in seeds):builders[t]=Samples(s,seeds)
 if builders:
  extractor=Extractor(root,device);images=[n for n in (root/'assets/rwd/ImageSets'/scene/'train.txt').read_text().splitlines() if any(n in b.s['train_images'] for b in builders.values())];start=time.monotonic()
  for i,name in enumerate(images):
   boxes,pf,features=extractor(scene,name)
   for builder in builders.values():builder.add(name,boxes,pf,features)
   progress(report,i+1,len(images),'B5 selected train streaming extraction',start)
   if i%32==0:space(root,cfg)
  for t,builder in builders.items():
   for variant,calfn in [('strict',strict_calibrate),('adaptive',adaptive_calibrate)]:
    if jobs[(t,variant)]!='pending':continue
    folder=work/t/variant
    try:
     data=builder.calibration();params=calfn(data,folder,len(builder.s['classes']),lambda n,total:progress(report,n,total,f'{t}/{variant} OOF',time.monotonic()))
     write(folder/'parameters.json',params);artifacts={p.name:sha(p) for p in folder.iterdir() if p.name in ['parameters.json','gallery.npz','quality.joblib','validation_predictions.npz']};write(folder/'CAL_LOCK.json',dict(parent_lock=record['identity'],files=artifacts,test_access_before_lock=False));jobs[(t,variant)]='ready'
    except ValueError as e:write(folder/'BLOCKED.json',dict(status='blocked_original_requirements' if variant=='strict' else 'blocked_adaptive_requirements',reason=str(e),test_access_before_lock=False));jobs[(t,variant)]='blocked'
   if any(jobs[(t,v)]=='ready' for v in ['strict','adaptive']):
    for seed in seeds:
     try:train_head(builder,seed,work/t,cfg,report,device)
     except ValueError as e:write(work/t/f'head_{seed}_BLOCKED.json',dict(reason=str(e)))
  del builders;gc.collect()
 active={};summary=[]
 for (t,variant),state in jobs.items():
  if state!='ready':summary.append(dict(task=t,variant=variant,status='blocked',detail=read(work/t/variant/'BLOCKED.json')));continue
  folder=work/t/variant;params=read(folder/'parameters.json');gallery=dict(np.load(folder/'gallery.npz',allow_pickle=False));quality=joblib.load(folder/'quality.joblib')
  for seed in seeds:
   if (work/t/f'head_{seed}_BLOCKED.json').exists():
    summary.append(dict(task=t,variant=variant,seed=seed,status='blocked_head',detail=read(work/t/f'head_{seed}_BLOCKED.json')));continue
   if sha(work/t/f'head_{seed}.pt')!=read(work/t/f'head_{seed}_TRAIN.json')['sha256']:raise RuntimeError('Saved B5 head changed')
   dest=report/t/variant/f'seed_{seed}';pred=dest/'predictions.json';saved=torch.load(work/t/f'head_{seed}.pt',map_location='cpu',weights_only=False);head=RegionHead(saved['dimension'],saved['n_classes']).to(device);head.load_state_dict(saved['state_dict']);head.eval();partial=dest/'partial.jsonl';dest.mkdir(parents=True,exist_ok=True);old=[json.loads(x) for x in partial.read_text().splitlines() if x.strip()] if partial.exists() else []
   if [x['image'] for x in old]!=specs[t]['test_images'][:len(old)]:raise RuntimeError('Test prefix does not match locked split')
   active[(t,variant,seed)]=dict(head=head,saved=saved,params=params,gallery=gallery,quality=quality,dest=dest,predictions=old)
 if active:
  if extractor is None:extractor=Extractor(root,device)
  images=specs['task1']['test_images'];start=time.monotonic()
  for i,name in enumerate(images):
   needed=[(k,v) for k,v in active.items() if len(v['predictions'])<=i]
   if needed:
    boxes,pf,features=extractor(scene,name)
    for (t,variant,seed),v in needed:
     row=dict(image=name,**decide(boxes,pf,features,v['head'],v['saved'],v['gallery'],v['quality'],v['params'],device));v['predictions'].append(row)
     with (v['dest']/'partial.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
   progress(report,i+1,len(images),'B5 official test streaming (no test labels)',start)
   if torch.cuda.max_memory_allocated(device)>cfg['max_gpu_gib']*2**30:raise RuntimeError('GPU budget exceeded')
   if i%32==0:space(root,cfg)
  for (t,variant,seed),v in active.items():
   write(v['dest']/'predictions.json',v['predictions']);metrics=evaluate(root,v['dest']/'predictions.json',specs[t],v['dest']);summary.append(dict(task=t,variant=variant,seed=seed,status='completed',metrics=metrics))
 result=dict(complete=True,scene=scene,source=source(root),results=summary,no_test_feature_cache=True,network_trainable=False);write(report/'RESULT.json',result);return result
