from pathlib import Path
import json,zipfile,hashlib,collections,statistics as st,xml.etree.ElementTree as ET,time,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'reports/reference_net/reference_net_diagnostics_20261024T171119Z.zip'
OUT=ROOT/'reports/reference_net/reference_net_v1'
def meanstd(v):return dict(mean=st.mean(v),std=st.stdev(v) if len(v)>1 else None,values=v)
def quantiles(v):return [float(x) for x in np.quantile(v,[0,.25,.5,.75,1])] if v else []
def iou(a,b):
 a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
 inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1)
 return inter/np.maximum(1e-9,np.maximum(0,a[:,2:]-a[:,:2]).prod(-1)[:,None]+np.maximum(0,b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter)
def main():
 start=time.monotonic();integrity={};results=[];evidence=[];gtcache={};codecheck={}
 with zipfile.ZipFile(BUNDLE) as z:
  manifest=json.loads(z.read('BUNDLE_MANIFEST.json'))
  for idx,(n,want) in enumerate(manifest.items(),1):
   h=hashlib.sha256()
   with z.open(n) as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
   if h.hexdigest()!=want:raise RuntimeError('ZIP member SHA mismatch: '+n)
   if idx%100==0:print('Verified',idx,'/',len(manifest),flush=True)
  integrity=dict(passed=True,verified_members=len(manifest),bundle_sha256=hashlib.sha256(BUNDLE.read_bytes()).hexdigest())
  for n in z.namelist():
   if n.startswith('src/refnet/') and n.endswith('.py'):
    remote=z.read(n).decode().replace('\r\n','\n').replace('\r','\n')
    codecheck[n]=ast.dump(ast.parse(remote))==ast.dump(ast.parse((ROOT/n).read_text(encoding='utf8')))
  for n in z.namelist():
   if not n.startswith('reports/reference_net/reference_net_v1/') or not n.endswith('/RESULT.json'):continue
   result=json.loads(z.read(n));assert result['complete'];results.append(result)
   if result['task']!='task1':continue
   scene=result['scene'];seed=result['seed'];prefix=n[:-len('RESULT.json')]
   meta=json.loads((ROOT/'prepared/b5_suite/b5_five_scene_v1'/(scene+'.json')).read_text())
   classes=meta['PK'];K=len(classes)
   cal=json.loads(z.read(prefix+'terminal/CALIBRATION.json'));curve=json.loads(z.read(prefix+'CURVE.json'))
   counts=collections.Counter();confusion=collections.Counter();unknown_classes={};tp_scores=[];fp_scores=[]
   with z.open(prefix+'terminal/partial.jsonl') as stream:
    for line in stream:
     row=json.loads(line);key=(scene,row['image'])
     if key not in gtcache:
      tree=ET.parse(ROOT/'assets/rwd/Annotations'/scene/(Path(row['image']).stem+'.xml')).getroot()
      gtcache[key]=[(o.findtext('name'),[float(o.find('bndbox').findtext(t))-1 for t in ['xmin','ymin','xmax','ymax']],int(o.findtext('difficult','0'))) for o in tree.findall('object')]
     allgt=gtcache[key];gt=[g for g in allgt if not g[2]];labels=np.array(row['labels']);scores=np.array(row['scores']);ov=iou(row['boxes'],[g[1] for g in gt]);u=labels==K;known=(labels>=0)&(labels<K)
     counts['images']+=1;counts['outputs']+=len(labels);counts['unknown_outputs']+=int(u.sum());counts['known_outputs']+=int(known.sum())
     for j,(name,box,_) in enumerate(gt):
      good=ov[:,j]>=.5;hit=bool(good.any());hu=bool((good&u).any());hk=bool((good&known).any())
      if name in classes:continue
      stats=unknown_classes.setdefault(name,collections.Counter());stats['gt']+=1;counts['unknown_gt']+=1
      counts['unknown_gt_any_final_box']+=hit;counts['unknown_gt_unknown_final_box']+=hu;counts['unknown_gt_known_final_box']+=hk
      if hk and not hu:
       counts['unknown_gt_known_only']+=1;stats['known_only']+=1
       ids=np.flatnonzero(good&known);best=ids[np.argmax(scores[ids])];confusion[classes[int(labels[best])]]+=1
      if not hit:counts['unknown_gt_no_final_box']+=1;stats['no_final_box']+=1
      stats['unknown_hit']+=hu
     allov=iou(row['boxes'],[g[1] for g in allgt]);bad=allov.max(1)<.5 if allgt else np.ones(len(labels),bool)
     for k in np.flatnonzero(u):
      true=bool(any(name not in classes and ov[k,j]>=.5 for j,(name,_,_) in enumerate(gt)))
      (tp_scores if true else fp_scores).append(float(scores[k]))
     counts['unknown_low_iou_outputs']+=int((bad&u).sum())
   first=curve['monitor'][0];last=curve['monitor'][-1]
   train=[v for row in curve['training'][-50:] for v in row['stats']]
   row=dict(scene=scene,seed=seed,counts=dict(counts),unknown_classes={k:dict(v) for k,v in unknown_classes.items()},unknown_to_known_classes=dict(confusion.most_common()),calibration={k:v for k,v in cal.items() if k not in ['per_class']},cal_class_coverage={classes[int(k)]:dict(gt=v['gt'],matched=v['matched'],unique_images=len(set(v['images']))) for k,v in cal['per_class'].items()},monitor_first=first,monitor_last=last,monitor_all=curve['monitor'],last50_losses={k:st.mean(v[k] for v in train) for k in ['pf','preserve','refine','quality','identity','teacher_matches']},unknown_matching_output_score_quantiles=quantiles(tp_scores),unknown_other_output_score_quantiles=quantiles(fp_scores))
   evidence.append(row);print('Analyzed',scene,seed,'GT',counts['unknown_gt'],'known only',counts['unknown_gt_known_only'],'no final box',counts['unknown_gt_no_final_box'],flush=True)
  task2=[]
  for scene in ['Aquatic','Aerial','Game','Medical','Surgical']:
   rr=[r for r in results if r['scene']==scene and r['task']=='task2']
   task2.append(dict(scene=scene,seeds=[r['seed'] for r in rr],metrics={p:{k:meanstd([r['evaluations'][p][k] for r in rr]) for k in ['PK_AP50','CK_AP50']} for p in ['terminal','selected']}))
  macro={p:{k:meanstd([st.mean(r['evaluations'][p][k] for r in results if r['task']=='task2' and r['seed']==seed) for seed in [42,43,44]]) for k in ['PK_AP50','CK_AP50']} for p in ['terminal','selected']}
  out=dict(integrity=integrity,remote_source_ast_matches_local=codecheck,completed_formal_runs=len(results),evidence_scope='Final predictions, training/monitor curves and calibration only; pre-filter candidates/logits not stored. IoU >=.5 nonexclusive final-box coverage diagnostic, not official one-to-one AP recall.',task1=evidence,task2=task2,task2_macro=macro)
  (OUT/'BUNDLE_DIAGNOSIS.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
  print('Task2',json.dumps(task2));print('Macro',json.dumps(macro));print('Code',codecheck);print('Seconds',time.monotonic()-start)
if __name__=='__main__':main()
