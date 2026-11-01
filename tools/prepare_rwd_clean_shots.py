"""Select clean known train ROIs, excluding exact encoded test-image duplicates."""
from pathlib import Path
import json,hashlib,collections,random,xml.etree.ElementTree as ET,runpy
ROOT=Path(__file__).resolve().parents[1]
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if json.loads(p.read_text(encoding='utf8'))!=v:raise RuntimeError('Clean selection changed; use a new tag: '+str(p))
  return
 p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def run(r=ROOT):
 r=Path(r);base=r/'assets/rwd';dataset=json.loads((base/'DATASET_MANIFEST.json').read_text());out=r/'prepared/rwd_task1/clean_shots_v1';visible=runpy.run_path(str(r/'tools/refine_rwd_training_rois.py'),run_name='visible_roi')['visible'];files={};rows=[]
 for scene in dataset['scenes']:
  name=scene['directory'];meta=base/'ImageSets'/name;train=(meta/'train.txt').read_text().splitlines();test=(meta/'test.txt').read_text().splitlines();known=(meta/'known_classnames.txt').read_text().splitlines();g=json.loads((meta/'groups.json').read_text());dims=g['dimensions']
  testsha={dataset['files'][f'JPEGImages/{name}/{n}']['sha256'] for n in test};excluded=[n for n in train if dataset['files'][f'JPEGImages/{name}/{n}']['sha256'] in testsha];allowed=set(train)-set(excluded);byclass={c:[] for c in known}
  for n in train:
   if n not in allowed:continue
   w,h=dims[n];x=ET.parse(base/'Annotations'/name/(Path(n).stem+'.xml')).getroot()
   for j,o in enumerate(x.findall('object')):
    c=o.findtext('name')
    if c not in byclass or int(o.findtext('difficult','0')):continue
    b=[float(o.find('bndbox').findtext(k))-1 for k in ['xmin','ymin','xmax','ymax']];b=visible(b,w,h)
    if b is not None:byclass[c].append([n,[v/s for v,s in zip(b,[w,h,w,h])]])
  selected={};rng=random.Random(20261023)
  for c,objects in byclass.items():
   by=collections.defaultdict(list)
   for obj in objects:by[obj[0]].append(obj)
   names=sorted(by);rng.shuffle(names);picked=[]
   while names and len(picked)<100:
    nxt=[]
    for n in names:
     if len(picked)<100:picked.append(by[n].pop(0))
     if by[n]:nxt.append(n)
    names=nxt
   selected[c]=picked
  if name=='Aquatic':selected=json.loads((r/'assets/inputs/task1_shots.json').read_text())
  assert all(n in allowed and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1 for rs in selected.values() for n,b in rs)
  folder=out/name
  for fn,val in [('task1_shots_100.json',selected),('EXCLUDED_EXACT_TEST_DUPLICATES.json',excluded)]:
   p=folder/fn;put(p,val);files[p.relative_to(r).as_posix()]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  rows.append(dict(scene=name,excluded_training_images=len(excluded),selected_shots=sum(len(x) for x in selected.values()),below100={c:len(x) for c,x in selected.items() if len(x)<100},selected_train_test_exact_hash_overlap=0))
 m=dict(version='clean_shots_v1',dataset_identity=dataset['identity'],files=files,scenes=rows,policy='Freeze official outer train/test files. For custom clean training, remove train images with exact encoded SHA appearing in test, using no test annotation labels. Rebuild known visible-ROI shots with the same fixed selection rule. Aquatic original400 unchanged. This is a documented clean variant, not an exact CODE training sample reproduction.',internal_fit_cal_dev_split_created=False)
 m['identity']=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest();put(out/'MANIFEST.json',m);print('Clean selection reproduced:',m['identity']);return m
if __name__=='__main__':run()
