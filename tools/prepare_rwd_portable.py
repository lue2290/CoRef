"""Build portable RWD Task1 data in-place on E, without duplicate image copies."""
from pathlib import Path,PurePosixPath
import json,hashlib,os,zipfile,shutil,collections,xml.etree.ElementTree as ET,random,re,time,math
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
RAW=Path(r'E:\deeplearn\data\RWD\ROOT')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);payload=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf8');put(p,payload)
def put(p,payload):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=payload:raise RuntimeError('Refuse overwrite of different data: '+str(p))
  return
 tmp=p.with_name(p.name+'.tmp')
 if tmp.exists():raise RuntimeError('Unresolved temporary file: '+str(tmp))
 with tmp.open('xb') as f:f.write(payload)
 os.replace(tmp,p)
def safe_name(name):
 if not name or Path(name).name!=name or '/' in name or chr(92) in name or name in ('.','..'):raise ValueError('Unsafe filename: '+name)
 return name

def coco_xml(image,anns,categories):
 root=ET.Element('annotation');ET.SubElement(root,'filename').text=safe_name(image['file_name'])
 size=ET.SubElement(root,'size')
 for key,value in [('width',image['width']),('height',image['height']),('depth',3)]:ET.SubElement(size,key).text=str(value)
 for a in anns:
  x,y,w,h=map(float,a['bbox'])
  if not all(math.isfinite(v) for v in [x,y,w,h]):raise ValueError('Nonfinite COCO box')
  obj=ET.SubElement(root,'object');ET.SubElement(obj,'name').text=categories[a['category_id']];ET.SubElement(obj,'difficult').text='0';b=ET.SubElement(obj,'bndbox')
  # Exact frozen FOMO COCO converter convention. No clipping or class aliases.
  for k,v in zip(['xmin','ymin','xmax','ymax'],[x+1,y+1,x+w+1,y+h+1]):ET.SubElement(b,k).text=str(int(v))
 return ET.tostring(root,encoding='utf8')
def surgical_xml(payload):
 src=ET.fromstring(payload);root=ET.Element('annotation');ET.SubElement(root,'filename').text=src.findtext('filename').split()[0];size=ET.SubElement(root,'size')
 for key,path in [('width','imagesize/ncols'),('height','imagesize/nrows')]:ET.SubElement(size,key).text=src.findtext(path)
 ET.SubElement(size,'depth').text='3'
 for old in src.findall('.//object'):
  pts=[(int(p.findtext('x')),int(p.findtext('y'))) for p in old.findall('polygon/pt')]
  if not pts:raise ValueError('Empty surgical polygon')
  xs,ys=zip(*pts);obj=ET.SubElement(root,'object');ET.SubElement(obj,'name').text=old.findtext('name');ET.SubElement(obj,'difficult').text='0';box=ET.SubElement(obj,'bndbox')
  for k,v in zip(['xmin','ymin','xmax','ymax'],[min(xs),min(ys),max(xs),max(ys)]):ET.SubElement(box,k).text=str(v)
 return ET.tostring(root,encoding='utf8')
def source_group(scene,name):
 stem=Path(name).stem
 if scene=='Surgical':return re.split(r'_frame',stem,flags=re.I)[0]
 if scene=='Aerial':return 'image_'+stem
 return re.split(r'\.rf\.[0-9a-f]+',stem,flags=re.I)[0]
def link_checked(source,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  if sha(source)!=sha(target):raise RuntimeError('Existing link payload differs: '+str(target))
 elif source.drive.lower()==target.drive.lower():os.link(source,target)
 else:raise RuntimeError('Cross-drive data copy is disabled')

def build(root=ROOT,raw=RAW):
 root=Path(root);raw=Path(raw);dest=root/'assets/rwd';report=root/'reports/h_five_scene/data_prepare_v1';report.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(root).free<12*2**30:raise RuntimeError('Need at least 12GiB free before preparation')
 started=time.monotonic();all_files={};scene_rows=[];raw_sources={}
 scenes=[('Aquatic','Aquatic'),('Aerial','Aerial'),('Game','Game'),('Medical','Medical'),('Surgery','Surgical')]
 for canonical,scene in scenes:
  meta=root/'assets/rwd_protocol/task1'/canonical;train=(meta/'train.txt').read_text().splitlines();test=(meta/'test.txt').read_text().splitlines();train=[safe_name(x.strip()) for x in train if x.strip()];test=[safe_name(x.strip()) for x in test if x.strip()]
  if len(train)!=len(set(train)) or len(test)!=len(set(test)) or set(train)&set(test):raise RuntimeError('Split overlap/duplicates')
  wanted=set(train+test);imgdir=dest/'JPEGImages'/scene;anndir=dest/'Annotations'/scene;setdir=dest/'ImageSets'/scene
  known=(meta/'known_classnames.txt').read_text().strip().splitlines();archive_paths=[];extracted=0;linked=0;cache_dimensions={}
  if scene=='Aquatic':
   for n in sorted(wanted):
    link_checked(root/'assets/data/images'/n,imgdir/n);link_checked(root/'assets/data/annotations'/(Path(n).stem+'.xml'),anndir/(Path(n).stem+'.xml'));linked+=1
  else:
   if scene=='Aerial':archive_paths=[raw/scene/n for n in ['JPEGImages-test.zip','JPEGImages-trainval.zip','Annotations.zip']]
   elif scene=='Game':archive_paths=[raw/scene/'team fight tactics.v3i.coco.zip']
   elif scene=='Medical':archive_paths=[raw/scene/'x-ray rheumatology.v2-release.coco.zip']
   else:archive_paths=[raw/scene/'NeuroSurgicalToolsDataset.zip']
   for archive in archive_paths:
    stamp=(archive.stat().st_size,archive.stat().st_mtime_ns);raw_sources[str(archive)]=dict(bytes=stamp[0],sha256=sha(archive))
    with zipfile.ZipFile(archive) as z:
     for member in z.infolist():
      if member.is_dir():continue
      n=PurePosixPath(member.filename).name;ext=Path(n).suffix.lower()
      if n in wanted and ext in ('.jpg','.jpeg','.png'):
       put(imgdir/n,z.read(member));extracted+=1
      elif ext=='.xml' and (Path(n).stem+'.png' in wanted if scene=='Surgical' else Path(n).stem+'.jpg' in wanted):
       if scene=='Aerial' and '/Horizontal Bounding Boxes/' not in member.filename:continue
       value=z.read(member);put(anndir/n,surgical_xml(value) if scene=='Surgical' else value)
      elif scene in ('Game','Medical') and n=='_annotations.coco.json':
       data=json.loads(z.read(member));by=collections.defaultdict(list);categories={a['id']:a['name'] for a in data['categories']}
       for a in data['annotations']:by[a['image_id']].append(a)
       for im in data['images']:
        if im['file_name'] in wanted:put(anndir/(Path(im['file_name']).stem+'.xml'),coco_xml(im,by[im['id']],categories))
    if stamp!=(archive.stat().st_size,archive.stat().st_mtime_ns):raise RuntimeError('Archive changed during build')
  # Every required image is fully decoded; expected dimensions must match XML.
  eligible={c:[] for c in known};invalid=[];groups={};dims={};class_names=set();pixel_hash_inputs=collections.defaultdict(list)
  for i,n in enumerate(sorted(wanted),1):
   image=imgdir/n;annotation=anndir/(Path(n).stem+'.xml')
   if not image.is_file() or not annotation.is_file():raise RuntimeError('Missing converted pair: '+n)
   with Image.open(image) as im:
    im.load();w,h=im.size;orientation=im.getexif().get(274,1)
    if orientation not in (None,1):raise RuntimeError('Nontrivial EXIF orientation requires explicit conversion: '+n)
   tree=ET.parse(annotation).getroot();size=tree.find('size')
   if size is None or (int(size.findtext('width')),int(size.findtext('height')))!=(w,h):raise RuntimeError('Image/XML dimensions differ: '+n)
   dims[n]=[w,h];groups[n]=source_group(scene,n)
   for j,obj in enumerate(tree.findall('object')):
    label=obj.findtext('name');class_names.add(label)
    if n not in train or label not in eligible:continue
    bb=obj.find('bndbox');box=[float(bb.findtext(k))-1 for k in ['xmin','ymin','xmax','ymax']]
    if not all(math.isfinite(v) for v in box) or not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):invalid.append(dict(image=n,object_index=j,class_name=label,box=box,dimensions=[w,h]));continue
    if int(obj.findtext('difficult','0')):continue
    eligible[label].append(dict(image=n,box=box,object_index=j,group=groups[n]))
   for p in [image,annotation]:all_files[p.relative_to(dest).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
   pixel_hash_inputs[all_files[image.relative_to(dest).as_posix()]['sha256']].append(n)
   if i%100==0 or i==len(wanted):print(f'\r{scene} full image decode and coordinate check {i}/{len(wanted)} | elapsed {time.monotonic()-started:.0f}s'.ljust(145),end='',flush=True)
  print()
  # Exact duplicate images share a group. Filename groups remain conservative, not confirmed acquisition IDs.
  parent={g:g for g in set(groups.values())}
  def find(g):
   while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
   return g
  for names in pixel_hash_inputs.values():
   roots=sorted({find(groups[n]) for n in names})
   for g in roots[1:]:parent[g]=roots[0]
  groups={n:find(g) for n,g in groups.items()}
  # Stratify by image to avoid taking 100 adjacent objects from one image.
  selected={};counts={};rng=random.Random(20261023)
  for label,objects in eligible.items():
   by=collections.defaultdict(list)
   for obj in objects:by[obj['image']].append(obj)
   names=sorted(by);rng.shuffle(names);picked=[]
   while len(picked)<100 and names:
    next_names=[]
    for n in names:
     if len(picked)<100:picked.append(by[n].pop(0))
     if by[n]:next_names.append(n)
    names=next_names
   selected[label]=[[o['image'],[v/s for v,s in zip(o['box'],dims[o['image']]*2)]] for o in picked]
   counts[label]=dict(eligible_train_objects=len(objects),selected=len(picked),selected_images=len({o['image'] for o in picked}),selected_groups=len({groups[o['image']] for o in picked}))
  if scene=='Aquatic':
   # Existing audited 400-shot inputs must stay exact, including order and normalization.
   selected=json.loads((root/'assets/inputs/task1_shots.json').read_text())
   for label,rows in selected.items():
    for n,box in rows:
     if n not in train or not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise RuntimeError('Historical Aquatic shot invalid')
    counts[label].update(selected=len(rows),selected_images=len({x[0] for x in rows}),selected_groups=len({groups[x[0]] for x in rows}))
  put(setdir/'train.txt',(meta/'train.txt').read_bytes());put(setdir/'test.txt',(meta/'test.txt').read_bytes());put(setdir/'known_classnames.txt',(meta/'known_classnames.txt').read_bytes())
  put(setdir/'known_shots_upstream_reference.json',(meta/'known_shots_upstream.json').read_bytes())
  write_json(setdir/'task1_shots_100.json',selected)
  write_json(setdir/'groups.json',dict(groups=groups,dimensions=dims,method='Surgical video prefix; Roboflow original filename before .rf; Aerial image ID; exact encoded image SHA unions. Conservative heuristic, not certified independent acquisition groups.',official_train_test_group_overlap=sorted({groups[n] for n in train}&{groups[n] for n in test})))
  write_json(setdir/'training_box_exclusions.json',invalid)
  put(setdir/'classnames_present.txt',('\n'.join(sorted(class_names))+'\n').encode())
  row=dict(scene=canonical,directory=scene,train=len(train),test=len(test),fully_decoded_images=len(wanted),source_archive_sha256={p.name:raw_sources[str(p)]['sha256'] for p in archive_paths},known_counts=counts,training_boxes_excluded=len(invalid),train_source_groups=len({groups[n] for n in train}),test_source_groups=len({groups[n] for n in test}),official_split_group_overlap=len({groups[n] for n in train}&{groups[n] for n in test}),image_files_linked=linked,image_files_extracted=extracted,internal_fit_cal_dev_split_created=False,selection_policy='Aquatic exact existing audited shots; other scenes deterministic image-round-robin, seed20261023, up to100 valid known train XML objects/class; no replacement, no test label selection. Original upstream references retained separately. This is a documented local protocol, not exact CODE shot equivalence.')
  scene_rows.append(row)
  for p in setdir.iterdir():
   if p.is_file():all_files[p.relative_to(dest).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
  write_json(report/(canonical+'.json'),row)
  if shutil.disk_usage(root).free<8*2**30:raise RuntimeError('Less than 8GiB free; stop before next scene')
 manifest=dict(version='rwd_portable_task1_v1',scenes=scene_rows,files=all_files,conversion=dict(coco='Frozen FOMO int(x+1), int(y+1), int(x+w+1), int(y+h+1), difficult0; preserve all annotation categories',surgical='Frozen FOMO polygon min/max, integer coordinates unchanged, difficult0',aerial='Original DIOR horizontal-box XML bytes preserved',training='XML coordinates minus1, known train only, finite positive in-bounds; exclusions recorded, no clipping'),training_ready=False,scope='Portable data integrity and official outer splits; source-group audit and internal fit/cal/dev splitting still require protocol confirmation; no model training or test AP')
 manifest['identity']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest();write_json(dest/'DATASET_MANIFEST.json',manifest)
 write_json(report/'RAW_SOURCES.json',raw_sources);write_json(report/'RESULT.json',dict(complete=True,identity=manifest['identity'],dataset=str(dest),files=len(all_files),logical_gib=sum(x['bytes'] for x in all_files.values())/2**30,scenes=scene_rows,elapsed_seconds=time.monotonic()-started,training_updates=0,new_checkpoints=0,training_ready=False))
 print(json.dumps(dict(complete=True,dataset=str(dest),identity=manifest['identity'],files=len(all_files),logical_gib=sum(x['bytes'] for x in all_files.values())/2**30,elapsed_seconds=time.monotonic()-started),indent=2))
 return manifest
if __name__=='__main__':build()
