from pathlib import Path
from collections import OrderedDict,defaultdict
import gc
import numpy as np
import torch
from PIL import Image,ImageOps
from cil.common import iou
from jointdet.geometry import xyxy,cxcywh
from current.progress import track,stage
from .protocol import read,write,sha,fp,lock,verify_file,locations,guard


def candidate_folder(root,cfg,case):return locations(root,cfg,case['scene'])[0]/'candidates'


@torch.inference_mode()
def native_candidates(pf,view,name,c,device):
    from jointdet.model import pixels,make_sample,pf_score
    x,d,meta=pixels(view,name,c,device);del d
    sample,_=make_sample(meta,[],device)
    out=pf.forward_transformer(pf.extract_feat(x),{},[sample])
    cls,boxes,ctr=pf.bbox_head(out['hidden_states'],out['references'],out['memory_text'],out['text_token_mask'])
    score=pf_score(cls[-1,0],ctr[-1,0,:,0]);ids=score.argsort(descending=True,stable=True)[:600]
    if len(ids)!=600:raise RuntimeError('160 requires600 real native PF candidates')
    boxes=xyxy(boxes[-1,0,ids]).clamp(0,1)
    wh=boxes.new_tensor([meta['ori_shape'][1],meta['ori_shape'][0]]*2)
    return dict(boxes=(boxes*wh).cpu().numpy().astype(np.float32),g=(score[ids]/2).cpu().numpy().astype(np.float32),
        proposal_ids=ids.cpu().numpy(),size=np.array([meta['ori_shape'][1],meta['ori_shape'][0]],np.int32))


def candidate_cache(root,cfg,case,pf,view,c,device,dataset,verified,extract_fn=None):
    extract_fn=extract_fn or native_candidates;folder=candidate_folder(root,cfg,case)
    record=lock(folder/'LOCK.json',dict(input=case['identity'],pf_weights=read(Path(root)/'MAINLINE_PARENT_INDEX.json')['base_weights'][c['pf_weights']],
        policy='native frozen PF; top600 pf_score stable descending; g=pf_score/2; no learned refinement, feedback, quality or NMS'))
    roles={name:role for role in ['fit','cal','monitor','test'] for name in case['spec'][role+'_images']}
    # Test images are used without opening their annotations in preparation.
    rows=[]
    for name,role in track(roles.items(),case['scene']+' shared frozen PF candidates',unit='images'):
        image=Path(root)/'assets/rwd/JPEGImages'/case['scene']/name
        ih=verify_file(image,dataset['files'][f'JPEGImages/{case["scene"]}/{name}']['sha256'],verified)
        path=folder/(fp(name)[:24]+'.npz');done=path.with_suffix('.json')
        binding=dict(cache=record['identity'],image=name,image_sha256=ih,role=role)
        if path.exists() and done.exists():
            row=read(done)
            if row['binding']!=binding:raise RuntimeError('PF candidate input identity changed')
            verify_file(path,row['sha256'],verified)
        else:
            guard(root,cfg);value=extract_fn(pf,view,name,c,device)
            if value['boxes'].shape!=(600,4) or len(value['g'])!=600 or not np.isfinite(value['boxes']).all() or not np.isfinite(value['g']).all():
                raise RuntimeError('Invalid native PF candidates')
            tmp=path.with_suffix('.npz.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,**value)
            tmp.replace(path);row=dict(binding=binding,file=path.name,sha256=sha(path));write(done,row)
        rows.append(row)
    result=dict(identity=record['identity'],rows=rows,files={r['file']:r['sha256'] for r in rows})
    path=folder/'MANIFEST.json'
    if path.exists() and read(path)!=result:raise RuntimeError('PF manifest changed')
    write(path,result)
    return folder


def candidates(folder,name):
    with np.load(Path(folder)/(fp(name)[:24]+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def build_plans(root,cfg,case,folder):
    s=case['spec'];by={}
    for obj in s['objects']:by.setdefault(obj['image'],[]).append(obj)
    class_images={cid:sorted({o['image'] for o in s['fit_objects'] if o['class_id']==cid}) for cid in range(len(s['classes']))}
    if any(not n for n in class_images.values()):raise RuntimeError('No legal fit reference for a known class')
    pair_cache={}
    for name in s['fit_images']:
        a=candidates(folder,name);matrix=iou(a['boxes'],[o['box'] for o in by[name]])
        pair_cache[name]=np.stack(np.where(matrix>=.5),1)
    plans={}
    for seed in cfg['seeds']:
        path=locations(root,cfg,case['scene'])[0]/'plans'/f'seed_{seed}.json';rng=np.random.default_rng(seed);steps=[]
        for step in range(1,cfg['updates']+1):
            micro=[]
            for _ in range(cfg['accumulate']):
                cid=int(rng.integers(len(s['classes'])));name=str(rng.choice(class_images[cid]));flip=bool(rng.integers(2));objects=by[name]
                queries=[dict(kind='gt',index=j,label=o['class_id'],instance_id=o['id']) for j,o in enumerate(objects)]
                pairs=pair_cache[name]
                if len(pairs):
                    for a,b in pairs[rng.permutation(len(pairs))[:16]]:
                        queries.append(dict(kind='pred',index=int(a),label=objects[b]['class_id'],matched_instance_id=objects[b]['id']))
                if len(queries)>32:queries=[queries[int(i)] for i in rng.choice(len(queries),32,replace=False)]
                seen=sorted({q['label'] for q in queries});held=int(rng.choice(seen));available=[c for c in range(len(s['classes'])) if c!=held]
                count=int(rng.integers(1,len(available)+1));support=sorted(map(int,rng.choice(available,count,replace=False)))
                micro.append(dict(image=name,flip=flip,queries=queries,episode=dict(held_class=held,support_classes=support)))
            steps.append(dict(step=step,microbatches=micro))
        plans[seed]=lock(path,dict(input=case['identity'],candidate_manifest_sha256=sha(Path(folder)/'MANIFEST.json'),seed=seed,steps=steps,
            augmentation='fixed native PF candidates/GT mirrored with the image; frozen OWL re-encodes mirrored pixels'))
    return plans


def feature_needs(spec,plans):
    needs={}
    for role in ['fit','cal','monitor']:
        for name in spec[role+'_images']:needs[name,False]=dict(role=role,pred=set(range(600)) if role=='cal' else set())
    for plan in plans.values():
        for row in plan['steps']:
            for micro in row['microbatches']:
                key=micro['image'],micro['flip'];item=needs.setdefault(key,dict(role='fit',pred=set()))
                item['pred'].update(q['index'] for q in micro['queries'] if q['kind']=='pred')
    return needs


def mirror(boxes,width):
    result=np.asarray(boxes,np.float32).reshape(-1,4).copy();result[:,[0,2]]=width-result[:,[2,0]]
    return result


def prepare_features(root,cfg,case,folder,plans,visual,processor,device,dataset,verified):
    from .model import read_regions
    spec=case['spec'];by={}
    for obj in spec['objects']:by.setdefault(obj['image'],[]).append(obj)
    out=locations(root,cfg,case['scene'])[0]/'features'
    binding=lock(out/'LOCK.json',dict(input=case['identity'],candidate_manifest=sha(Path(folder)/'MANIFEST.json'),
        plans={str(k):v['identity'] for k,v in plans.items()},owl_files=cfg['owl_files'],feature_layer=cfg['feature_layer'],
        context_scale=cfg['context_scale'],channels=visual.channels,raw_dim=visual.channels*5,dtype='float32'))
    rows=[];needs=feature_needs(spec,plans)
    for (name,flip),need in track(sorted(needs.items()),case['scene']+' shared OWL training/calibration ROI features',unit='views'):
        image=Path(root)/'assets/rwd/JPEGImages'/case['scene']/name
        ih=verify_file(image,dataset['files'][f'JPEGImages/{case["scene"]}/{name}']['sha256'],verified)
        slots=sorted(need['pred']);path=out/(fp([name,flip])[:24]+'.npz');done=path.with_suffix('.json')
        record=dict(cache=binding['identity'],image=name,flip=flip,image_sha256=ih,pred_slots=slots,role=need['role'])
        if path.exists() and done.exists():
            row=read(done)
            if row['binding']!=record:raise RuntimeError('OWL feature view binding changed')
            verify_file(path,row['sha256'],verified)
        else:
            guard(root,cfg)
            with Image.open(image) as raw:im=ImageOps.exif_transpose(raw).convert('RGB')
            if flip:im=ImageOps.mirror(im)
            pixels=processor(images=im,return_tensors='pt')['pixel_values'].to(device)
            with torch.inference_mode():
                fmap=visual(pixels);a=candidates(folder,name);w,h=a['size'];objects=by[name]
                gt=np.asarray([o['box'] for o in objects],np.float32).reshape(-1,4);pred=a['boxes'][slots]
                if flip:gt=mirror(gt,w);pred=mirror(pred,w)
                normalized=np.concatenate([gt,pred])/np.array([w,h,w,h],np.float32)
                raw=read_regions(fmap,cxcywh(torch.tensor(normalized,device=device)),cfg['context_scale']).cpu().numpy().astype(np.float32)
            if raw.shape!=(len(gt)+len(slots),visual.channels*5) or not np.isfinite(raw).all():raise RuntimeError('Invalid OWL ROI features')
            payload=dict(gt_raw=raw[:len(gt)],pred_raw=raw[len(gt):],pred_slots=np.asarray(slots,np.int64),
                labels=np.asarray([o['class_id'] for o in objects],np.int64),ids=np.array([o['id'] for o in objects]))
            tmp=path.with_suffix('.npz.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,**payload)
            tmp.replace(path);row=dict(binding=record,file=path.name,sha256=sha(path));write(done,row)
            del fmap,pixels,raw
        rows.append(row)
    result=dict(identity=binding['identity'],channels=visual.channels,raw_dim=visual.channels*5,rows=rows,files={r['file']:r['sha256'] for r in rows})
    path=out/'MANIFEST.json'
    if path.exists() and read(path)!=result:raise RuntimeError('OWL feature manifest changed')
    write(path,result)
    return out


class Data:
    def __init__(self,case,cfg,folder,feature_folder,device):
        self.case=case;self.spec=case['spec'];self.cfg=cfg;self.candidates=Path(folder);self.folder=Path(feature_folder);self.device=device
        self.manifest=read(self.folder/'MANIFEST.json');self.raw_dim=self.manifest['raw_dim'];self.rows={(r['binding']['image'],r['binding']['flip']):r for r in self.manifest['rows']}
        self.lru=OrderedDict();self.lru_bytes=0
        raw=[];labels=[];groups=[];ids=[]
        for name in self.spec['fit_images']:
            a=self.view(name,False);raw.append(a['gt_raw']);labels.extend(a['labels'].tolist());groups.extend([self.spec['groups'][name]]*len(a['labels']));ids.extend(a['ids'].tolist())
        self.bank=dict(raw=torch.tensor(np.concatenate(raw),device=device),labels=torch.tensor(labels,device=device,dtype=torch.long),groups=groups,ids=ids)
        if set(labels)!=set(range(len(self.spec['classes']))):raise RuntimeError('Missing full OWL fit reference class')

    def view(self,name,flip=False):
        key=name,flip
        if key in self.lru:
            result=self.lru.pop(key);self.lru[key]=result;return result
        row=self.rows[key]
        with np.load(self.folder/row['file'],allow_pickle=False) as z:result={k:z[k].copy() for k in z.files}
        size=sum(v.nbytes for v in result.values());self.lru[key]=result;self.lru_bytes+=size
        while len(self.lru)>1 and self.lru_bytes>self.cfg['lru_cache_mib']*2**20:
            _,old=self.lru.popitem(last=False);self.lru_bytes-=sum(v.nbytes for v in old.values())
        return result

    def query(self,micro):
        a=self.view(micro['image'],micro['flip']);where={int(v):j for j,v in enumerate(a['pred_slots'])};raw=[];labels=[]
        for q in micro['queries']:
            raw.append(a['gt_raw'][q['index']] if q['kind']=='gt' else a['pred_raw'][where[q['index']]]);labels.append(q['label'])
        raw=torch.tensor(np.asarray(raw),device=self.device);y=torch.tensor(labels,device=self.device,dtype=torch.long)
        group=self.spec['groups'][micro['image']];allowed=torch.tensor([g!=group for g in self.bank['groups']],device=self.device)[None].expand(len(raw),-1)
        return raw,y,allowed
