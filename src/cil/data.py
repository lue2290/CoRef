from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageOps
from .common import TASK_CLASSES, digest, fingerprint, iou, names, read_json, stable_seed, write_json

def image_source(name):
    stem = re.split(r"\.rf\.[0-9a-f]+", name, flags=re.I)[0]
    stem = re.sub(r"[_-](?:frame|imgframe)[_-]?\d+.*$", "", stem, flags=re.I)
    return stem

def group_images(root, image_names, overrides):
    parent = {n:n for n in image_names}
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    def union(a,b):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    thumbnails, hashes, dimensions = {}, {}, {}
    sources = defaultdict(list)
    for n in sorted(image_names):
        with Image.open(Path(root)/"assets/data/images"/n) as raw:
            im=ImageOps.exif_transpose(raw).convert("RGB")
            dimensions[n]=list(im.size)
            small=np.asarray(im.resize((64,64)),np.float32)
            gray=np.asarray(im.convert("L").resize((9,8)),np.int16)
            h=0
            for bit in (gray[:,1:]>gray[:,:-1]).flatten():h=(h<<1)|int(bit)
            hashes[n]=h
            thumbnails[n]=small
        sources[overrides.get(n,image_source(n))].append(n)
    for group in sources.values():
        for n in group[1:]:union(group[0],n)
    candidates=[]
    ordered=sorted(image_names)
    for j,n in enumerate(ordered):
        for other in ordered[:j]:
            dist=(hashes[n]^hashes[other]).bit_count()
            if dist>5:continue
            mae=float(np.abs(thumbnails[n]-thumbnails[other]).mean())
            if dist<=3 and mae<=6:
                union(n,other)
            elif mae<=20:
                candidates.append({"a":n,"b":other,"dhash_distance":dist,"thumbnail_mae":mae})
    roots=sorted({find(n) for n in ordered})
    group_id={n:f"source_{roots.index(find(n)):04d}" for n in ordered}
    return group_id,dimensions,candidates

def prepare(root, task, cfg):
    root=Path(root)
    out=root/"prepared"/task
    raw=read_json(root/f"assets/inputs/{task}_shots.json")
    classes=TASK_CLASSES[task]
    train=names(root/"assets/inputs/train.txt")
    test=names(root/"assets/inputs/test.txt")
    if len(train)!=318 or len(test)!=319 or set(train)&set(test):
        raise ValueError("Changed Aquatic train/test split")
    objects=[]
    for cid,c in enumerate(classes):
        for j,(name,box) in enumerate(raw[c]):
            if name not in train or name in test:raise ValueError("Shot outside training set")
            if len(box)!=4 or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
                raise ValueError("Invalid normalized xyxy")
            objects.append({"id":f"{c}_{j:03d}","image":name,"class_id":cid,"normalized_box":box})
    if task=="task1" and Counter(o["class_id"] for o in objects)!=dict.fromkeys(range(4),100):
        raise ValueError("Task1 must use exactly the existing 400 shots")
    selected=sorted({o["image"] for o in objects})
    overrides=read_json(root/"configs/group_overrides.json")
    groups,dims,candidates=group_images(root,selected,overrides)
    for o in objects:
        w,h=dims[o["image"]]
        o["box"]=[float(v*s) for v,s in zip(o.pop("normalized_box"),[w,h,w,h])]
        o["group"]=groups[o["image"]]
    ids=sorted(set(groups.values()))
    cls_groups=[{o["group"] for o in objects if o["class_id"]==c} for c in range(len(classes))]
    perclass={c:{"objects":sum(o["class_id"]==j for o in objects),
                 "images":len({o["image"] for o in objects if o["class_id"]==j}),
                 "source_groups":len(cls_groups[j])} for j,c in enumerate(classes)}
    requested=cfg["min_train_groups"]+cfg["min_val_groups"]
    if any(len(g)<requested for g in cls_groups):
        write_json(out/"split_diagnostic.json",{"counts":perclass,"near_duplicate_candidates":candidates,
            "error":"Insufficient source groups for locked split targets; do not split a source group across partitions"})
        raise RuntimeError("Insufficient independent source groups. See prepared split_diagnostic.json; revise data protocol explicitly.")
    best=None
    rng=np.random.default_rng(cfg["split_seed"])
    for _ in range(20000):
        shuffled=rng.permutation(ids)
        n=max(1,int(round(len(ids)*cfg["train_fraction"])))
        fit=set(shuffled[:n])
        good=all(len(g&fit)>=cfg["min_train_groups"] and len(g-fit)>=cfg["min_val_groups"] for g in cls_groups)
        if not good:continue
        score=sum(abs(sum(o["group"] in fit for o in objects if o["class_id"]==c)/sum(o["class_id"]==c for o in objects)-cfg["train_fraction"]) for c in range(len(classes)))
        if best is None or score<best[0]:best=(score,fit)
    if best is None:raise RuntimeError("No source-group-stratified split satisfies the locked targets")
    fit=best[1]
    manifest={"task":task,"classes":classes,"objects":objects,"groups":groups,"dimensions":dims,
              "fit_images":[n for n in selected if groups[n] in fit],
              "val_images":[n for n in selected if groups[n] not in fit],
              "test_images":test,"fixed_train_images":train,"counts":perclass,
              "input_sha256":{f"assets/inputs/{task}_shots.json":digest(root/f"assets/inputs/{task}_shots.json"),
                              "assets/inputs/train.txt":digest(root/"assets/inputs/train.txt"),
                              "assets/inputs/test.txt":digest(root/"assets/inputs/test.txt")},
              "split_config":{k:cfg[k] for k in ["split_seed","train_fraction","min_train_groups","min_val_groups"]},
              "group_override_sha256":digest(root/"configs/group_overrides.json"),
              "group_method":"source filename + conservative near duplicate union; candidate list for manual review, not guaranteed scene/video independence"}
    manifest["sha256"]=fingerprint(manifest)
    existing=out/"manifest.json"
    if existing.exists() and read_json(existing)!=manifest:
        raise RuntimeError("Prepared manifest differs. Preserve the old run and use a new project version; do not overwrite splits.")
    write_json(existing,manifest)
    write_json(out/"near_duplicate_candidates.json",candidates)
    return {"task":task,"objects":len(objects),"images":len(selected),"fit_images":len(manifest["fit_images"]),
            "val_images":len(manifest["val_images"]),"source_groups":len(ids),"counts":perclass,"near_candidates":len(candidates)}

def manifest(root,task):
    p=Path(root)/"prepared"/task/"manifest.json"
    m=read_json(p)
    expected=m.pop("sha256")
    if fingerprint(m)!=expected:raise RuntimeError("Prepared manifest was edited")
    m["sha256"]=expected
    for key,value in m["input_sha256"].items():
        if digest(Path(root)/key)!=value:raise RuntimeError("Changed input: "+key)
    return m

def relevant_objects(m, formal, held=None):
    active=[i for i,c in enumerate(m["classes"]) if c!=held]
    if len(active)<3:raise ValueError("At least three training classes are required")
    allowed=set(m["fixed_train_images"] if formal else m["fit_images"])
    objects=[o for o in m["objects"] if o["class_id"] in active and o["image"] in allowed]
    if any(o["class_id"] not in active for o in objects):raise AssertionError("Held-out class leaked")
    return objects,active

def proposal(root,name,split="train"):
    with np.load(Path(root)/"assets/proposals"/split/(Path(name).stem+".npz"),allow_pickle=False) as f:
        boxes=f["boxes"].astype(np.float32)
        scores=f["scores"].astype(np.float32)
    if len(boxes)>300 or boxes.shape!=(len(scores),4):raise ValueError("Invalid fixed PF cache")
    return boxes,scores

def semantic_valid(box,gt,cfg):
    box,gt=np.asarray(box),np.asarray(gt)
    wh=np.maximum(np.minimum(box[2:],gt[2:])-np.maximum(box[:2],gt[:2]),0)
    inter=float(wh.prod())
    return bool(iou([box],[gt])[0,0]>=cfg["semantic_iou"] and
                inter/max(float((gt[2:]-gt[:2]).prod()),1e-9)>=cfg["semantic_coverage"] and
                inter/max(float((box[2:]-box[:2]).prod()),1e-9)>=cfg["semantic_purity"])

def geometry_for_object(obj, all_available_objects, dimensions, cfg):
    """Only passed-in allowed labels can affect sampling; outer held class is absent."""
    gt=np.array(obj["box"],np.float32)
    others=np.array([o["box"] for o in all_available_objects],np.float32)
    owner=[o["id"] for o in all_available_objects].index(obj["id"])
    rng=np.random.default_rng(stable_seed(cfg["split_seed"],obj["id"],"geometry"))
    w,h=dimensions
    center=(gt[:2]+gt[2:])/2
    size=gt[2:]-gt[:2]
    rows=[{"box":gt.tolist(),"quality":1.0,"reliable":True,"owner":obj["id"],"class_id":obj["class_id"],"kind":"anchor"}]
    for lo,hi in [(0.75,1.000001),(0.5,0.75),(0.2,0.5),(0,0.2)]:
        accepted=[]
        for _ in range(cfg["geometry_attempts"]):
            # Target the requested bin, then accept only the ACTUAL clipped IoU.
            # A shared wide log-uniform sampler starved the reliable mild pairs.
            if lo>=0.75:
                scale=np.exp(rng.uniform(-0.06,0.06,2))
                shift=rng.uniform(-0.06,0.06,2)*size
            elif lo>=0.5:
                scale=np.exp(rng.uniform(-0.16,0.16,2))
                shift=rng.uniform(0.08,0.22,2)*rng.choice([-1,1],size=2)*size
            elif lo>=0.2:
                scale=np.exp(rng.uniform(-0.1,0.1,2))*np.sqrt(rng.uniform(0.22,0.48))
                shift=rng.uniform(-0.1,0.1,2)*size
            else:
                scale=rng.uniform(0.08,0.4,2)
                shift=rng.uniform(-0.3,0.3,2)*size
            b=np.r_[center+shift-size*scale/2,center+shift+size*scale/2]
            b=np.maximum(0,np.minimum(b,[w,h,w,h])).astype(np.float32)
            if np.any(b[2:]-b[:2]<1):continue
            ctr=(b[:2]+b[2:])/2
            if not np.all((ctr>=gt[:2])&(ctr<=gt[2:])):continue
            ov=iou([b],others)[0]
            q=float(ov[owner])
            if not lo<=q<hi or int(ov.argmax())!=owner:continue
            if any(np.allclose(b,x["box"],atol=1e-4) for x in accepted):continue
            accepted.append({"box":b.tolist(),"quality":q,"reliable":semantic_valid(b,gt,cfg),
                             "owner":obj["id"],"class_id":obj["class_id"],"kind":"geometry"})
            if len(accepted)>=cfg["variants_per_bin"]:break
        rows+=accepted
    return rows

class TrainingData:
    def __init__(self,root,m,cfg,formal=False,held=None):
        self.root,self.m,self.cfg=Path(root),m,cfg
        self.objects,self.active=relevant_objects(m,formal,held)
        self.by_image=defaultdict(list)
        for o in self.objects:self.by_image[o["image"]].append(o)
        self.images=sorted(self.by_image)
        self.class_images={c:sorted({o["image"] for o in self.objects if o["class_id"]==c}) for c in self.active}
        geometry_keys=["split_seed","variants_per_bin","geometry_attempts","semantic_iou","semantic_coverage","semantic_purity"]
        cache_key=fingerprint({"geometry_recipe":"bin_targeted_v2","objects":self.objects,"dimensions":{n:m["dimensions"][n] for n in self.images},
                               "config":{k:cfg[k] for k in geometry_keys}})
        cache=Path(root)/"prepared"/m["task"]/"geometry"/(cache_key+".json") if "task" in m else None
        if cache is not None and cache.exists():
            saved=read_json(cache)
            self.geometry=saved["geometry"]
            if saved["inputs_sha256"]!=cache_key or fingerprint(self.geometry)!=saved["geometry_sha256"]:
                raise RuntimeError("Geometric sample cache changed")
        else:
            self.geometry={o["id"]:geometry_for_object(o,self.by_image[o["image"]],m["dimensions"][o["image"]],cfg) for o in self.objects}
            if cache is not None:write_json(cache,{"inputs_sha256":cache_key,"geometry_sha256":fingerprint(self.geometry),"geometry":self.geometry})
        self.geometry_hash=fingerprint(self.geometry)

    def sample(self,seed,micro):
        rng=np.random.default_rng(stable_seed(seed,micro,"query"))
        c=int(rng.choice(self.active))
        name=str(rng.choice(self.class_images[c]))
        available=self.by_image[name]
        target=[o for o in available if o["class_id"]==c]
        chosen=[target[int(rng.integers(len(target)))]]
        remaining=[o for o in available if o["id"]!=chosen[0]["id"]]
        if remaining and self.cfg["max_objects_per_image"]>1:
            chosen+=list(rng.choice(remaining,size=min(len(remaining),self.cfg["max_objects_per_image"]-1),replace=False))
        rows=[]
        for o in chosen:rows+=self.geometry[o["id"]]
        boxes,scores=proposal(self.root,name)
        gt=np.array([o["box"] for o in available])
        ov=iou(boxes,gt)
        for j in np.argsort(-scores,kind="stable"):
            owner=int(ov[j].argmax())
            obj=available[owner]
            if obj["id"] not in {o["id"] for o in chosen}:continue
            b=boxes[j]
            ctr=(b[:2]+b[2:])/2
            g=gt[owner]
            if not np.all((ctr>=g[:2])&(ctr<=g[2:])):continue
            rows.append({"box":b.tolist(),"quality":float(ov[j,owner]),"reliable":semantic_valid(b,g,self.cfg),
                         "owner":obj["id"],"class_id":obj["class_id"],"kind":"proposal"})
            if len(rows)>=self.cfg["max_train_rois"]:break
        return name,rows[:self.cfg["max_train_rois"]]
