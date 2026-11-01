"""Use the archived official FOMO VOC functions, without its distributed stack."""
from __future__ import annotations
import ast
from collections import defaultdict
import contextlib
import functools
import io
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .common import TASK_CLASSES,digest,iou,names,read_json,write_json

def official_functions(root):
    source=Path(root)/"vendor/fomo_open_world_eval.py"
    tree=ast.parse(source.read_text(encoding="utf-8"))
    wanted={"voc_ap","parse_rec","voc_eval"}
    funcs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in wanted]
    if {n.name for n in funcs}!=wanted:raise RuntimeError("Official evaluator functions missing")
    scope={"np":np,"os":os,"ET":ET,"functools":functools}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),"exec"),scope)
    return scope

def evaluate(root,prediction_path,task,out=None):
    root=Path(root)
    prediction_path=Path(prediction_path)
    results=read_json(prediction_path)
    test=names(root/"assets/inputs/test.txt")
    if len(results)!=len(test) or len({p["image"] for p in results})!=len(test) or set(p["image"] for p in results)!=set(test):
        raise ValueError("Official evaluation requires exactly the fixed 319 unique test images")
    cls=TASK_CLASSES[task]
    parsed={}
    by_class=defaultdict(list)
    indices={name:i for i,name in enumerate(test)}
    errors=[]
    unknown_gt_count=0
    for name in test:
        xml=root/"assets/data/annotations"/(Path(name).stem+".xml")
        objects=[]
        for o in ET.parse(xml).getroot().findall("object"):
            label=o.findtext("name")
            box=[float(o.find("bndbox").findtext(k))-1 for k in ["xmin","ymin","xmax","ymax"]]
            objects.append((label if label in cls else "unknown",box))
        parsed[name]=objects
        unknown_gt_count+=sum(c=="unknown" for c,_ in objects)
    for pred in results:
        name=pred["image"]
        if not (len(pred["boxes"])==len(pred["scores"])==len(pred["labels"])):raise ValueError("Misaligned predictions")
        boxes=np.asarray(pred["boxes"],np.float32).reshape(-1,4)
        if not np.isfinite(boxes).all() or not np.isfinite(pred["scores"]).all():raise ValueError("Nonfinite predictions")
        gt=parsed[name]
        overlaps=iou(boxes,[b for _,b in gt])
        best=overlaps.max(1) if gt else np.zeros(len(boxes))
        own=overlaps.argmax(1) if gt else np.zeros(len(boxes),int)
        unknown=[]
        for j,(b,s,label) in enumerate(zip(boxes,pred["scores"],pred["labels"])):
            label=len(cls) if label==-1 else int(label)
            if not 0<=label<=len(cls):raise ValueError("Unknown label index")
            image_id=f"{indices[name]}-^-{Path(name).stem}"
            b=b.copy(); b[:2]+=1
            line=f"{image_id} {float(s):.3f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
            by_class[label].append(line)
            if label==len(cls):unknown.append({"score":float(f"{s:.3f}"),"bad":bool(best[j]<0.5)})
        errors.append({"image":name,"group":name,"background_or_low_iou_fp":int((best<0.5).sum()),
            "unknown_background_or_low_iou_fp":sum(x["bad"] for x in unknown),"unknown_predictions":unknown,
            "known_as_unknown":sum(int((l==len(cls) or l==-1) and v>=0.5 and gt[j][0]!="unknown") for l,v,j in zip(pred["labels"],best,own))})
    funcs=official_functions(root)
    aps,curves={},{}
    log=io.StringIO()
    annotations=[str(root/"assets/data/annotations"/(Path(n).stem+".xml")) for n in test]
    for j,c in enumerate(cls+["unknown"]):
        if c=="unknown" and unknown_gt_count==0:
            aps[c]=None;curves[c]={"recall":[],"precision":[],"scores":[]};continue
        with contextlib.redirect_stdout(log):
            rec,prec,ap,*_=funcs["voc_eval"](by_class[j],annotations,test,c,ovthresh=0.5,use_07_metric=True,known_classes=cls)
        aps[c]=float(ap*100)
        conf=np.array([float(line.split()[1]) for line in by_class[j]])
        ordered=np.argsort(-conf)
        curves[c]={"recall":rec.tolist(),"precision":prec.tolist(),"scores":conf[ordered].tolist()}
    metrics={"task":task,"per_class_AP50":aps,"K_AP50":float(np.mean([aps[c] for c in cls])),
             "U_AP50":aps["unknown"],"metric":"FOMO official VOC07 eleven-point AP50; inclusive coordinates, strict IoU>0.5, original rounding/tie sort",
             "prediction_sha256":digest(prediction_path),"official_evaluator_sha256":digest(root/"vendor/fomo_open_world_eval.py"),
             "background_fp_per_image":float(np.mean([e["background_or_low_iou_fp"] for e in errors])),
             "unknown_background_fp_per_image":float(np.mean([e["unknown_background_or_low_iou_fp"] for e in errors])),
             "known_as_unknown_detections":sum(e["known_as_unknown"] for e in errors),
             "matched_unknown_recall":{},"CODE_comparison":"Report values only; current backbone/pretraining and shot lists differ"}
    if task=="task2":
        metrics["PK_AP50"]=float(np.mean([aps[c] for c in cls[:4]]))
        metrics["CK_AP50"]=float(np.mean([aps[c] for c in cls[4:]]))
    for target in [0.6,0.7]:
        rec=np.array(curves["unknown"]["recall"])
        if not len(rec) or rec.max()<target:
            metrics["matched_unknown_recall"][str(target)]={"reached":False,"max_recall":float(rec.max()) if len(rec) else None}
        else:
            index=int(np.flatnonzero(rec>=target)[0])
            threshold=curves["unknown"]["scores"][index]
            tied=np.array(curves["unknown"]["scores"])>=threshold
            last=int(np.flatnonzero(tied)[-1])
            fp=sum(x["bad"] and x["score"]>=threshold for e in errors for x in e["unknown_predictions"])
            metrics["matched_unknown_recall"][str(target)]={"reached":True,"threshold":threshold,
                "actual_recall_including_ties":float(rec[last]),"background_or_low_iou_fp_per_image":fp/len(test)}
    out=Path(out) if out else prediction_path.parent/"evaluation"
    write_json(out/"metrics.json",metrics)
    write_json(out/"pr_curves.json",curves)
    write_json(out/"errors_by_image.json",errors)
    (out/"official_eval.log").write_text(log.getvalue(),encoding="utf-8")
    return metrics

def baseline_regression(root):
    expected=read_json(Path(root)/"assets/b5/results.json")
    checked={}
    for task in ["task1","task2"]:
        values=[]
        for seed in [42,43,44]:
            values.append(evaluate(root,Path(root)/f"assets/b5/{task}/seed_{seed}.json",task,
                                   Path(root)/"reports/b5_regression"/task/str(seed)))
        keys=["K_AP50","U_AP50"] if task=="task1" else ["PK_AP50","CK_AP50"]
        checked[task]={k:float(np.mean([v[k] for v in values])) for k in keys}
    # Independently frozen published local summaries; allow only original tensor rounding.
    targets={"task1":{"K_AP50":61.02,"U_AP50":31.78},"task2":{"PK_AP50":63.31,"CK_AP50":77.52}}
    for task,vals in targets.items():
        for key,target in vals.items():
            if abs(checked[task][key]-target)>0.011:raise AssertionError((task,key,checked[task][key],target))
    write_json(Path(root)/"reports/b5_regression/summary.json",{"passed":True,"mean":checked,"original_record_present":expected["status"]=="completed"})
    return checked
