from pathlib import Path
from collections import defaultdict
import contextlib,io,copy,xml.etree.ElementTree as ET
import numpy as np
from cil.evaluate import official_functions
from pclean.files import require
from .protocol import write

def projected_truth(root,split,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);truth={};paths=[]
 for name in split['val_images']:
  original=ET.parse(Path(root)/'assets/rwd/Annotations'/split['scene']/(Path(name).stem+'.xml')).getroot();projected=ET.Element('annotation');objects=[]
  for obj in original.findall('object'):
   label=obj.findtext('name')
   if label not in split['classes']:continue
   projected.append(copy.deepcopy(obj));b=[float(obj.find('bndbox').findtext(k))-1 for k in ['xmin','ymin','xmax','ymax']];objects.append((label,b))
  p=out/(Path(name).stem+'.xml');ET.ElementTree(projected).write(p,encoding='utf8');paths.append(str(p));truth[name]=objects
 return paths,truth

def evaluate_predictions(root, split, predictions, out, annotation_paths=None, truth=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if annotation_paths is None:
        annotation_paths,truth=projected_truth(root,split,out)
    active=[split['classes'][i] for i in split['active']]
    require([p['image'] for p in predictions]==split['val_images'],'Full image list/order must match development split')
    by=defaultdict(list); log=io.StringIO()
    for i,p in enumerate(predictions):
        require(len(p['boxes'])==len(p['scores'])==len(p['labels']),'Prediction dimensions differ')
        for box,score,label in zip(p['boxes'],p['scores'],p['labels']):
            name='unknown' if label==-1 else split['classes'][label]
            b=np.asarray(box).copy();b[:2]+=1
            by[name].append(f'{i}-^-{Path(p["image"]).stem} {score:.3f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}')
    funcs=official_functions(root); aps={};curves={}
    for name in active+['unknown']:
        with contextlib.redirect_stdout(log):
            rec,prec,ap,*_=funcs['voc_eval'](by[name],annotation_paths,split['val_images'],name,
                                           ovthresh=.5,use_07_metric=True,known_classes=active)
        aps[name]=float(ap*100);curves[name]=dict(recall=rec.tolist(),precision=prec.tolist())
    metrics=dict(K_dev_AP50=float(np.mean([aps[n] for n in active])),held_dev_AP50=aps['unknown'],per_class_AP50=aps,
                 images=len(predictions),formal=False,real_unknown_labels_used=False,
                 scope='Exploratory scene-specific known-class development projection; held class is pseudo-unknown. Unmatched predictions are not verified background.')
    write(out/'metrics.json',metrics);write(out/'pr_curves.json',curves);write(out/'predictions.json',predictions)
    (out/'official_eval.log').write_text(log.getvalue(),encoding='utf8')
    return metrics
