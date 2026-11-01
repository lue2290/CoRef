"""Portable scene/task inference for the released VRDD checkpoints."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,urllib.request,zipfile

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor/pfrpn'),str(ROOT/'versions/joint_detector'),
              str(ROOT/'versions/joint_network_merge'),str(ROOT/'versions/project_cleanup')]
SCENES=['Aquatic','Aerial','Game','Medical','Surgical']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)


def checkpoint(scene,task):
    return next(r for r in read(ROOT/'checkpoints/index.json')['models'] if (r['scene'],r['task'])==(scene,task))


def link_image(view,source):
    source=Path(source).resolve(strict=True);target=view/'assets/data/images'/source.name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file():
        if sha(target)==sha(source):return source.name
        raise ValueError('Image basename collision in view: '+source.name)
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)
    return source.name


def load_model(scene,task,device):
    import torch
    from refnet.model import ReferenceDetector
    from jointdet.protocol import load_torch
    from cil.common import configure_torch
    row=checkpoint(scene,task);path=ROOT/row['path']
    if not path.is_file():raise FileNotFoundError('Run: python run_vrdd.py download-weights')
    if sha(path)!=row['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
    configure_torch(42,2)
    c=read(ROOT/'configs/reference_net.json')['model']
    model=ReferenceDetector(ROOT,c,list(range(len(row['classes'])))).to(device)
    state=load_torch(path);model.load_adapted(state['model']);model.eval();model.requires_grad_(False)
    return model,c,row


def build_bank(args,model,c):
    import numpy as np
    import torch
    from jointdet.model import pixels,make_sample
    from jointdet.geometry import cxcywh
    from refnet.model import region
    from current.progress import track
    spec=read(ROOT/f'protocols/{args.scene}/{args.task}.json')
    refs=read(ROOT/f'checkpoints/{args.scene}/{args.task}/reference_bank.json')
    objects={o['id']:o for o in spec['fit_objects']};by={}
    for row in refs['rows']:
        if row['id'] not in objects:raise ValueError('Reference identity missing from legal fit split')
        by.setdefault(row['image'],[]).append(row)
    view=ROOT/f'.cache/views/{args.scene}';raw=[];labels=[];ids=[];images=[]
    with torch.inference_mode():
        for name,rows in track(list(by.items()),'Extract legal fit references',unit='images'):
            link_image(view,args.data/'JPEGImages'/args.scene/name)
            _,d,meta=pixels(view,name,c,args.device)
            selected=[objects[r['id']] for r in rows];_,gt=make_sample(meta,selected,args.device)
            features=region(model.feature_map(d),cxcywh(gt)).cpu().numpy()
            raw.extend(features);labels.extend([r['class_id'] for r in rows]);ids.extend([r['id'] for r in rows]);images.extend([name]*len(rows))
    path=ROOT/f'.cache/reference_banks/{args.scene}/{args.task}/bank.npz';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:np.savez_compressed(f,raw=np.asarray(raw),labels=np.asarray(labels),ids=np.asarray(ids))
    write(path.with_suffix('.json'),dict(sha256=sha(path),reference_bank_sha256=sha(ROOT/f'checkpoints/{args.scene}/{args.task}/reference_bank.json'),
        dino_sha256=sha(ROOT/'assets/models/dinov3_large/model.safetensors'),images=images,reference_ids=ids))
    return path


def load_bank(args,device):
    import numpy as np
    import torch
    path=ROOT/f'.cache/reference_banks/{args.scene}/{args.task}/bank.npz'
    if not path.is_file():raise FileNotFoundError('Build the reference bank first with build-bank')
    manifest=read(path.with_suffix('.json'))
    if sha(path)!=manifest['sha256'] or sha(ROOT/f'checkpoints/{args.scene}/{args.task}/reference_bank.json')!=manifest['reference_bank_sha256']:
        raise ValueError('Reference bank identity changed; rebuild bank')
    if sha(ROOT/'assets/models/dinov3_large/model.safetensors')!=manifest['dino_sha256']:
        raise ValueError('DINO checkpoint changed; rebuild bank')
    with np.load(path,allow_pickle=False) as z:
        return dict(raw=torch.tensor(z['raw'],device=device),labels=torch.tensor(z['labels'],device=device,dtype=torch.long))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','download-weights','build-bank','infer','evaluate'])
    p.add_argument('--scene',choices=SCENES,default='Aquatic');p.add_argument('--task',choices=['task1','task2'],default='task1')
    p.add_argument('--device',default='cuda:0');p.add_argument('--data',type=Path,default=ROOT/'assets/rwd')
    p.add_argument('--image',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    from current.progress_io import install
    from current.progress import track,stage
    install()
    if a.command=='check':
        missing=[]
        for name,want in read(ROOT/'MAINLINE_PARENT_INDEX.json').get('base_weights',{}).items():
            path=ROOT/name
            if not path.is_file():missing.append(name)
            elif sha(path)!=want:raise ValueError('Backbone SHA mismatch: '+name)
        print(json.dumps(dict(missing_backbones=missing,checkpoint_present=(ROOT/checkpoint(a.scene,a.task)['path']).is_file()),indent=2));return
    if a.command=='download-weights':
        record=read(ROOT/'checkpoints/download.json');path=ROOT/'.cache/VRDD_weights.zip';path.parent.mkdir(parents=True,exist_ok=True)
        with stage('Download scene/task checkpoint bundle'):
            if not path.is_file() or sha(path)!=record['sha256']:urllib.request.urlretrieve(record['url'],path)
            if sha(path)!=record['sha256']:raise ValueError('Downloaded archive SHA mismatch')
        with zipfile.ZipFile(path) as z:
            for name in track(z.namelist(),'Extract verified checkpoint bundle',unit='files'):
                target=(ROOT/name).resolve()
                if ROOT not in target.parents or not name.startswith('checkpoints/'):raise ValueError('Unsafe checkpoint member')
                z.extract(name,ROOT)
        print('Checkpoints ready in checkpoints/<scene>/<task>/model.pt');return
    import torch
    from refnet_f1.runtime import candidate_values,decode
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; choose --device cpu explicitly')
    with stage('Load '+a.scene+'/'+a.task):model,c,row=load_model(a.scene,a.task,a.device)
    if a.command=='build-bank':print(build_bank(a,model,c));return
    bank=load_bank(a,a.device);cal=read(ROOT/f'checkpoints/{a.scene}/{a.task}/calibration.json')
    spec=read(ROOT/f'protocols/{a.scene}/{a.task}.json')
    names=spec['test_images'] if a.command=='evaluate' else [a.image.name if a.image else p.error('--image is required for infer')]
    predictions=[];view=ROOT/f'.cache/views/{a.scene}'
    with torch.inference_mode():
        for name in track(names,'VRDD '+a.scene+'/'+a.task,unit='images'):
            source=a.data/'JPEGImages'/a.scene/name if a.command=='evaluate' else a.image
            link_image(view,source);out=model.forward_image(view,name,[],a.device)
            values=candidate_values(model,out,bank,dict(relation_batch=600),c)
            pred,_=decode(values,'F1',cal,c);predictions.append(dict(image=name,**pred));del out,values
    folder=a.output or ROOT/f'outputs/{a.scene}/{a.task}';folder.mkdir(parents=True,exist_ok=True)
    write(folder/'predictions.json',predictions)
    write(folder/'classes.json',row['classes']+['unknown'])
    if a.command=='evaluate':
        from refnet.evaluation import evaluate
        # The official evaluator resolves XML beneath this view root; no data copies.
        evalroot=ROOT/f'.cache/eval/{a.scene}/{a.task}';target=evalroot/'assets/rwd/Annotations'/a.scene
        target.mkdir(parents=True,exist_ok=True)
        evaluator=evalroot/'vendor/fomo_open_world_eval.py';evaluator.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'vendor/fomo_open_world_eval.py',evaluator)
        for name in track(spec['test_images'],'Bind evaluation XML',unit='files'):
            src=a.data/'Annotations'/a.scene/(Path(name).stem+'.xml');dst=target/src.name
            if dst.exists():
                if sha(dst)!=sha(src):raise ValueError('Evaluation annotation changed: '+src.name)
            else:
                try:os.link(src,dst)
                except OSError:shutil.copyfile(src,dst)
        print(json.dumps(evaluate(evalroot,folder/'predictions.json',spec,folder),indent=2))
    print('Saved:',folder)


if __name__=='__main__':main()
