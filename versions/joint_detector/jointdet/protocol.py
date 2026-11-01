from __future__ import annotations
from collections import Counter
from pathlib import Path
import os
import random
import shutil
import time
import importlib.metadata
import sys
import numpy as np
from cil.common import digest, fingerprint, read_json, write_json, stable_seed


def config(root, path=None):
    c = read_json(Path(root) / "configs/joint_detector.json")
    if path:
        c.update(read_json(Path(path)))
    if c['precision'] != 'fp32' or c['accumulate'] < 1:
        raise ValueError('v1 requires FP32 and positive accumulation')
    return c


def environment():
    # The local 3.10/2.1 profile remains reproducible; remote 3.12/2.8
    # is admitted for verification, not treated as already CUDA-validated.
    required={'torch':{'2.1.0','2.8.0'},'torchvision':{'0.16.0','0.23.0'},
              'mmcv':{'2.1.0','2.2.0'},'mmengine':{'0.10.7'},'mmdet':{'3.3.0'},
              'timm':{'1.0.20'},'numpy':{'1.26.4'},'fairscale':{'0.4.13'}}
    actual={}
    for name,allowed in required.items():
        try:actual[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual[name]=None
        if actual[name] is None or actual[name].split('+')[0] not in allowed:
            raise RuntimeError(f'{name}: expected one of {sorted(allowed)}, found {actual[name]}. Run 13 Joint install current dependencies with the existing interpreter.')
    pair=(sys.version_info[:2],actual['torch'].split('+')[0],actual['torchvision'].split('+')[0])
    if pair not in [((3,10),'2.1.0','0.16.0'),((3,12),'2.8.0','0.23.0')]:
        raise RuntimeError('Unverified Python/PyTorch/torchvision combination: '+repr(pair)+'. Preserve the existing environment and inspect 13 --probe output.')
    return dict(python=sys.version,packages=actual)


def split(root, task, held, formal, c):
    from cil.data import manifest
    m = manifest(root, task)
    if formal and held is not None:
        raise ValueError('Formal run cannot hold out a class')
    if not formal and held not in m['classes']:
        raise ValueError('Development run requires a held class')
    active = [i for i, n in enumerate(m['classes']) if n != held]
    outer = set(m['fixed_train_images'] if formal else m['fit_images'])
    objects = [o for o in m['objects'] if o['image'] in outer and o['class_id'] in active]
    groups = sorted({o['group'] for o in objects})
    rng = np.random.default_rng(c['inner_split_seed'])
    counts=np.array([[sum(o['group']==g and o['class_id']==j for o in objects)
                       for j in active] for g in groups],dtype=np.int64)
    total=counts.sum(0)
    n=max(1,min(len(groups)-1,round(len(groups)*c['inner_fit_fraction'])))
    best = None
    for _ in range(20000):
        order = rng.permutation(len(groups))
        selected=counts[order[:n]];fit_counts=selected.sum(0)
        # Library must support five neighbors after source-group exclusion.
        if np.any(total-fit_counts==0):
            continue
        if np.any(fit_counts-selected.max(0)<c['k']):
            continue
        err=float(np.abs(fit_counts/total-c['inner_fit_fraction']).sum())
        if best is None or err < best[0]:
            best = (err, set(groups[i] for i in order[:n]))
    if best is None:
        raise RuntimeError('No legal inner group split; do not lower k or split groups silently')
    a=[o for o in objects if o['group'] in best[1]]
    b=[o for o in objects if o['group'] not in best[1]]
    result = dict(task=task, held=held, formal=formal, active=active, classes=m['classes'],
                  fit_objects=a, cal_objects=b,
                  fit_images=sorted({o['image'] for o in a}),
                  cal_images=sorted({o['image'] for o in b}),
                  val_images=[] if formal else m['val_images'],
                  manifest_sha256=m['sha256'], groups=m['groups'])
    result['counts'] = {key:dict(Counter(o['class_id'] for o in result[key]))
                        for key in ['fit_objects','cal_objects']}
    if set(result['fit_images']) & set(result['cal_images']):
        raise AssertionError('Group split leakage')
    result['identity'] = fingerprint(result)
    return result


def source_hashes(root):
    root = Path(root)
    paths = [root/'run_joint_detector.py', root/'configs/joint_detector.json',root/'configs/joint_assets.json',root/'requirements-joint.txt']
    paths += sorted((root/'versions/joint_detector').rglob('*.py'))
    paths += sorted((root/'vendor/pfrpn/mmdet').rglob('*.py'))
    paths += sorted((root/'vendor/pfrpn/configs').rglob('*.py'))
    paths += sorted((root/'src').rglob('*.py'))+[root/'vendor/fomo_open_world_eval.py']
    return {p.relative_to(root).as_posix():digest(p) for p in paths}


def atomic_save(value, path):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)


def load_torch(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def check_disk(root,c):
    root=Path(root)
    if shutil.disk_usage(root).free < c['min_free_disk_gib']*1024**3:
        raise RuntimeError('Project disk free space below configured minimum')
    folders=[root/'runs/joint_detector',root/'reports/joint_detector',root/'prepared/joint_detector']
    used=sum(p.stat().st_size for f in folders if f.exists() for p in f.rglob('*') if p.is_file())
    if used > c['disk_budget_gib']*1024**3:
        raise RuntimeError('Joint experiment output budget exceeded; preserve results and review storage')
    return used


def gpu(spec,c):
    import torch
    if not spec.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Training needs the remote CUDA GPU; local 4GB/CPU training is blocked')
    d=torch.device(spec)
    if torch.cuda.get_device_properties(d).total_memory < 24*1024**3:
        raise RuntimeError('Joint detector training requires at least 24GiB total GPU memory')
    torch.cuda.set_device(d);torch.cuda.reset_peak_memory_stats(d)
    return d


def memory_check(d,c):
    import torch
    torch.cuda.synchronize(d)
    allocated=torch.cuda.max_memory_allocated(d)
    reserved=torch.cuda.max_memory_reserved(d)
    if allocated > c['max_allocated_gib']*1024**3:
        raise RuntimeError(f'Peak allocated {allocated/1024**3:.2f}GiB exceeds the common configuration budget')
    return dict(peak_allocated_bytes=allocated,peak_reserved_bytes=reserved)


class Progress:
    def __init__(self,total,label):
        self.total,self.label=total,label;self.start=time.monotonic()
    def show(self,n,extra=''):
        elapsed=time.monotonic()-self.start
        eta=elapsed*(self.total-n)/n if n else None
        fmt=lambda s:time.strftime('%H:%M:%S',time.gmtime(max(0,s)))
        line=f'{self.label} {n}/{self.total} | elapsed {fmt(elapsed)} | ETA {fmt(eta) if eta is not None else "--:--:--"} {extra}'
        print('\r'+line.ljust(120),end='\n' if n>=self.total else '',flush=True)
