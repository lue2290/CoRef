"""Mainline environment, code integrity, parent inventory and saved results only."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def selected_parents(root, scope):
    rows = read(root/'MAINLINE_PARENT_INDEX.json')['models']
    return [r for r in rows if r['scene'] in ['Aquatic','Game'] and r['seed']==42] if scope=='probe' else rows


def check(root, gpu=False):
    value = dict(python=sys.version, packages={})
    for name in ['torch','torchvision','numpy','timm','mmcv','mmengine','mmdet','fairscale','scipy','safetensors','Pillow']:
        try:
            value['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value['packages'][name] = None
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; enable the remote GPU')
        d = torch.device('cuda:0')
        x = torch.randn(32,32,device=d,requires_grad=True)
        x.square().mean().backward()
        if not bool(torch.isfinite(x.grad).all()):
            raise RuntimeError('CUDA backward is nonfinite')
        try:
            bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            bf16 = torch.cuda.is_bf16_supported()
        value['gpu'] = dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                            memory_bytes=torch.cuda.get_device_properties(0).total_memory, cuda=torch.version.cuda,
                            compiled_arches=torch.cuda.get_arch_list(), native_bf16=bf16)
        import mmdet
        value['mmdet_source'] = str(mmdet.__file__)
        spec = importlib.util.spec_from_file_location('_mainline_ops_probe',root/'tools/install_joint_env.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = subprocess.run([sys.executable,'-c',module.OPS_CHECK],cwd=root,capture_output=True,text=True)
        log = root/'reports/mainline/mmcv_cuda_probe.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:
            raise RuntimeError('MMCV CUDA forward/backward failed; see '+str(log))
        value['mmcv_cuda_forward_backward_passed'] = True
    return value


def artifacts(root, full=False, models=False, scope='probe'):
    manifest = read(root/'MAINLINE_MANIFEST.json')
    for name, want in manifest['files'].items():
        p = root/name
        if not p.is_file() or sha(p)!=want:
            raise RuntimeError('Mainline runtime file missing/changed: '+name)
    index = read(root/'MAINLINE_PARENT_INDEX.json')
    for name, want in index['base_weights'].items():
        if not (root/name).is_file() or (full and sha(root/name)!=want):
            raise RuntimeError('Base model missing/changed: '+name)
    checked = []
    if models:
        verified_cache = set()
        for row in selected_parents(root,scope):
            item = row['terminal']
            p = root/item['checkpoint']
            if not p.is_file() or sha(p)!=item['sha256']:
                raise RuntimeError('Download matching fixed500 parent first: '+item['checkpoint'])
            split = read(root/row['split'])
            if split['identity']!=row['split_identity']:
                raise RuntimeError('Parent split identity changed: '+row['split'])
            cache = root/row['reference_cache']
            if cache not in verified_cache:
                if sha(cache/'MANIFEST.json')!=row['reference_manifest_sha256']:
                    raise RuntimeError('Reference manifest changed: '+str(cache))
                for name,want in read(cache/'MANIFEST.json')['files'].items():
                    if sha(cache/name)!=want:
                        raise RuntimeError('Reference/teacher cache changed: '+str(cache/name))
                verified_cache.add(cache)
            cal = read(root/item['calibration'])
            if sha(root/item['calibration'])!=item['calibration_sha256'] or cal['checkpoint_sha256']!=item['sha256'] or cal['bank_manifest']!=row['reference_manifest_sha256']:
                raise RuntimeError('Calibration does not match parent/cache: '+row['id'])
            checked.append(row['id'])
    return dict(passed=True,runtime_files=len(manifest['files']),base_weight_sha_checked=full,
                model_validation_requested=models,verified_terminal_models=checked,
                local_terminal_files_present=sum((root/r['terminal']['checkpoint']).is_file() for r in index['models']),
                model_inventory_total=30,training_updates=0)


def main(root):
    root = Path(root)
    parser = argparse.ArgumentParser(description='Reference Net mainline. Old exploratory launchers are archived.')
    sub = parser.add_subparsers(dest='command')
    q = sub.add_parser('check');q.add_argument('--gpu',action='store_true')
    q = sub.add_parser('artifacts');q.add_argument('--full',action='store_true');q.add_argument('--models',action='store_true');q.add_argument('--scope',choices=['probe','terminal'],default='probe')
    q = sub.add_parser('weights');q.add_argument('--scope',choices=['probe','terminal'],default='probe')
    sub.add_parser('results');sub.add_parser('baseline')
    args = parser.parse_args()
    if not args.command:
        parser.print_help();return
    if args.command=='check':
        value = check(root,args.gpu)
    elif args.command=='artifacts':
        value = artifacts(root,args.full,args.models,args.scope)
    elif args.command=='weights':
        value = dict(models=[dict(id=r['id'],checkpoint=r['terminal']['checkpoint'],sha256=r['terminal']['sha256'],
                    present=(root/r['terminal']['checkpoint']).is_file(),reference_cache=r['reference_cache'],calibration=r['terminal']['calibration'])
                    for r in selected_parents(root,args.scope)],download_guide='REMOTE_DOWNLOAD_GUIDE.md')
    elif args.command=='baseline':
        value = dict(scope='Saved historical B5 Aquatic Task1/Task2 only; no new inference',results=read(root/'assets/b5/results.json'))
    else:
        saved = read(root/'docs/handover_20261025/metrics.json')
        value = dict(scope='Saved official results; no inference or test selection',
                     reference_net=saved['reference_net']['macro'],b5_task1=saved['b5']['strict_task1_macro'])
    print(json.dumps(value,ensure_ascii=False,indent=2))
