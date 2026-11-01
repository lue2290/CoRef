from pathlib import Path
import shutil
import importlib.metadata
from .files import read, write, sha, signed, check_identity, require, safe, walk

MODELS = [(s, h, a) for s in (43, 44) for h in ('fish', 'jellyfish') for a in ('P0', 'M1')]


def retained(root):
    return Path(root) / 'retained_models/repeat_v1'


def run(root, seed, held, phase, arm):
    return Path(root) / f'runs/joint_repeat/repeat_v1/main/seed_{seed}/held_{held}/{phase}_{arm}'


def runtime_files(root):
    root=Path(root); result=[]
    for folder in ('vendor/pfrpn','versions/joint_detector/jointdet'):
        result += [p for p in walk(root/folder) if p.suffix=='.py' and p.name not in ('cli.py','orchestrate.py')]
    for name in ('__init__.py','common.py','data.py','evaluate.py','model.py'):
        result.append(root/'src/cil'/name)
    result += [root/('versions/joint_network_merge/jnet/'+n) for n in ('__init__.py','model.py')]
    return sorted(set(result))


def low_memory_load(path):
    import torch
    # All source checkpoints were saved with the zip-based torch.save format.
    return torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)


def atomic_save(value,path):
    import torch,os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def memory_status():
    out={}
    for name in ('memory.max','memory.current','memory.events'):
        p=Path('/sys/fs/cgroup')/name
        if p.is_file():out[name]=p.read_text().strip()
    return out


def check_bases(root, spec):
    for name, want in spec['base_files'].items():
        require(sha(safe(root, name)) == want, 'Base asset changed: ' + name)


def check_package(root, verify_runtime=True):
    base = retained(root); index = check_identity(read(base / 'INDEX.json'))
    require(len(index['models']) == 8 and {r['id'] for r in index['models']} == {f'{s}/{h}/{a}' for s, h, a in MODELS}, 'Expected eight retained models')
    for name, want in index['files'].items():
        require(sha(safe(base, name)) == want, 'Retained package file missing/changed: ' + name)
    for model in index['models']:
        spec = read(base / model['folder'] / 'model.json'); check_bases(root, spec)
    if verify_runtime:
        for name, want in index['runtime'].items():
            require(sha(safe(root, name)) == want, 'Retained runtime changed: ' + name)
    return index


def pack_state(arm, lean, parent_s=None, tail_start=22):
    """Exact tensor copy; no cast, quantization or optimizer retention."""
    state = dict(lean)
    if arm == 'M1':
        require(parent_s is not None, 'M1 frozen S tail is missing')
        for name, tensor in parent_s.items():
            for j in range(2):
                prefix = f'dino.blocks.{tail_start+j}.'
                if name.startswith(prefix):
                    state[f'open_blocks.{j}.' + name[len(prefix):]] = tensor
            if name.startswith('dino.norm.'):
                state['open_norm.' + name[len('dino.norm.'):]] = tensor
        require(any(n.startswith('open_blocks.0.') for n in state) and any(n.startswith('open_blocks.1.') for n in state)
                and any(n.startswith('open_norm.') for n in state), 'Frozen S tail parameter set incomplete')
    return state


def prepare(root):
    import torch
    load_torch=low_memory_load
    import gc
    torch.set_num_threads(2)
    print("CPU-only mmap export; cgroup memory:", memory_status(), flush=True)
    root = Path(root); base = retained(root); base.mkdir(parents=True, exist_ok=True)
    if (base / 'INDEX.json').exists():
        return check_package(root)
    require(shutil.disk_usage(root).free >= 8 * 2**30, 'Need8GiB free while retaining/validating; originals preserved')
    control = read(root / 'reports/joint_repeat/repeat_v1/REPEAT_LOCK.json')
    check_identity(control)
    for name, want in control['source'].items():
        require(sha(safe(root, name)) == want, 'Historical source changed before retention: ' + name)
    records = []; files = {}
    for seed, held, arm in MODELS:
        print(f'Packing {seed}/{held}/{arm} (CPU, no training)', flush=True)
        source = run(root, seed, held, 'compare', arm); lock = read(source / 'RUN_LOCK.json')
        require(lock['control_identity'] == control['identity'] and lock['arm'] == arm and lock['seed'] == seed and lock['held'] == held, 'Source run selector differs')
        ex = read(source / 'execution.json'); require(ex['complete'], 'Cannot clean incomplete runs')
        selection = read(source / 'selection.json')['selected']['balanced']
        require(selection == ex['selected']['balanced'], 'Selection record differs')
        source_weight = safe(source, selection['weights_file'])
        require(sha(source_weight) == selection['weights_sha256'], 'Selected source weight changed')
        saved = load_torch(source_weight)
        require(saved['step'] == selection['step'] and saved['lock_identity'] == lock['identity'], 'Selected tensor boundary differs')
        s_state = None; parent_sha = None
        if arm == 'M1':
            parent = run(root, seed, held, 'parent', 'S'); pe = read(parent / 'execution.json')
            require(pe['complete'] and pe['step'] == 100, 'S parent is incomplete')
            require(pe['weights_sha256'] == lock['parents']['S']['weights_sha256'], 'S parent reference differs')
            pp = safe(parent, pe['weights_file']); require(sha(pp) == pe['weights_sha256'], 'S parent changed')
            s_state = load_torch(pp)['model']; parent_sha = pe['weights_sha256']
        original = source / selection['evaluation']; original_cal = read(original / 'calibration.json')
        if arm == 'M1':
            bank = safe(root, selection['gallery']['S_bank'])
        else:
            bank = safe(original, original_cal['folder'] + '/bank.pt')
        require(sha(bank) == original_cal['bank_sha256'], 'Selected gallery changed')
        ident = f'{seed}/{held}/{arm}'; folder = f'seed_{seed}/held_{held}/{arm}'; dest = base / folder; dest.mkdir(parents=True, exist_ok=True)
        tail_start = max(int(n.split('.')[2]) for n in s_state if n.startswith('dino.blocks.')) - 1 if s_state else 22
        state = pack_state(arm, saved['model'], s_state, tail_start)
        target = dest / 'model.pt'
        atomic_save(dict(model=state, kind='standalone_adapted_state_no_optimizer', arm=arm, source_weights_sha256=selection['weights_sha256']), target)
        copied = load_torch(target)['model']
        require(set(copied) == set(state) and all(torch.equal(copied[n], state[n]) for n in state), 'Retained tensors changed')
        require(all(torch.equal(copied[n], t) for n, t in saved['model'].items()), 'Selected learned parameters changed')
        for src, name in ((bank, 'bank.pt'), (original / 'calibration.json', 'calibration.json'),
                          (original / 'validation_metrics.json', 'expected_metrics.json'),
                          (original / ('validation/' + selection['weights_sha256'][:16] + '/predictions.json'), 'expected_predictions.json'),
                          (original / ('validation/' + selection['weights_sha256'][:16] + '/pr_curves.json'), 'expected_pr.json'),
                          (original / ('calibration/' + selection['weights_sha256'][:16] + '/cal_candidates.jsonl'), 'expected_cal_candidates.jsonl'),
                          (original / ('validation/' + selection['weights_sha256'][:16] + '/candidates.jsonl'), 'expected_candidates.jsonl')):
            require(src.is_file(), 'Missing selected evaluation evidence: ' + str(src)); shutil.copyfile(src, dest / name)
        spec = dict(id=ident, seed=seed, held=held, arm=arm, step=selection['step'], config=lock['config'], split=lock['split'],
                    original_lock_identity=lock['identity'], original_weights_sha256=selection['weights_sha256'],
                    S_parent_original_sha256=parent_sha, model_sha256=sha(target), bank_sha256=sha(dest / 'bank.pt'),
                    base_files={lock['config']['pf_weights']: lock['base_weights']['pf'],
                                'assets/models/dinov3_large/model.safetensors': lock['base_weights']['dino']},
                    environment=lock['environment'], exact_tensor_copy=True, optimizer_included=False,
                    frozen_open_tail_included=arm == 'M1', historical_runs_required_for_loading=False,
                    checksum_mapping='Original calibration stays immutable; package hash maps to original selected checkpoint.')
        check_bases(root, spec); write(dest / 'model.json', spec)
        for p in walk(dest): files[p.relative_to(base).as_posix()] = sha(p)
        records.append(dict(id=ident, folder=folder, source_step=selection['step']))
        print(f'Retained {len(records)}/8 {ident}', flush=True)
        del state, copied, saved, s_state
        gc.collect()
    index = signed(dict(version='retained_repeat_best_v1', source_control_identity=control['identity'], models=records, files=files,
                        runtime={p.relative_to(root).as_posix(): sha(p) for p in runtime_files(root)}))
    write(base / 'INDEX.json', index); return check_package(root)


def environment():
    from jointdet.protocol import environment as native
    value = native(); value['repair_packages'] = {n: importlib.metadata.version(n) for n in ('scipy', 'Pillow', 'safetensors')}
    return value


def load_model(root, folder, device):
    from jointdet.model import JointDetector
    from jnet.model import NetworkMerge
    from jointdet.protocol import load_torch
    spec = read(Path(folder) / 'model.json'); check_bases(root, spec)
    require(sha(Path(folder) / 'model.pt') == spec['model_sha256'], 'Retained model changed')
    if spec['arm'] == 'P0':
        model = JointDetector(root, spec['config'], spec['split']['active'], 'J0')
    else:
        model = NetworkMerge(root, spec['config'], spec['split']['active'], 'M1', initial=None)
    model.load_adapted(load_torch(Path(folder) / 'model.pt')['model']); model.configure_stage(True)
    for q in model.parameters(): q.requires_grad_(False)
    model.eval(); return model.to(device), spec
