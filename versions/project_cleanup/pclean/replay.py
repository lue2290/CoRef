"""Cold replay with historical run-directory IO forbidden throughout inference."""
from pathlib import Path
from collections import defaultdict
import json
import os
import sys
import time
import types
from .files import read, write, sha, signed, require
from .retain import retained, check_package, load_model, environment


class IndependenceGate:
    def __init__(self, root, split):
        root = Path(root).resolve(); self.old_runs = root / 'runs'; self.annotation_root = root / 'assets/data/annotations'
        self.allowed = {(self.annotation_root / (Path(n).stem + '.xml')).resolve() for n in split['val_images']}
        self.forbidden = [(root / p).resolve() for p in ('assets/inputs/test.txt', 'assets/inputs/task2_shots.json', 'prepared/task2')]
        self.phase = 'inference'; sys.addaudithook(self.audit)

    def audit(self, event, args):
        if self.phase == 'closed' or event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        require(not (p == self.old_runs or self.old_runs in p.parents), 'Independent replay tried to open historical runs: ' + str(p))
        require(not any(p == q or q in p.parents for q in self.forbidden), 'Formal test/Task2 access during development replay')
        if p == self.annotation_root or self.annotation_root in p.parents:
            require(self.phase == 'projection' and p in self.allowed, 'Annotation access outside four-class development projection')


def raw_arrays(out, c, arm):
    import torch
    from jointdet.scoring import raw_arrays as native
    from jointdet.model import pf_score
    raw = native(out, c)
    if arm == 'M1':
        ids = torch.as_tensor(raw['query_index'], device=out['open_cls'].device)
        cpu = lambda v: v.detach().cpu().numpy()
        raw.update(pf=cpu(pf_score(out['open_cls'][ids], out['open_ctr'][ids])),
                   object_logit=cpu(out['open_cls'][ids, 1]), pf_token_logits=cpu(out['open_cls'][ids]), centerness=cpu(out['open_ctr'][ids]))
    return raw


def candidates(path):
    by = defaultdict(list)
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line); by[row['image']].append(row)
    return dict(by)


def compare_raw(raw, rows):
    import numpy as np
    by = {r['query_index']: r for r in rows}
    require(set(map(int, raw['query_index'])) == set(by), 'Top300 query set changed')
    ordered = [by[int(q)] for q in raw['query_index']]
    errors = {}
    for key, oldkey, tolerance in [('boxes', 'box', .001), ('pf', 'pf', 1e-5), ('q', 'quality', 1e-5),
                                   ('known_logits', 'known_logits', 1e-5), ('centerness', 'centerness', 1e-5),
                                   ('pf_token_logits', 'pf_token_logits', 1e-5)]:
        error = float(np.max(np.abs(raw[key] - np.asarray([r[oldkey] for r in ordered]))))
        require(error <= tolerance, 'Retained candidate changed: ' + key + ' ' + str(error)); errors[key] = error
    return errors



def checked_raw(raw, rows, work, image, phase, spec, out):
    """Keep strict verification; write actionable evidence before propagating a failure."""
    try:
        return compare_raw(raw, rows)
    except RuntimeError as exc:
        import numpy as np
        import torch
        from jointdet.model import pf_score
        old = {int(r['query_index']): r for r in rows}
        new = {int(q): i for i, q in enumerate(raw['query_index'])}
        common = sorted(set(old) & set(new))
        errors = {}
        for key, oldkey in [('boxes','box'), ('pf','pf'), ('q','quality'),
                            ('known_logits','known_logits'), ('centerness','centerness'),
                            ('pf_token_logits','pf_token_logits')]:
            if common:
                errors[key] = float(np.max(np.abs(np.asarray([raw[key][new[q]] for q in common]) - np.asarray([old[q][oldkey] for q in common]))))
        full_pf = pf_score(out['cls'][-1], out['ctr'][-1]).detach().cpu().numpy()
        report = dict(passed=False, error=str(exc), model=spec['id'], phase=phase, image=image,
                      expected_count=len(old), actual_count=len(new), common_count=len(common),
                      missing_queries=sorted(set(old)-set(new)), added_queries=sorted(set(new)-set(old)),
                      common_max_abs_error=errors,
                      expected_pf_range=[float(min(r['pf'] for r in rows)),float(max(r['pf'] for r in rows))],
                      actual_selection_pf_range=[float(full_pf[list(new)].min()),float(full_pf[list(new)].max())],
                      original_queries_current_pf={str(q):float(full_pf[q]) for q in sorted(old)},
                      device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                      float32_matmul_precision=torch.get_float32_matmul_precision(), training_updates=0)
        write(work/'REPLAY_MISMATCH.json',report)
        raise RuntimeError(str(exc)+'; diagnostic: '+str(work/'REPLAY_MISMATCH.json')) from exc


def serialized_predictions(rows):
    """Match original VOC score/coordinate serialization without tie reordering."""
    result = []
    for image in rows:
        values = []
        for box, score, label in zip(image['boxes'], image['scores'], image['labels']):
            b = list(box); b[0] += 1; b[1] += 1
            values.append((int(label), f'{score:.3f}', *(f'{v:.1f}' for v in b)))
        result.append((image['image'], values))
    return result


def one(root, folder, device_spec, ordinal=1, total=8, measure_drift=False):
    import torch
    import numpy as np
    from cil.common import configure_torch
    from jointdet.protocol import gpu, load_torch, memory_check
    root = Path(root); folder = Path(folder); spec = read(folder / 'model.json'); c = spec['config']; s = spec['split']
    actual_env=environment(); expected_env=spec['environment']
    require(actual_env['packages']==expected_env['packages'] and actual_env['repair_packages']==expected_env['repair_packages'], 'Replay dependency versions differ; see saved model environment')
    require(actual_env['python'].split()[0].split('.')[:2]==expected_env['python'].split()[0].split('.')[:2], 'Python major/minor differs')
    require(sha(folder / 'bank.pt') == spec['bank_sha256'], 'Retained bank changed')
    configure_torch(spec['seed'], c['threads']); d = gpu(device_spec, c)
    gate = IndependenceGate(root, s); started = time.monotonic()
    try:
        model, _ = load_model(root, folder, d); bank = load_torch(folder / 'bank.pt'); cal = read(folder / 'calibration.json')
        expected = candidates(folder / 'expected_candidates.jsonl'); expected_cal = candidates(folder / 'expected_cal_candidates.jsonl')
        work = root / 'reports/project_cleanup/replay' / f"seed_{spec['seed']}" / spec['held'] / spec['arm']
        if measure_drift: work = root/'reports/drift_measurement'/f"seed_{spec['seed']}"/spec['held']/spec['arm']
        source = root / 'versions/joint_detector/jointdet/evaluation.py'; module = types.ModuleType('jointdet._retained_replay')
        module.__package__ = 'jointdet'; module.__file__ = str(source)
        text = source.read_text(encoding='utf8')
        cached = "    if (out/'metrics.json').exists():return read_json(out/'metrics.json')\n"
        require(text.count(cached) == 1, 'Native evaluator cache hook changed')
        exec(compile(text.replace(cached, '    # Always replay this independent verification.\n'), str(source), 'exec'), module.__dict__)
        original_read, original_digest, projection = module.read_json, module.digest, module.projected_annotations
        view = dict(config=dict(c, updates=spec['step']), split=s, method=spec['arm'])
        module.read_json = lambda p: view if Path(p).resolve() == (work / 'RUN_LOCK.json').resolve() else original_read(p)
        module.digest = lambda p: spec['original_weights_sha256'] if Path(p).resolve() == (work / 'last.pt').resolve() else original_digest(p)
        module.load_run = lambda *a: (model, dict(step=spec['step']), view)
        module.calibrate = lambda *a: (bank, cal)
        maximum = defaultdict(float)
        violations = []
        def verify(condition, message):
            if not condition:
                if not measure_drift: require(False, message)
                violations.append(message)
        def compare(raw, rows, name, phase, out):
            if not measure_drift: return checked_raw(raw, rows, work, name, phase, spec, out)
            old = {int(r['query_index']):r for r in rows}
            new = {int(q):i for i,q in enumerate(raw['query_index'])}
            verify(set(old)==set(new), phase+'/'+name+': query set changed')
            common = sorted(set(old)&set(new)); errors = {}
            for key, oldkey, tol in [('boxes','box',.001),('pf','pf',1e-5),('q','quality',1e-5),('known_logits','known_logits',1e-5),('centerness','centerness',1e-5),('pf_token_logits','pf_token_logits',1e-5)]:
                error = float(np.max(np.abs(np.asarray([raw[key][new[q]] for q in common])-np.asarray([old[q][oldkey] for q in common])))) if common else float('inf')
                errors[key] = error
                verify(error<=tol,phase+'/'+name+': '+key+' error='+str(error))
            return errors
        def checked(out, config):
            raw = raw_arrays(out, config, spec['arm']); name = Path(out['meta']['img_path']).name
            for key, value in compare(raw, expected[name], name, 'validation', out).items(): maximum[key] = max(maximum[key], value)
            return raw
        module.raw_arrays = checked
        def project(*args, **kwargs):
            gate.phase = 'projection'
            try: return projection(*args, **kwargs)
            finally: gate.phase = 'inference'
        module.projected_annotations = project
        class Progress:
            def __init__(self, count, label): self.count = count
            def show(self, n, extra=''):
                elapsed = time.monotonic()-started
                print(f'\rReplay {ordinal}/{total} {spec["id"]} | images {n}/{self.count} | elapsed {elapsed:.0f}s | image ETA ~{elapsed/max(1,n)*(self.count-n):.0f}s'.ljust(160), end='', flush=True)
        module.Progress = Progress
        # Independently check the calibration image readout and fixed threshold.
        from jointdet.geometry import overlaps
        scores = []
        with torch.inference_mode():
            for name in s['cal_images']:
                out = model.forward_image(root, name, [], d)
                raw = raw_arrays(out, c, spec['arm'])
                for key, value in compare(raw, expected_cal[name], name, 'calibration', out).items(): maximum[key] = max(maximum[key], value)
                objects = [o for o in s['cal_objects'] if o['image'] == name]
                ov = overlaps(torch.tensor(raw['boxes']), torch.tensor([o['box'] for o in objects], dtype=torch.float32)).numpy()
                obj = raw['pf'] * (1-c['quality_alpha']+c['quality_alpha']*raw['q'])
                for j in range(len(objects)):
                    i = int(ov[:, j].argmax())
                    if ov[i, j] >= .5: scores.append(float(obj[i]))
            verify(len(scores) == cal['qualified_cal_objects'], 'Calibration qualified objects changed')
            threshold = float(np.quantile(scores, 1-c['object_retention'], method='lower'))
            verify(abs(threshold-cal['object_threshold']) <= 1e-7, 'Calibration threshold changed')
            metrics = module.validate(root, work, str(d))
        predictions = read(work / ('validation/' + spec['original_weights_sha256'][:16] + '/predictions.json'))
        verify(serialized_predictions(predictions) == serialized_predictions(read(folder / 'expected_predictions.json')), 'Serialized final predictions changed')
        previous = read(folder / 'expected_metrics.json')
        for key in ('K_dev_AP50', 'held_dev_AP50'):
            verify(abs(metrics[key]-previous[key]) <= 1e-8, 'Retained AP changed: ' + key)
        require(not any(q.requires_grad or q.grad is not None for q in model.parameters()), 'Replay created gradients')
        result = dict(id=spec['id'], passed=not violations, measurement_only=measure_drift, strict_violations=violations, metric_delta={k:metrics[k]-previous[k] for k in ('K_dev_AP50','held_dev_AP50')}, calibration_threshold_delta=threshold-cal['object_threshold'], model_sha256=spec['model_sha256'], bank_sha256=spec['bank_sha256'],
                      historical_runs_access_blocked=True, tensor_copy_recorded=spec['exact_tensor_copy'], serialized_predictions_identical=serialized_predictions(predictions)==serialized_predictions(read(folder/'expected_predictions.json')), actual_environment=actual_env, original_environment=expected_env,
                      K_dev_AP50=metrics['K_dev_AP50'], held_dev_AP50=metrics['held_dev_AP50'], max_candidate_error=dict(maximum), **memory_check(d, c))
        write(work / ('DRIFT_RESULT.json' if measure_drift else 'REPLAY_OK.json'), result)
        print(); del model, bank; torch.cuda.empty_cache(); return result
    finally:
        gate.phase = 'closed'

