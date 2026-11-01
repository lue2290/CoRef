"""B5: train-only calibration, immutable lock, then fixed-test ablations.

Calibration never reads XML annotations or test feature caches.  It reads only
the selected known 100-shot JSON, train images, and train feature caches.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.ops import batched_nms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "b3_b4"))
from train_eval import RegionHead, cache_path, read_feature_cache, selected_shots, GATE_FLOOR
from common import config, paths, names, iou_matrix, write_json, sha256, read_gt
from evaluate_detection import official_eval, error_counts
from scoring import (STAGES, REASONS, representation, class_distances, fit_support,
                     novelty, sigmoid, fit_quality, objectness, retention_threshold)

OUT = HERE / "outputs"
CAL = OUT / "calibration"
SOURCE = ROOT / "experiments" / "b3_b4" / "outputs" / "B4D3L"
WEIGHTS = (0.0, .25, .5, 1.0)
PROTOCOL = {
    "version": "B5-static-v1", "encoder": "B4D3L", "knn": 5,
    "folds": 5, "split_seed": 42, "radius_quantile": .95,
    "known_rejection_budget": .05, "quality_positive_iou": .5,
    "quality_negative_iou": .2, "quality_samples_per_object_per_group": 3,
    "quality_features": ["spatial_cosine_mean", "spatial_cosine_min",
                         "spatial_cosine_std", "region_context_cosine_distance"],
    "quality_model": "StandardScaler + balanced LogisticRegression(C=1)",
    "quality_weights": list(WEIGHTS),
    "objectness_formula": "pf * ((1-weight) + weight*quality)",
    "unknown_formula": "sigmoid(min_class_normalized_knn_distance - known_shift)",
    "stages": {
        "support_only": "old foreground gate; kNN novelty; no new objectness gate",
        "quality_only": "new objectness gate; original entropy-vs-pmax decision",
        "full": "new objectness gate; kNN novelty; factorized known/unknown scores"},
    "decision_reasons": REASONS,
    "retention_unit": "best IoU>=0.5 proposal per selected known object",
    "calibration_labels": "selected known 100-shot only; no true unknown XML",
}


def collect_training(cfg, p):
    shots, counts = selected_shots(cfg, p)
    refs, ref_struct, ref_labels, ref_images, ref_ids, ref_pf = [], [], [], [], [], []
    qx, qy, qc, qi, qids, qpf = [], [], [], [], [], []
    all_classes, all_images = [], []
    missed = Counter()
    hashes = {}
    for image_name, items in shots.items():
        source_file = cache_path("B4D3L", "train", image_name)
        boxes, pf, features = read_feature_cache(source_file)
        hashes[str(source_file)] = sha256(source_file)
        hashes[str(p["images"] / image_name)] = sha256(p["images"] / image_name)
        z, structural = representation(features)
        with Image.open(p["images"] / image_name) as im:
            w, h = im.size
        gt = np.asarray([[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for _, b in items], np.float32)
        overlap = iou_matrix(boxes, gt)
        centers = (boxes[:, :2]+boxes[:, 2:])/2
        contains = ((centers[:, None, :] >= gt[None, :, :2]).all(axis=2) &
                    (centers[:, None, :] <= gt[None, :, 2:]).all(axis=2))
        eligible = np.where(contains, overlap, -1)
        owner = eligible.argmax(axis=1)
        has_owner = contains.any(axis=1)
        for j, (c, _) in enumerate(items):
            object_id = f"{image_name}#{j}"
            all_classes.append(c)
            all_images.append(image_name)
            best = int(overlap[:, j].argmax())
            if overlap[best, j] >= .5:
                refs.append(z[best]); ref_struct.append(structural[best]); ref_labels.append(c)
                ref_images.append(image_name); ref_ids.append(object_id); ref_pf.append(pf[best])
            else:
                missed[c] += 1
            owned = np.flatnonzero(has_owner & (owner == j))
            for good in (1, 0):
                if good:
                    ids = owned[overlap[owned, j] >= .5]
                else:
                    ids = owned[overlap[owned, j] <= .2]
                # Closest qualified boxes; hardest low-IoU localization proxies.
                ids = ids[np.argsort(-overlap[ids, j], kind="stable")[:3]]
                for index in ids:
                    qx.append(structural[index]); qy.append(good); qc.append(c)
                    qi.append(image_name); qids.append(object_id); qpf.append(pf[index])
    data = {"z": np.asarray(refs, np.float32), "ref_struct": np.asarray(ref_struct, np.float32),
            "labels": np.asarray(ref_labels, np.int64), "images": np.asarray(ref_images),
            "object_ids": np.asarray(ref_ids), "pf": np.asarray(ref_pf, np.float32),
            "qx": np.asarray(qx, np.float32), "qy": np.asarray(qy, np.int64),
            "qc": np.asarray(qc, np.int64), "qi": np.asarray(qi),
            "q_object_ids": np.asarray(qids), "qpf": np.asarray(qpf, np.float32)}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    image_fold = {}
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(all_classes)),
                                                  all_classes, all_images)):
        train_images = {all_images[i] for i in tr}
        val_images = {all_images[i] for i in va}
        if train_images & val_images:
            raise ValueError("Image leakage between support and query folds")
        image_fold.update({image: fold for image in val_images})
    data["fold"] = np.asarray([image_fold[n] for n in data["images"]], np.int64)
    data["qfold"] = np.asarray([image_fold[n] for n in data["qi"]], np.int64)
    meta = {"shots": counts, "reference_objects": len(refs), "reference_images": len(set(ref_images)),
            "reference_classes": dict(Counter(ref_labels)), "unmatched": dict(missed),
            "quality_good": int(sum(qy)), "quality_poor": int(len(qy)-sum(qy)),
            "quality_by_class_and_label": dict(Counter(f"{c}:{y}" for c, y in zip(qc, qy))),
            "image_folds": image_fold,
            "quality_note": "Poor localization around selected known objects, not verified background"}
    return data, meta, hashes


def calibrate():
    if (CAL / "LOCK.json").exists():
        raise FileExistsError("Calibration is already locked; use existing lock for evaluation")
    cfg = config(); p = paths(cfg)
    CAL.mkdir(parents=True, exist_ok=True)
    data, meta, input_hashes = collect_training(cfg, p)
    if len(data["z"]) != 390:
        raise ValueError(f"Expected 390 matched reference objects, got {len(data['z'])}")
    np.savez_compressed(CAL / "training_samples.npz", **data)
    write_json(CAL / "training_manifest.json", meta)
    n = len(data["z"]); nq = len(data["qx"])
    known_novelty = np.full(n, np.nan, np.float32)
    pseudo_novelty = np.full(n, np.nan, np.float32)
    ref_q_known = np.full(n, np.nan, np.float32)
    ref_q_pseudo = np.full(n, np.nan, np.float32)
    quality_oof = np.full(nq, np.nan, np.float32)
    quality_loco = np.full(nq, np.nan, np.float32)
    for fold in range(5):
        tr = data["fold"] != fold; va = ~tr
        qt = data["qfold"] != fold; qv = ~qt
        if set(data["labels"][tr]) != set(range(4)):
            raise ValueError(f"Fold {fold} support is missing a known class")
        radius, scale = fit_support(data["z"][tr], data["labels"][tr], data["images"][tr])
        distances = class_distances(data["z"][va], data["z"][tr], data["labels"][tr])
        known_novelty[va] = novelty(distances, radius, scale)
        removed = distances.copy()
        removed[np.arange(va.sum()), data["labels"][va]] = np.inf
        pseudo_novelty[va] = novelty(removed, radius, scale)
        model = fit_quality(data["qx"][qt], data["qy"][qt])
        quality_oof[qv] = model.predict_proba(data["qx"][qv])[:, 1]
        ref_q_known[va] = model.predict_proba(data["ref_struct"][va])[:, 1]
        for held in range(4):
            train_quality = qt & (data["qc"] != held)
            held_model = fit_quality(data["qx"][train_quality], data["qy"][train_quality])
            held_ref = va & (data["labels"] == held)
            held_quality = qv & (data["qc"] == held)
            if held_ref.any():
                ref_q_pseudo[held_ref] = held_model.predict_proba(data["ref_struct"][held_ref])[:, 1]
            if held_quality.any():
                quality_loco[held_quality] = held_model.predict_proba(data["qx"][held_quality])[:, 1]
        print(f"Calibration fold {fold+1}/5: {tr.sum()} support / {va.sum()} queries", flush=True)
    for array in (known_novelty, pseudo_novelty, ref_q_known, ref_q_pseudo, quality_oof, quality_loco):
        if not np.isfinite(array).all():
            raise ValueError("Incomplete or invalid out-of-fold calibration")
    shift = max(0.0, float(np.quantile(known_novelty, .95, method="higher")))
    y_novelty = np.r_[np.zeros(n), np.ones(n)]
    score_novelty = np.r_[known_novelty, pseudo_novelty]
    trials = []
    for weight in WEIGHTS:
        qscore = objectness(data["qpf"], quality_loco, weight)
        known_o = objectness(data["pf"], ref_q_known, weight)
        pseudo_o = objectness(data["pf"], ref_q_pseudo, weight)
        threshold = min(retention_threshold(known_o), retention_threshold(pseudo_o))
        aps = []
        for c in range(4):
            mask = data["qc"] == c
            if len(np.unique(data["qy"][mask])) == 2:
                aps.append(float(average_precision_score(data["qy"][mask], qscore[mask])))
        if not aps:
            raise ValueError("No held-out class has both quality labels")
        trials.append({"weight": weight, "threshold": threshold,
                       "macro_loco_quality_AP": float(np.mean(aps)),
                       "known_retention": float((known_o >= threshold).mean()),
                       "pseudo_unknown_retention": float((pseudo_o >= threshold).mean()),
                       "poor_localization_pass_rate": float((qscore[data["qy"] == 0] >= threshold).mean())})
    base = trials[0]
    eligible = [t for t in trials if t["known_retention"] >= .95 and t["pseudo_unknown_retention"] >= .95
                and t["macro_loco_quality_AP"] > base["macro_loco_quality_AP"]+1e-6]
    chosen = max(eligible, key=lambda t: (t["macro_loco_quality_AP"], -t["weight"])) if eligible else base
    radius, scale = fit_support(data["z"], data["labels"], data["images"])
    np.savez_compressed(CAL / "gallery.npz", z=data["z"], labels=data["labels"], images=data["images"],
                        object_ids=data["object_ids"], radius=radius, scale=scale)
    quality_model = fit_quality(data["qx"], data["qy"])
    joblib.dump(quality_model, CAL / "quality.joblib")
    np.savez_compressed(CAL / "validation_predictions.npz", known_novelty=known_novelty,
                        pseudo_novelty=pseudo_novelty, quality_oof=quality_oof,
                        quality_loco=quality_loco, ref_q_known=ref_q_known, ref_q_pseudo=ref_q_pseudo)
    calibration = {"known_shift": shift, "unknown_threshold": .5,
                   "known_reference_false_unknown_rate": float((known_novelty > shift).mean()),
                   "pseudo_unknown_recall": float((pseudo_novelty > shift).mean()),
                   "novelty_AUROC": float(roc_auc_score(y_novelty, score_novelty)),
                   "novelty_AUPR": float(average_precision_score(y_novelty, score_novelty)),
                   "weight_trials": trials, "chosen": chosen,
                   "quality_effective": chosen["weight"] > 0,
                   "radius": radius.tolist(), "scale": scale.tolist(),
                   "validation_caveat": "Four known-class holdouts do not validate fine-grained near-known novelty"}
    write_json(CAL / "parameters.json", calibration)
    input_hashes[str(p["sets"] / "v2_shots_100.json")] = sha256(p["sets"] / "v2_shots_100.json")
    input_hashes[str(p["sets"] / "train.txt")] = sha256(p["sets"] / "train.txt")
    input_hashes[str(p["sets"] / "test.txt")] = sha256(p["sets"] / "test.txt")
    for seed in cfg["seeds"]:
        path = SOURCE / f"seed_{seed}" / "head.pth"
        input_hashes[str(path)] = sha256(path)
    artifacts = {str(path): sha256(path) for path in CAL.iterdir() if path.is_file()}
    code_files = (HERE / "run_b5.py", HERE / "scoring.py",
                  ROOT / "experiments" / "b3_b4" / "train_eval.py",
                  ROOT / "baseline" / "scripts" / "common.py",
                  ROOT / "baseline" / "scripts" / "evaluate_detection.py")
    code = {str(path): sha256(path) for path in code_files}
    lock = {"locked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "protocol": PROTOCOL, "baseline_config": cfg, "training_inputs": input_hashes,
            "calibration_artifacts": artifacts, "code": code,
            "test_feature_or_annotation_read_before_lock": False}
    write_json(CAL / "LOCK.json", lock)
    print(json.dumps(calibration, ensure_ascii=True), flush=True)


def verified_lock():
    lock_file = CAL / "LOCK.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    for group in ("training_inputs", "calibration_artifacts", "code"):
        for filename, expected in lock[group].items():
            if sha256(filename) != expected:
                raise ValueError(f"Locked input changed: {filename}")
    if config() != lock["baseline_config"]:
        raise ValueError("Baseline config changed after calibration")
    return lock, sha256(lock_file)


def unknown_error_counts(results, p, cfg):
    counts = Counter()
    for pred in results:
        gt = read_gt(p["xml"] / (Path(pred["image"]).stem+".xml"), cfg["known_classes"])
        ub = [box for c, box in gt if c == "unknown"]
        kb = [box for c, box in gt if c != "unknown"]
        boxes = np.asarray(pred["boxes"], np.float32).reshape(-1, 4)
        labels = np.asarray(pred["labels"], np.int64)
        scores = np.asarray(pred["scores"], np.float32)
        ui = np.flatnonzero(labels == 4)
        iu = iou_matrix(boxes, ub); ik = iou_matrix(boxes, kb)
        counts["unknown_gt"] += len(ub)
        counts["unknown_predictions"] += len(ui)
        matched = set()
        for j in ui[np.argsort(-scores[ui])]:
            best_u = int(iu[j].argmax()) if len(ub) else -1
            if len(ub) and iu[j, best_u] >= .5:
                if best_u not in matched:
                    matched.add(best_u); counts["unknown_true_positive"] += 1
                else:
                    counts["unknown_duplicate_fp"] += 1
            elif len(kb) and ik[j].max() >= .5:
                counts["unknown_overlaps_known_fp"] += 1
            else:
                counts["unknown_background_or_low_iou_fp"] += 1
        if len(ub) and (labels != 4).any():
            counts["unknown_gt_hit_by_known_detection"] += int(
                (iu[labels != 4].max(axis=0) >= .5).sum())
    out = dict(counts)
    out["unknown_background_fp_per_image"] = counts["unknown_background_or_low_iou_fp"] / len(results)
    out["diagnostic_unknown_recall_percent"] = 100*counts["unknown_true_positive"]/max(counts["unknown_gt"], 1)
    return out


def official_extra(log_path):
    out = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        for prefix, key in (("Unknown Recall50:", "unknown_recall_percent"),
                            ("Unknown Precisions50:", "unknown_precision_percent"),
                            ("Known Recall50:", "known_recall_percent")):
            if line.startswith(prefix):
                out[key] = float(line.split(":", 1)[1].strip())
    return out


def decide(stage, boxes, pf, old_gate, probs, entropy, u, quality, params, cfg):
    if stage == "support_only":
        o = pf*old_gate
        accepted = np.ones(len(boxes), bool)
    else:
        o = objectness(pf, quality, params["chosen"]["weight"])
        accepted = o >= params["chosen"]["threshold"]
    if stage == "quality_only":
        evidence = entropy
        known_scores = o*probs.max(axis=1)
        unknown_scores = o*entropy
        labels = np.where(entropy > probs.max(axis=1), 4, probs.argmax(axis=1)).astype(np.int64)
    else:
        evidence = u
        known_scores = o*(1-u)*probs.max(axis=1)
        unknown_scores = o*u
        labels = np.where(u > .5, 4, probs.argmax(axis=1)).astype(np.int64)
    scores = np.where(labels == 4, unknown_scores, known_scores).astype(np.float32)
    candidates = np.flatnonzero(accepted)
    nms = batched_nms(torch.from_numpy(boxes[candidates]), torch.from_numpy(scores[candidates]),
                      torch.from_numpy(labels[candidates]), cfg["nms_iou"]).numpy()
    kept_nms = candidates[nms]
    kept = kept_nms[:cfg["detection_limit"]]
    reasons = np.zeros(len(boxes), np.int8)
    reasons[candidates] = 1; reasons[kept_nms] = 2
    reasons[kept] = np.where(labels[kept] == 4, 4, 3)
    return kept, labels, scores, o, evidence, reasons


def evaluate(seeds, device):
    lock, lock_sha = verified_lock()
    cfg = lock["baseline_config"]; p = paths(cfg)
    if not set(seeds).issubset(cfg["seeds"]):
        raise ValueError("Only precommitted seeds 42/43/44 are allowed")
    train_manifest = json.loads((SOURCE / "cache_train_manifest.json").read_text(encoding="utf-8"))
    test_manifest = json.loads((SOURCE / "cache_test_manifest.json").read_text(encoding="utf-8"))
    if train_manifest["encoder_signature"] != test_manifest["encoder_signature"] or test_manifest["count"] != 319:
        raise ValueError("Test feature cache differs from calibrated encoder or split")
    params = json.loads((CAL / "parameters.json").read_text(encoding="utf-8"))
    with np.load(CAL / "gallery.npz", allow_pickle=False) as gallery:
        gz, gy = gallery["z"].copy(), gallery["labels"].copy()
        radius, scale = gallery["radius"].copy(), gallery["scale"].copy()
    quality_model = joblib.load(CAL / "quality.joblib")
    test_images = names(p["sets"] / "test.txt")
    for seed in seeds:
        completed = [OUT / stage / f"seed_{seed}" / "metrics.json" for stage in STAGES]
        if all(path.exists() for path in completed):
            for path in completed:
                if json.loads(path.read_text(encoding="utf-8"))["calibration_lock_sha256"] != lock_sha:
                    raise ValueError("Existing result belongs to a different lock")
            print(f"Seed {seed} already complete with this lock; skipped", flush=True)
            continue
        if any(path.exists() for path in completed):
            raise FileExistsError(f"Partial seed {seed} results exist; preserve and inspect before resuming")
        saved = torch.load(SOURCE / f"seed_{seed}" / "head.pth", map_location="cpu", weights_only=False)
        head = RegionHead(saved["dimension"]).to(device).eval()
        head.load_state_dict(saved["state_dict"])
        predictions = {stage: [] for stage in STAGES}
        counters = {stage: Counter() for stage in STAGES}
        evidence_dir = OUT / "proposal_evidence" / f"seed_{seed}"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for index, image_name in enumerate(test_images):
            boxes, pf, features = read_feature_cache(cache_path("B4D3L", "test", image_name))
            start = time.perf_counter()
            z, structure = representation(features)
            distances = class_distances(z, gz, gy, device=device)
            u = sigmoid(novelty(distances, radius, scale)-params["known_shift"])
            quality = quality_model.predict_proba(structure)[:, 1].astype(np.float32)
            x = torch.from_numpy(((features-saved["mean"])/saved["std"]).astype(np.float32)).to(device)
            with torch.inference_mode():
                logits, fg_logits = head(x)
                probs = logits.softmax(dim=1).cpu().numpy()
                old_gate = GATE_FLOOR+(1-GATE_FLOOR)*fg_logits.sigmoid().cpu().numpy()
            entropy = -(probs*np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)/math.log(4)
            common_seconds = time.perf_counter()-start
            per_stage = []
            for stage in STAGES:
                stage_start = time.perf_counter()
                kept, labels, scores, o, evidence, reasons = decide(
                    stage, boxes, pf, old_gate, probs, entropy, u, quality, params, cfg)
                predictions[stage].append({"image": image_name,
                    "image_id": f"{index}-^-{Path(image_name).stem}",
                    "boxes": boxes[kept].tolist(), "labels": labels[kept].tolist(),
                    "scores": scores[kept].tolist(),
                    "decision_seconds": common_seconds+time.perf_counter()-stage_start})
                counters[stage].update({REASONS[c]: int((reasons == c).sum()) for c in REASONS})
                per_stage.append((labels, scores, o, evidence, reasons))
            np.savez_compressed(evidence_dir / (Path(image_name).stem+".npz"),
                boxes=boxes, pf_objectness=pf, old_foreground_gate=old_gate,
                quality=quality, class_distances=distances,
                known_support=1-u, unknown_evidence=u, known_probabilities=probs,
                stage_names=np.asarray(STAGES), stage_labels=np.stack([v[0] for v in per_stage], axis=1),
                stage_scores=np.stack([v[1] for v in per_stage], axis=1),
                stage_objectness=np.stack([v[2] for v in per_stage], axis=1),
                stage_unknown_evidence=np.stack([v[3] for v in per_stage], axis=1),
                decision_reason=np.stack([v[4] for v in per_stage], axis=1),
                calibration_lock_sha256=np.asarray(lock_sha))
            if (index+1) % 100 == 0:
                print(f"Seed {seed}: {index+1}/319 images, three frozen ablations", flush=True)
        peak = float(torch.cuda.max_memory_allocated(device)/2**30) if str(device).startswith("cuda") else 0.0
        write_json(evidence_dir / "manifest.json", {"images": len(test_images), "seed": seed,
            "calibration_lock_sha256": lock_sha, "stage_columns": list(STAGES),
            "decision_reason_codes": REASONS, "reason_counts": {k: dict(v) for k, v in counters.items()},
            "peak_vram_gib": peak, "memory_scope": "cached scoring only, excludes PF-RPN and DINO encoders"})
        for stage in STAGES:
            target = OUT / stage / f"seed_{seed}"
            target.mkdir(parents=True, exist_ok=True)
            results = predictions[stage]
            write_json(target / "predictions.json", results)
            metrics = official_eval(results, p, cfg, target / "fomo_eval.log")
            metrics.update(official_extra(target / "fomo_eval.log"))
            errors = error_counts(results, p, cfg)
            metrics.update({k: v for k, v in errors.items() if k != "per_image"})
            metrics.update(unknown_error_counts(results, p, cfg))
            metrics.update({"stage": stage, "seed": seed, "calibration_lock_sha256": lock_sha,
                "decision_seconds_per_image": float(np.mean([x["decision_seconds"] for x in results])),
                "peak_scoring_vram_gib": peak, "detections": sum(len(x["boxes"]) for x in results),
                "decision_reason_counts": dict(counters[stage])})
            write_json(target / "errors_by_image.json", errors["per_image"])
            write_json(target / "metrics.json", metrics)
            print(f"{stage} seed {seed}: K {metrics['K_AP50']:.3f}, U {metrics['U_AP50']:.3f}, "
                  f"U-recall {metrics['unknown_recall_percent']:.2f}, "
                  f"U-background FP/image {metrics['unknown_background_fp_per_image']:.2f}", flush=True)


def summarize():
    lock, lock_sha = verified_lock()
    cfg = lock["baseline_config"]; p = paths(cfg)
    params = json.loads((CAL / "parameters.json").read_text(encoding="utf-8"))
    summary = {"calibration_lock_sha256": lock_sha, "calibration": params, "methods": {}}
    fields = ("K_AP50", "U_AP50", "unknown_recall_percent", "background_fp_per_image",
              "unknown_background_fp_per_image", "known_gt_flagged_unknown_rate",
              "decision_seconds_per_image")
    for stage in ("B4D3L", *STAGES):
        rows = []
        for seed in cfg["seeds"]:
            target = SOURCE / f"seed_{seed}" if stage == "B4D3L" else OUT / stage / f"seed_{seed}"
            row = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
            if stage == "B4D3L":
                row.update(official_extra(target / "fomo_eval.log"))
                preds = json.loads((target / "predictions.json").read_text(encoding="utf-8"))
                row.update(unknown_error_counts(preds, p, cfg))
            elif row["calibration_lock_sha256"] != lock_sha:
                raise ValueError("Result lock mismatch")
            rows.append(row)
        stats = {f: {"mean": float(np.mean([r[f] for r in rows])),
                     "std": float(np.std([r[f] for r in rows], ddof=1))} for f in fields}
        summary["methods"][stage] = {"seeds": list(cfg["seeds"]), "metrics": rows, "mean_std": stats}
    baseline = summary["methods"]["B4D3L"]["mean_std"]
    for stage in STAGES:
        stats = summary["methods"][stage]["mean_std"]
        tests = {"U_AP50_above_8_60": stats["U_AP50"]["mean"] > 8.60,
                 "K_AP50_at_least_57_58": stats["K_AP50"]["mean"] >= 57.58,
                 "unknown_background_FP_reduced": stats["unknown_background_fp_per_image"]["mean"] <
                                                   baseline["unknown_background_fp_per_image"]["mean"]}
        summary["methods"][stage]["acceptance"] = {**tests, "qualified_main_baseline": all(tests.values())}
    write_json(OUT / "B5_RESULTS.json", summary)
    lines = ["# B5 分离评分：Aquatic Task 1 实验结果", "",
        "固定 B0 候选、DINOv3-L/16 特征及现有三种子四分类头；100-shot 训练标注，FOMO 官方 VOC07 AP50。",
        "校准只读取选定已知类训练标注；五折按图像分组，轮流留类验证。测试集真值只用于最后评测和误差分析。", "",
        f"参数锁定时间（UTC）：{lock['locked_at_utc']}；锁文件 SHA-256：`{lock_sha}`。", "",
        "| 方法 | K-AP50 | U-AP50 | 未知召回 % | 未知背景/定位不足 FP/图 | 已知 GT 被未知框命中 % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage, item in summary["methods"].items():
        stat = item["mean_std"]
        cells = []
        for field in ("K_AP50", "U_AP50", "unknown_recall_percent", "unknown_background_fp_per_image",
                      "known_gt_flagged_unknown_rate"):
            mult = 100 if field == "known_gt_flagged_unknown_rate" else 1
            cells.append(f"{stat[field]['mean']*mult:.2f} ± {stat[field]['std']*mult:.2f}")
        lines.append(f"| {stage} | "+" | ".join(cells)+" |")
    lines += ["", "以上为均值 ± 样本标准差；三种子复用原分类头，参考库和校准划分固定。", "",
        "## 校准记录", "",
        f"390 个参考目标，五折已知误拒率 {params['known_reference_false_unknown_rate']*100:.2f}%；"
        f"留类伪未知召回 {params['pseudo_unknown_recall']*100:.2f}%，AUROC {params['novelty_AUROC']:.4f}。",
        f"选择质量混合权重 {params['chosen']['weight']}，目标性阈值 {params['chosen']['threshold']:.6f}。",
        "权重 0 表示定位质量模块未通过训练验证的增益要求，不能宣称该模块有效。", "",
        "## 固定消融定义", "",
        "- support_only：仅改已知支持与未知评分，保留旧前景门控，不增加目标性拒绝。",
        "- quality_only：替换旧前景门控并增加目标性拒绝，保留原熵与最大概率的类别比较。",
        "- full：定位质量目标性、独立背景拒绝、距离式未知证据及分层评分同时使用。", "",
        "## 验收", ""]
    for stage in STAGES:
        check = summary["methods"][stage]["acceptance"]
        lines.append(f"- {stage}：U>8.60 {'通过' if check['U_AP50_above_8_60'] else '未通过'}；"
            f"K≥57.58 {'通过' if check['K_AP50_at_least_57_58'] else '未通过'}；"
            f"未知背景误检减少 {'通过' if check['unknown_background_FP_reduced'] else '未通过'}；"
            f"{'达到' if check['qualified_main_baseline'] else '未达到'}新主基线标准。")
    lines += ["", "## 限制与证据文件", "",
        "定位质量的负例是已知框附近定位不足的候选，不是真实背景标注；类间留出也不充分模拟近已知类别。",
        "未知背景误检按与全部已标注 GT 的 IoU<0.5 定义，可能包含未标注物体或定位不足的框。",
        "评分耗时和显存只覆盖读取缓存后的评分计算，不包含原始 PF-RPN/DINO 图像编码。",
        "所有参数在测试评测前锁定；无论结果是否通过验收，保留当前结果，不继续按测试集调参。", "",
        "逐框证据在 `proposal_evidence/seed_*/`，各列及拒绝原因在其 manifest.json 中说明；"
        "各消融目录保存 predictions.json、metrics.json、fomo_eval.log 和 errors_by_image.json。", ""]
    (OUT / "B5_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(OUT / "B5_RESULTS.md"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "evaluate", "summarize"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "calibrate":
        def train_only_guard(event, arguments):
            if event == "open" and isinstance(arguments[0], (str, bytes)):
                filename = str(arguments[0]).replace("\\", "/").lower()
                if filename.endswith(".xml") or "/b4d3l/cache/test/" in filename:
                    raise RuntimeError(f"Calibration attempted forbidden annotation/test access: {filename}")
        sys.addaudithook(train_only_guard)
        calibrate()
    elif args.command == "evaluate":
        evaluate(args.seeds, args.device)
    else:
        summarize()


if __name__ == "__main__":
    main()
