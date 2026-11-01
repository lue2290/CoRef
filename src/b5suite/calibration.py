import numpy as np,joblib
from sklearn.metrics import average_precision_score,roc_auc_score
from .scoring import fit_support,class_distances,novelty,fit_quality,objectness,retention_threshold
WEIGHTS=(0.,.25,.5,1.)
def calibrate(data,CAL,n_classes,progress=None):
    CAL.mkdir(parents=True,exist_ok=True)
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
        if set(data["labels"][tr]) != set(range(n_classes)):
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
        for held in range(n_classes):
            train_quality = qt & (data["qc"] != held)
            held_model = fit_quality(data["qx"][train_quality], data["qy"][train_quality])
            held_ref = va & (data["labels"] == held)
            held_quality = qv & (data["qc"] == held)
            if held_ref.any():
                ref_q_pseudo[held_ref] = held_model.predict_proba(data["ref_struct"][held_ref])[:, 1]
            if held_quality.any():
                quality_loco[held_quality] = held_model.predict_proba(data["qx"][held_quality])[:, 1]
        if progress:progress(fold+1,5)
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
        for c in range(n_classes):
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
    return calibration
