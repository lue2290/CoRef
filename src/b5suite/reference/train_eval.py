"""Train and evaluate B3/B4 using fixed B0 boxes and known-class 100-shot labels.

The unknown class has no training examples. Weak proxy-background regions are
chosen solely from low-ranked PF-RPN proposals on training images; the gate has
a nonzero floor so unseen objects are not hard-rejected by that proxy task.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.ops import batched_nms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline" / "scripts"))
from common import config, iou_matrix, names, paths, read_cache, write_json  # noqa: E402
from evaluate_detection import error_counts, official_eval  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
EPOCHS = 40
BATCH = 128
LR = 1e-3
NEG_PER_IMAGE = 10
NEG_WEIGHT = 0.15
GATE_FLOOR = 0.25


class RegionHead(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(dimension, 256), nn.LayerNorm(256),
                                    nn.GELU(), nn.Dropout(0.3))
        self.known = nn.Linear(256, 4)
        self.foreground = nn.Linear(256, 1)

    def forward(self, x):
        z = self.shared(x)
        return self.known(z), self.foreground(z).squeeze(1)


def cache_path(variant, split, image_name):
    return OUT / variant / "cache" / split / f"{Path(image_name).stem}.npz"


def read_feature_cache(path):
    with np.load(path, allow_pickle=False) as z:
        return (z["boxes"].astype(np.float32), z["scores"].astype(np.float32),
                z["features"].astype(np.float32))


def check_complete(variant, p, require_test=True):
    signatures = []
    splits = (("train", 318), ("test", 319)) if require_test else (("train", 318),)
    for split, expected in splits:
        image_names = names(p["sets"] / f"{split}.txt")
        if len(image_names) != expected:
            raise ValueError(f"Unexpected {split} set: {len(image_names)}")
        missing = [n for n in image_names if not cache_path(variant, split, n).is_file()]
        if missing:
            raise FileNotFoundError(f"{variant}/{split}: {len(missing)} caches missing, first={missing[0]}")
        manifest_file = OUT / variant / f"cache_{split}_manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(manifest_file)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest["count"] != expected or manifest["split"] != split:
            raise ValueError(f"Incomplete feature manifest: {manifest_file}")
        signatures.append(manifest["encoder_signature"])
    if len(signatures) == 2 and signatures[0] != signatures[1]:
        raise ValueError(f"Train/test feature encoders differ for {variant}")


def selected_shots(cfg, p):
    raw = json.loads((p["sets"] / "v2_shots_100.json").read_text(encoding="utf-8"))
    train_set = set(names(p["sets"] / "train.txt"))
    test_set = set(names(p["sets"] / "test.txt"))
    by_image = defaultdict(list)
    counts = {}
    for class_id, class_name in enumerate(cfg["known_classes"]):
        shots = raw[class_name][:cfg["shots_per_class"]]
        counts[class_name] = len(shots)
        for image_name, box in shots:
            if image_name not in train_set or image_name in test_set:
                raise ValueError(f"Training shot outside fixed train set: {image_name}")
            by_image[image_name].append((class_id, box))
    return by_image, counts


def collect_training(cfg, p, variant, seed):
    by_image, shot_counts = selected_shots(cfg, p)
    rng = np.random.default_rng(seed)
    positives, positive_labels, weak_negatives = [], [], []
    matched, unmatched = Counter(), Counter()
    for image_name in names(p["sets"] / "train.txt"):
        if image_name not in by_image:
            continue
        boxes, scores, features = read_feature_cache(cache_path(variant, "train", image_name))
        base_boxes, _, _ = read_cache(p["output"] / "B0" / "cache" / "train" /
                                      f"{Path(image_name).stem}.npz")
        if not np.array_equal(boxes, base_boxes):
            raise ValueError(f"Changed B0 boxes: {image_name}")
        with Image.open(p["images"] / image_name) as im:
            width, height = im.size
        items = by_image[image_name]
        gt_boxes = np.asarray([[b[0] * width, b[1] * height,
                                b[2] * width, b[3] * height] for _, b in items], dtype=np.float32)
        overlaps = iou_matrix(boxes, gt_boxes)
        assignments = {}
        for shot_idx, (class_id, _) in enumerate(items):
            eligible = np.flatnonzero(overlaps[:, shot_idx] >= cfg["positive_iou"])
            class_name = cfg["known_classes"][class_id]
            if len(eligible) == 0:
                unmatched[class_name] += 1
                continue
            matched[class_name] += 1
            for idx in eligible[np.argsort(-overlaps[eligible, shot_idx])[:3]]:
                quality = float(overlaps[idx, shot_idx])
                if idx not in assignments or quality > assignments[idx][1]:
                    assignments[int(idx)] = (class_id, quality)
        for idx, (class_id, _) in assignments.items():
            positives.append(features[idx])
            positive_labels.append(class_id)
        # Weak proxy negatives only. Unknown objects are not annotated or used.
        low_cutoff = np.quantile(scores, 0.2)
        candidates = np.flatnonzero((scores <= low_cutoff) &
                                    (overlaps.max(axis=1) < 0.05))
        if len(candidates):
            chosen = rng.choice(candidates, size=min(NEG_PER_IMAGE, len(candidates)), replace=False)
            weak_negatives.extend(features[chosen])
    if not positives or not weak_negatives or len(set(positive_labels)) != 4:
        raise ValueError("Insufficient known positives or proxy background samples")
    pos = np.stack(positives).astype(np.float32)
    neg = np.stack(weak_negatives).astype(np.float32)
    x = np.concatenate([pos, neg], axis=0)
    mean = x.mean(axis=0)
    std = x.std(axis=0).clip(min=1e-4)
    x = (x - mean) / std
    known_labels = np.concatenate([np.asarray(positive_labels, dtype=np.int64),
                                   np.full(len(neg), -1, dtype=np.int64)])
    foreground_labels = np.concatenate([np.ones(len(pos), dtype=np.float32),
                                        np.zeros(len(neg), dtype=np.float32)])
    report = {"shot_counts": shot_counts, "matched_shots": dict(matched),
              "unmatched_shots": dict(unmatched), "positive_proposals": len(pos),
              "positive_class_counts": dict(Counter(positive_labels)),
              "weak_proxy_background": len(neg),
              "proxy_negative_rule": "bottom 20% PF-RPN score and IoU<0.05 to selected known shots; <=10/image",
              "negative_loss_weight": NEG_WEIGHT, "gate_floor": GATE_FLOOR}
    return x, known_labels, foreground_labels, mean, std, report


def train(cfg, p, variant, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    x, y_known, y_fg, mean, std, report = collect_training(cfg, p, variant, seed)
    features = torch.from_numpy(x).to(device)
    known = torch.from_numpy(y_known).to(device)
    fg = torch.from_numpy(y_fg).to(device)
    head = RegionHead(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-3)
    losses = []
    for _epoch in range(EPOCHS):
        head.train()
        order = torch.randperm(len(features), device=device)
        total = 0.0
        for ids in order.split(BATCH):
            logits, fg_logits = head(features[ids])
            is_positive = known[ids] >= 0
            cls_loss = F.cross_entropy(logits[is_positive], known[ids][is_positive]) if is_positive.any() else logits.sum() * 0
            fg_loss = F.binary_cross_entropy_with_logits(fg_logits, fg[ids], reduction="none")
            weights = torch.where(is_positive, 1.0, NEG_WEIGHT)
            fg_loss = (fg_loss * weights).sum() / weights.sum()
            loss = cls_loss + 0.5 * fg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        losses.append(total / len(features))
    target = OUT / variant / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.cpu().state_dict(), "mean": mean, "std": std,
                "dimension": x.shape[1], "seed": seed, "epochs": EPOCHS,
                "gate_floor": GATE_FLOOR}, target / "head.pth")
    report.update({"seed": seed, "variant": variant, "epochs": EPOCHS, "losses": losses,
                   "feature_dimension": x.shape[1]})
    write_json(target / "training.json", report)
    return target, report


def predict(cfg, p, variant, seed, device):
    target = OUT / variant / f"seed_{seed}"
    saved = torch.load(target / "head.pth", map_location="cpu", weights_only=False)
    head = RegionHead(saved["dimension"]).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()
    mean = saved["mean"]
    std = saved["std"]
    results = []
    for image_idx, image_name in enumerate(names(p["sets"] / "test.txt")):
        boxes, object_scores, features = read_feature_cache(cache_path(variant, "test", image_name))
        base_boxes, _, _ = read_cache(p["output"] / "B0" / "cache" / "test" /
                                      f"{Path(image_name).stem}.npz")
        if not np.array_equal(boxes, base_boxes):
            raise ValueError(f"Changed B0 boxes: {image_name}")
        started = time.perf_counter()
        x = torch.from_numpy(((features - mean) / std).astype(np.float32)).to(device)
        with torch.inference_mode():
            logits, fg_logits = head(x)
            probs = logits.softmax(dim=1).cpu().numpy()
            fg_prob = fg_logits.sigmoid().cpu().numpy()
        pmax = probs.max(axis=1)
        best_class = probs.argmax(axis=1)
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1) / math.log(4)
        gate = GATE_FLOOR + (1 - GATE_FLOOR) * fg_prob
        known_score = object_scores * gate * pmax
        unknown_score = object_scores * gate * entropy
        labels = np.where(unknown_score > known_score, 4, best_class).astype(np.int64)
        scores = np.where(labels == 4, unknown_score, known_score).astype(np.float32)
        keep = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores),
                           torch.from_numpy(labels), cfg["nms_iou"]).numpy()
        keep = keep[:cfg["detection_limit"]]
        results.append({"image": image_name, "image_id": f"{image_idx}-^-{Path(image_name).stem}",
                        "boxes": boxes[keep].tolist(), "scores": scores[keep].tolist(),
                        "labels": labels[keep].tolist(),
                        "decision_seconds": time.perf_counter() - started})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["B3", "B4", "B4D3", "B4D3L", "B4LB"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    cfg = config()
    if args.seed not in cfg["seeds"]:
        ap.error(f"Seed must be one of {cfg['seeds']}")
    p = paths(cfg)
    if args.skip_train and args.train_only:
        ap.error("--skip-train and --train-only cannot be combined")
    check_complete(args.variant, p, require_test=not args.train_only)
    target = OUT / args.variant / f"seed_{args.seed}"
    if not args.skip_train:
        target, report = train(cfg, p, args.variant, args.seed, args.device)
        print(f"{args.variant} seed {args.seed}: {report['positive_proposals']} positives, "
              f"{report['weak_proxy_background']} weak negatives", flush=True)
    if args.train_only:
        return
    predictions = predict(cfg, p, args.variant, args.seed, args.device)
    if len(predictions) != 319:
        raise ValueError(f"Expected 319 fixed test images, got {len(predictions)}")
    write_json(target / "predictions.json", predictions)
    metrics = official_eval(predictions, p, cfg, target / "fomo_eval.log")
    errors = error_counts(predictions, p, cfg)
    metrics.update({key: value for key, value in errors.items() if key != "per_image"})
    metrics["decision_seconds_per_image"] = float(np.mean([x["decision_seconds"] for x in predictions]))
    metrics["variant"] = args.variant
    metrics["seed"] = args.seed
    write_json(target / "errors_by_image.json", errors["per_image"])
    write_json(target / "metrics.json", metrics)
    print(args.variant, args.seed, metrics, flush=True)


if __name__ == "__main__":
    main()
