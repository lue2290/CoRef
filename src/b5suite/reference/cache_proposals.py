"""Frozen PF-RPN B0 proposals and single-level PF-RPN neck RoI features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import roi_align

from common import config, names, paths, sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--max-images", type=int, default=None, help="Only for smoke tests; omitted for official run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image", type=Path, help="Independent one-image smoke check")
    ap.add_argument("--force", action="store_true", help="Recompute even validated existing caches")
    args = ap.parse_args()
    cfg = config()
    p = paths(cfg)
    model_root = Path(cfg["pf_rpn_root"])
    sys.path.insert(0, str(model_root))
    from mmdet.apis import inference_detector, init_detector

    checkpoint = model_root / cfg["pf_rpn_checkpoint"]
    model_cfg = model_root / cfg["pf_rpn_config"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    config_hash = sha256(model_cfg)
    model = init_detector(str(model_cfg), str(checkpoint), device=args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured = []
    hook = model.neck.register_forward_hook(lambda _m, _i, o: captured.append(o[0]))

    if args.image:
        items = [(args.image.name, args.image)]
    else:
        items = [(n, p["images"] / n) for n in names(p["sets"] / f"{args.split}.txt")]
    if args.max_images:
        items = items[:args.max_images]
    smoke = bool(args.max_images or args.image)
    run_root = p["output"] / ("smoke" if smoke else "B0")
    out_dir = run_root / "cache" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for index, (name, image_path) in enumerate(items, 1):
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            output_file = out_dir / f"{Path(name).stem}.npz"
            image_hash = sha256(image_path)
            if output_file.is_file() and not args.force:
                try:
                    with np.load(output_file, allow_pickle=False) as cached:
                        valid = (str(cached["image_sha256"]) == image_hash and
                                 str(cached["checkpoint_sha256"]) == checkpoint_hash and
                                 str(cached["config_sha256"]) == config_hash and
                                 cached["boxes"].shape[0] <= cfg["proposal_limit"] and
                                 cached["features"].shape == (cached["boxes"].shape[0], 256))
                        if valid:
                            record = {"image": name, "image_id": Path(name).stem,
                                      "proposals": int(cached["boxes"].shape[0]),
                                      "seconds": float(cached["seconds"]),
                                      "pf_rpn_seconds": float(cached["pf_rpn_seconds"]),
                                      "roi_feature_seconds": float(cached["roi_feature_seconds"]),
                                      "peak_vram_gib": float(cached["peak_vram_gib"]),
                                      "image_sha256": image_hash, "cache": str(output_file),
                                      "reused_cache": True}
                            records.append(record)
                            print(f"{index}/{len(items)} {name}: validated cache reused", flush=True)
                            continue
                except (KeyError, ValueError, OSError):
                    pass
            captured.clear()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = inference_detector(model, str(image_path))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if len(captured) != 1:
                raise RuntimeError(f"Expected one neck feature, got {len(captured)}")
            feat = captured.pop()
            predictions = result.pred_instances
            scores = predictions.scores
            order = torch.argsort(scores, descending=True, stable=True)[:cfg["proposal_limit"]]
            boxes = predictions.bboxes[order]
            scores = scores[order]
            scale = result.metainfo["scale_factor"]
            scale_factor = boxes.new_tensor([scale[0], scale[1], scale[0], scale[1]])
            resized_boxes = boxes * scale_factor
            # Swin stage 1 is stride 8 and the first ChannelMapper map retains it.
            pooled = roi_align(feat, [resized_boxes], output_size=(3, 3),
                               spatial_scale=1 / 8, sampling_ratio=2, aligned=True)
            features = pooled.float().mean(dim=(2, 3))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_elapsed = time.perf_counter() - start
            peak_vram = round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else -1.0
            np.savez_compressed(output_file,
                                boxes=boxes.detach().cpu().numpy().astype(np.float32),
                                scores=scores.detach().cpu().numpy().astype(np.float32),
                                features=features.detach().cpu().numpy().astype(np.float16),
                                image_sha256=image_hash, checkpoint_sha256=checkpoint_hash,
                                config_sha256=config_hash, seconds=total_elapsed,
                                pf_rpn_seconds=elapsed, roi_feature_seconds=total_elapsed - elapsed,
                                peak_vram_gib=peak_vram)
            record = {
                "image": name, "image_id": Path(name).stem, "proposals": len(boxes),
                "seconds": total_elapsed, "pf_rpn_seconds": elapsed,
                "roi_feature_seconds": total_elapsed - elapsed,
                "peak_vram_gib": peak_vram if peak_vram >= 0 else None,
                "image_sha256": image_hash,
                "cache": str(output_file), "reused_cache": False,
            }
            records.append(record)
            print(f"{index}/{len(items)} {name}: {len(boxes)} boxes, {total_elapsed:.2f}s, peak {record['peak_vram_gib']} GiB", flush=True)
            del result, feat, predictions, scores, boxes, pooled, features
    finally:
        hook.remove()
    write_json(run_root / f"{args.split}_inference.json", {
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "split": args.split, "smoke_only": smoke,
        "records": records,
    })


if __name__ == "__main__":
    main()
