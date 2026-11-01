"""Cache B3 spatial PF-RPN or B4 frozen visual-encoder features on B0 boxes.

No labels are read here. Both variants use the exact B0 proposal coordinates.
All cache and Hugging Face paths are on E: to avoid filling the system disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline" / "scripts"))
from common import config, names, paths, read_cache, sha256, write_json  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"


def expanded(boxes: torch.Tensor, width: float, height: float, factor=1.5):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = (boxes[:, 2:] - boxes[:, :2]) * factor / 2
    lo = torch.maximum(centers - sizes, boxes.new_zeros(2))
    hi = torch.minimum(centers + sizes, boxes.new_tensor([width, height]))
    return torch.cat([lo, hi], dim=1)


def pool_spatial(feature_map: torch.Tensor, boxes: torch.Tensor, stride: int,
                 width: float, height: float):
    center = roi_align(feature_map, [boxes], output_size=(2, 2),
                       spatial_scale=1 / stride, sampling_ratio=2, aligned=True)
    context_boxes = expanded(boxes, width, height)
    context = roi_align(feature_map, [context_boxes], output_size=(1, 1),
                        spatial_scale=1 / stride, sampling_ratio=2, aligned=True)
    return torch.cat([center.flatten(1), context.flatten(1)], dim=1)


def b3_extractor(cfg, device):
    model_root = Path(cfg["pf_rpn_root"])
    sys.path.insert(0, str(model_root))
    from mmdet.apis import inference_detector, init_detector

    ckpt = model_root / cfg["pf_rpn_checkpoint"]
    model_cfg = model_root / cfg["pf_rpn_config"]
    model = init_detector(str(model_cfg), str(ckpt), device=device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    captured = []
    hook = model.neck.register_forward_hook(lambda _m, _i, o: captured.append(o[0]))
    signature = sha256(ckpt) + ":" + sha256(model_cfg) + ":spatial2-context1-v1"

    def extract(image_path, boxes, old_features):
        captured.clear()
        result = inference_detector(model, str(image_path))
        if len(captured) != 1:
            raise RuntimeError(f"Expected one PF-RPN neck map, got {len(captured)}")
        feat = captured.pop()
        scale = result.metainfo["scale_factor"]
        with Image.open(image_path) as im:
            width, height = im.size
        scaled = torch.from_numpy(boxes).to(feat.device) * feat.new_tensor(
            [scale[0], scale[1], scale[0], scale[1]])
        with torch.inference_mode():
            new = pool_spatial(feat, scaled, 8, width * scale[0], height * scale[1])
            old = torch.from_numpy(old_features).to(new.device)
            combined = torch.cat([old, new.float()], dim=1)
        del result, feat
        return combined.cpu().numpy()

    return extract, signature, hook


def b4_extractor(model_dir: Path, device: str, max_side: int):
    from transformers import AutoImageProcessor, AutoModel

    if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Visual encoder weights missing in {model_dir}")
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                                      use_safetensors=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    use_half = str(device).startswith("cuda")
    if use_half:
        model.half()
    mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32).view(3, 1, 1)
    patch = int(getattr(model.config, "patch_size", 14))
    signature = sha256(model_dir / "model.safetensors") + ":" + sha256(model_dir / "config.json")
    signature += f":full-image-patches-{max_side}-spatial2-context1-v1"

    def extract(image_path, boxes, _old_features):
        with Image.open(image_path) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = im.size
            ratio = max_side / max(width, height)
            new_width = max(patch, round(width * ratio / patch) * patch)
            new_height = max(patch, round(height * ratio / patch) * patch)
            im = im.resize((new_width, new_height), Image.Resampling.BICUBIC)
            pixels = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        pixels = ((pixels - mean) / std).unsqueeze(0).to(device)
        if use_half:
            pixels = pixels.half()
        with torch.inference_mode():
            tokens = model(pixel_values=pixels).last_hidden_state
            grid_h, grid_w = new_height // patch, new_width // patch
            # Works for DINOv2 (CLS) and DINOv3 (CLS + register tokens).
            patches = tokens[:, -grid_h * grid_w:, :]
            if patches.shape[1] != grid_h * grid_w:
                raise ValueError("Unexpected visual-encoder patch count")
            feat = patches.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous()
            mapped = torch.from_numpy(boxes).to(device=device, dtype=feat.dtype) * feat.new_tensor(
                [new_width / width, new_height / height] * 2)
            pooled = pool_spatial(feat, mapped, patch, new_width, new_height)
        return pooled.float().cpu().numpy()

    return extract, signature, None


def dinov3_timm_extractor(model_dir: Path, device: str, max_side: int,
                          architecture: str, precision: str):
    """Load a public timm DINOv3 checkpoint without contacting the Hub."""
    import timm
    from safetensors.torch import load_file

    config_file = model_dir / "config.json"
    weights_file = model_dir / "model.safetensors"
    if not config_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError(f"timm DINOv3 config/weights missing in {model_dir}")
    model_config = json.loads(config_file.read_text(encoding="utf-8"))
    if (model_config.get("architecture") != architecture or
            model_config.get("pretrained_cfg", {}).get("tag") != "lvd1689m"):
        raise ValueError(f"Unexpected timm DINOv3 architecture or pretraining: {config_file}")
    if precision == "fp16" and not str(device).startswith("cuda"):
        raise ValueError("fp16 DINOv3 extraction requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = timm.create_model(architecture, pretrained=False, num_classes=0)
    model.load_state_dict(load_file(str(weights_file), device="cpu"), strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    patch = int(model.patch_embed.patch_size[0])
    mean = torch.tensor(model_config["pretrained_cfg"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(model_config["pretrained_cfg"]["std"], dtype=torch.float32).view(3, 1, 1)
    signature = (sha256(weights_file) + ":" + sha256(config_file) +
                 f":timm-{timm.__version__}:{precision}-full-image-patches-{max_side}-spatial2-context1-v1")

    def extract(image_path, boxes, _old_features):
        with Image.open(image_path) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = im.size
            ratio = max_side / max(width, height)
            new_width = max(patch, round(width * ratio / patch) * patch)
            new_height = max(patch, round(height * ratio / patch) * patch)
            im = im.resize((new_width, new_height), Image.Resampling.BICUBIC)
            pixels = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        pixels = ((pixels - mean) / std).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.inference_mode():
            tokens = model.forward_features(pixels)
            grid_h, grid_w = new_height // patch, new_width // patch
            patch_count = grid_h * grid_w
            if tokens.ndim != 3 or tokens.shape[1] != patch_count + model.num_prefix_tokens:
                raise ValueError(f"Unexpected DINOv3 token shape: {tuple(tokens.shape)}")
            patches = tokens[:, -patch_count:, :]
            feature_map = patches.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous()
            mapped = torch.from_numpy(boxes).to(device=device, dtype=feature_map.dtype) * feature_map.new_tensor(
                [new_width / width, new_height / height] * 2)
            pooled = pool_spatial(feature_map, mapped, patch, new_width, new_height)
        return pooled.float().cpu().numpy()

    return extract, signature, None


def lingbot_large_extractor(model_dir: Path, device: str, max_side: int):
    """Pool frozen LingBot-Vision-Large patch tokens on the unchanged B0 boxes."""
    source_root = ROOT / "upstream" / "lingbot_vision"
    if not (source_root / "lingbot_vision" / "__init__.py").is_file():
        raise FileNotFoundError(f"LingBot-Vision source missing in {source_root}")
    sys.path.insert(0, str(source_root))
    from lingbot_vision import extract_patch_tokens, load_pretrained_backbone

    weights_file = model_dir / "model.pt"
    config_file = source_root / "lingbot_vision" / "configs" / "lbot_vision_vitl.yaml"
    if not weights_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"LingBot-Vision-Large weights/config missing: {weights_file}, {config_file}")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model, embed_dim = load_pretrained_backbone(
        repo_id_or_path=str(model_dir), variant="large", device=device, dtype=dtype,
        local_files_only=True, verbose=True)
    patch = int(model.patch_size)
    if embed_dim != 1024 or patch != 16:
        raise ValueError(f"Unexpected LingBot-Vision-Large dimensions: {embed_dim}, patch {patch}")
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    signature = (sha256(weights_file) + ":" + sha256(config_file) +
                 f":lingbot-large-{dtype}-full-image-patches-{max_side}-spatial2-context1-v1")

    def extract(image_path, boxes, _old_features):
        with Image.open(image_path) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = im.size
            ratio = max_side / max(width, height)
            new_width = max(patch, round(width * ratio / patch) * patch)
            new_height = max(patch, round(height * ratio / patch) * patch)
            im = im.resize((new_width, new_height), Image.Resampling.BICUBIC)
            pixels = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        pixels = ((pixels - mean) / std).unsqueeze(0)
        with torch.inference_mode():
            tokens, (grid_h, grid_w) = extract_patch_tokens(model, pixels, str(device), dtype)
            if (grid_h, grid_w) != (new_height // patch, new_width // patch):
                raise ValueError(f"Unexpected LingBot patch grid: {grid_h}x{grid_w}")
            feature_map = tokens.reshape(1, grid_h, grid_w, embed_dim).permute(0, 3, 1, 2).contiguous()
            mapped = torch.from_numpy(boxes).to(device=device, dtype=feature_map.dtype) * feature_map.new_tensor(
                [new_width / width, new_height / height] * 2)
            pooled = pool_spatial(feature_map, mapped, patch, new_width, new_height)
        return pooled.float().cpu().numpy()

    return extract, signature, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["B3", "B4", "B4D3", "B4D3L", "B4LB"], required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="Local model directory; defaults to the matching upstream encoder directory")
    ap.add_argument("--max-side", type=int, default=518)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="timm DINOv3 extraction precision; fp32 matches the existing B/16 run")
    ap.add_argument("--max-images", type=int, default=None, help="Smoke check only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.variant in ("B4", "B4D3", "B4D3L", "B4LB") and args.max_side < 224:
        ap.error("--max-side must be >= 224")
    cfg = config()
    p = paths(cfg)
    default_dirs = {"B4": "dinov2_base", "B4D3": "dinov3", "B4D3L": "dinov3_large",
                    "B4LB": "lingbot_vision_large"}
    model_dir = args.model_dir or ROOT / "upstream" / default_dirs.get(args.variant, "dinov2_base")
    if args.variant == "B3":
        extractor, signature, hook = b3_extractor(cfg, args.device)
    elif args.variant == "B4":
        extractor, signature, hook = b4_extractor(model_dir, args.device, args.max_side)
    elif args.variant in ("B4D3", "B4D3L"):
        architecture = ("vit_large_patch16_dinov3" if args.variant == "B4D3L"
                        else "vit_base_patch16_dinov3")
        extractor, signature, hook = dinov3_timm_extractor(
            model_dir, args.device, args.max_side, architecture, args.precision)
    else:
        extractor, signature, hook = lingbot_large_extractor(model_dir, args.device, args.max_side)
    items = names(p["sets"] / f"{args.split}.txt")
    if args.max_images is not None:
        items = items[:args.max_images]
    dest = OUT / args.variant / "cache" / args.split
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for index, name in enumerate(items, 1):
            image_path = p["images"] / name
            base_path = p["output"] / "B0" / "cache" / args.split / f"{Path(name).stem}.npz"
            if not image_path.is_file() or not base_path.is_file():
                raise FileNotFoundError(f"Missing image or B0 cache: {image_path}, {base_path}")
            image_hash, base_hash = sha256(image_path), sha256(base_path)
            output_file = dest / f"{Path(name).stem}.npz"
            if output_file.is_file() and not args.force:
                try:
                    with np.load(output_file, allow_pickle=False) as z:
                        valid = (str(z["image_sha256"]) == image_hash and
                                 str(z["base_sha256"]) == base_hash and
                                 str(z["encoder_signature"]) == signature and
                                 z["features"].shape[0] == z["boxes"].shape[0])
                    if valid:
                        records.append({"image": name, "cache": str(output_file), "reused": True})
                        print(f"{index}/{len(items)} {name}: reused", flush=True)
                        continue
                except (KeyError, ValueError, OSError):
                    pass
            boxes, scores, old_features = read_cache(base_path)
            if str(args.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            features = extractor(image_path, boxes, old_features)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not np.isfinite(features).all() or features.shape[0] != len(boxes):
                raise ValueError(f"Invalid features for {name}: {features.shape}")
            peak = torch.cuda.max_memory_allocated() / 2**30 if str(args.device).startswith("cuda") else 0
            np.savez_compressed(output_file, boxes=boxes, scores=scores,
                                features=features.astype(np.float16),
                                image_sha256=image_hash, base_sha256=base_hash,
                                encoder_signature=signature, seconds=elapsed, peak_vram_gib=peak)
            records.append({"image": name, "cache": str(output_file), "reused": False,
                            "seconds": elapsed, "peak_vram_gib": peak, "feature_dim": features.shape[1]})
            print(f"{index}/{len(items)} {name}: {elapsed:.2f}s, {features.shape[1]}D, {peak:.2f}GiB", flush=True)
    finally:
        if hook is not None:
            hook.remove()
    write_json(OUT / args.variant / f"cache_{args.split}_manifest.json", {
        "variant": args.variant, "split": args.split, "count": len(records),
        "encoder_signature": signature, "model_dir": str(model_dir) if args.variant != "B3" else None,
        "records": records,
    })


if __name__ == "__main__":
    main()
