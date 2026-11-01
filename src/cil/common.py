from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

TASK_CLASSES = {
    "task1": ["fish", "jellyfish", "penguin", "shark"],
    "task2": ["fish", "jellyfish", "penguin", "shark", "puffin", "starfish", "stingray"],
}

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def stable_seed(*parts):
    return int(fingerprint(parts)[:8], 16)

def names(path):
    return [n.strip() for n in Path(path).read_text(encoding="utf-8").splitlines() if n.strip()]

def load_config(root, path=None):
    c = read_json(Path(root) / "configs/default.json")
    if path:
        c.update(read_json(path))
    if c["microbatch_images"] != 1:
        raise ValueError("This implementation uses one image per microstep; change accumulate for effective batch")
    return c

def iou(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(hi-lo, 0)
    inter = wh.prod(-1)
    union = np.maximum(a[:, 2:]-a[:, :2], 0).prod(-1)[:, None] + np.maximum(b[:, 2:]-b[:, :2], 0).prod(-1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)

def auc(pos, neg):
    """Mann-Whitney AUROC, exactly half credit for ties, O(N log N)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    ordered = np.sort(neg)
    left = np.searchsorted(ordered, pos, side="left")
    right = np.searchsorted(ordered, pos, side="right")
    return float(np.mean((left+right)*0.5/len(neg)))

def fp_at_recall(pos, neg, recall=0.8):
    if len(pos) == 0 or len(neg) == 0:
        return None
    p = np.sort(np.asarray(pos))[::-1]
    threshold = float(p[min(len(p)-1, max(0, math.ceil(recall*len(p))-1))])
    return {"threshold": threshold, "recall": float((np.asarray(pos)>=threshold).mean()),
            "false_positive_rate": float((np.asarray(neg)>=threshold).mean())}

def install_label_guard(root, task):
    """Process-local IO guard. Training and prediction never open evaluation XML."""
    forbidden = [str((Path(root)/"assets/data/annotations").resolve()).casefold()]
    if task == "task1":
        forbidden += [str((Path(root)/"assets/inputs/task2_shots.json").resolve()).casefold(),
                      str((Path(root)/"prepared/task2").resolve()).casefold()]
    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve()).casefold()
        if any(path == p or path.startswith(p+os.sep) for p in forbidden):
            raise PermissionError("Evaluation/other-task labels are forbidden in this process: " + path)
    sys.addaudithook(audit)

def source_snapshot(root):
    root = Path(root)
    files = [root/"run.py"] + sorted((root/"src").rglob("*.py"))
    return {p.relative_to(root).as_posix(): digest(p) for p in files}

def atomic_torch_save(obj, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])

def configure_torch(seed, threads=2):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)

def choose_device(spec, training=False):
    import torch
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if spec == "auto" else spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this interpreter")
        if training and torch.cuda.get_device_properties(device).total_memory < 12*1024**3:
            raise RuntimeError("Full DINO training is blocked below 12GiB; use the remote GPU, not the local 4GiB GPU")
    elif training:
        raise RuntimeError("Full DINO training requires the remote GPU. CPU tests use the tiny test encoder.")
    return device

def native_bf16_supported(device):
    import torch
    device = torch.device(device)
    # Newer PyTorch can report BF16 emulation on Volta; it is not native support.
    return (device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
            and torch.cuda.is_bf16_supported())

def resolve_precision(requested, device):
    if requested not in {"auto", "bf16", "fp32"}:
        raise ValueError("Precision must be auto, bf16 or fp32")
    native = native_bf16_supported(device)
    if requested == "auto":
        return "bf16" if native else "fp32"
    if requested == "bf16" and not native:
        raise RuntimeError("Native BF16 is unavailable on this device; use precision auto or fp32 (V100 uses fp32)")
    return requested

def check_assets(root, full=False):
    root = Path(root)
    manifest = read_json(root/"assets/manifest.json")
    bad = []
    for row in manifest["files"]:
        p = root/row["path"]
        if not p.is_file() or p.stat().st_size != row["bytes"]:
            bad.append(row["path"])
        elif full and digest(p) != row["sha256"]:
            bad.append(row["path"]+" hash")
    if bad:
        raise RuntimeError("Missing/changed upload assets: " + str(bad[:10]))
    return {"files": len(manifest["files"]), "logical_bytes": sum(r["bytes"] for r in manifest["files"]), "full_hashes": full}
