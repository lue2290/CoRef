"""Static B5 scoring; no annotation access or model downloads."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STAGES = ("support_only", "quality_only", "full")
REASONS = {0: "background_gate", 1: "nms_suppressed", 2: "outside_top100",
           3: "retained_known", 4: "retained_unknown"}


def representation(features):
    """Cache layout is channel-major Cx2x2 followed by C context values."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] % 5:
        raise ValueError("Expected N x (4C+C) spatial/context features")
    c = features.shape[1] // 5
    spatial = features[:, :4*c].reshape(-1, c, 4)
    z = features[:, :4*c].copy()
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    tokens = spatial / np.maximum(np.linalg.norm(spatial, axis=1, keepdims=True), 1e-9)
    pairs = np.stack([(tokens[:, :, a] * tokens[:, :, b]).sum(axis=1)
                      for a in range(4) for b in range(a+1, 4)], axis=1)
    center = spatial.mean(axis=2)
    context = features[:, 4*c:]
    similarity = (center*context).sum(axis=1) / np.maximum(
        np.linalg.norm(center, axis=1)*np.linalg.norm(context, axis=1), 1e-9)
    structure = np.column_stack([pairs.mean(axis=1), pairs.min(axis=1),
                                 pairs.std(axis=1), 1-similarity]).astype(np.float32)
    if not np.isfinite(z).all() or not np.isfinite(structure).all():
        raise ValueError("Nonfinite cached features")
    return z, structure


def class_distances(query, gallery, classes, query_images=None, gallery_images=None,
                    active=(0, 1, 2, 3), k=5, device="cpu"):
    with torch.inference_mode():
        q = torch.as_tensor(query, dtype=torch.float32, device=device)
        g = torch.as_tensor(gallery, dtype=torch.float32, device=device)
        d = (1-q@g.T).clamp_(0, 2).cpu().numpy()
    if query_images is not None:
        if gallery_images is None:
            raise ValueError("Image exclusion requires gallery image IDs")
        d[np.asarray(query_images)[:, None] == np.asarray(gallery_images)[None, :]] = np.inf
    out = np.full((len(query), 4), np.inf, np.float32)
    for c in active:
        values = d[:, classes == c]
        if values.shape[1] < k:
            raise ValueError(f"Class {c} has fewer than {k} references")
        nearest = np.partition(values, k-1, axis=1)[:, :k]
        if not np.isfinite(nearest).all():
            raise ValueError(f"Insufficient image-independent references for class {c}")
        out[:, c] = nearest.mean(axis=1)
    return out


def fit_support(z, labels, images):
    d = class_distances(z, z, labels, images, images)
    radius, scale = [], []
    for c in range(4):
        values = d[labels == c, c]
        radius.append(float(np.quantile(values, .95, method="higher")))
        scale.append(float(max(np.quantile(values, .75)-np.quantile(values, .25), .001)))
    return np.asarray(radius, np.float32), np.asarray(scale, np.float32)


def novelty(distances, radius, scale):
    return ((distances-radius[None, :])/scale[None, :]).min(axis=1)


def sigmoid(values):
    return (1/(1+np.exp(-np.clip(values, -30, 30)))).astype(np.float32)


def fit_quality(x, y):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Quality training requires both good and poor-localization boxes")
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=42))
    return model.fit(x, y)


def objectness(pf, quality, weight):
    # A soft modulation, not an extra known-category foreground classifier.
    return np.asarray(pf, np.float32)*((1-weight)+weight*np.asarray(quality, np.float32))


def retention_threshold(values, risk=.05):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not len(values):
        raise ValueError("Cannot calibrate retention on an empty set")
    # Reject strictly below this value: at most floor(risk*n) observations.
    return float(values[min(int(np.floor(risk*len(values))), len(values)-1)])
