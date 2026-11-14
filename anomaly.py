"""PatchCore-style anomaly scoring on a memory bank of good-part patch features (numpy only).

fit: collect patch features from good images, reduce them with a greedy k-center coreset so
the bank stays small, store the bank. score: for every patch of a new image, the distance to
its nearest bank patch; the image score is the largest patch distance; the heatmap is the
patch grid upsampled and smoothed. calibrate: the threshold is a quantile of the scores of
held-out good parts (the sensor-anomaly recipe), so the operator chooses the false-alarm rate
rather than a magic number; labelled defects, when available, only report what that
threshold catches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics import auroc, best_f1_threshold, quantile, threshold_metrics


def coreset(features, fraction=0.1, seed=0, projection_dim=64):
    """Greedy k-center subsampling on a random projection; returns the selected row indices."""
    features = np.asarray(features, np.float32)
    count = len(features)
    target = max(1, min(count, int(round(count * fraction))))
    if target >= count:
        return list(range(count))
    rng = np.random.default_rng(seed)
    if features.shape[1] > projection_dim:
        projection = rng.normal(0, 1, (features.shape[1], projection_dim)).astype(np.float32) / np.sqrt(projection_dim)
        reduced = features @ projection
    else:
        reduced = features
    selected = [int(rng.integers(count))]
    distances = np.linalg.norm(reduced - reduced[selected[0]], axis=1)
    for _ in range(target - 1):
        nxt = int(np.argmax(distances))
        selected.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(reduced - reduced[nxt], axis=1))
    return selected


def nearest_distances(queries, bank, chunk=2048):
    """Euclidean distance from each query row to its nearest bank row, without materialising all pairs at once."""
    queries = np.asarray(queries, np.float32)
    bank = np.asarray(bank, np.float32)
    bank_norms = (bank * bank).sum(axis=1)
    out = np.empty(len(queries), np.float32)
    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        squared = (block * block).sum(axis=1)[:, None] + bank_norms[None, :] - 2.0 * block @ bank.T
        out[start:start + chunk] = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    return out


def _gaussian_blur(array, sigma):
    if sigma <= 0:
        return array
    radius = max(1, int(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2).astype(np.float32)
    kernel /= kernel.sum()
    padded = np.pad(array, radius, mode="reflect")
    out = np.zeros_like(array)
    for i, weight in enumerate(kernel):
        out += weight * padded[i:i + array.shape[0], radius:radius + array.shape[1]]
    padded = np.pad(out, radius, mode="reflect")
    out2 = np.zeros_like(array)
    for i, weight in enumerate(kernel):
        out2 += weight * padded[radius:radius + array.shape[0], i:i + array.shape[1]]
    return out2


def upsample(grid, size):
    """Bilinear upsampling of an HxW float grid to (height, width) with numpy."""
    height, width = size
    gh, gw = grid.shape
    ys = (np.arange(height) + 0.5) * gh / height - 0.5
    xs = (np.arange(width) + 0.5) * gw / width - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, gh - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, gw - 1)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
    bottom = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def heatmap(patch_scores, grid_shape, size, sigma=4.0):
    grid = np.asarray(patch_scores, np.float32).reshape(grid_shape)
    return _gaussian_blur(upsample(grid, size), sigma)


class MemoryBank:
    def __init__(self, bank, grid_shape, feature_info, coreset_info=None):
        self.bank = np.asarray(bank, np.float32)
        self.grid_shape = tuple(grid_shape)
        self.feature_info = feature_info
        self.coreset_info = coreset_info or {}
        self.threshold = None
        self.calibration = None

    @classmethod
    def fit(cls, patch_features, grid_shape, feature_info, fraction=0.1, seed=0):
        """patch_features: array (images*patches, dim) from good parts only."""
        features = np.asarray(patch_features, np.float32)
        keep = coreset(features, fraction, seed)
        bank = features[keep]
        return cls(bank, grid_shape, feature_info, {"patches_seen": int(len(features)), "kept": int(len(keep)), "fraction": fraction, "seed": seed})

    def score(self, patch_features, size, sigma=4.0):
        """Returns (image score, heatmap array of shape size, patch score grid)."""
        distances = nearest_distances(patch_features, self.bank)
        grid = distances.reshape(self.grid_shape)
        return float(distances.max()), heatmap(distances, self.grid_shape, size, sigma), grid

    def calibrate(self, good_scores, target_false_alarm=0.05, defect_scores=None):
        """Threshold from held-out good parts; defect scores, when given, only measure the recall this buys."""
        if len(good_scores) < 10:
            raise ValueError("calibration needs at least 10 held-out good images")
        if not 0 < target_false_alarm < 1:
            raise ValueError("target_false_alarm must be in (0, 1)")
        self.threshold = float(quantile(list(good_scores), 1.0 - target_false_alarm))
        report = {"good_images": len(good_scores), "target_false_alarm": target_false_alarm, "threshold": self.threshold,
                  "good_score_median": float(np.median(good_scores)), "good_score_max": float(max(good_scores))}
        if defect_scores:
            scores = list(good_scores) + list(defect_scores)
            labels = [0] * len(good_scores) + [1] * len(defect_scores)
            report["defect_images"] = len(defect_scores)
            report["calibration_auroc"] = auroc(scores, labels)
            report["at_threshold"] = threshold_metrics(scores, labels, self.threshold)
            report["best_f1_reference"] = best_f1_threshold(scores, labels)
        self.calibration = report
        return report

    def verdict(self, image_score):
        if self.threshold is None:
            raise ValueError("bank is not calibrated")
        return {"score": float(image_score), "threshold": self.threshold, "defect": bool(image_score > self.threshold),
                "margin": float(image_score - self.threshold)}

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "bank.npy", self.bank)
        meta = {"grid_shape": list(self.grid_shape), "feature_info": self.feature_info, "coreset": self.coreset_info,
                "threshold": self.threshold, "calibration": self.calibration, "bank_shape": list(self.bank.shape)}
        (folder / "bank.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return str(folder)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        meta = json.loads((folder / "bank.json").read_text(encoding="utf-8"))
        bank = cls(np.load(folder / "bank.npy"), meta["grid_shape"], meta["feature_info"], meta.get("coreset"))
        bank.threshold = meta.get("threshold")
        bank.calibration = meta.get("calibration")
        return bank
