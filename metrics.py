"""Ranking and mask metrics in pure Python / numpy: AUROC, threshold statistics, IoU."""
from __future__ import annotations

import math


def auroc(scores, labels):
    """Area under the ROC curve by the Mann-Whitney rank statistic; ties get average ranks."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be equal non-empty sequences")
    positives = sum(1 for l in labels if l == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    rank_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def threshold_metrics(scores, labels, threshold):
    tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s > threshold)
    fp = sum(1 for s, l in zip(scores, labels) if l == 0 and s > threshold)
    fn = sum(1 for s, l in zip(scores, labels) if l == 1 and s <= threshold)
    tn = sum(1 for s, l in zip(scores, labels) if l == 0 and s <= threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None}


def quantile(values, q):
    """Upper empirical quantile as in sensor-anomaly: the value at ceil(n*q), never interpolated."""
    if not values:
        raise ValueError("no values")
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


def best_f1_threshold(scores, labels):
    candidates = sorted(set(scores))
    best = None
    for i, value in enumerate(candidates):
        threshold = value if i == len(candidates) - 1 else (value + candidates[i + 1]) / 2
        report = threshold_metrics(scores, labels, threshold)
        if best is None or report["f1"] > best["f1"]:
            best = report
    return best


def mask_metrics(predicted, truth):
    """predicted/truth: boolean arrays of the same shape (numpy)."""
    import numpy as np

    predicted = np.asarray(predicted, bool)
    truth = np.asarray(truth, bool)
    if predicted.shape != truth.shape:
        raise ValueError("mask shapes differ")
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    union = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / union if union else None,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}


def dilate(mask, radius=2):
    """Binary dilation with a square structuring element (numpy shifts)."""
    import numpy as np

    mask = np.asarray(mask, bool)
    padded = np.pad(mask, radius, mode="constant")
    out = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= padded[radius + dy:radius + dy + mask.shape[0], radius + dx:radius + dx + mask.shape[1]]
    return out


def tolerant_mask_metrics(predicted, truth, radius=2):
    """Precision against the truth dilated by `radius`, recall against the prediction dilated by `radius`.

    Thin structures such as cracks are a few pixels wide, so a one-pixel offset halves the plain IoU;
    the tolerant form, common in crack benchmarks, forgives offsets up to `radius` pixels.
    """
    import numpy as np

    predicted = np.asarray(predicted, bool)
    truth = np.asarray(truth, bool)
    tp_p = int(np.logical_and(predicted, dilate(truth, radius)).sum())
    tp_r = int(np.logical_and(truth, dilate(predicted, radius)).sum())
    precision = tp_p / predicted.sum() if predicted.sum() else None
    recall = tp_r / truth.sum() if truth.sum() else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {"precision": precision, "recall": recall, "f1": f1, "radius": radius}


def pixel_auroc(score_maps, masks, sample=200000, seed=0):
    """AUROC over pixels, subsampled for speed; score_maps and masks are lists of equal-shape arrays."""
    import numpy as np

    rng = np.random.default_rng(seed)
    scores, labels = [], []
    for score_map, mask in zip(score_maps, masks):
        flat_scores = np.asarray(score_map, np.float32).ravel()
        flat_mask = np.asarray(mask, bool).ravel()
        scores.append(flat_scores)
        labels.append(flat_mask)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    if len(scores) > sample:
        pick = rng.choice(len(scores), sample, replace=False)
        scores, labels = scores[pick], labels[pick]
    return auroc(scores.tolist(), labels.astype(int).tolist())
