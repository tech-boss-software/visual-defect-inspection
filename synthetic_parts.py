"""Synthetic inspection fixtures with pixel masks: three product categories for the anomaly mode and concrete cracks.

Categories (anomaly mode, trained on good parts only):
- plate: brushed-metal plate with four drilled holes; defects: scratch, dent, stain, missing hole.
- cap: round bottle cap with ridges on a dark tray; defects: scratch, chip at the rim, stain.
- tile: ceramic tile with an embossed grid; defects: crack, stain, dent.
Cracks (segmentation mode): concrete texture with random-walk cracks, plus crack-free negatives.

Every image is drawn, not photographed, with per-image random texture, lighting and noise so that
no two parts are identical. The point is an exact mask for every defect, which real datasets with
research-only licences (MVTec AD) could not provide for a commercial portfolio.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

SIZE = 256
CATEGORIES = ["plate", "cap", "tile"]
DEFECTS = {"plate": ["scratch", "dent", "stain", "missing_hole"], "cap": ["scratch", "chip", "stain"], "tile": ["crack", "stain", "dent"]}


def _rng_noise(rng, shape, sigma):
    return np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, sigma, shape).astype(np.float32)


def _gaussian_blur(array, sigma):
    """Separable Gaussian blur on a float32 HxW (or HxWxC) array, numpy only."""
    if sigma <= 0:
        return array
    radius = max(1, int(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2).astype(np.float32)
    kernel /= kernel.sum()
    padded = np.pad(array, [(radius, radius), (radius, radius)] + [(0, 0)] * (array.ndim - 2), mode="reflect")
    out = np.zeros_like(array)
    for i, weight in enumerate(kernel):
        out += weight * padded[i:i + array.shape[0], radius:radius + array.shape[1]]
    padded = np.pad(out, [(radius, radius), (radius, radius)] + [(0, 0)] * (array.ndim - 2), mode="reflect")
    out2 = np.zeros_like(array)
    for i, weight in enumerate(kernel):
        out2 += weight * padded[radius:radius + array.shape[0], i:i + array.shape[1]]
    return out2


def _grid(size=SIZE):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    return xs, ys


def _lighting(rng, size=SIZE):
    xs, ys = _grid(size)
    angle = rng.uniform(0, 2 * math.pi)
    ramp = (xs * math.cos(angle) + ys * math.sin(angle)) / size
    return (1.0 + rng.uniform(-0.12, 0.12) * (ramp - 0.5)).astype(np.float32)


# ---------------------------------------------------------------- good parts

def good_plate(rng, size=SIZE):
    xs, ys = _grid(size)
    base = np.full((size, size, 3), rng.uniform(150, 175), np.float32)
    streaks = _gaussian_blur(_rng_noise(rng, (size, size), 1.0), 0.8)
    streaks = np.cumsum(streaks, axis=1) / 6.0
    base += (streaks - streaks.mean())[..., None] * rng.uniform(4, 9)
    base += _rng_noise(rng, (size, size, 1), 3.0)
    holes = []
    for cx, cy in ((56, 56), (200, 56), (56, 200), (200, 200)):
        cx += rng.uniform(-3, 3)
        cy += rng.uniform(-3, 3)
        holes.append((cx, cy, 13.0))
    for cx, cy, r in holes:
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        base[d < r] = rng.uniform(28, 40)
        ring = (d >= r) & (d < r + 2.5)
        base[ring] = base[ring] * 0.75
    base *= _lighting(rng, size)[..., None]
    return base, {"holes": holes}


def good_cap(rng, size=SIZE):
    xs, ys = _grid(size)
    base = np.full((size, size, 3), rng.uniform(40, 60), np.float32)
    base += _rng_noise(rng, (size, size, 1), 4.0)
    cx, cy, r = size / 2 + rng.uniform(-4, 4), size / 2 + rng.uniform(-4, 4), 96 + rng.uniform(-3, 3)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    color = np.array([rng.uniform(150, 190), rng.uniform(40, 70), rng.uniform(40, 60)], np.float32)
    inside = d < r
    base[inside] = color + _rng_noise(rng, (int(inside.sum()), 3), 3.0)
    angle = np.arctan2(ys - cy, xs - cx)
    ridges = (np.sin(angle * 36) > 0.2) & (d > r - 14) & inside
    base[ridges] *= 0.82
    top = (d < r - 16)
    base[top] *= 1.08
    base[(d >= r) & (d < r + 2)] *= 0.5
    base *= _lighting(rng, size)[..., None]
    return base, {"center": (cx, cy), "radius": r}


def good_tile(rng, size=SIZE):
    xs, ys = _grid(size)
    tone = rng.uniform(190, 215)
    base = np.full((size, size, 3), tone, np.float32)
    base[..., 2] += rng.uniform(-10, 5)
    base += _gaussian_blur(_rng_noise(rng, (size, size, 1), 6.0), 2.0)
    pitch = 32
    offset = rng.uniform(0, pitch)
    grooves = (np.abs(((xs + offset) % pitch) - pitch / 2) < 1.5) | (np.abs(((ys + offset) % pitch) - pitch / 2) < 1.5)
    base[grooves] *= 0.86
    base += _rng_noise(rng, (size, size, 1), 2.5)
    base *= _lighting(rng, size)[..., None]
    return base, {"pitch": pitch, "offset": offset}


GOOD = {"plate": good_plate, "cap": good_cap, "tile": good_tile}


# ---------------------------------------------------------------- defects (return image, mask)

def _line_mask(rng, size, length, width, inside=None):
    xs, ys = _grid(size)
    for _ in range(20):
        x0, y0 = rng.uniform(30, size - 30), rng.uniform(30, size - 30)
        angle = rng.uniform(0, math.pi)
        x1, y1 = x0 + length * math.cos(angle), y0 + length * math.sin(angle)
        px, py = xs - x0, ys - y0
        dx, dy = x1 - x0, y1 - y0
        t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
        dist = np.sqrt((px - t * dx) ** 2 + (py - t * dy) ** 2)
        mask = dist < width / 2
        if inside is not None:
            mask &= inside
        if mask.sum() > 40:
            return mask
    return mask


def _blob_mask(rng, size, radius, inside=None, margin=24):
    xs, ys = _grid(size)
    for _ in range(20):
        cx, cy = rng.uniform(margin, size - margin), rng.uniform(margin, size - margin)
        rx, ry = radius * rng.uniform(0.7, 1.3), radius * rng.uniform(0.7, 1.3)
        angle = rng.uniform(0, math.pi)
        u = (xs - cx) * math.cos(angle) + (ys - cy) * math.sin(angle)
        v = -(xs - cx) * math.sin(angle) + (ys - cy) * math.cos(angle)
        mask = (u / rx) ** 2 + (v / ry) ** 2 < 1
        if inside is not None:
            mask &= inside
        if mask.sum() > 40:
            return mask
    return mask


def crack_mask(rng, size, steps=None, width=2.0, start=None):
    """Random-walk crack: a jagged polyline with a slowly turning heading and a few branches."""
    xs, ys = _grid(size)
    mask = np.zeros((size, size), bool)
    x, y = start or (rng.uniform(20, size - 20), rng.uniform(20, size - 20))
    heading = rng.uniform(0, 2 * math.pi)
    points = [(x, y)]
    for _ in range(steps or rng.randint(25, 60)):
        heading += rng.uniform(-0.6, 0.6)
        step = rng.uniform(3, 7)
        x, y = x + step * math.cos(heading), y + step * math.sin(heading)
        if not (2 <= x < size - 2 and 2 <= y < size - 2):
            break
        points.append((x, y))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        px, py = xs - x0, ys - y0
        dx, dy = x1 - x0, y1 - y0
        t = np.clip((px * dx + py * dy) / max(1e-6, dx * dx + dy * dy), 0, 1)
        dist = np.sqrt((px - t * dx) ** 2 + (py - t * dy) ** 2)
        local = width * rng.uniform(0.6, 1.4)
        mask |= dist < local / 2
    if len(points) > 10 and rng.random() < 0.5:
        branch_start = points[rng.randint(3, len(points) - 3)]
        mask |= crack_mask(rng, size, steps=rng.randint(8, 20), width=width * 0.8, start=branch_start)
    return mask


def apply_defect(kind, image, info, rng, category, size=SIZE):
    xs, ys = _grid(size)
    out = image.copy()
    inside = None
    if category == "cap":
        cx, cy = info["center"]
        inside = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) < info["radius"] - 4
    if kind == "scratch":
        mask = _line_mask(rng, size, rng.uniform(40, 110), rng.uniform(2, 4), inside)
        out[mask] = out[mask] * rng.choice([0.55, 1.35]) + rng.uniform(-5, 5)
    elif kind == "dent":
        mask = _blob_mask(rng, size, rng.uniform(9, 16), inside)
        soft = _gaussian_blur(mask.astype(np.float32), 2.0)
        out = out * (1 - 0.45 * soft[..., None])
    elif kind == "stain":
        mask = _blob_mask(rng, size, rng.uniform(10, 18), inside)
        tint = np.array([rng.uniform(60, 120), rng.uniform(40, 90), rng.uniform(20, 60)], np.float32)
        soft = _gaussian_blur(mask.astype(np.float32), 1.5)[..., None]
        out = out * (1 - 0.6 * soft) + tint * 0.6 * soft
    elif kind == "missing_hole":
        cx, cy, r = info["holes"][rng.randint(0, 3)]
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        mask = d < r + 2.5
        fill = np.median(image[(d > r + 6) & (d < r + 14)], axis=0)
        out[mask] = fill + _rng_noise(rng, (int(mask.sum()), 3), 3.0)
    elif kind == "chip":
        cx, cy = info["center"]
        angle = rng.uniform(0, 2 * math.pi)
        px, py = cx + info["radius"] * math.cos(angle), cy + info["radius"] * math.sin(angle)
        d = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
        mask = (d < rng.uniform(10, 18)) & (np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) < info["radius"])
        out[mask] = np.array([rng.uniform(40, 60), rng.uniform(40, 60), rng.uniform(40, 60)], np.float32) + _rng_noise(rng, (int(mask.sum()), 3), 4.0)
    elif kind == "crack":
        mask = crack_mask(rng, size, width=rng.uniform(1.5, 3.0))
        out[mask] = out[mask] * 0.45
    else:
        raise ValueError(f"unknown defect {kind}")
    return out, mask


def finish(array, rng):
    from PIL import Image

    array = array + _rng_noise(rng, array.shape[:2] + (1,), rng.uniform(1.0, 3.0))
    if rng.random() < 0.3:
        array = _gaussian_blur(array, rng.uniform(0.3, 0.7))
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def render_part(category, rng, defect=None):
    image, info = GOOD[category](rng)
    mask = np.zeros((SIZE, SIZE), bool)
    if defect:
        image, mask = apply_defect(defect, image, info, rng, category)
    return finish(image, rng), mask


def render_concrete(rng, with_crack=True):
    xs, ys = _grid(SIZE)
    base = np.full((SIZE, SIZE, 3), rng.uniform(140, 175), np.float32)
    base += _gaussian_blur(_rng_noise(rng, (SIZE, SIZE, 1), 14.0), 1.2)
    base += _gaussian_blur(_rng_noise(rng, (SIZE, SIZE, 1), 30.0), 6.0) * 0.5
    pores = _rng_noise(rng, (SIZE, SIZE), 1.0) > 2.6
    base[pores] *= 0.7
    base *= _lighting(rng, SIZE)[..., None]
    mask = np.zeros((SIZE, SIZE), bool)
    if with_crack:
        for _ in range(rng.randint(1, 2)):
            mask |= crack_mask(rng, SIZE, width=rng.uniform(1.5, 4.0))
        soft = _gaussian_blur(mask.astype(np.float32), 0.7)[..., None]
        base = base * (1 - 0.6 * np.clip(soft, 0, 1))
    return finish(base, rng), mask


# ---------------------------------------------------------------- writers

def _save(image, mask, folder, stem):
    from PIL import Image

    image.save(folder / f"{stem}.png")
    if mask is not None:
        Image.fromarray((mask.astype(np.uint8) * 255), "L").save(folder / f"{stem}_mask.png")


def write_anomaly_dataset(root, train_good=60, calib_good=20, test_good=20, test_defect_each=8, seed=5):
    root = Path(root)
    rng = random.Random(seed)
    index = {"seed": seed, "categories": {}}
    for category in CATEGORIES:
        entry = {"train_good": [], "calibration_good": [], "test": []}
        for split, count in (("train_good", train_good), ("calibration_good", calib_good)):
            folder = root / category / split
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                image, _ = render_part(category, rng)
                stem = f"good_{i:03d}"
                _save(image, None, folder, stem)
                entry[split].append({"file": f"{stem}.png", "label": 0})
        folder = root / category / "test"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(test_good):
            image, _ = render_part(category, rng)
            stem = f"good_{i:03d}"
            _save(image, None, folder, stem)
            entry["test"].append({"file": f"{stem}.png", "label": 0, "defect": None})
        for defect in DEFECTS[category]:
            for i in range(test_defect_each):
                image, mask = render_part(category, rng, defect)
                stem = f"{defect}_{i:03d}"
                _save(image, mask, folder, stem)
                entry["test"].append({"file": f"{stem}.png", "label": 1, "defect": defect, "mask": f"{stem}_mask.png",
                                      "mask_pixels": int(mask.sum())})
        index["categories"][category] = entry
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return index


def write_crack_dataset(root, train=120, val=30, test=40, negative_fraction=0.25, seed=9):
    root = Path(root)
    rng = random.Random(seed)
    index = {"seed": seed, "splits": {}}
    for split, count in (("train", train), ("val", val), ("test", test)):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for i in range(count):
            with_crack = rng.random() >= negative_fraction
            image, mask = render_concrete(rng, with_crack)
            stem = f"{'crack' if with_crack else 'clean'}_{i:03d}"
            _save(image, mask, folder, stem)
            entries.append({"file": f"{stem}.png", "mask": f"{stem}_mask.png", "crack": with_crack, "mask_pixels": int(mask.sum())})
        index["splits"][split] = entries
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomaly", help="write the three-category anomaly dataset here")
    parser.add_argument("--cracks", help="write the concrete crack dataset here")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.anomaly and not args.cracks:
        parser.error("give --anomaly and/or --cracks")
    if args.anomaly:
        index = write_anomaly_dataset(args.anomaly, seed=args.seed)
        print(json.dumps({c: {k: len(v) for k, v in e.items()} for c, e in index["categories"].items()}))
    if args.cracks:
        index = write_crack_dataset(args.cracks, seed=args.seed + 4)
        print(json.dumps({k: len(v) for k, v in index["splits"].items()}))


if __name__ == "__main__":
    main()
