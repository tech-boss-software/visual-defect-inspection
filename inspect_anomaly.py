"""Anomaly mode: fit a memory bank on good parts, calibrate a threshold on held-out good parts, score new images.

    python inspect_anomaly.py fit --good data/plate/train_good --output artifacts/plate
    python inspect_anomaly.py calibrate --bank artifacts/plate --good data/plate/calibration_good --false-alarm 0.05
    python inspect_anomaly.py score --bank artifacts/plate --input data/plate/test --output artifacts/plate/scored [--index index.json --category plate]
    python inspect_anomaly.py benchmark --data data --output artifacts/anomaly   (all categories, synthetic index, dashboard)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from anomaly import MemoryBank
from metrics import auroc, pixel_auroc, threshold_metrics
from render import anomaly_dashboard, overlay, score_histogram_svg, side_by_side

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.endswith("_mask"))


def open_image(path):
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def make_extractor(args):
    from features import ResNetPatchFeatures

    return ResNetPatchFeatures(input_size=args.input_size, threads=args.threads, cache_dir=args.cache_dir)


def fit(extractor, good_paths, fraction, seed):
    features = []
    started = time.perf_counter()
    for path in good_paths:
        features.append(extractor.extract(open_image(path)))
    stacked = np.concatenate(features, axis=0)
    bank = MemoryBank.fit(stacked, extractor.grid_shape, extractor.info(), fraction=fraction, seed=seed)
    bank.coreset_info["images"] = len(good_paths)
    bank.coreset_info["fit_seconds"] = round(time.perf_counter() - started, 1)
    return bank


def score_paths(extractor, bank, paths, sigma=4.0):
    results = []
    for path in paths:
        image = open_image(path)
        t0 = time.perf_counter()
        score, heat, grid = bank.score(extractor.extract(image), (image.height, image.width), sigma)
        results.append({"path": path, "image": image, "score": score, "heat": heat, "ms": (time.perf_counter() - t0) * 1000})
    return results


def write_scored(results, bank, output, truth_masks=None, limit_tiles=None):
    """Per-image JSON and overlay PNGs; returns the records."""
    from PIL import Image

    output = Path(output)
    (output / "heatmaps").mkdir(parents=True, exist_ok=True)
    records = []
    low = bank.calibration["good_score_median"] if bank.calibration else float(min(r["score"] for r in results))
    high = max(bank.threshold or 0.0, max(r["score"] for r in results)) * 1.05
    for result in results:
        verdict = bank.verdict(result["score"]) if bank.threshold is not None else {"score": result["score"], "defect": None}
        mask = truth_masks.get(result["path"].name) if truth_masks else None
        heat_image = overlay(result["image"], result["heat"], low, high, mask=mask)
        panel = side_by_side([overlay(result["image"], np.zeros_like(result["heat"]), 0, 1, alpha=0, mask=mask), heat_image],
                             [result["path"].name, f"score {result['score']:.3f}" + (f"  thr {bank.threshold:.3f}" if bank.threshold else "")])
        name = f"heatmaps/{result['path'].stem}.png"
        panel.save(output / name)
        records.append({"file": result["path"].name, **verdict, "heatmap": name, "score_ms": round(result["ms"], 1),
                        "heat_max": float(result["heat"].max()), "heat_argmax_xy": [int(v) for v in np.unravel_index(int(np.argmax(result["heat"])), result["heat"].shape)[::-1]]})
    return records


def load_masks(folder, entries):
    from PIL import Image

    masks = {}
    for entry in entries:
        if entry.get("mask"):
            with Image.open(Path(folder) / entry["mask"]) as image:
                masks[entry["file"]] = np.asarray(image.convert("L")) > 127
    return masks


def benchmark(args):
    extractor = make_extractor(args)
    data = Path(args.data)
    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"features": extractor.info(), "coreset_fraction": args.coreset, "target_false_alarm": args.false_alarm, "sigma": args.sigma,
              "categories": {}, "timing": {}}
    all_ms, tiles = [], []
    for category, entry in index["categories"].items():
        folder = data / category
        bank = fit(extractor, [folder / "train_good" / e["file"] for e in entry["train_good"]], args.coreset, args.seed)
        calibration = score_paths(extractor, bank, [folder / "calibration_good" / e["file"] for e in entry["calibration_good"]], args.sigma)
        bank.calibrate([r["score"] for r in calibration], args.false_alarm)
        bank.save(output / category)
        test_entries = entry["test"]
        results = score_paths(extractor, bank, [folder / "test" / e["file"] for e in test_entries], args.sigma)
        masks = load_masks(folder / "test", test_entries)
        records = write_scored(results, bank, output / category, masks)
        labels = [e["label"] for e in test_entries]
        scores = [r["score"] for r in results]
        at_threshold = threshold_metrics(scores, labels, bank.threshold)
        # Pixel AUROC needs masks; datasets with image-level labels only (real photos) skip it.
        defect_maps = [r["heat"] for r, e in zip(results, test_entries) if e["label"] == 1 and e["file"] in masks]
        defect_masks = [masks[e["file"]] for e in test_entries if e["label"] == 1 and e["file"] in masks]
        good_maps = [r["heat"] for r, e in zip(results, test_entries) if e["label"] == 0]
        px_auroc = pixel_auroc(defect_maps + good_maps, defect_masks + [np.zeros_like(m, bool) for m in good_maps]) if defect_masks else None
        per_defect = {}
        for record, e in zip(records, test_entries):
            if e["label"] == 1:
                slot = per_defect.setdefault(e["defect"], {"count": 0, "caught": 0, "scores": []})
                slot["count"] += 1
                slot["caught"] += int(record["defect"])
                slot["scores"].append(record["score"])
        for slot in per_defect.values():
            slot["mean_score"] = float(np.mean(slot.pop("scores")))
        report["categories"][category] = {
            "train_good": len(entry["train_good"]), "calibration_good": len(entry["calibration_good"]),
            "test": {"good": labels.count(0), "defective": labels.count(1)}, "bank_patches": int(len(bank.bank)),
            "fit_seconds": bank.coreset_info["fit_seconds"], "calibration": bank.calibration, "threshold": bank.threshold,
            "image_auroc": auroc(scores, labels), "pixel_auroc": px_auroc, "at_threshold": at_threshold, "per_defect": per_defect,
            "histogram_svg": score_histogram_svg([s for s, l in zip(scores, labels) if l == 0], [s for s, l in zip(scores, labels) if l == 1], bank.threshold),
            "records": records}
        all_ms.extend(r["ms"] for r in results)
        for record, e in zip(records, test_entries):
            status = ("miss" if not record["defect"] else "hit") if e["label"] == 1 else ("false alarm" if record["defect"] else "good")
            tiles.append({"category": category, "name": f"{category}/{e['file']}", "image": f"{category}/{record['heatmap']}", "score": record["score"],
                          "threshold": bank.threshold, "status": status, "class": {"miss": "miss", "false alarm": "fa", "hit": "hit", "good": "muted"}[status]})
        print(f"{category}: image AUROC {report['categories'][category]['image_auroc']:.3f}, pixel AUROC {px_auroc if px_auroc is None else round(px_auroc, 3)}, "
              f"recall {at_threshold['recall']:.2f} at FPR {at_threshold['false_positive_rate']:.2f} (threshold {bank.threshold:.3f})")
    report["timing"] = {"median_score_ms": statistics.median(all_ms) if all_ms else None, "threads": args.threads}
    shown = [t for t in tiles if t["status"] in ("miss", "false alarm")]
    hits = [t for t in tiles if t["status"] == "hit"]
    goods = [t for t in tiles if t["status"] == "good"]
    shown += hits[::max(1, len(hits) // 9)][:9] + goods[::max(1, len(goods) // 3)][:3]
    (output / "report.json").write_text(json.dumps({k: v for k, v in report.items()}, indent=2) + "\n", encoding="utf-8")
    (output / "dashboard.html").write_text(anomaly_dashboard(report, shown), encoding="utf-8")
    summary = {c: {"image_auroc": round(r["image_auroc"], 4), "pixel_auroc": None if r["pixel_auroc"] is None else round(r["pixel_auroc"], 4),
                   "recall": r["at_threshold"]["recall"], "fpr": r["at_threshold"]["false_positive_rate"]} for c, r in report["categories"].items()}
    print(json.dumps({"median_score_ms": report["timing"]["median_score_ms"], **summary}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=4.0, help="heatmap smoothing in pixels")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fit")
    p.add_argument("--good", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("calibrate")
    p.add_argument("--bank", required=True)
    p.add_argument("--good", required=True, help="held-out good images, never the ones used for fit")
    p.add_argument("--defective", default=None, help="optional labelled defects, reported only")
    p.add_argument("--false-alarm", type=float, default=0.05)
    p = commands.add_parser("score")
    p.add_argument("--bank", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("benchmark")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--false-alarm", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        return benchmark(args)
    extractor = make_extractor(args)
    if args.command == "fit":
        bank = fit(extractor, list_images(args.good), args.coreset, args.seed)
        bank.save(args.output)
        print(json.dumps({"images": bank.coreset_info["images"], "bank_patches": int(len(bank.bank)), "seconds": bank.coreset_info["fit_seconds"]}))
    elif args.command == "calibrate":
        bank = MemoryBank.load(args.bank)
        good = [r["score"] for r in score_paths(extractor, bank, list_images(args.good), args.sigma)]
        defective = [r["score"] for r in score_paths(extractor, bank, list_images(args.defective), args.sigma)] if args.defective else None
        report = bank.calibrate(good, args.false_alarm, defective)
        bank.save(args.bank)
        print(json.dumps(report))
    elif args.command == "score":
        bank = MemoryBank.load(args.bank)
        results = score_paths(extractor, bank, list_images(args.input), args.sigma)
        records = write_scored(results, bank, args.output)
        (Path(args.output) / "scores.json").write_text(json.dumps({"bank": str(args.bank), "threshold": bank.threshold, "images": records}, indent=2) + "\n", encoding="utf-8")
        for record in records:
            print(f"{record['file']}: score {record['score']:.3f} -> {'DEFECT' if record.get('defect') else 'ok' if record.get('defect') is False else 'uncalibrated'}")


if __name__ == "__main__":
    main()
