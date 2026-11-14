"""Segmentation mode: fine-tune DeepLabV3 MobileNetV3-Large for crack masks, pick the threshold on validation, report IoU on test.

    python inspect_cracks.py train --data data/cracks --output artifacts/cracks --epochs 6
    python inspect_cracks.py predict --model artifacts/cracks/cracks.pt --input photos/ --output artifacts/cracks/predicted
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

from metrics import mask_metrics, tolerant_mask_metrics
from render import crack_dashboard, curve_svg, side_by_side

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def configure(cache_dir, threads, seed):
    if cache_dir:
        os.environ["TORCH_HOME"] = str(Path(cache_dir).resolve())
    import torch

    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return torch


def build_model(pretrained=True):
    import torch
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large

    weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    model = deeplabv3_mobilenet_v3_large(weights=weights, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)     # one logit: crack probability
    model.aux_classifier = None
    return model, weights


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(image, size):
    import torch
    from PIL import Image

    resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    array = (np.asarray(resized, np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def load_split(root, split):
    index = json.loads((Path(root) / "index.json").read_text(encoding="utf-8"))
    return [(Path(root) / split / e["file"], Path(root) / split / e["mask"], e) for e in index["splits"][split]]


def read_pair(image_path, mask_path):
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
    with Image.open(mask_path) as mask:
        mask = np.asarray(mask.convert("L")) > 127
    return image, mask


def predict_probabilities(model, images, size, torch):
    """images: list of PIL; returns list of HxW float32 probability maps at the original size."""
    out = []
    with torch.no_grad():
        for image in images:
            logits = model(to_tensor(image, size).unsqueeze(0))["out"]
            logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
            out.append(torch.sigmoid(logits)[0, 0].numpy().astype(np.float32))
    return out


def dataset_iou(probabilities, masks, threshold):
    tp = fp = fn = 0
    for prob, mask in zip(probabilities, masks):
        m = mask_metrics(prob > threshold, mask)
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
    return {"threshold": threshold, "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0}


def train(args):
    torch = configure(args.cache_dir, args.threads, args.seed)
    model, weights = build_model()
    train_items = load_split(args.data, "train")
    val_items = load_split(args.data, "val")
    test_items = load_split(args.data, "test")
    train_data = [read_pair(i, m) for i, m, _ in train_items]
    val_data = [read_pair(i, m) for i, m, _ in val_items]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    positive_weight = torch.tensor([args.positive_weight], dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    def loss_fn(logits, target):
        # Weighted BCE for the rare crack pixels plus a soft Dice term, which optimises overlap directly
        # and keeps thin lines from being traded for easy background pixels.
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum()
        dice = 1 - (2 * intersection + 1) / (probabilities.sum() + target.sum() + 1)
        return bce(logits, target) + dice
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log = {"model": "deeplabv3_mobilenet_v3_large", "weights": str(weights), "url": weights.url, "input_size": args.size,
           "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "positive_weight": args.positive_weight, "seed": args.seed,
                      "augmentation": "random horizontal and vertical flips", "loss": "weighted BCE + soft Dice"},
           "train_images": len(train_data), "val_images": len(val_data), "test_images": len(test_items), "epochs": []}
    best_iou, best_state = -1.0, None
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [train_data[i] for i in order[start:start + args.batch_size]]
            xs, ys = [], []
            for image, mask in batch:
                x = to_tensor(image, args.size)
                y = torch.from_numpy(np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(mask.astype(np.uint8) * 255).resize((args.size, args.size)), np.float32) / 255.0 > 0.5).float()
                if rng.random() < 0.5:
                    x, y = x.flip(-1), y.flip(-1)
                if rng.random() < 0.5:
                    x, y = x.flip(-2), y.flip(-2)
                xs.append(x)
                ys.append(y)
            x = torch.stack(xs)
            y = torch.stack(ys).unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_fn(model(x)["out"], y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        model.eval()
        probabilities = predict_probabilities(model, [i for i, _ in val_data], args.size, torch)
        val = dataset_iou(probabilities, [m for _, m in val_data], 0.5)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_iou": val["iou"], "seconds": time.perf_counter() - started}
        log["epochs"].append(entry)
        print(json.dumps(entry))
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    # Threshold sweep on validation, applied once to test.
    probabilities = predict_probabilities(model, [i for i, _ in val_data], args.size, torch)
    sweep = [dataset_iou(probabilities, [m for _, m in val_data], t / 100) for t in range(10, 95, 5)]
    best = max(sweep, key=lambda s: s["iou"])
    log["threshold_sweep"] = sweep
    log["threshold"] = best["threshold"]
    log["best_epoch"] = max(log["epochs"], key=lambda e: e["val_iou"])["epoch"]
    log["sweep_svg"] = curve_svg([(s["threshold"], s["iou"]) for s in sweep], "probability threshold", "validation IoU", marker=(best["threshold"], best["iou"]))
    test_data = [read_pair(i, m) for i, m, _ in test_items]
    timings = []
    test_probabilities = []
    for image, _ in test_data:
        t0 = time.perf_counter()
        test_probabilities.extend(predict_probabilities(model, [image], args.size, torch))
        timings.append((time.perf_counter() - t0) * 1000)
    per_image, tiles = [], []
    (output / "test").mkdir(exist_ok=True)
    for (image, mask), prob, (image_path, _, entry) in zip(test_data, test_probabilities, test_items):
        predicted = prob > best["threshold"]
        m = mask_metrics(predicted, mask)
        tolerant = tolerant_mask_metrics(predicted, mask, 2)
        per_image.append({"file": image_path.name, "crack": entry["crack"], "iou": m["iou"], "tolerant_f1": tolerant["f1"],
                          "tolerant_precision": tolerant["precision"], "tolerant_recall": tolerant["recall"],
                          "predicted_pixels": int(predicted.sum()), "truth_pixels": int(mask.sum())})
    overall = dataset_iou(test_probabilities, [m for _, m in test_data], best["threshold"])
    crack_ious = [p["iou"] for p in per_image if p["crack"] and p["iou"] is not None]
    tolerant_f1 = [p["tolerant_f1"] for p in per_image if p["crack"] and p["tolerant_f1"] is not None]
    clean = [p for p in per_image if not p["crack"]]
    log["test"] = {"images": len(per_image), "threshold": best["threshold"], "dataset_iou": overall["iou"], "precision": overall["precision"], "recall": overall["recall"],
                   "mean_iou_crack_images": float(np.mean(crack_ious)) if crack_ious else None, "median_iou_crack_images": float(np.median(crack_ious)) if crack_ious else None,
                   "mean_tolerant_f1_crack_images": float(np.mean(tolerant_f1)) if tolerant_f1 else None, "tolerance_px": 2,
                   "clean_images": len(clean), "clean_false_positive_images": sum(1 for p in clean if p["predicted_pixels"] > 0),
                   "clean_false_pixels_mean": float(np.mean([p["predicted_pixels"] for p in clean])) if clean else None,
                   "median_inference_ms": statistics.median(timings), "per_image": per_image}
    from PIL import Image

    ranked = sorted([(p, i) for i, p in enumerate(per_image) if p["crack"]], key=lambda t: t[0]["iou"])
    chosen = ranked[:4] + ranked[-4:] + [(p, i) for i, p in enumerate(per_image) if not p["crack"]][:2]
    for p, i in chosen:
        image, mask = test_data[i]
        predicted = test_probabilities[i] > best["threshold"]
        panel = side_by_side([image, Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB"), Image.fromarray(predicted.astype(np.uint8) * 255).convert("RGB")],
                             [p["file"], "truth", f"prediction (IoU {p['iou'] if p['iou'] is None else round(p['iou'], 3)})"])
        name = f"test/{Path(p['file']).stem}.png"
        panel.save(output / name)
        tiles.append({"name": p["file"], "image": name, "iou": p["iou"], "predicted_pixels": p["predicted_pixels"], "truth_pixels": p["truth_pixels"]})
    torch.save({"state_dict": model.state_dict(), "input_size": args.size, "threshold": best["threshold"], "weights": str(weights)}, output / "cracks.pt")
    (output / "training_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    (output / "dashboard.html").write_text(crack_dashboard(log, tiles), encoding="utf-8")
    print(json.dumps({"best_epoch": log["best_epoch"], "threshold": best["threshold"], "test_dataset_iou": round(overall["iou"], 4),
                      "mean_iou_crack_images": round(log["test"]["mean_iou_crack_images"], 4), "mean_tolerant_f1": round(log["test"]["mean_tolerant_f1_crack_images"], 4),
                      "clean_false_positive_images": log["test"]["clean_false_positive_images"],
                      "median_inference_ms": round(log["test"]["median_inference_ms"], 1)}))
    return log


def predict(args):
    torch = configure(args.cache_dir, args.threads, 0)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model, _ = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    from PIL import Image

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(args.input)] if Path(args.input).is_file() else sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.endswith("_mask"))
    records = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
        prob = predict_probabilities(model, [image], checkpoint["input_size"], torch)[0]
        predicted = prob > checkpoint["threshold"]
        Image.fromarray(predicted.astype(np.uint8) * 255, "L").save(output / f"{path.stem}_mask.png")
        side_by_side([image, Image.fromarray(predicted.astype(np.uint8) * 255).convert("RGB")], [path.name, f"crack pixels {int(predicted.sum())}"]).save(output / f"{path.stem}_panel.png")
        records.append({"file": path.name, "crack_pixels": int(predicted.sum()), "crack_fraction": float(predicted.mean()), "max_probability": float(prob.max())})
        print(f"{path.name}: {records[-1]['crack_pixels']} crack pixels ({records[-1]['crack_fraction']:.2%})")
    (output / "predictions.json").write_text(json.dumps({"model": str(args.model), "threshold": checkpoint["threshold"], "images": records}, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--positive-weight", type=float, default=8.0, help="BCE weight on crack pixels (they are rare)")
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
