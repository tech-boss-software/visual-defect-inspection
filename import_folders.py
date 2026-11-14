"""Build the anomaly benchmark layout from plain folders of good and defective images (no masks needed).

    python import_folders.py --good concrete/Negative --defective concrete/Positive --category concrete --output artifacts/concrete \\
        --train 1000 --calibration 100 --test-good 100 --test-defective 400 --defect-name crack

Images are copied (or linked) into <output>/<category>/{train_good,calibration_good,test} and an index.json
is written in the shape `inspect_anomaly.py benchmark` expects. Without masks the pixel AUROC is skipped;
image AUROC, calibrated threshold, recall and false alarms are still measured.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def build(good, defective, category, output, train, calibration, test_good, test_defective, defect_name="defect", seed=0, source=None):
    rng = random.Random(seed)
    good_files = list_images(good)
    bad_files = list_images(defective)
    if len(good_files) < train + calibration + test_good:
        raise ValueError(f"need {train + calibration + test_good} good images, found {len(good_files)}")
    if len(bad_files) < test_defective:
        raise ValueError(f"need {test_defective} defective images, found {len(bad_files)}")
    rng.shuffle(good_files)
    rng.shuffle(bad_files)
    splits = {"train_good": good_files[:train], "calibration_good": good_files[train:train + calibration],
              "test_good": good_files[train + calibration:train + calibration + test_good], "test_defective": bad_files[:test_defective]}
    root = Path(output) / category
    entry = {"train_good": [], "calibration_good": [], "test": []}
    for split in ("train_good", "calibration_good"):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        for path in splits[split]:
            target = folder / f"good_{path.stem}{path.suffix.lower()}"
            shutil.copyfile(path, target)
            entry[split].append({"file": target.name, "label": 0, "source": str(path)})
    folder = root / "test"
    folder.mkdir(parents=True, exist_ok=True)
    for path in splits["test_good"]:
        target = folder / f"good_{path.stem}{path.suffix.lower()}"
        shutil.copyfile(path, target)
        entry["test"].append({"file": target.name, "label": 0, "defect": None, "source": str(path)})
    for path in splits["test_defective"]:
        target = folder / f"{defect_name}_{path.stem}{path.suffix.lower()}"
        shutil.copyfile(path, target)
        entry["test"].append({"file": target.name, "label": 1, "defect": defect_name, "source": str(path)})
    index_path = Path(output) / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"seed": seed, "categories": {}}
    index["categories"][category] = entry
    index.setdefault("sources", {})[category] = source or f"{good} (good), {defective} (defective)"
    index_path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return {k: len(v) for k, v in entry.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--good", required=True)
    parser.add_argument("--defective", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--calibration", type=int, default=30)
    parser.add_argument("--test-good", type=int, default=30)
    parser.add_argument("--test-defective", type=int, default=30)
    parser.add_argument("--defect-name", default="defect")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default=None, help="provenance text recorded in index.json")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.good, args.defective, args.category, args.output, args.train, args.calibration, args.test_good,
                           args.test_defective, args.defect_name, args.seed, args.source)))


if __name__ == "__main__":
    main()
