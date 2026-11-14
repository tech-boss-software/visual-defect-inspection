# System card: visual defect inspection

**Project owner:** Tech Boss. **Components:** a frozen ImageNet ResNet-18 (TorchVision, unchanged) feeding a memory-bank anomaly scorer, calibration and reporting written by Tech Boss; a DeepLabV3 MobileNetV3-Large fine-tuned by Tech Boss for crack masks. **Images used for the numbers below:** real concrete photographs (CC BY 4.0) for the anomaly mode, and synthetic parts and synthetic cracks drawn by `synthetic_parts.py` for the exact-mask measurements.

## Intended use

Anomaly mode flags parts that look unlike the good parts it was shown, with a heatmap of where, at a false-alarm rate the operator chooses; it is a screening aid for a line or a QC bench, and every flag is meant to be looked at. Segmentation mode outlines cracks on a surface for measurement and reporting. Neither is a safety certification and neither replaces an inspector on parts that matter.

## Anomaly mode on the synthetic categories

Fit on 60 good images per category, threshold at the 95% quantile of 20 held-out good images (target false-alarm rate 5%), test on 20 good plus 8 images per defect type. Features: ResNet-18 layer2+layer3, 224 px, 28x28 grid of 384-d patches, 10% coreset (4,704 patches per bank).

| Category | Defect types | Image AUROC | Pixel AUROC | Threshold | Recall at threshold | False alarms (of 20 good) | Fit time |
|---|---|---|---|---|---|---|---|
| plate | scratch, dent, stain, missing hole | 1.000 | 0.990 | 1.424 | 32 / 32 | 1 | 107 s |
| cap | scratch, chip, stain | 0.967 | 0.996 | 1.229 | 23 / 24 | 0 | 101 s |
| tile | crack, stain, dent | 1.000 | 0.998 | 1.166 | 24 / 24 | 0 | 93 s |

Every defect type was caught on every image except one cap stain (`cap/stain_003`, score 1.09 against the 1.23 threshold): a sliver of stain a few pixels wide at the rim, where the mask clipped it against the cap's edge. The one plate false alarm (`plate/good_000`, 1.52) is what a 5% target means on 20 images; the best-F1 threshold on the test set is reported in `report.json` for reference but was not used. Median scoring time 131 ms per 256 px image on 4 CPU threads (feature extraction included); the fit time is dominated by the greedy coreset.

Mean score by defect type (threshold about 1.2 to 1.4): missing hole 2.1, dent 2.8 to 4.8, scratch 3.0 to 3.6, stain 3.5 to 5.0, crack 3.4. Missing holes score lowest because the filled hole looks like plain brushed metal, which the bank has plenty of; they are caught because the neighbourhood of a hole is also in the bank.

## Real concrete: Concrete Crack Images for Classification

Photographs of concrete surfaces (Özgenel, Mendeley Data 10.17632/5y9wdsg2zt.2, CC BY 4.0; 227 x 227 crops of bridge decks, walls and pavements), used as an image-level anomaly task: 300 crack-free images for the fit, 100 held-out crack-free images for calibration at a 5% target, test on 100 crack-free and 400 cracked images. Same features and code as the synthetic run; coreset 5% (11,760 bank patches, 1338 s to fit, almost all of it the greedy coreset). No masks exist, so there is no pixel AUROC.

| Measure | Result |
|---|---|
| Image AUROC | 0.995 |
| Cracks caught at the calibrated threshold | 398 / 400 |
| Crack-free images flagged | 10 / 100 |
| Precision / recall / F1 at the threshold | 0.976 / 0.995 / 0.985 |
| Scoring time | 188 ms per image on 4 threads |

Two misses are hairline cracks (scores 2.54 and 2.58 against a 2.64 threshold). The false-alarm rate on the test images (10%) is twice the 5% target: 100 calibration images set the threshold at their 95th percentile, and the test crack-free images are rougher than that sample (pitted surfaces at scores up to 3.25), which is exactly why the calibration set must be as varied as production. The best-F1 threshold on the test set itself (2.70, F1 0.990) is reported in `report.json` for reference only.

Earlier the same category was fitted on 1,000 crack-free images; that run was abandoned because the greedy coreset over 784,000 patches would have taken hours on this CPU. The coreset is the scaling limit of this implementation: a few hundred good images per category is the practical ceiling without an approximate nearest-neighbour library.

## Segmentation mode on synthetic cracks

DeepLabV3 MobileNetV3-Large fine-tuned from the TorchVision COCO/VOC checkpoint on 120 images, best epoch by validation IoU, probability threshold chosen on the 30 validation images, reported on 40 test images (27 with cracks, 13 clean).

| Run | Input | Epochs | Loss | Threshold | Test IoU (dataset) | Mean IoU, crack images | Tolerant F1 (2 px) | Clean images with false crack pixels |
|---|---|---|---|---|---|---|---|---|
| first | 256 px | 6 | weighted BCE | 0.50 | 0.178 | 0.155 | not measured | 0 / 13 |
| used | 384 px | 15 (best 14) | weighted BCE + Dice | 0.65 | 0.411 | 0.384 | 0.914 | 0 / 13 |

The first run is kept (`examples/cracks/training_log-256px-6epochs.json`) because it shows the failure mode: the loss was still falling, and 2 to 4 px cracks at stride 8 came out as blobs. Longer training at a larger input with a Dice term is the used configuration. Plain IoU on lines a few pixels wide punishes one-pixel offsets heavily, which is why the tolerant F1 (offsets up to 2 px forgiven, as in crack benchmarks) is reported next to it. Median inference 87 ms per image on 4 threads.

## What was not measured

- Photographs of manufactured parts: the real run is on concrete surfaces, which vary in texture but not in shape; specular highlights, assembly variation and fixed-position parts were not tested. The bank must be fitted on the client's own good parts from the production camera, and the threshold on a separate good set from the same camera.
- Defects smaller than a stride-8 patch (8 px at 224) or textures finer than the backbone sees; `--input-size` trades speed for resolution and was not swept here.
- Drift over time, re-calibration schedules, and the effect of fitting on fewer than 60 good images.
- MVTec AD, NEU-DET and SDNET2018: not used (non-commercial licences, or a download blocked for scripts); the concrete set above is the real benchmark here.
- The crack model on real concrete, asphalt or walls.

## Privacy and licence

Everything runs locally; the demo server binds to localhost. TorchVision code and checkpoints are BSD-3-Clause; original code is MIT.

## Before client deployment

Photograph 50 to 100 good parts and whatever defective parts exist with the production camera, fit and calibrate on disjoint good sets, agree the false-alarm rate with the line owner, mask regions that are allowed to vary (labels, serial numbers), check the heatmaps of every flagged part for a week, and measure throughput on the target machine. Cracks: collect and mask 100 to 300 real images of the surfaces in question and retrain with the same script.
