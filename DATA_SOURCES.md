# Data and output provenance

## Images

No photographs are used. `synthetic_parts.py` draws every image with numpy and Pillow (256x256):

- **plate**: brushed-metal texture with four drilled holes at slightly random positions; defects scratch, dent, stain, missing hole.
- **cap**: red ridged bottle cap on a dark tray; defects scratch, rim chip, stain.
- **tile**: ceramic tile with an embossed grid at a random offset; defects crack, stain, dent.
- **concrete**: mottled concrete with pores; cracks are random-walk polylines with a slowly turning heading, variable width and occasional branches; a quarter of the images are crack-free negatives.

Each image gets its own texture noise, lighting gradient, sensor noise and sometimes blur, so good parts differ from one another the way parts from one line do. Every defect is applied through an explicit mask, saved as `<name>_mask.png`; dents and stains are blended softly, so the mask marks the region that was changed.

Anomaly dataset (`--anomaly`, seed 5), per category: 60 good images for the fit, 20 held-out good images for calibration, a test split of 20 good images plus 8 images per defect type (32 defective for plate, 24 for cap and tile). Crack dataset (`--cracks`, seed 9): 120 training, 30 validation and 40 test images with masks.

## Models

- Anomaly mode: TorchVision `resnet18` with `ResNet18_Weights.IMAGENET1K_V1`, frozen; its SHA-256 is recorded in `report.json` under `features`. Nothing is trained; the memory bank is built from the good images' patch features (10% coreset).
- Segmentation mode: TorchVision `deeplabv3_mobilenet_v3_large` with `DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1` as the starting point, fine-tuned by Tech Boss on the 120 synthetic crack images; the checkpoint `cracks.pt` is not committed, the training log is.

## Executed runs

The committed `examples/` were produced on 14 November 2026 on a laptop CPU (Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu, numpy 2.5.4, Pillow 13.0.0):

```powershell
python synthetic_parts.py --anomaly artifacts/data --cracks artifacts/cracks --seed 5
python inspect_anomaly.py --cache-dir 'D:\Tech Boss\cv-cache' --threads 4 benchmark --data artifacts/data --output artifacts/anomaly
python inspect_cracks.py --cache-dir 'D:\Tech Boss\cv-cache' --threads 4 train --data artifacts/cracks --output artifacts/cracks-model --epochs 6
```

- `examples/anomaly/`: `dashboard.html`, `report.json` (features, per-category calibration, AUROC, per-defect recall, every test record with its score), `<category>/bank.json` (bank metadata; the `.npy` banks are not committed) and `<category>/heatmaps/` panels for every test image.
- `examples/cracks/`: `dashboard.html`, `training_log.json` (epochs, threshold sweep, test metrics per image) and `test/` panels for the worst and best test images.
- `examples/data/`: a few input images with their masks from each category.

Timings are from this one machine and run and are not a hardware benchmark.
