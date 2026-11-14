"""Heatmap overlays, calibration plots as SVG, and HTML galleries (Pillow + numpy, no matplotlib)."""
from __future__ import annotations

import html

import numpy as np


def colormap(values):
    """Blue-cyan-yellow-red colour ramp for values in [0, 1]; returns uint8 HxWx3."""
    v = np.clip(np.asarray(values, np.float32), 0, 1)
    r = np.clip(1.5 * v - 0.25, 0, 1)
    g = np.clip(1.0 - np.abs(v - 0.5) * 2.2 + 0.4, 0, 1) * np.clip(2.5 * v, 0, 1)
    b = np.clip(1.0 - 2.2 * v, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image, heat, low, high, alpha=0.55, mask=None, contour_color=(255, 255, 255)):
    """Blend a heatmap normalised between `low` and `high` over the image; draw the truth mask outline when given."""
    from PIL import Image

    array = np.asarray(image.convert("RGB"), np.float32)
    normalised = (np.asarray(heat, np.float32) - low) / max(1e-6, high - low)
    colour = colormap(normalised).astype(np.float32)
    weight = np.clip(normalised, 0, 1)[..., None] * alpha
    blended = array * (1 - weight) + colour * weight
    if mask is not None:
        mask = np.asarray(mask, bool)
        edge = mask & ~_erode(mask)
        blended[edge] = contour_color
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")


def _erode(mask):
    padded = np.pad(mask, 1, mode="constant")
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return out


def side_by_side(images, labels, font_size=14):
    from PIL import Image, ImageDraw

    width = sum(i.width for i in images) + 8 * (len(images) - 1)
    height = max(i.height for i in images) + font_size + 8
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for image, label in zip(images, labels):
        canvas.paste(image, (x, font_size + 6))
        draw.text((x + 2, 2), label, fill=(30, 30, 30))
        x += image.width + 8
    return canvas


def score_histogram_svg(good, defect, threshold, width=520, height=200, bins=30):
    """Two histograms (good parts, defective parts) with the calibrated threshold."""
    values = list(good) + list(defect)
    low, high = min(values), max(values)
    span = max(1e-6, high - low)
    edges = [low + span * i / bins for i in range(bins + 1)]

    def counts(data):
        out = [0] * bins
        for v in data:
            index = min(bins - 1, int((v - low) / span * bins))
            out[index] += 1
        return out

    good_counts, defect_counts = counts(good), counts(defect)
    top = max(good_counts + defect_counts + [1])
    left, bottom, plot_w, plot_h = 40, height - 30, width - 60, height - 50
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' style='background:#fff;border:1px solid #e2e8f0'>"]
    bar_w = plot_w / bins
    for i in range(bins):
        for data, colour, shift in ((good_counts, "#16a34a", 0), (defect_counts, "#dc2626", bar_w / 2)):
            h = plot_h * data[i] / top
            parts.append(f"<rect x='{left + i * bar_w + shift:.1f}' y='{bottom - h:.1f}' width='{bar_w / 2:.1f}' height='{h:.1f}' fill='{colour}' opacity='0.8'/>")
    tx = left + (threshold - low) / span * plot_w
    parts.append(f"<line x1='{tx:.1f}' y1='{bottom - plot_h}' x2='{tx:.1f}' y2='{bottom}' stroke='#111' stroke-width='2' stroke-dasharray='4 3'/>")
    parts.append(f"<text x='{tx + 4:.1f}' y='{bottom - plot_h + 12}' font-size='11' fill='#111'>threshold {threshold:.3f}</text>")
    parts.append(f"<text x='{left}' y='{height - 8}' font-size='11' fill='#64748b'>{low:.2f}</text><text x='{left + plot_w - 30}' y='{height - 8}' font-size='11' fill='#64748b'>{high:.2f}</text>")
    parts.append(f"<text x='{left + 50}' y='{height - 8}' font-size='11' fill='#16a34a'>good</text><text x='{left + 90}' y='{height - 8}' font-size='11' fill='#dc2626'>defective</text>")
    parts.append("</svg>")
    return "".join(parts)


def curve_svg(points, x_label, y_label, width=420, height=220, marker=None):
    """Simple line chart for (x, y) points in [0, 1] x [0, 1]."""
    left, bottom, plot_w, plot_h = 40, height - 30, width - 60, height - 50
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' style='background:#fff;border:1px solid #e2e8f0'>"]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - plot_h * frac
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#e2e8f0'/><text x='4' y='{y + 4:.1f}' font-size='10' fill='#64748b'>{frac:.2f}</text>")
    path = " ".join(f"{left + plot_w * x:.1f},{bottom - plot_h * y:.1f}" for x, y in points)
    parts.append(f"<polyline points='{path}' fill='none' stroke='#2563eb' stroke-width='2'/>")
    if marker is not None:
        mx, my = marker
        parts.append(f"<circle cx='{left + plot_w * mx:.1f}' cy='{bottom - plot_h * my:.1f}' r='5' fill='#dc2626'/>")
    parts.append(f"<text x='{left + plot_w / 2 - 30}' y='{height - 6}' font-size='11' fill='#64748b'>{html.escape(x_label)}</text>")
    parts.append(f"<text x='{left + 4}' y='{bottom - plot_h - 6}' font-size='11' fill='#64748b'>{html.escape(y_label)}</text></svg>")
    return "".join(parts)


STYLE = """body{font-family:system-ui,sans-serif;margin:24px;color:#1f2937;background:#f8fafc} h1{margin:0 0 4px} h2{margin-top:28px} .muted{color:#64748b}
table{border-collapse:collapse;margin:12px 0 20px;background:#fff} td,th{border:1px solid #e2e8f0;padding:5px 9px;text-align:left;font-size:14px}
img{display:block;border:1px solid #e2e8f0} .gallery{display:flex;flex-wrap:wrap;gap:14px} .tile{background:#fff;border:1px solid #e2e8f0;padding:8px;max-width:560px}
.hit{color:#16a34a;font-weight:600} .miss{color:#dc2626;font-weight:600} .fa{color:#d97706;font-weight:600} code{background:#eef2ff;padding:1px 4px}"""


def anomaly_dashboard(report, tiles):
    rows = "".join(f"<tr><td>{html.escape(c)}</td><td>{r['train_good']}</td><td>{r['calibration_good']}</td><td>{r['test']['good']}</td><td>{r['test']['defective']}</td>"
                   f"<td>{r['image_auroc']:.3f}</td><td>{'' if r['pixel_auroc'] is None else f'{r['pixel_auroc']:.3f}'}</td><td>{r['threshold']:.3f}</td>"
                   f"<td>{r['at_threshold']['recall']:.2f}</td><td>{r['at_threshold']['false_positive_rate']:.2f}</td><td>{r['bank_patches']}</td></tr>"
                   for c, r in report["categories"].items())
    per_defect = "".join(f"<tr><td>{html.escape(c)}</td><td>{html.escape(d)}</td><td>{v['count']}</td><td>{v['caught']}</td><td>{v['mean_score']:.3f}</td></tr>"
                         for c, r in report["categories"].items() for d, v in r["per_defect"].items())
    plots = "".join(f"<div class='tile'><div><b>{html.escape(c)}</b> score distributions on the test set</div>{r['histogram_svg']}</div>" for c, r in report["categories"].items())
    gallery = "".join(f"<div class='tile'><img src='{html.escape(t['image'])}' alt='{html.escape(t['name'])}'><div><span class='{t['class']}'>{html.escape(t['status'])}</span> · "
                      f"{html.escape(t['name'])} · score {t['score']:.3f} vs {t['threshold']:.3f}</div></div>" for t in tiles)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Visual defect inspection</title><style>{STYLE}</style></head><body>
<h1>Visual defect inspection: anomaly mode</h1>
<p class="muted">Trained on good parts only · features {html.escape(report['features']['backbone'])} layers {html.escape(', '.join(report['features']['layers']))} at {report['features']['input_size']} px ·
coreset {report['coreset_fraction']:.0%} · threshold at {report['target_false_alarm']:.0%} false alarms on held-out good parts · median scoring time {report['timing']['median_score_ms']:.0f} ms per image</p>
<table><tr><th>Category</th><th>Train good</th><th>Calib. good</th><th>Test good</th><th>Test defective</th><th>Image AUROC</th><th>Pixel AUROC</th><th>Threshold</th><th>Recall</th><th>False alarm rate</th><th>Bank patches</th></tr>{rows}</table>
<table><tr><th>Category</th><th>Defect</th><th>Images</th><th>Caught</th><th>Mean score</th></tr>{per_defect}</table>
<div class="gallery">{plots}</div>
<h2>Every miss and false alarm, then a sample of hits</h2>
<p class="muted">Left: input with the truth mask outlined in white. Right: heatmap of the patch distance to the nearest good-part patch.</p>
<div class="gallery">{gallery}</div>
</body></html>
"""


def crack_dashboard(report, tiles):
    rows = "".join(f"<tr><td>{r['epoch']}</td><td>{r['train_loss']:.4f}</td><td>{r['val_iou']:.3f}</td><td>{r['seconds']:.0f}</td></tr>" for r in report["epochs"])
    sweep = "".join(f"<tr><td>{s['threshold']:.2f}</td><td>{s['iou']:.3f}</td><td>{s['precision']:.3f}</td><td>{s['recall']:.3f}</td></tr>" for s in report["threshold_sweep"])
    gallery = "".join(f"<div class='tile'><img src='{html.escape(t['image'])}' alt='{html.escape(t['name'])}'><div>{html.escape(t['name'])} · IoU {'' if t['iou'] is None else f'{t['iou']:.3f}'}"
                      f" · predicted {t['predicted_pixels']} px, truth {t['truth_pixels']} px</div></div>" for t in tiles)
    t = report["test"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Crack segmentation</title><style>{STYLE}</style></head><body>
<h1>Visual defect inspection: crack segmentation</h1>
<p class="muted">DeepLabV3 MobileNetV3-Large fine-tuned from the TorchVision checkpoint · {report['train_images']} training images · {report['input_size']} px · threshold {t['threshold']:.2f} chosen on the validation split</p>
<table><tr><th>Test split</th><th>Images</th><th>Mean IoU (crack images)</th><th>Dataset IoU</th><th>Pixel precision</th><th>Pixel recall</th><th>Clean images with false crack pixels</th></tr>
<tr><td>test</td><td>{t['images']}</td><td>{t['mean_iou_crack_images']:.3f}</td><td>{t['dataset_iou']:.3f}</td><td>{t['precision']:.3f}</td><td>{t['recall']:.3f}</td><td>{t['clean_false_positive_images']} / {t['clean_images']}</td></tr></table>
<p class="muted">Tolerant F1 (offsets up to {t.get('tolerance_px', 2)} px forgiven, the usual crack-benchmark convention): {t.get('mean_tolerant_f1_crack_images', 0):.3f} mean over crack images.</p>
<div class="gallery"><div class='tile'><b>Validation IoU against probability threshold</b>{report['sweep_svg']}</div></div>
<table><tr><th>Threshold</th><th>Val IoU</th><th>Precision</th><th>Recall</th></tr>{sweep}</table>
<table><tr><th>Epoch</th><th>Train loss</th><th>Val IoU</th><th>Seconds</th></tr>{rows}</table>
<h2>Worst and best test images</h2>
<p class="muted">Left: input. Middle: truth mask. Right: prediction (crack probability above the threshold).</p>
<div class="gallery">{gallery}</div>
</body></html>
"""
