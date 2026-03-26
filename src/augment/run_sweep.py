"""
Turbidity robustness sweep for SUIM-Net segmentation.

What this script does
─────────────────────
For each turbidity level in [0.0, 0.25, 0.5, 0.75, 1.0]:
  1. Apply apply_turbidity(img, level) to every image in --images_dir
  2. Run SUIM-Net inference on the degraded images
  3. Compute IoU / Dice / Precision / Recall against ground-truth masks
  4. Write a CSV to  --out_dir/level_<X.XX>/metrics.csv
After all levels, write a summary CSV:  --out_dir/turbidity_summary.csv

The summary CSV has one row per turbidity level × class, which can be plotted
directly to show how accuracy degrades as turbidity increases.

Usage (on ARC Great Lakes)
──────────────────────────
    cd /path/to/rob472-underwater-danger-map

    python -m src.augment.run_sweep \\
        --images_dir $DATA_ROOT/suim/test/images \\
        --masks_dir  $DATA_ROOT/suim/test/masks \\
        --weights    vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \\
        --out_dir    reports/turbidity

    # Produces:
    #   reports/turbidity/level_0.00/metrics.csv
    #   reports/turbidity/level_0.25/metrics.csv
    #   ...
    #   reports/turbidity/turbidity_summary.csv

Notes
─────
- Mask directory must follow the SUIM layout: <masks_dir>/<CLASS>/*.bmp or .png
  where CLASS ∈ {RO, FV, HD, RI, WR}.
- Images are processed in-memory — no augmented images are written to disk.
- The sweep reuses one SUIM-Net model instance across all levels for efficiency.
- Per-class thresholds default to 0.5. Override with --thr.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import skimage.transform as sktf
from PIL import Image
from skimage import io as skio

# ── Repo-relative imports ─────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
SUIMNET_ROOT = REPO_ROOT / "vendor" / "SUIM-Net"
sys.path.append(str(SUIMNET_ROOT))

# Keras 2.13+ compatibility shim (mirrors src/suimnet/run_infer.py)
import keras
import keras.models as _km
if not hasattr(_km, "Input"):
    _km.Input = keras.layers.Input
_OrigModel = _km.Model
class _ModelShim(_OrigModel):
    def __init__(self, *args, **kwargs):
        if "input" in kwargs and "inputs" not in kwargs:
            kwargs["inputs"] = kwargs.pop("input")
        if "output" in kwargs and "outputs" not in kwargs:
            kwargs["outputs"] = kwargs.pop("output")
        super().__init__(*args, **kwargs)
_km.Model = _ModelShim

from model import SUIM_Net  # type: ignore  (vendor/SUIM-Net/model.py)

from src.augment.turbidity import apply_turbidity
from src.suimnet.run_infer import suimnet_rgb_from_logits
from src.suimnet.metric_calc import (
    CLASSES,
    extract_class_mask,
    iou_binary,
    dice_binary,
    precision_recall,
)

# ── Constants ─────────────────────────────────────────────────────────────────
SUIM_H, SUIM_W = 240, 320
DEFAULT_LEVELS  = [0.0, 0.25, 0.5, 0.75, 1.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def _run_inference(model, img: np.ndarray, threshold: float) -> np.ndarray:
    """
    Run SUIM-Net on a single image and return an RGB prediction mask.

    Args:
        model:     Loaded Keras SUIM-Net model.
        img:       (H, W, 3) uint8 RGB image.
        threshold: Sigmoid threshold applied to all classes.

    Returns:
        (H, W, 3) uint8 RGB mask (SUIM colour encoding).
    """
    img_rs = sktf.resize(img, (SUIM_H, SUIM_W, 3),
                         preserve_range=False, anti_aliasing=True)
    x = np.expand_dims(img_rs, axis=0)
    logits = model.predict(x, verbose=0)[0]   # (H, W, 5)
    thresholds = {cls: threshold for cls in ["RO", "FV", "HD", "RI", "WR"]}
    return suimnet_rgb_from_logits(logits, thresholds)   # (H, W, 3) uint8


def _compute_metrics_for_level(
    model,
    images_dir: Path,
    masks_root: Path,
    level: float,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """
    Apply turbidity augmentation at `level` to all images, run SUIM-Net,
    and compute per-class metrics against ground-truth masks.

    Returns:
        {class_name: {"iou": float, "dice": float, "precision": float, "recall": float}}
    """
    # Accumulate per-class metric lists
    results: dict[str, list[tuple[float, float, float, float]]] = {
        c: [] for c in CLASSES
    }

    for cls_name, (r, g, b) in CLASSES.items():
        gt_dir = masks_root / cls_name
        if not gt_dir.exists():
            continue

        gt_files = sorted(gt_dir.glob("*.bmp")) + sorted(gt_dir.glob("*.png"))
        for gt_path in gt_files:
            img_path = _find_image(images_dir, gt_path.stem)
            if img_path is None:
                continue

            # Load and augment
            img = skio.imread(str(img_path))
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.shape[-1] > 3:
                img = img[..., :3]

            img_aug = apply_turbidity(img, level)

            # Inference
            pred_rgb = _run_inference(model, img_aug, threshold)

            # Metrics
            gt_img  = Image.open(gt_path).convert("L")
            pred_img = Image.fromarray(pred_rgb)
            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.NEAREST)

            gt_mask   = np.array(gt_img) > 128
            pred_mask = extract_class_mask(np.array(pred_img), r, g, b)

            iou   = iou_binary(gt_mask, pred_mask)
            dice  = dice_binary(gt_mask, pred_mask)
            prec, rec = precision_recall(gt_mask, pred_mask)
            results[cls_name].append((iou, dice, prec, rec))

    # Aggregate per class
    agg: dict[str, dict[str, float]] = {}
    for cls_name, rows in results.items():
        if not rows:
            agg[cls_name] = {"iou": float("nan"), "dice": float("nan"),
                             "precision": float("nan"), "recall": float("nan")}
        else:
            agg[cls_name] = {
                "iou":       float(np.mean([r[0] for r in rows])),
                "dice":      float(np.mean([r[1] for r in rows])),
                "precision": float(np.mean([r[2] for r in rows])),
                "recall":    float(np.mean([r[3] for r in rows])),
            }
    return agg


def _find_image(images_dir: Path, stem: str) -> Path | None:
    """Search images_dir for a file whose stem matches `stem` (any extension)."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    # Fall back to recursive search (handles sub-directories)
    hits = list(images_dir.rglob(stem + ".*"))
    hits = [h for h in hits if h.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    return hits[0] if hits else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep turbidity levels and evaluate SUIM-Net segmentation accuracy at each level."
    )
    ap.add_argument("--images_dir", required=True,
                    help="Directory of input images (e.g. SUIM test set images/).")
    ap.add_argument("--masks_dir", required=True,
                    help="Ground-truth mask root. Must contain sub-folders RO/, FV/, HD/, RI/, WR/ "
                         "with per-image .bmp or .png masks.")
    ap.add_argument("--weights",
                    default=str(SUIMNET_ROOT / "sample_test" / "ckpt_seg_5obj.hdf5"),
                    help="Path to SUIM-Net .hdf5 weights.")
    ap.add_argument("--out_dir", default="reports/turbidity",
                    help="Output directory for per-level CSVs and summary. Default: reports/turbidity")
    ap.add_argument("--levels", nargs="+", type=float, default=DEFAULT_LEVELS,
                    help=f"Turbidity levels to sweep. Default: {DEFAULT_LEVELS}")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="SUIM-Net sigmoid threshold. Default: 0.5")
    args = ap.parse_args()

    images_dir = Path(args.images_dir).resolve()
    masks_root = Path(args.masks_dir).resolve()
    out_dir    = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks dir not found: {masks_root}")

    print(f"Images  : {images_dir}")
    print(f"Masks   : {masks_root}")
    print(f"Levels  : {args.levels}")
    print(f"Output  : {out_dir}")
    print()

    # Load model once — shared across all levels
    print("Loading SUIM-Net…")
    model = SUIM_Net(im_res=(SUIM_H, SUIM_W), n_classes=5).model
    model.load_weights(str(args.weights))
    print()

    # Per-level evaluation
    all_rows: list[dict] = []

    for level in args.levels:
        print(f"── Turbidity level {level:.2f} ──────────────────────")
        metrics = _compute_metrics_for_level(
            model, images_dir, masks_root, level, args.thr
        )

        # Print summary
        iou_vals = [v["iou"] for v in metrics.values() if not np.isnan(v["iou"])]
        miou = float(np.mean(iou_vals)) if iou_vals else float("nan")
        print(f"  mIoU = {miou:.4f}")
        for cls_name, m in metrics.items():
            print(f"  {cls_name}: IoU={m['iou']:.4f}  Dice={m['dice']:.4f}  "
                  f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")

        # Write per-level CSV
        level_dir = out_dir / f"level_{level:.2f}"
        level_dir.mkdir(parents=True, exist_ok=True)
        csv_path = level_dir / "metrics.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["level", "class", "iou", "dice", "precision", "recall"])
            writer.writeheader()
            for cls_name, m in metrics.items():
                writer.writerow({"level": level, "class": cls_name, **m})
        print(f"  → {csv_path}")
        print()

        # Accumulate for summary
        for cls_name, m in metrics.items():
            all_rows.append({"level": level, "class": cls_name, **m})

    # Write summary CSV (all levels × classes in one file — easy to load into pandas)
    summary_path = out_dir / "turbidity_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "class", "iou", "dice", "precision", "recall"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Summary CSV → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
