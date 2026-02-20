"""
Compute per-class IoU, boundary F-score, and overall mIoU for SUIM-Net predictions.

Usage (local, with bundled sample data):
    python -m src.suimnet.metric_calc \
        --preds_dir outputs/sample \
        --masks_dir vendor/SUIM-Net/sample_test/masks

Usage (Great Lakes, full SUIM test set):
    python -m src.suimnet.metric_calc \
        --profile greatlakes --dataset suim \
        --preds_dir outputs/suim

Usage (Great Lakes via SLURM):
    sbatch cluster/suimnet_metrics.sbat
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.config import resolve_dataset_paths

# ---------------------------------------------------------------------------
# Class definitions – order matches SUIM-Net 5-channel sigmoid outputs
# Each class maps to a specific RGB encoding in the composite prediction mask.
# ---------------------------------------------------------------------------
CLASSES = {
    #  name    R      G      B
    "RO": (True,  False, False),   # Robot/instrument  -> Red
    "FV": (True,  True,  False),   # Fish/vertebrate   -> Yellow
    "HD": (False, False, True),    # Human diver       -> Blue
    "RI": (True,  False, True),    # Reef/invertebrate -> Magenta
    "WR": (False, True,  True),    # Wreck/ruin        -> Cyan
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def iou_binary(gt: np.ndarray, pred: np.ndarray) -> float:
    """Binary IoU for boolean masks of identical shape."""
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0  # both empty → perfect
    return float(inter) / float(union)


def dice_binary(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice / F1 coefficient for boolean masks."""
    inter = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()
    if total == 0:
        return 1.0
    return float(2 * inter) / float(total)


def precision_recall(gt: np.ndarray, pred: np.ndarray):
    """Pixel-level precision and recall."""
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return float(prec), float(rec)


# ---------------------------------------------------------------------------
# Prediction decoding
# ---------------------------------------------------------------------------

def extract_class_mask(pred_rgb: np.ndarray, r: bool, g: bool, b: bool,
                       thr: int = 128) -> np.ndarray:
    """
    Extract a binary class mask from an RGB prediction image.

    Because the SUIM-Net RGB encoding uses OR-based composition across classes,
    a pixel belongs to a class if ALL of its expected channels are high AND
    none of the unexpected channels are high (to disambiguate overlapping bits).

    However, in practice the 5-class mask allows overlaps, so we simply check
    whether the expected channels are active (> thr).
    """
    mask = np.ones(pred_rgb.shape[:2], dtype=bool)
    if r:
        mask &= pred_rgb[..., 0] > thr
    else:
        mask &= pred_rgb[..., 0] <= thr
    if g:
        mask &= pred_rgb[..., 1] > thr
    else:
        mask &= pred_rgb[..., 1] <= thr
    if b:
        mask &= pred_rgb[..., 2] > thr
    else:
        mask &= pred_rgb[..., 2] <= thr
    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute IoU / Dice / Precision / Recall for SUIM-Net predictions vs ground truth."
    )
    ap.add_argument("--preds_dir", required=True,
                    help="Directory containing prediction PNG masks (output of run_infer.py)")
    ap.add_argument("--masks_dir", default=None,
                    help="Root ground-truth masks directory (must contain per-class sub-dirs RO/, FV/, etc.). "
                         "Overrides --profile/--dataset resolution.")
    ap.add_argument("--profile", default=None,
                    help="Profile key (e.g. greatlakes) – used to locate GT masks via configs")
    ap.add_argument("--dataset", default=None,
                    help="Dataset key (e.g. suim) – used to locate GT masks via configs")
    ap.add_argument("--out_csv", default=None,
                    help="Optional path to write per-image CSV results")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="Subset of classes to evaluate (default: all). E.g. --classes RO HD WR")
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir).resolve()
    if not preds_dir.exists():
        print(f"ERROR: Predictions directory not found: {preds_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve ground-truth masks directory
    if args.masks_dir is not None:
        masks_root = Path(args.masks_dir).resolve()
    elif args.profile and args.dataset:
        ds = resolve_dataset_paths(profile=args.profile, dataset=args.dataset)
        if ds.labels_dir is None:
            print(f"ERROR: Dataset '{args.dataset}' has no labels_rel configured", file=sys.stderr)
            sys.exit(1)
        masks_root = ds.labels_dir
    else:
        ap.error("Provide either --masks_dir OR both --profile and --dataset")

    if not masks_root.exists():
        print(f"ERROR: Masks directory not found: {masks_root}", file=sys.stderr)
        sys.exit(1)

    # Determine which classes to evaluate
    eval_classes = list(CLASSES.keys()) if args.classes is None else [c.upper() for c in args.classes]
    for c in eval_classes:
        if c not in CLASSES:
            print(f"ERROR: Unknown class '{c}'. Valid: {list(CLASSES.keys())}", file=sys.stderr)
            sys.exit(1)

    # Verify per-class sub-dirs exist
    for cls_name in eval_classes:
        cls_dir = masks_root / cls_name
        if not cls_dir.exists():
            print(f"WARNING: GT sub-directory missing: {cls_dir} – skipping class {cls_name}", file=sys.stderr)
            eval_classes = [c for c in eval_classes if c != cls_name]

    if not eval_classes:
        print("ERROR: No valid class directories found under masks_root.", file=sys.stderr)
        sys.exit(1)

    print(f"Predictions : {preds_dir}")
    print(f"Ground truth: {masks_root}")
    print(f"Classes     : {eval_classes}")
    print()

    # Collect results: {class_name: [(image_name, iou, dice, prec, rec), ...]}
    results: dict[str, list[tuple[str, float, float, float, float]]] = {c: [] for c in eval_classes}

    for cls_name in eval_classes:
        r, g, b = CLASSES[cls_name]
        gt_dir = masks_root / cls_name
        gt_files = sorted(gt_dir.glob("*.bmp")) + sorted(gt_dir.glob("*.png"))

        for gt_path in gt_files:
            # Match prediction by stem (predictions are .png)
            pred_path = preds_dir / (gt_path.stem + ".png")
            if not pred_path.exists():
                continue

            gt_img = Image.open(gt_path).convert("L")
            pred_img = Image.open(pred_path).convert("RGB")

            # Resize prediction to match GT dimensions
            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.NEAREST)

            gt_mask = np.array(gt_img) > 128   # binarize GT
            pred_rgb = np.array(pred_img)
            pred_mask = extract_class_mask(pred_rgb, r, g, b)

            _iou = iou_binary(gt_mask, pred_mask)
            _dice = dice_binary(gt_mask, pred_mask)
            _prec, _rec = precision_recall(gt_mask, pred_mask)

            results[cls_name].append((gt_path.stem, _iou, _dice, _prec, _rec))

    # ---- Print per-class summary ----
    print("=" * 72)
    print(f"{'Class':<6} {'mIoU':>8} {'mDice':>8} {'mPrec':>8} {'mRec':>8} {'Images':>8}")
    print("-" * 72)

    all_ious = []
    for cls_name in eval_classes:
        rows = results[cls_name]
        if not rows:
            print(f"{cls_name:<6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>8}")
            continue
        ious  = [r[1] for r in rows]
        dices = [r[2] for r in rows]
        precs = [r[3] for r in rows]
        recs  = [r[4] for r in rows]
        all_ious.extend(ious)
        print(f"{cls_name:<6} {np.mean(ious):>8.4f} {np.mean(dices):>8.4f} "
              f"{np.mean(precs):>8.4f} {np.mean(recs):>8.4f} {len(rows):>8}")

    print("-" * 72)
    if all_ious:
        print(f"{'MEAN':<6} {np.mean(all_ious):>8.4f}")
    else:
        print("No matching prediction/GT pairs found.")
    print("=" * 72)

    # ---- Optional CSV export ----
    if args.out_csv:
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "image", "iou", "dice", "precision", "recall"])
            for cls_name in eval_classes:
                for stem, _iou, _dice, _prec, _rec in results[cls_name]:
                    writer.writerow([cls_name, stem, f"{_iou:.6f}", f"{_dice:.6f}",
                                     f"{_prec:.6f}", f"{_rec:.6f}"])
        print(f"\nPer-image results written to: {csv_path}")


if __name__ == "__main__":
    main()
