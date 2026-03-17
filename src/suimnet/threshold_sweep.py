"""
Sweep sigmoid thresholds for each SUIM-Net class using saved logit NPZ files.

For each class, tests a range of thresholds and reports which maximises mean IoU.
Requires running run_infer.py --save_logits first.

Usage (with profile/dataset):
    python -m src.suimnet.threshold_sweep \\
        --profile greatlakes --dataset deepfish \\
        --logits_dir outputs/deepfish/logits

Usage (with explicit paths):
    python -m src.suimnet.threshold_sweep \\
        --logits_dir outputs/deepfish/logits \\
        --masks_dir /data/deepfish/masks \\
        --classes FV

Usage (Great Lakes via SLURM):
    sbatch cluster/suimnet_sweep.sbat
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.common.config import resolve_dataset_paths

mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

CLASS_ORDER = ["RO", "FV", "HD", "RI", "WR"]
CLASS_LABELS = {
    "RO": "Robot / Instrument",
    "FV": "Fish / Vertebrate",
    "HD": "Human Diver",
    "RI": "Reef / Invertebrate",
    "WR": "Wreck / Ruin",
}
CLASS_COLORS = {
    "RO": "#c0392b",
    "FV": "#d4ac0d",
    "HD": "#2980b9",
    "RI": "#8e44ad",
    "WR": "#17a589",
}

DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()


def iou_binary(gt: np.ndarray, pred: np.ndarray) -> float:
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def load_gt_mask(gt_dir: Path, stem: str) -> np.ndarray | None:
    """Load a binary GT mask by stem, trying .bmp then .png."""
    for ext in (".bmp", ".png"):
        p = gt_dir / (stem + ext)
        if p.exists():
            from PIL import Image
            return np.array(Image.open(p).convert("L")) > 128
    return None


def sweep_class(logit_files: list[Path], gt_dir: Path, cls_idx: int,
                thresholds: list[float]) -> tuple[np.ndarray, int]:
    """
    For one class channel, sweep thresholds and return mean IoU per threshold.
    Returns (mean_iou_per_threshold, n_evaluated).
    """
    iou_per_thr = np.zeros(len(thresholds), dtype=np.float64)
    count = 0

    for npz_path in logit_files:
        gt_mask = load_gt_mask(gt_dir, npz_path.stem)
        if gt_mask is None:
            continue

        data = np.load(npz_path)
        logits = data["logits"].astype(np.float32)   # H x W x 5

        chan = logits[..., cls_idx]

        # Resize GT to match logit spatial dims if needed
        if gt_mask.shape != chan.shape[:2]:
            from skimage.transform import resize
            gt_mask = resize(gt_mask.astype(np.float32), chan.shape[:2],
                             order=0, preserve_range=True, anti_aliasing=False) > 0.5

        for j, thr in enumerate(thresholds):
            pred = chan > thr
            iou_per_thr[j] += iou_binary(gt_mask, pred)

        count += 1

    if count > 0:
        iou_per_thr /= count

    return iou_per_thr, count


def plot_sweep(results: dict[str, np.ndarray], thresholds: list[float],
               out_path: Path, dpi: int, title: str) -> None:
    """One subplot per class: mean IoU vs threshold, with optimal point marked."""
    classes = list(results.keys())
    n = len(classes)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, cls in enumerate(classes):
        ax = axes[idx // ncols][idx % ncols]
        ious = results[cls]
        color = CLASS_COLORS[cls]

        ax.plot(thresholds, ious, color=color, linewidth=2, marker="o",
                markersize=4, label=CLASS_LABELS[cls])

        best_idx = int(np.argmax(ious))
        best_thr = thresholds[best_idx]
        best_iou = ious[best_idx]
        ax.axvline(best_thr, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.scatter([best_thr], [best_iou], color=color, s=100, zorder=5,
                   edgecolors="black", linewidths=0.8,
                   label=f"Optimal thr={best_thr:.2f} (IoU={best_iou:.3f})")

        ax.set_title(f"{CLASS_LABELS[cls]} ({cls})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Mean IoU")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(ious.max() * 1.15, 0.1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.legend(fontsize=9, loc="best")

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{title} — Threshold Sweep", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep sigmoid thresholds over saved SUIM-Net logit NPZ files."
    )
    ap.add_argument("--logits_dir", required=True,
                    help="Directory of .npz logit files (output of run_infer.py --save_logits)")
    ap.add_argument("--masks_dir", default=None,
                    help="Root GT masks directory (sub-dirs RO/, FV/, etc.). "
                         "Overrides --profile/--dataset.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--classes", nargs="*", default=None,
                    help="Classes to sweep (default: all present in masks_dir). "
                         "E.g. --classes FV RI WR")
    ap.add_argument("--thresholds", nargs="*", type=float, default=None,
                    help="Threshold values to test (default: 0.05 0.10 … 0.95).")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for chart + CSV "
                         "(default: reports/suimnet/sweep/<dataset>/)")
    ap.add_argument("--title", default=None,
                    help="Title prefix for charts.")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    logits_dir = Path(args.logits_dir).resolve()
    if not logits_dir.exists():
        raise FileNotFoundError(f"Logits directory not found: {logits_dir}")

    # Resolve GT masks directory
    if args.masks_dir is not None:
        masks_root = Path(args.masks_dir).resolve()
        dataset_name = args.dataset or logits_dir.parent.name
    elif args.profile and args.dataset:
        ds = resolve_dataset_paths(profile=args.profile, dataset=args.dataset)
        if ds.labels_dir is None:
            raise ValueError(f"Dataset '{args.dataset}' has no labels_rel configured")
        masks_root = ds.labels_dir
        dataset_name = args.dataset
    else:
        ap.error("Provide --masks_dir OR both --profile and --dataset")

    thresholds = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS
    thresholds = sorted(set(thresholds))

    # Determine which classes to sweep
    if args.classes:
        eval_classes = [c.upper() for c in args.classes]
    else:
        eval_classes = [c for c in CLASS_ORDER if (masks_root / c).exists()]

    if not eval_classes:
        raise RuntimeError(f"No class subdirectories found under: {masks_root}")

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("reports/suimnet/sweep") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or f"SUIM-Net — {dataset_name}"

    # Collect all logit files
    logit_files = sorted(logits_dir.rglob("*.npz"))
    if not logit_files:
        raise RuntimeError(f"No .npz files found under: {logits_dir}")

    print(f"Logits dir : {logits_dir}  ({len(logit_files)} files)")
    print(f"Masks root : {masks_root}")
    print(f"Classes    : {eval_classes}")
    print(f"Thresholds : {thresholds[0]:.2f} … {thresholds[-1]:.2f}  ({len(thresholds)} values)")
    print(f"Output dir : {out_dir}")
    print()

    results: dict[str, np.ndarray] = {}
    optimal: dict[str, tuple[float, float]] = {}   # cls → (best_thr, best_iou)

    for cls in eval_classes:
        cls_idx = CLASS_ORDER.index(cls)
        gt_dir = masks_root / cls
        if not gt_dir.exists():
            print(f"  [{cls}] skipped — GT dir missing: {gt_dir}")
            continue

        print(f"  [{cls}] sweeping {len(thresholds)} thresholds …", flush=True)
        mean_ious, n = sweep_class(logit_files, gt_dir, cls_idx, thresholds)
        results[cls] = mean_ious

        best_idx = int(np.argmax(mean_ious))
        best_thr = thresholds[best_idx]
        best_iou = float(mean_ious[best_idx])
        optimal[cls] = (best_thr, best_iou)

        # Also report IoU at default 0.5
        default_iou = float(mean_ious[thresholds.index(0.5)]) if 0.5 in thresholds else float("nan")
        gain = best_iou - default_iou
        print(f"         n={n}  IoU@0.5={default_iou:.4f}  best IoU={best_iou:.4f} "
              f"@ thr={best_thr:.2f}  (Δ{gain:+.4f})")

    print()

    # ── Summary table ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"{'Class':<6} {'Opt thr':>8} {'Best IoU':>10} {'IoU@0.5':>10} {'Δ gain':>8}")
    print("-" * 60)
    for cls in eval_classes:
        if cls not in results:
            continue
        best_thr, best_iou = optimal[cls]
        iou_at_half = float(results[cls][thresholds.index(0.5)]) if 0.5 in thresholds else float("nan")
        print(f"{cls:<6} {best_thr:>8.2f} {best_iou:>10.4f} {iou_at_half:>10.4f} "
              f"{best_iou - iou_at_half:>+8.4f}")
    print("=" * 60)

    # ── Recommended thresholds block (paste into datasets.yaml) ───────────────
    print("\nRecommended datasets.yaml thresholds block:")
    print("    thresholds:")
    for cls in eval_classes:
        if cls in optimal:
            print(f"      {cls}: {optimal[cls][0]:.2f}")
    print()

    # ── CSV export ─────────────────────────────────────────────────────────────
    csv_path = out_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "threshold", "mean_iou"])
        for cls, ious in results.items():
            for thr, iou in zip(thresholds, ious):
                writer.writerow([cls, f"{thr:.2f}", f"{iou:.6f}"])
    print(f"  CSV  → {csv_path}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    if results:
        chart_path = out_dir / "threshold_sweep.png"
        plot_sweep(results, thresholds, chart_path, args.dpi, title)

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
