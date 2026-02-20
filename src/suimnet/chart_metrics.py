"""
Chart SUIM-Net benchmark metrics from a CSV produced by metric_calc.py.

Usage:
    python -m src.suimnet.chart_metrics                                 # defaults: reports/suim_metrics.csv
    python -m src.suimnet.chart_metrics --csv reports/sample_metrics.csv
    python -m src.suimnet.chart_metrics --csv reports/suim_metrics.csv --out reports/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Nice readable names
CLASS_LABELS = {
    "RO": "Robot / Instrument",
    "FV": "Fish / Vertebrate",
    "HD": "Human Diver",
    "RI": "Reef / Invertebrate",
    "WR": "Wreck / Ruin",
}
CLASS_COLORS = {
    "RO": "#e74c3c",   # red
    "FV": "#f1c40f",   # yellow
    "HD": "#3498db",   # blue
    "RI": "#9b59b6",   # purple
    "WR": "#1abc9c",   # teal
}
METRIC_COLS = ["iou", "dice", "precision", "recall"]
METRIC_LABELS = {"iou": "IoU", "dice": "Dice (F1)", "precision": "Precision", "recall": "Recall"}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Chart 1: per-class mean bar chart ─────────────────────────────────
def plot_class_means(df: pd.DataFrame, out_dir: Path) -> None:
    means = df.groupby("class")[METRIC_COLS].mean().reindex(CLASS_LABELS.keys())

    x = np.arange(len(means))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, col in enumerate(METRIC_COLS):
        bars = ax.bar(x + i * width, means[col], width, label=METRIC_LABELS[col])
        for bar, val in zip(bars, means[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([CLASS_LABELS[c] for c in means.index], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("SUIM-Net — Per-Class Mean Metrics")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "class_means.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 2: IoU box-plots per class ──────────────────────────────────
def plot_iou_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    classes = [c for c in CLASS_LABELS if c in df["class"].unique()]
    data = [df.loc[df["class"] == c, "iou"].values for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], fontsize=9)
    ax.set_ylabel("IoU")
    ax.set_ylim(-0.05, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("SUIM-Net — IoU Distribution per Class")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "iou_boxplots.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 3: overall mIoU horizontal summary ─────────────────────────
def plot_overall_summary(df: pd.DataFrame, out_dir: Path) -> None:
    means = df.groupby("class")[METRIC_COLS].mean().reindex(CLASS_LABELS.keys())
    overall = means.mean()

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(
        [METRIC_LABELS[m] for m in METRIC_COLS],
        [overall[m] for m in METRIC_COLS],
        color=["#2ecc71", "#3498db", "#e67e22", "#9b59b6"],
        height=0.5,
    )
    for bar, val in zip(bars, [overall[m] for m in METRIC_COLS]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("SUIM-Net — Overall Mean Metrics (macro-avg across classes)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "overall_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 4: per-image IoU heatmap (classes × images) ────────────────
def plot_iou_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(index="class", columns="image", values="iou")
    pivot = pivot.reindex([c for c in CLASS_LABELS if c in pivot.index])

    # Sort columns by mean IoU so worst images are on the left
    col_order = pivot.mean(axis=0).sort_values().index
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.12), 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([CLASS_LABELS[c] for c in pivot.index], fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel(f"Images (sorted by mean IoU, n={len(pivot.columns)})")
    ax.set_title("SUIM-Net — Per-Image IoU Heatmap")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("IoU")
    fig.tight_layout()

    path = out_dir / "iou_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 5: precision-recall scatter per class ───────────────────────
def plot_precision_recall(df: pd.DataFrame, out_dir: Path) -> None:
    classes = [c for c in CLASS_LABELS if c in df["class"].unique()]
    fig, ax = plt.subplots(figsize=(7, 6))

    for cls in classes:
        sub = df[df["class"] == cls]
        ax.scatter(sub["recall"], sub["precision"],
                   c=CLASS_COLORS[cls], label=CLASS_LABELS[cls],
                   alpha=0.45, s=18, edgecolors="none")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.15)
    ax.set_title("SUIM-Net — Precision vs Recall (per image)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / "precision_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Chart SUIM-Net metrics from CSV.")
    ap.add_argument("--csv", default="reports/suim_metrics.csv",
                    help="Path to metrics CSV (default: reports/suim_metrics.csv)")
    ap.add_argument("--out", default=None,
                    help="Output directory for PNGs (default: reports/figures)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out) if args.out else csv_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    n_images = df["image"].nunique()
    n_classes = df["class"].nunique()
    print(f"Loaded {len(df)} rows  ({n_images} images × {n_classes} classes)  from {csv_path}")

    plot_class_means(df, out_dir)
    plot_iou_boxplots(df, out_dir)
    plot_overall_summary(df, out_dir)
    plot_iou_heatmap(df, out_dir)
    plot_precision_recall(df, out_dir)

    print(f"\nAll charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
