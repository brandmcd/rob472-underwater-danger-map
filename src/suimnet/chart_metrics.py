"""
Chart SUIM-Net benchmark metrics from a CSV produced by metric_calc.py.

Usage:
    python -m src.suimnet.chart_metrics
    python -m src.suimnet.chart_metrics --csv reports/suimnet/suim_metrics.csv
    python -m src.suimnet.chart_metrics --csv reports/suimnet/suim_metrics.csv \\
        --out reports/suimnet/figures --title "SUIM TEST Split (110 images)" --dpi 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Global style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,   # screen preview; overridden by --dpi at save time
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

# ── Class metadata ─────────────────────────────────────────────────────────────
CLASS_LABELS = {
    "RO": "Robot / Instrument",
    "FV": "Fish / Vertebrate",
    "HD": "Human Diver",
    "RI": "Reef / Invertebrate",
    "WR": "Wreck / Ruin",
}
# Colors match the SUIM RGB encoding (adjusted for print legibility)
CLASS_COLORS = {
    "RO": "#c0392b",   # deep red
    "FV": "#d4ac0d",   # gold-yellow
    "HD": "#2980b9",   # strong blue
    "RI": "#8e44ad",   # purple
    "WR": "#17a589",   # teal
}
METRIC_COLS   = ["iou", "dice", "precision", "recall"]
METRIC_LABELS = {"iou": "IoU", "dice": "Dice (F1)", "precision": "Precision", "recall": "Recall"}
METRIC_COLORS = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 1: per-class mean grouped bar chart ─────────────────────────────────
def plot_class_means(df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    means = df.groupby("class")[METRIC_COLS].mean().reindex(CLASS_LABELS.keys())

    x     = np.arange(len(means))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, col in enumerate(METRIC_COLS):
        bars = ax.bar(x + i * width, means[col], width,
                      label=METRIC_LABELS[col], color=METRIC_COLORS[i], alpha=0.88)
        for bar, val in zip(bars, means[col]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([CLASS_LABELS[c] for c in means.index])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(f"{title_prefix} — Per-Class Mean Metrics")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / "class_means.png", dpi)


# ── Chart 2: IoU box-plots per class ──────────────────────────────────────────
def plot_iou_boxplots(df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    classes = [c for c in CLASS_LABELS if c in df["class"].unique()]
    data    = [df.loc[df["class"] == c, "iou"].dropna().values for c in classes]
    colors  = [CLASS_COLORS[c] for c in classes]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(
        data, patch_artist=True, showmeans=True, widths=0.55,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="#333", markersize=6),
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticklabels([CLASS_LABELS[c] for c in classes])
    ax.set_ylabel("IoU")
    ax.set_ylim(-0.05, 1.12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(f"{title_prefix} — IoU Distribution per Class")
    ax.axhline(0.5,  color="orange", linestyle="--", linewidth=0.9, alpha=0.7, label="Acceptable (0.50)")
    ax.axhline(0.7,  color="green",  linestyle="--", linewidth=0.9, alpha=0.7, label="Good (0.70)")
    ax.axhline(0.85, color="navy",   linestyle="--", linewidth=0.9, alpha=0.7, label="Excellent (0.85)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / "iou_boxplots.png", dpi)


# ── Chart 3: overall macro-average horizontal summary ─────────────────────────
def plot_overall_summary(df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    means   = df.groupby("class")[METRIC_COLS].mean().reindex(CLASS_LABELS.keys())
    overall = means.mean()

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels  = [METRIC_LABELS[m] for m in METRIC_COLS]
    values  = [overall[m] for m in METRIC_COLS]
    bars    = ax.barh(labels, values, color=METRIC_COLORS, height=0.5, alpha=0.88)

    for bar, val in zip(bars, values):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 1.18)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(f"{title_prefix} — Macro-Averaged Metrics (all classes)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    _save(fig, out_dir / "overall_summary.png", dpi)


# ── Chart 4: per-image IoU heatmap (classes × images) ─────────────────────────
def plot_iou_heatmap(df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    pivot = df.pivot_table(index="class", columns="image", values="iou")
    pivot = pivot.reindex([c for c in CLASS_LABELS if c in pivot.index])

    col_order = pivot.mean(axis=0).sort_values().index
    pivot     = pivot[col_order]

    n_images = len(pivot.columns)
    fig_w    = max(12, n_images * 0.11)
    fig, ax  = plt.subplots(figsize=(fig_w, 3.8))

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([CLASS_LABELS[c] for c in pivot.index])
    ax.set_xticks([])
    ax.set_xlabel(f"Images sorted by mean IoU  (n = {n_images})", labelpad=6)
    ax.set_title(f"{title_prefix} — Per-Image IoU Heatmap")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("IoU", fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    fig.tight_layout()
    _save(fig, out_dir / "iou_heatmap.png", dpi)


# ── Chart 5: precision-recall scatter per class ───────────────────────────────
def plot_precision_recall(df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    classes = [c for c in CLASS_LABELS if c in df["class"].unique()]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for cls in classes:
        sub = df[df["class"] == cls]
        ax.scatter(sub["recall"], sub["precision"],
                   c=CLASS_COLORS[cls], label=CLASS_LABELS[cls],
                   alpha=0.50, s=22, edgecolors="none", zorder=3)

        # class-level mean crosshair marker
        mx, my = sub["recall"].mean(), sub["precision"].mean()
        ax.scatter(mx, my, c=CLASS_COLORS[cls], s=120, marker="*",
                   edgecolors="black", linewidths=0.5, zorder=5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.15, linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"{title_prefix} — Precision vs Recall (per image)\n"
                 r"$\bigstar$ = class mean")
    ax.legend(loc="lower left", framealpha=0.92)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    fig.tight_layout()
    _save(fig, out_dir / "precision_recall.png", dpi)


# ── Chart 6: image-level prediction breakdown (TP / TN / FP / FN) ────────────
def plot_prediction_breakdown(df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int) -> None:
    """
    Categorise every image × class row by whether GT and prediction are present.

    Uses the edge-case conventions in metric_calc.py:
      prec=1 when nothing is predicted (TP+FP=0); rec=1 when GT is empty (TP+FN=0).

    Categories (image-level, not pixel-level):
      TP  GT present,  model fires   → real detection
      FP  GT absent,   model fires   → false alarm
      FN  GT present,  model silent  → missed detection
      TN  GT absent,   model silent  → correct background suppression
    """
    classes = [c for c in CLASS_LABELS if c in df["class"].unique()]

    def categorise(row: pd.Series) -> str:
        prec, rec = row["precision"], row["recall"]
        pred_empty = prec > 0.99   # TP+FP == 0  →  prec = 1.0 by convention
        gt_empty   = rec  > 0.99   # TP+FN == 0  →  rec  = 1.0 by convention
        if gt_empty and pred_empty:
            return "TN"
        if gt_empty and not pred_empty:
            return "FP"
        if not gt_empty and pred_empty:
            return "FN"
        return "TP"

    df2 = df.copy()
    df2["category"] = df2.apply(categorise, axis=1)

    cat_order  = ["TP", "TN", "FN", "FP"]
    cat_colors = {"TP": "#27ae60", "TN": "#bdc3c7", "FN": "#e67e22", "FP": "#e74c3c"}
    cat_labels = {
        "TP": "GT present, detected",
        "TN": "GT absent,  suppressed (correct)",
        "FN": "GT present, missed (false negative)",
        "FP": "GT absent,  fired (false positive)",
    }

    x      = np.arange(len(classes))
    width  = 0.55
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 2.2), 5.5))
    bottoms = np.zeros(len(classes))

    for cat in cat_order:
        fracs = []
        for cls in classes:
            cls_df = df2[df2["class"] == cls]
            n      = len(cls_df)
            fracs.append((cls_df["category"] == cat).sum() / n if n else 0.0)
        fracs = np.array(fracs)
        bars  = ax.bar(x, fracs, width, bottom=bottoms,
                       color=cat_colors[cat], label=cat_labels[cat], alpha=0.88)
        for bar, frac, bot in zip(bars, fracs, bottoms):
            if frac > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + frac / 2,
                        f"{frac:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottoms += fracs

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes])
    ax.set_ylabel("Fraction of Images")
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(f"{title_prefix} — Image-Level Prediction Breakdown\n"
                 "(categorised by GT presence vs model activation)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    _save(fig, out_dir / "prediction_breakdown.png", dpi)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Chart SUIM-Net metrics from CSV.")
    ap.add_argument("--csv", default="reports/suimnet/suim_metrics.csv",
                    help="Path to metrics CSV (default: reports/suimnet/suim_metrics.csv)")
    ap.add_argument("--out", default=None,
                    help="Output directory for PNGs (default: <csv_dir>/figures)")
    ap.add_argument("--title", default=None,
                    help="Dataset label shown in chart titles, e.g. 'SUIM TEST Split (110 images)'")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Output DPI — use 300 for report/print quality (default: 300)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Default: reports/suimnet/figures/<dataset>/ derived from CSV stem
    # e.g. reports/suimnet/suim_metrics.csv → reports/suimnet/figures/suim/
    if args.out:
        out_dir = Path(args.out)
    else:
        dataset_tag = csv_path.stem.replace("_metrics", "")
        out_dir = csv_path.parent / "figures" / dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-derive title from CSV filename if not given
    title_prefix = args.title or f"SUIM-Net — {csv_path.stem.replace('_', ' ').title()}"

    df = load(csv_path)
    n_images  = df["image"].nunique()
    n_classes = df["class"].nunique()
    print(f"Loaded {len(df)} rows  ({n_images} images × {n_classes} classes)  from {csv_path}")
    print(f"Output dir : {out_dir}  |  DPI : {args.dpi}")
    print()

    plot_class_means(df, out_dir, title_prefix, args.dpi)
    plot_iou_boxplots(df, out_dir, title_prefix, args.dpi)
    plot_overall_summary(df, out_dir, title_prefix, args.dpi)
    plot_iou_heatmap(df, out_dir, title_prefix, args.dpi)
    plot_precision_recall(df, out_dir, title_prefix, args.dpi)
    plot_prediction_breakdown(df, out_dir, title_prefix, args.dpi)

    print(f"\nAll charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
