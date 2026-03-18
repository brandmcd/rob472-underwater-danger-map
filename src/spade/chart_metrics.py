"""Chart SPADE benchmark metrics from a CSV produced by run_eval.py.

Usage:
    # Single dataset
    python -m src.spade.chart_metrics --csv reports/spade/seathru_metrics.csv

    # Multiple datasets → individual charts + cross-dataset comparison
    python -m src.spade.chart_metrics \\
        --csvs reports/spade/flsea_demo_metrics.csv \\
               reports/spade/seathru_metrics.csv \\
               reports/spade/kskin_metrics.csv \\
        --title "SPADE Benchmark"

Charts saved to reports/spade/figures/<dataset>/ at 300 DPI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

# Primary error metrics (lower = better)
ERROR_METRICS = ["mae", "rmse", "abs_rel", "silog"]
# Threshold-based accuracy (higher = better)
ACCURACY_METRICS = ["a1", "a2", "a3"]

METRIC_LABELS: dict[str, str] = {
    "mae":     "MAE (m)",
    "rmse":    "RMSE (m)",
    "abs_rel": "AbsRel",
    "silog":   "SILog",
    "a1":      "δ < 1.25",
    "a2":      "δ < 1.25²",
    "a3":      "δ < 1.25³",
    "mdae":    "MdAE (m)",
    "p90":     "P90 err (m)",
    "iRMSE":   "iRMSE",
    "iMAE":    "iMAE",
}

# One colour per evaluation range (up to 5 ranges)
RANGE_COLORS = ["#2980b9", "#27ae60", "#e74c3c", "#8e44ad", "#e67e22"]


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {path}")


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in list(METRIC_LABELS.keys()) + ["iAbsRel", "sq_rel", "log_10", "rmse_log"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Chart 1: error metrics per evaluation range ───────────────────────────────
def plot_errors_by_range(
    df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int
) -> None:
    ranges  = sorted(df["range_m"].unique())
    metrics = [m for m in ERROR_METRICS if m in df.columns]
    if not metrics:
        return

    x     = np.arange(len(metrics))
    width = 0.8 / max(len(ranges), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, rng in enumerate(ranges):
        sub  = df[df["range_m"] == rng]
        vals = [sub[m].mean() for m in metrics]
        bars = ax.bar(
            x + i * width, vals, width,
            label=f"≤ {rng:.0f} m",
            color=RANGE_COLORS[i % len(RANGE_COLORS)],
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x + width * (len(ranges) - 1) / 2)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax.set_ylabel("Error  (lower = better)")
    ax.set_title(f"{title_prefix} — Error Metrics by Evaluation Range")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / "errors_by_range.png", dpi)


# ── Chart 2: δ-accuracy per evaluation range ─────────────────────────────────
def plot_accuracy_by_range(
    df: pd.DataFrame, out_dir: Path, title_prefix: str, dpi: int
) -> None:
    acc_cols = [c for c in ACCURACY_METRICS if c in df.columns]
    if not acc_cols:
        return

    ranges = sorted(df["range_m"].unique())
    x      = np.arange(len(acc_cols))
    width  = 0.8 / max(len(ranges), 1)
    labels = {"a1": "δ < 1.25", "a2": "δ < 1.25²", "a3": "δ < 1.25³"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, rng in enumerate(ranges):
        sub  = df[df["range_m"] == rng]
        vals = [sub[c].mean() for c in acc_cols]
        bars = ax.bar(
            x + i * width, vals, width,
            label=f"≤ {rng:.0f} m",
            color=RANGE_COLORS[i % len(RANGE_COLORS)],
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x + width * (len(ranges) - 1) / 2)
    ax.set_xticklabels([labels.get(c, c) for c in acc_cols])
    ax.set_ylabel("Accuracy  (higher = better)")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(f"{title_prefix} — δ-Accuracy by Evaluation Range")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / "accuracy_by_range.png", dpi)


# ── Chart 3: multi-dataset comparison ────────────────────────────────────────
def plot_dataset_comparison(
    dfs: list[tuple[str, pd.DataFrame]],
    out_dir: Path,
    title_prefix: str,
    dpi: int,
    rng_filter: float = 10.0,
) -> None:
    """Bar chart comparing primary error metrics across datasets at rng_filter m."""
    datasets = [name for name, _ in dfs]
    metrics  = [m for m in ERROR_METRICS
                if all(m in df.columns for _, df in dfs)]
    if not metrics:
        return

    x      = np.arange(len(datasets))
    width  = 0.8 / max(len(metrics), 1)
    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(max(9, len(datasets) * 2.5), 5.5))
    for i, metric in enumerate(metrics):
        vals = []
        for _, df in dfs:
            row_sel = df[df["range_m"] == rng_filter]
            if row_sel.empty:
                row_sel = df
            vals.append(row_sel[metric].mean() if metric in row_sel.columns else float("nan"))
        bars = ax.bar(
            x + i * width, vals, width,
            label=METRIC_LABELS.get(metric, metric),
            color=colors[i % len(colors)],
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(f"Error at ≤ {rng_filter:.0f} m  (lower = better)")
    ax.set_title(f"{title_prefix} — Dataset Comparison (≤ {rng_filter:.0f} m)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / "dataset_comparison.png", dpi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Chart SPADE metrics from CSV.")
    ap.add_argument(
        "--csv", default=None,
        help="Single metrics CSV (default: reports/spade/seathru_metrics.csv)",
    )
    ap.add_argument(
        "--csvs", nargs="+", default=None,
        help="Multiple CSVs — generates per-dataset charts + comparison chart",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output directory (default: reports/spade/figures/<dataset>/)",
    )
    ap.add_argument("--title", default=None,
                    help="Title prefix for all charts")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--compare_range", type=float, default=10.0,
        help="Evaluation range (m) used for the multi-dataset comparison chart",
    )
    args = ap.parse_args()

    if args.csvs:
        csv_paths = [Path(c) for c in args.csvs]
    elif args.csv:
        csv_paths = [Path(args.csv)]
    else:
        csv_paths = [Path("reports/spade/seathru_metrics.csv")]

    loaded: list[tuple[str, pd.DataFrame]] = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            continue

        dataset_tag = csv_path.stem.replace("_metrics", "")
        out_dir = (
            Path(args.out) if args.out
            else csv_path.parent / "figures" / dataset_tag
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        title = args.title or f"SPADE — {dataset_tag.replace('_', ' ').title()}"
        df    = load(csv_path)
        n_rows = len(df)
        print(f"Loaded {n_rows} rows from {csv_path}")
        print(f"Output dir : {out_dir}  |  DPI : {args.dpi}")

        plot_errors_by_range(df, out_dir, title, args.dpi)
        plot_accuracy_by_range(df, out_dir, title, args.dpi)
        loaded.append((dataset_tag, df))

    # Multi-dataset comparison
    if len(loaded) > 1:
        compare_dir = csv_paths[0].parent / "figures" / "comparison"
        compare_dir.mkdir(parents=True, exist_ok=True)
        compare_title = args.title or "SPADE — Multi-Dataset Comparison"
        plot_dataset_comparison(
            loaded, compare_dir, compare_title, args.dpi, args.compare_range
        )
        print(f"Comparison chart → {compare_dir}/dataset_comparison.png")

    print("\nAll charts saved.")


if __name__ == "__main__":
    main()
