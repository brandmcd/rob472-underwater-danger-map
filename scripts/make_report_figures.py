"""
Generate all report figures from available CSV results.

Run from the repo root:
    python scripts/make_report_figures.py

Outputs (all in figures/charts/):
    turbidity_miou.png          — mIoU vs turbidity level, per class + mean
    turbidity_heatmap.png       — IoU heatmap (class × turbidity level)
    latency_breakdown.png       — Per-component latency bar chart (CPU + GPU projections)
    spade_accuracy.png          — SPADE depth accuracy (a1, abs_rel, RMSE) by range × dataset
    suimnet_iou.png             — SUIM-Net per-class IoU on SUIM test set
    system_summary.png          — One-page results summary card
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parents[1]
OUT    = REPO / "figures" / "charts"
TURB   = REPO / "reports" / "turbidity" / "turbidity_summary.csv"
LAT    = REPO / "reports" / "latency_local.csv"
SPADE_FLSEA   = REPO / "reports" / "spade" / "flsea_metrics.csv"
SPADE_SEATHRU = REPO / "reports" / "spade" / "seathru_metrics.csv"
SUIM_METRICS  = REPO / "reports" / "suimnet" / "suim_metrics.csv"

OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG   = "#16161a"
PANEL_BG  = "#1e1e24"
GRID_COL  = "#2e2e38"
TEXT_COL  = "#e0e0e0"
MUT_TEXT  = "#888"
ACCENT    = "#4ea8de"
PALETTE   = ["#4ea8de", "#f6b84b", "#6fcf97", "#e05c5c", "#b47fe5"]

CLASS_ORDER  = ["RO", "FV", "HD", "RI", "WR"]
CLASS_LABELS = {
    "RO": "Robot/Inst.",
    "FV": "Fish/Vertebrate",
    "HD": "Human Diver",
    "RI": "Reef/Invert.",
    "WR": "Wreck/Ruin",
}
CLASS_RISK = {"HD": 1.0, "WR": 0.9, "RI": 0.8, "FV": 0.5, "RO": 0.2}

def _style(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(DARK_BG)

def _ax_style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    if title:   ax.set_title(title, fontsize=11, pad=8, color=TEXT_COL, fontweight="bold")
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)


# ── 1. Turbidity mIoU line chart ──────────────────────────────────────────────
def plot_turbidity_miou() -> None:
    df = pd.read_csv(TURB)
    levels = sorted(df["level"].unique())
    classes = CLASS_ORDER

    fig, ax = plt.subplots(figsize=(8, 5))
    _style(fig)

    # Per-class lines
    for i, cls in enumerate(classes):
        sub = df[df["class"] == cls].sort_values("level")
        ax.plot(sub["level"], sub["iou"], marker="o", color=PALETTE[i],
                linewidth=2, markersize=6, label=CLASS_LABELS[cls])
        ax.fill_between(sub["level"], sub["iou"], alpha=0.07, color=PALETTE[i])

    # Mean IoU line
    mean_iou = df.groupby("level")["iou"].mean().reset_index()
    ax.plot(mean_iou["level"], mean_iou["iou"], marker="s", color="white",
            linewidth=2.5, markersize=7, linestyle="--", label="Mean IoU", zorder=10)

    _ax_style(ax, "SUIM-Net: Segmentation Accuracy vs Turbidity",
              "Turbidity Level", "IoU")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(levels)
    ax.set_xticklabels([str(l) for l in levels])

    # Danger zone shading
    ax.axvspan(0.5, 1.0, color="#e05c5c", alpha=0.06)
    ax.text(0.73, 0.03, "High turbidity\nzone", color="#e05c5c",
            fontsize=7.5, ha="center", transform=ax.get_xaxis_transform())

    legend = ax.legend(loc="upper right", fontsize=8.5, framealpha=0.3,
                       facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    fig.tight_layout(pad=1.4)
    out = OUT / "turbidity_miou.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── 2. Turbidity heatmap ──────────────────────────────────────────────────────
def plot_turbidity_heatmap() -> None:
    df = pd.read_csv(TURB)
    levels  = sorted(df["level"].unique())
    classes = CLASS_ORDER

    matrix = np.zeros((len(classes), len(levels)))
    for ci, cls in enumerate(classes):
        for li, lv in enumerate(levels):
            row = df[(df["class"] == cls) & (df["level"] == lv)]
            matrix[ci, li] = row["iou"].values[0] if len(row) else np.nan

    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_facecolor(PANEL_BG)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(l) for l in levels], color=TEXT_COL, fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([CLASS_LABELS[c] for c in classes], color=TEXT_COL, fontsize=9)
    ax.set_xlabel("Turbidity Level", fontsize=9, color=TEXT_COL)
    ax.set_title("IoU Heatmap: Class × Turbidity Level", fontsize=11,
                 color=TEXT_COL, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT_COL)

    # Annotate cells
    for ci in range(len(classes)):
        for li in range(len(levels)):
            val = matrix[ci, li]
            text_col = "black" if val > 0.55 else "white"
            ax.text(li, ci, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color=text_col, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_COL, labelsize=8)
    cbar.set_label("IoU", color=TEXT_COL, fontsize=9)
    cbar.outline.set_edgecolor(GRID_COL)

    fig.tight_layout(pad=1.4)
    out = OUT / "turbidity_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── 3. Latency breakdown ──────────────────────────────────────────────────────
def plot_latency() -> None:
    df = pd.read_csv(LAT)

    # CPU measured values (mean over 8 frames)
    cpu_suim   = df["suimnet_ms"].mean()
    cpu_spade  = df["spade_ms"].mean()
    cpu_fusion = df["fusion_ms"].mean()

    # GPU projected values — SUIM-Net ~15ms, SPADE ~55ms, fusion stays ~5ms
    # (based on typical A40 speedup for these model sizes)
    gpu_suim   = 15.0
    gpu_spade  = 55.0
    gpu_fusion = 5.0

    labels = ["SUIM-Net\n(Segmentation)", "SPADE\n(Depth Est.)", "Danger\nFusion"]
    cpu_vals = [cpu_suim, cpu_spade, cpu_fusion]
    gpu_vals = [gpu_suim, gpu_spade, gpu_fusion]

    x = np.arange(len(labels))
    w = 0.32

    fig, (ax_bar, ax_fps) = plt.subplots(1, 2, figsize=(10, 5),
                                          gridspec_kw={"width_ratios": [3, 1]})
    _style(fig)

    # Grouped bar chart
    bars_cpu = ax_bar.bar(x - w/2, cpu_vals, w, label="CPU (local WSL2)",
                          color=PALETTE[0], alpha=0.85)
    bars_gpu = ax_bar.bar(x + w/2, gpu_vals, w, label="GPU est. (ARC A40)",
                          color=PALETTE[1], alpha=0.85)

    # Value labels on bars
    for bar in bars_cpu:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 20, f"{h:.0f}ms",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COL)
    for bar in bars_gpu:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 20, f"{h:.0f}ms",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COL)

    _ax_style(ax_bar, "Per-Component Latency", "Pipeline Stage", "Time (ms)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, color=TEXT_COL, fontsize=9)
    ax_bar.set_ylim(0, max(cpu_vals) * 1.28)
    legend = ax_bar.legend(fontsize=8.5, framealpha=0.3, facecolor=DARK_BG,
                           edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # FPS summary panel
    cpu_fps = 1000 / sum(cpu_vals)
    gpu_fps = 1000 / sum(gpu_vals)

    ax_fps.set_facecolor(PANEL_BG)
    for spine in ax_fps.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax_fps.set_xticks([])
    ax_fps.set_yticks([])

    ax_fps.text(0.5, 0.85, "Effective FPS", ha="center", va="center",
                transform=ax_fps.transAxes, fontsize=10, color=MUT_TEXT)
    ax_fps.text(0.5, 0.62, f"{cpu_fps:.1f}", ha="center", va="center",
                transform=ax_fps.transAxes, fontsize=32, color=PALETTE[0],
                fontweight="bold")
    ax_fps.text(0.5, 0.48, "CPU (WSL2)", ha="center", va="center",
                transform=ax_fps.transAxes, fontsize=8.5, color=PALETTE[0])
    ax_fps.text(0.5, 0.32, f"{gpu_fps:.1f}", ha="center", va="center",
                transform=ax_fps.transAxes, fontsize=32, color=PALETTE[1],
                fontweight="bold")
    ax_fps.text(0.5, 0.18, "GPU est. (A40)", ha="center", va="center",
                transform=ax_fps.transAxes, fontsize=8.5, color=PALETTE[1])
    ax_fps.set_title("FPS", fontsize=11, color=TEXT_COL, fontweight="bold", pad=8)

    fig.suptitle("End-to-End Latency: SUIM-Net + SPADE + Danger Map Fusion",
                 fontsize=12, color=TEXT_COL, y=1.01)
    fig.tight_layout(pad=1.4)
    out = OUT / "latency_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── 4. SPADE depth accuracy ───────────────────────────────────────────────────
def plot_spade_accuracy() -> None:
    fl = pd.read_csv(SPADE_FLSEA)
    st = pd.read_csv(SPADE_SEATHRU)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    _style(fig)

    metrics = [
        ("a1",      r"$\delta < 1.25$  (↑ better)",   True),
        ("abs_rel", "Abs. Rel. Error   (↓ better)",    False),
        ("rmse",    "RMSE (m)   (↓ better)",            False),
    ]

    ranges = [2.0, 5.0, 10.0]
    x = np.arange(len(ranges))
    w = 0.32

    for ax, (metric, title, higher_better) in zip(axes, metrics):
        fl_vals = [fl[fl["range_m"] == r][metric].values[0] for r in ranges]
        st_vals = [st[st["range_m"] == r][metric].values[0] for r in ranges]

        bars_fl = ax.bar(x - w/2, fl_vals, w, label="FLSea", color=PALETTE[0], alpha=0.85)
        bars_st = ax.bar(x + w/2, st_vals, w, label="SeaThru", color=PALETTE[2], alpha=0.85)

        for bar, val in zip(list(bars_fl) + list(bars_st), fl_vals + st_vals):
            h = bar.get_height()
            fmt = f"{val:.2f}" if val < 2 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.02, fmt,
                    ha="center", va="bottom", fontsize=7.5, color=TEXT_COL)

        _ax_style(ax, title, "Depth Range (m)")
        ax.set_xticks(x)
        ax.set_xticklabels(["0–2m", "0–5m", "0–10m"], color=TEXT_COL, fontsize=9)
        ax.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG,
                  edgecolor=GRID_COL, labelcolor=TEXT_COL)
        ax.set_ylim(0, max(max(fl_vals), max(st_vals)) * 1.35)

    fig.suptitle("SPADE Depth Estimation Accuracy  ·  FLSea vs SeaThru",
                 fontsize=12, color=TEXT_COL, y=1.01)
    fig.tight_layout(pad=1.4)
    out = OUT / "spade_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── 5. SUIM-Net per-class IoU ──────────────────────────────────────────────────
def plot_suimnet_iou() -> None:
    df = pd.read_csv(SUIM_METRICS)

    per_class = df.groupby("class")["iou"].agg(["mean", "median", "std"]).reindex(CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5))
    _style(fig)

    x = np.arange(len(CLASS_ORDER))
    w = 0.5
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(CLASS_ORDER))]

    bars = ax.bar(x, per_class["mean"], w, color=colors, alpha=0.85)

    # Error bars (std)
    ax.errorbar(x, per_class["mean"], yerr=per_class["std"],
                fmt="none", color="white", capsize=5, linewidth=1.5, capthick=1.5)

    # Median dots
    ax.scatter(x, per_class["median"], color="white", zorder=5,
               s=40, marker="D", label="Median")

    # Risk weight annotation
    for i, cls in enumerate(CLASS_ORDER):
        risk = CLASS_RISK[cls]
        ax.text(i, -0.09, f"risk w={risk}", ha="center", fontsize=7.5,
                color=MUT_TEXT, transform=ax.get_xaxis_transform())

    # Value labels
    for bar, (cls, row) in zip(bars, per_class.iterrows()):
        h = row["mean"]
        ax.text(bar.get_x() + bar.get_width()/2, h + row["std"] + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COL)

    _ax_style(ax, "SUIM-Net Segmentation Accuracy  (SUIM Test Set)",
              "Class", "IoU (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], color=TEXT_COL, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="#e05c5c", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(len(x) - 0.5, 0.52, "IoU = 0.5 threshold", color="#e05c5c",
            fontsize=7.5, va="bottom", ha="right")
    ax.legend(fontsize=8.5, framealpha=0.3, facecolor=DARK_BG,
              edgecolor=GRID_COL, labelcolor=TEXT_COL)

    fig.tight_layout(pad=1.4)
    out = OUT / "suimnet_iou.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── 6. System summary card ────────────────────────────────────────────────────
def plot_system_summary() -> None:
    df_turb = pd.read_csv(TURB)
    df_lat  = pd.read_csv(LAT)
    fl      = pd.read_csv(SPADE_FLSEA)
    df_suim = pd.read_csv(SUIM_METRICS)

    # Key numbers
    mean_iou_0   = df_turb[(df_turb["level"] == 0.0)]["iou"].mean()
    mean_iou_1   = df_turb[(df_turb["level"] == 1.0)]["iou"].mean()
    turb_drop    = (mean_iou_0 - mean_iou_1) / mean_iou_0 * 100
    gpu_fps      = 1000 / (15 + 55 + 5)
    spade_a1     = fl[fl["range_m"] == 5.0]["a1"].values[0]
    suim_miou    = df_suim.groupby("class")["iou"].mean().mean()

    fig = plt.figure(figsize=(12, 6))
    _style(fig)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.35,
                           left=0.04, right=0.96, top=0.88, bottom=0.08)

    def _metric_card(ax, value: str, label: str, sub: str, color: str) -> None:
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.70, value, ha="center", va="center",
                transform=ax.transAxes, fontsize=26, color=color, fontweight="bold")
        ax.text(0.5, 0.35, label, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=TEXT_COL)
        ax.text(0.5, 0.12, sub, ha="center", va="center",
                transform=ax.transAxes, fontsize=7.5, color=MUT_TEXT)

    ax0 = fig.add_subplot(gs[0, 0])
    _metric_card(ax0, f"{suim_miou:.2f}", "SUIM-Net mIoU",
                 "SUIM test set (5 classes)", PALETTE[0])

    ax1 = fig.add_subplot(gs[0, 1])
    _metric_card(ax1, f"{spade_a1:.3f}", "SPADE δ < 1.25",
                 "FLSea dataset, 0–5m range", PALETTE[2])

    ax2 = fig.add_subplot(gs[0, 2])
    _metric_card(ax2, f"{mean_iou_0:.2f}", "Seg. IoU (clear)",
                 "Turbidity level 0.0", PALETTE[1])

    ax3 = fig.add_subplot(gs[0, 3])
    _metric_card(ax3, f"{mean_iou_1:.2f}", "Seg. IoU (max turb.)",
                 f"Level 1.0 — {turb_drop:.0f}% degradation", "#e05c5c")

    # Turbidity mini chart (bottom left, spans 2)
    ax_turb = fig.add_subplot(gs[1, :2])
    levels = sorted(df_turb["level"].unique())
    mean_by_level = df_turb.groupby("level")["iou"].mean()
    ax_turb.fill_between(levels, mean_by_level.values, alpha=0.25, color=PALETTE[0])
    ax_turb.plot(levels, mean_by_level.values, color=PALETTE[0], marker="o",
                 linewidth=2, markersize=6)
    _ax_style(ax_turb, "Mean IoU vs Turbidity", "Turbidity Level", "mIoU")
    ax_turb.set_ylim(0, 1.0)
    ax_turb.set_xticks(levels)

    # Latency mini chart (bottom right, spans 2)
    ax_lat = fig.add_subplot(gs[1, 2:])
    components = ["SUIM-Net", "SPADE", "Fusion"]
    gpu_vals   = [15, 55, 5]
    cpu_vals   = [df_lat["suimnet_ms"].mean(), df_lat["spade_ms"].mean(), df_lat["fusion_ms"].mean()]
    x = np.arange(3)
    ax_lat.barh(x + 0.2, cpu_vals, 0.35, color=PALETTE[0], alpha=0.8, label="CPU")
    ax_lat.barh(x - 0.2, gpu_vals, 0.35, color=PALETTE[1], alpha=0.8, label="GPU est.")
    _ax_style(ax_lat, "Latency by Stage", "Time (ms)", "")
    ax_lat.set_yticks(x)
    ax_lat.set_yticklabels(components, color=TEXT_COL, fontsize=9)
    ax_lat.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG,
                  edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax_lat.text(max(cpu_vals) * 0.92, 2.45, f"{gpu_fps:.0f} FPS (GPU)",
                color=PALETTE[1], fontsize=8.5, ha="right")

    fig.suptitle("ROB 472 — Underwater Danger Map  ·  Results Summary",
                 fontsize=14, color=TEXT_COL, fontweight="bold", y=0.97)
    out = OUT / "system_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating report figures → {OUT}/\n")
    plot_turbidity_miou()
    plot_turbidity_heatmap()
    plot_latency()
    plot_spade_accuracy()
    plot_suimnet_iou()
    plot_system_summary()
    print("\nAll figures generated.")
