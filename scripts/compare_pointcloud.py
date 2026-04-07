"""
Assemble a side-by-side comparison figure for the report:

    [ Danger Map 3-panel  |  Point Cloud (perspective)  |  Point Cloud (top-down) ]

This script is meant to be run *after* quick_test.py (which generates the
danger map panels and the .ply file) so you can see exactly what the
navigation system "sees" in both 2D and 3D for the same frame.

Usage
─────
    # Use the defaults (quick_test sample):
    python scripts/compare_pointcloud.py

    # Custom inputs:
    python scripts/compare_pointcloud.py \\
        --danger_png  reports/danger_map/quick_test/w_r_147__danger.png \\
        --ply         reports/danger_map/quick_test/sample_cloud.ply \\
        --out         figures/pointcloud/comparison_w_r_147.png

Outputs
    <out>   — single wide PNG combining danger map + both cloud views
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image


def _parse_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (pts [N,3], colors [N,3] float) from an ASCII PLY."""
    pts, colors = [], []
    in_body = False
    with open(ply_path) as f:
        for line in f:
            if line.strip() == "end_header":
                in_body = True
                continue
            if not in_body:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            colors.append([int(parts[3]) / 255.0,
                           int(parts[4]) / 255.0,
                           int(parts[5]) / 255.0])
    pts    = np.array(pts,    dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    return pts, colors


def _subsample(pts: np.ndarray, colors: np.ndarray, n: int = 40000):
    if len(pts) > n:
        idx = np.random.default_rng(42).choice(len(pts), n, replace=False)
        return pts[idx], colors[idx]
    return pts, colors


def make_comparison(danger_png: Path, ply_path: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    danger_img = np.array(Image.open(danger_png).convert("RGB"))
    pts, colors = _parse_ply(ply_path)
    pts, colors = _subsample(pts, colors)
    print(f"PLY: {len(pts)} points  |  danger map: {danger_img.shape[:2]}")

    # Camera frame: X=right, Y=down, Z=forward
    # We plot as:  horiz=X, depth=Z, vert=-Y (up)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    # ── Figure layout ─────────────────────────────────────────────────────────
    bg = "#16161a"
    fig = plt.figure(figsize=(20, 6), facecolor=bg)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05,
                            left=0.02, right=0.98, top=0.88, bottom=0.05)

    # Panel 1: Danger map (just display the PNG)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(danger_img)
    ax0.set_title("Danger Map + Navigation", color="white", fontsize=12, pad=6)
    ax0.axis("off")

    # Panel 2: Perspective view (3D scatter)
    ax1 = fig.add_subplot(gs[1], projection="3d", facecolor=bg)
    ax1.scatter(X, Z, -Y, c=colors, s=0.4, alpha=0.85, edgecolors="none", rasterized=True)
    ax1.set_xlabel("X (right)", color="#aaa", fontsize=8, labelpad=4)
    ax1.set_ylabel("Z (forward)", color="#aaa", fontsize=8, labelpad=4)
    ax1.set_zlabel("Y (up)", color="#aaa", fontsize=8, labelpad=4)
    ax1.tick_params(colors="#666", labelsize=6)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor("#333")
    ax1.yaxis.pane.set_edgecolor("#333")
    ax1.zaxis.pane.set_edgecolor("#333")
    ax1.view_init(elev=20, azim=-55)
    ax1.set_title("Risk Point Cloud — Perspective", color="white", fontsize=12, pad=6)
    ax1.set_facecolor(bg)

    # Panel 3: Top-down view (2D scatter, X vs Z)
    ax2 = fig.add_subplot(gs[2], facecolor=bg)
    ax2.scatter(X, Z, c=colors, s=0.4, alpha=0.85, edgecolors="none", rasterized=True)
    ax2.set_xlabel("X (right)", color="#aaa", fontsize=9)
    ax2.set_ylabel("Z (forward / depth)", color="#aaa", fontsize=9)
    ax2.tick_params(colors="#666", labelsize=7)
    ax2.set_aspect("equal")
    ax2.spines[:].set_color("#333")
    ax2.set_title("Risk Point Cloud — Top-Down", color="white", fontsize=12, pad=6)

    # Shared legend: green=safe, red=danger
    for ax in (ax1, ax2):
        pass  # color is self-evident from the scatter

    fig.suptitle(
        f"Underwater Danger Map  ·  {danger_png.stem}",
        color="white", fontsize=13, y=0.97,
    )

    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"Saved → {out}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(
        description="Combine a danger map panel with 3D point cloud views into one comparison figure."
    )
    ap.add_argument(
        "--danger_png",
        default=str(repo / "reports/danger_map/quick_test/w_r_147__danger.png"),
        help="Path to the 3-panel danger map PNG from quick_test.py",
    )
    ap.add_argument(
        "--ply",
        default=str(repo / "reports/danger_map/quick_test/sample_cloud.ply"),
        help="Path to the .ply risk point cloud file from quick_test.py",
    )
    ap.add_argument(
        "--out",
        default=str(repo / "figures/pointcloud/comparison.png"),
        help="Output path for the comparison figure",
    )
    args = ap.parse_args()

    make_comparison(Path(args.danger_png), Path(args.ply), Path(args.out))
    print("Done.")


if __name__ == "__main__":
    main()
