"""Shared utilities for SPADE dataset converters.

Extracted from convert_seathru.py and convert_kskin.py to avoid duplication.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

# ── Sparse-feature coordinate space ──────────────────────────────────────────
# Must match SPADE's sparse_feature_height / sparse_feature_width config values.
SPARSE_H: int         = 240
SPARSE_W: int         = 320
MAX_CORNERS: int      = 500
CORNER_QUALITY: float = 0.01
CORNER_MIN_DIST: int  = 5

# Depth file extensions handled by load_depth()
DEPTH_EXTS = {".npy", ".tif", ".tiff", ".png"}


def load_depth(depth_path: Path, squeeze: bool = False) -> np.ndarray | None:
    """Load a depth map and return a float32 array with values in metres.

    Zero and NaN indicate invalid/missing data.
    Handles .npy, .tif/.tiff (float32), and 16-bit PNG (auto-converted from mm).

    Args:
        depth_path: Path to the depth file.
        squeeze:    If True, squeeze extra dimensions and validate shape is 2-D
                    (needed for stereo archives that store (H, W, 1) arrays).
    """
    ext = depth_path.suffix.lower()
    try:
        if ext == ".npy":
            arr = np.load(str(depth_path)).astype(np.float32)
        elif ext in (".tif", ".tiff"):
            arr = tifffile.imread(str(depth_path)).astype(np.float32)
        elif ext == ".png":
            img = Image.open(depth_path)
            arr = np.array(img, dtype=np.float32)
            # 16-bit PNG heuristic: values >> 200 suggest mm → convert to m
            if arr.max() > 200:
                arr /= 1000.0
        else:
            print(f"  WARNING: unsupported depth format {depth_path}")
            return None
    except Exception as exc:
        print(f"  WARNING: cannot load depth {depth_path}: {exc}")
        return None

    if squeeze:
        arr = arr.squeeze()
        if arr.ndim != 2:
            print(f"  WARNING: unexpected depth shape {arr.shape} for {depth_path}")
            return None
    return arr


def generate_sparse_csv(
    img_bgr: np.ndarray,
    depth: np.ndarray,
    out_csv: Path,
    resize_depth: bool = False,
) -> int:
    """Detect Shi-Tomasi corners in an image, sample dense depth → sparse CSV.

    CSV coordinates are in SPARSE_H × SPARSE_W (240 × 320) space, matching
    SPADE's ``sparse_feature_height`` / ``sparse_feature_width`` config values.

    Args:
        img_bgr:      BGR image as a numpy array (from cv2 or converted from PIL).
        depth:        Dense float32 depth map in metres.
        out_csv:      Output path for the sparse CSV file.
        resize_depth: If True, resize the depth map to match the image dimensions
                      before sampling (needed when stereo GT has different resolution).

    Returns:
        Number of sparse points written (0 if no corners detected or image is None).
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUALITY,
        minDistance=CORNER_MIN_DIST,
    )
    if corners is None:
        return 0

    if resize_depth:
        depth_h, depth_w = depth.shape[:2]
        if (depth_h, depth_w) != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    rows_out: list[tuple[float, float, float]] = []
    for c in corners:
        x, y     = c[0]
        row_orig = int(np.clip(round(y), 0, H - 1))
        col_orig = int(np.clip(round(x), 0, W - 1))
        d = float(depth[row_orig, col_orig])
        if d <= 0 or not np.isfinite(d):
            continue
        # Scale to SPARSE coordinate space
        row_s = row_orig * SPARSE_H / H
        col_s = col_orig * SPARSE_W / W
        rows_out.append((row_s, col_s, d))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "column", "depth"])
        writer.writerows(rows_out)
    return len(rows_out)
