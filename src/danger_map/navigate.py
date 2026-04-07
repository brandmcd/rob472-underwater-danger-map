"""
Navigation command from a danger map.

Takes a per-pixel risk map (and optionally metric depth) and returns a
plain-English direction command for the AUV plus a per-sector risk breakdown.

Core function
─────────────
    result = nav_command(risk_map)
    result = nav_command(risk_map, depth_m, fx=400, fy=400, cx=320, cy=240,
                         ply_path=Path("risk_cloud.ply"))

Outputs
    result.command      str   — "PROCEED" | "GO LEFT" | "GO RIGHT" |
                                "ASCEND"  | "DESCEND" | "ASCEND LEFT" |
                                "ASCEND RIGHT" | "DESCEND LEFT" |
                                "DESCEND RIGHT" | "STOP"
    result.sector_risks dict  — mean risk for each of the 9 spatial sectors
    result.safest       str   — name of the lowest-risk sector
    result.risk_level   str   — "CLEAR" | "CAUTION" | "DANGER"
    result.overall_risk float — mean risk across the full frame

Sector layout (forward-facing AUV camera, 3x3 grid)
────────────────────────────────────────────────────
    ┌──────────┬──────────┬──────────┐
    │  top-l   │  top-c   │  top-r   │
    │ (0,0)    │ (0,1)    │ (0,2)    │
    ├──────────┼──────────┼──────────┤
    │  mid-l   │  center  │  mid-r   │
    │ (1,0)    │ (1,1)    │ (1,2)    │
    ├──────────┼──────────┼──────────┤
    │  bot-l   │  bot-c   │  bot-r   │
    │ (2,0)    │ (2,1)    │ (2,2)    │
    └──────────┴──────────┴──────────┘

  The AUV steers *toward* the safest (lowest-risk) sector:

    Safest = top-l   → ASCEND LEFT      Safest = top-c   → ASCEND
    Safest = top-r   → ASCEND RIGHT     Safest = mid-l   → GO LEFT
    Safest = center  → PROCEED          Safest = mid-r   → GO RIGHT
    Safest = bot-l   → DESCEND LEFT     Safest = bot-c   → DESCEND
    Safest = bot-r   → DESCEND RIGHT
  ALL sectors dangerous → STOP

3-D point cloud (optional)
──────────────────────────
When `depth_m` and camera intrinsics (fx, fy, cx, cy) are supplied, a PLY
point cloud is written to `ply_path` with each point coloured green (safe)
→ red (dangerous) based on risk.  No external library needed — the PLY is
written directly.

Usage example
─────────────
    from src.danger_map import danger_map
    from src.danger_map.navigate import nav_command, draw_nav_overlay

    risk_map, overlay = danger_map(rgb, seg_logits, depth_m)
    result  = nav_command(risk_map, depth_m, fx=400, fy=400)
    overlay = draw_nav_overlay(overlay, result)

    print(result.command)   # → "ASCEND"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Tuning ────────────────────────────────────────────────────────────────────
# Risk level below which a sector is considered "clear".
CLEAR_THRESHOLD: float = 0.20

# Risk level above which the overall scene is "DANGER" and we recommend STOP
# even if one sector looks safer.
DANGER_THRESHOLD: float = 0.55

# Minimum depth (m) for a pixel to be included in the point cloud.
MIN_DEPTH_M: float = 0.1


# ── Data class ────────────────────────────────────────────────────────────────
@dataclass
class NavResult:
    command:      str                      # AUV command string
    sector_risks: dict[str, float]         # mean risk per sector
    safest:       str                      # lowest-risk sector name
    overall_risk: float                    # mean risk over whole frame
    risk_level:   str                      # "CLEAR" | "CAUTION" | "DANGER"
    ply_path:     Optional[Path] = field(default=None, repr=False)


# ── Sector helpers ────────────────────────────────────────────────────────────

# The 9 sector names, in row-major order matching the 3x3 grid.
SECTOR_NAMES = [
    "top-l",  "top-c",  "top-r",
    "mid-l",  "center", "mid-r",
    "bot-l",  "bot-c",  "bot-r",
]

# sector name → (grid_row, grid_col)
SECTOR_GRID: dict[str, tuple[int, int]] = {
    "top-l": (0, 0), "top-c": (0, 1), "top-r": (0, 2),
    "mid-l": (1, 0), "center": (1, 1), "mid-r": (1, 2),
    "bot-l": (2, 0), "bot-c": (2, 1), "bot-r": (2, 2),
}


def _sector_masks(H: int, W: int) -> dict[str, np.ndarray]:
    """Return boolean masks for each of the 9 spatial sectors (3x3 grid)."""
    r1, r2 = H // 3, 2 * H // 3
    c1, c2 = W // 3, 2 * W // 3

    masks: dict[str, np.ndarray] = {}

    # Top row
    m = np.zeros((H, W), dtype=bool); m[:r1, :c1] = True
    masks["top-l"] = m
    m = np.zeros((H, W), dtype=bool); m[:r1, c1:c2] = True
    masks["top-c"] = m
    m = np.zeros((H, W), dtype=bool); m[:r1, c2:] = True
    masks["top-r"] = m

    # Middle row
    m = np.zeros((H, W), dtype=bool); m[r1:r2, :c1] = True
    masks["mid-l"] = m
    m = np.zeros((H, W), dtype=bool); m[r1:r2, c1:c2] = True
    masks["center"] = m
    m = np.zeros((H, W), dtype=bool); m[r1:r2, c2:] = True
    masks["mid-r"] = m

    # Bottom row
    m = np.zeros((H, W), dtype=bool); m[r2:, :c1] = True
    masks["bot-l"] = m
    m = np.zeros((H, W), dtype=bool); m[r2:, c1:c2] = True
    masks["bot-c"] = m
    m = np.zeros((H, W), dtype=bool); m[r2:, c2:] = True
    masks["bot-r"] = m

    return masks


# "Safest sector" → recommended direction for the AUV to move toward.
_SECTOR_TO_COMMAND: dict[str, str] = {
    "top-l":  "ASCEND LEFT",
    "top-c":  "ASCEND",
    "top-r":  "ASCEND RIGHT",
    "mid-l":  "GO LEFT",
    "center": "PROCEED",
    "mid-r":  "GO RIGHT",
    "bot-l":  "DESCEND LEFT",
    "bot-c":  "DESCEND",
    "bot-r":  "DESCEND RIGHT",
}

# Arrow glyphs for the HUD label
_COMMAND_ARROWS: dict[str, str] = {
    "ASCEND LEFT":   "^\\ ",
    "ASCEND":        "^^",
    "ASCEND RIGHT":  " /^",
    "GO LEFT":       "<<",
    "PROCEED":       ">>",
    "GO RIGHT":      ">>",
    "DESCEND LEFT":  "v/ ",
    "DESCEND":       "vv",
    "DESCEND RIGHT": " \\v",
    "STOP":          "!!",
}


# ── Public API ────────────────────────────────────────────────────────────────

def nav_command(
    risk_map: np.ndarray,
    depth_m: Optional[np.ndarray] = None,
    *,
    fx: float = 400.0,
    fy: float = 400.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    ply_path: Optional[Path] = None,
) -> NavResult:
    """
    Compute a navigation command from a per-pixel risk map.

    Args:
        risk_map:  (H, W) float32 in [0, 1] — output of danger_map().
        depth_m:   (H, W) float32 — metric depth in metres (optional).
                   Required if ply_path is set.
        fx, fy:    Camera focal lengths in pixels. FLSea-VI: ~400.
                   Only used for the PLY point cloud.
        cx, cy:    Camera principal point. Defaults to image centre.
        ply_path:  If given, writes a coloured 3-D risk point cloud to this path.

    Returns:
        NavResult — see module docstring.
    """
    H, W = risk_map.shape[:2]
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    # ── Per-sector mean risk ─────────────────────────────────────────────────
    masks = _sector_masks(H, W)
    sector_risks: dict[str, float] = {}
    for name, mask in masks.items():
        vals = risk_map[mask]
        sector_risks[name] = float(vals.mean()) if vals.size else 0.0

    overall_risk = float(risk_map.mean())

    # ── Risk level ───────────────────────────────────────────────────────────
    if overall_risk < CLEAR_THRESHOLD:
        risk_level = "CLEAR"
    elif overall_risk < DANGER_THRESHOLD:
        risk_level = "CAUTION"
    else:
        risk_level = "DANGER"

    # ── Navigation command ───────────────────────────────────────────────────
    safest = min(sector_risks, key=lambda s: sector_risks[s])

    if risk_level == "DANGER" and sector_risks[safest] > CLEAR_THRESHOLD:
        # Even the best path is risky — halt
        command = "STOP"
    else:
        command = _SECTOR_TO_COMMAND[safest]

    # ── Optional 3-D point cloud ─────────────────────────────────────────────
    saved_ply: Optional[Path] = None
    if ply_path is not None and depth_m is not None:
        saved_ply = _write_risk_ply(risk_map, depth_m, fx, fy, cx, cy, ply_path)

    return NavResult(
        command=command,
        sector_risks=sector_risks,
        safest=safest,
        overall_risk=overall_risk,
        risk_level=risk_level,
        ply_path=saved_ply,
    )


def draw_nav_overlay(
    canvas: np.ndarray,
    result: NavResult,
) -> np.ndarray:
    """
    Render a full-image sector risk panel for the 3-panel presentation layout.

    Draws translucent green-to-red coloured regions over the entire image
    (one per sector in the 3x3 grid), with risk values centred in each cell,
    white grid lines, the safest sector highlighted, and a prominent nav
    command + risk level banner at the bottom.

    Designed to be placed as the third panel in:
        [ Original  |  Danger Map  |  Sector Risk + Nav ]

    Args:
        canvas:   (H, W, 3) uint8 image (typically the danger map overlay).
        result:   NavResult from nav_command().

    Returns:
        Annotated copy of canvas.
    """
    out = canvas.copy().astype(np.float32)
    H, W = canvas.shape[:2]

    r1, r2 = H // 3, 2 * H // 3
    c1, c2 = W // 3, 2 * W // 3

    row_bounds = [(0, r1), (r1, r2), (r2, H)]
    col_bounds = [(0, c1), (c1, c2), (c2, W)]

    alpha = 0.45  # translucency of the coloured sector tint

    for sector, (gr, gc) in SECTOR_GRID.items():
        risk = result.sector_risks.get(sector, 0.0)
        # Green (safe) → Red (dangerous)
        r_val = int(np.clip(risk * 2, 0, 1) * 255)
        g_val = int(np.clip(2 - risk * 2, 0, 1) * 200)
        tint = np.array([r_val, g_val, 0], dtype=np.float32)  # RGB

        y0, y1 = row_bounds[gr]
        x0, x1 = col_bounds[gc]
        out[y0:y1, x0:x1] = (1 - alpha) * out[y0:y1, x0:x1] + alpha * tint

    out = out.clip(0, 255).astype(np.uint8)

    # ── Grid lines (white, thin) ─────────────────────────────────────────────
    line_color = (200, 200, 200)
    cv2.line(out, (c1, 0), (c1, H), line_color, 1)
    cv2.line(out, (c2, 0), (c2, H), line_color, 1)
    cv2.line(out, (0, r1), (W, r1), line_color, 1)
    cv2.line(out, (0, r2), (W, r2), line_color, 1)

    # ── Safest sector highlight ──────────────────────────────────────────────
    sg_r, sg_c = SECTOR_GRID[result.safest]
    sy0, sy1 = row_bounds[sg_r]
    sx0, sx1 = col_bounds[sg_c]
    cv2.rectangle(out, (sx0 + 1, sy0 + 1), (sx1 - 1, sy1 - 1),
                  (255, 255, 255), 2)

    # ── Per-sector risk labels ───────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.45, min(W, H) / 500.0)
    thickness = max(1, int(font_scale * 2))

    _SECTOR_LABELS = {
        "top-l": "TL", "top-c": "TC", "top-r": "TR",
        "mid-l": "ML", "center": "C",  "mid-r": "MR",
        "bot-l": "BL", "bot-c": "BC", "bot-r": "BR",
    }

    for sector, (gr, gc) in SECTOR_GRID.items():
        risk = result.sector_risks.get(sector, 0.0)
        y0, y1 = row_bounds[gr]
        x0, x1 = col_bounds[gc]
        cx_cell = (x0 + x1) // 2
        cy_cell = (y0 + y1) // 2

        # Sector label
        lbl = _SECTOR_LABELS[sector]
        (lw, lh), _ = cv2.getTextSize(lbl, font, font_scale * 0.6, 1)
        cv2.putText(out, lbl,
                    (cx_cell - lw // 2, cy_cell - lh),
                    font, font_scale * 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        # Risk value
        txt = f"{risk:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
        tx = cx_cell - tw // 2
        ty = cy_cell + th + 2
        # Dark outline for readability
        cv2.putText(out, txt, (tx, ty), font, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, txt, (tx, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # ── Navigation command banner (bottom) ───────────────────────────────────
    level_color = {
        "CLEAR":   (50, 200, 50),
        "CAUTION": (255, 165, 0),
        "DANGER":  (230, 0, 0),
    }
    color = level_color.get(result.risk_level, (200, 200, 200))

    arrow = _COMMAND_ARROWS.get(result.command, "")
    label = f"{arrow} {result.command}  [{result.risk_level}]"

    ban_font_scale = max(0.6, min(W, H) / 400.0)
    ban_thick = max(1, int(ban_font_scale * 2))
    (bw, bh), bbl = cv2.getTextSize(label, font, ban_font_scale, ban_thick)

    ban_h = bh + bbl + 16
    ban_y0 = H - ban_h
    # Semi-transparent black banner
    banner = out[ban_y0:, :].astype(np.float32)
    banner *= 0.3
    out[ban_y0:, :] = banner.clip(0, 255).astype(np.uint8)

    tx = (W - bw) // 2
    ty = H - bbl - 6
    cv2.putText(out, label, (tx, ty), font, ban_font_scale,
                color, ban_thick, cv2.LINE_AA)

    return out


# ── PLY writer ────────────────────────────────────────────────────────────────

def _write_risk_ply(
    risk_map: np.ndarray,
    depth_m:  np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    ply_path: Path,
) -> Path:
    """
    Unproject risk_map pixels to 3-D and write a coloured PLY point cloud.

    Each point is coloured on a green (safe, risk=0) → red (dangerous, risk=1)
    gradient so the cloud gives an immediate spatial sense of where the AUV
    faces collision risk.

    Point cloud coordinate frame: +Z forward, +X right, +Y down (camera frame).
    """
    H, W = risk_map.shape[:2]

    # Valid pixels only — must have positive, finite depth
    valid = (depth_m > MIN_DEPTH_M) & np.isfinite(depth_m)

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    d = depth_m.copy()
    d[~valid] = 0.0

    # Camera-frame unprojection
    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d

    pts_x = X[valid].ravel()
    pts_y = Y[valid].ravel()
    pts_z = Z[valid].ravel()
    risk  = risk_map[valid].ravel()

    # Green → red colour by risk
    r_ch = np.clip(risk * 255, 0, 255).astype(np.uint8)
    g_ch = np.clip((1.0 - risk) * 180, 0, 255).astype(np.uint8)
    b_ch = np.zeros_like(r_ch)

    n = pts_x.shape[0]
    ply_path = Path(ply_path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    with ply_path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{pts_x[i]:.4f} {pts_y[i]:.4f} {pts_z[i]:.4f} "
                    f"{r_ch[i]} {g_ch[i]} {b_ch[i]}\n")

    return ply_path
