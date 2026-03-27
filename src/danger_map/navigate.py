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
                                "ASCEND"  | "DESCEND" | "STOP"
    result.sector_risks dict  — mean risk for each of the 5 spatial sectors
    result.safest       str   — name of the lowest-risk sector
    result.risk_level   str   — "CLEAR" | "CAUTION" | "DANGER"
    result.overall_risk float — mean risk across the full frame

Sector layout (forward-facing AUV camera)
─────────────────────────────────────────
    ┌──────────────────────────────┐
    │         UP  (top 1/3)        │
    ├────────┬────────────┬────────┤
    │  LEFT  │   CENTER   │  RIGHT │
    │(left   │ (mid 1/3   │(right  │
    │ 1/3)   │  × mid 1/3)│ 1/3)  │
    ├────────┴────────────┴────────┤
    │        DOWN (bot 1/3)        │
    └──────────────────────────────┘

  HIGH risk in LEFT   → recommend  GO RIGHT
  HIGH risk in RIGHT  → recommend  GO LEFT
  HIGH risk in UP     → recommend  DESCEND   (obstacle above → go lower)
  HIGH risk in DOWN   → recommend  ASCEND    (seafloor/reef below → go up)
  HIGH risk in CENTER → recommend  STOP
  ALL sectors low     → recommend  PROCEED

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

def _sector_masks(H: int, W: int) -> dict[str, np.ndarray]:
    """Return boolean masks for each of the 5 spatial sectors."""
    r1, r2 = H // 3, 2 * H // 3
    c1, c2 = W // 3, 2 * W // 3

    masks = {}
    masks["up"]     = np.zeros((H, W), dtype=bool)
    masks["up"][:r1, :]          = True

    masks["down"]   = np.zeros((H, W), dtype=bool)
    masks["down"][r2:, :]        = True

    masks["left"]   = np.zeros((H, W), dtype=bool)
    masks["left"][:, :c1]        = True

    masks["right"]  = np.zeros((H, W), dtype=bool)
    masks["right"][:, c2:]       = True

    masks["center"] = np.zeros((H, W), dtype=bool)
    masks["center"][r1:r2, c1:c2] = True

    return masks


# "Safest sector" → recommended direction for the AUV to move toward.
# The AUV moves *toward* the safest region in the image.
_SECTOR_TO_COMMAND: dict[str, str] = {
    "left":   "GO LEFT",
    "right":  "GO RIGHT",
    "up":     "ASCEND",
    "down":   "DESCEND",
    "center": "PROCEED",
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
    position: str = "bottom-left",
) -> np.ndarray:
    """
    Render the navigation command and sector risk bar onto the canvas.

    Draws two things:
      1. A prominent command label (e.g. "↑ ASCEND") with risk-level colour.
      2. A small 3×3 sector risk grid in the corner showing per-sector heat.

    Args:
        canvas:   (H, W, 3) uint8 image (typically the danger map overlay).
        result:   NavResult from nav_command().
        position: Where to place the HUD — "bottom-left" or "top-left".

    Returns:
        Annotated copy of canvas.
    """
    out = canvas.copy()
    H, W = out.shape[:2]

    # Colour by risk level
    level_color_bgr = {
        "CLEAR":   (50, 200, 50),    # green
        "CAUTION": (0, 165, 255),    # orange
        "DANGER":  (0, 0, 230),      # red
    }
    color = level_color_bgr.get(result.risk_level, (200, 200, 200))

    # ── Command label ────────────────────────────────────────────────────────
    arrow = {"GO LEFT": "<<", "GO RIGHT": ">>", "ASCEND": "^^",
             "DESCEND": "vv", "PROCEED": ">>", "STOP": "!!"}.get(result.command, "")
    label = f"{arrow} {result.command}"

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.5, min(W, H) / 600.0)
    thickness  = max(1, int(font_scale * 2))

    (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad = 6
    if position == "top-left":
        lx, ly = pad + 2, pad + lh + 2
    else:  # bottom-left
        lx, ly = pad + 2, H - baseline - pad - 2

    # Background box
    cv2.rectangle(out,
                  (lx - pad, ly - lh - pad),
                  (lx + lw + pad, ly + baseline + pad),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(out, label, (lx, ly), font, font_scale, color, thickness, cv2.LINE_AA)

    # ── 3×3 sector mini-map ──────────────────────────────────────────────────
    cell = max(18, W // 20)
    grid_w, grid_h = cell * 3, cell * 3
    gx = W - grid_w - pad
    gy = (H - grid_h - pad) if position == "bottom-left" else (pad)

    # Grid background
    cv2.rectangle(out, (gx - 2, gy - 2), (gx + grid_w + 2, gy + grid_h + 2),
                  (30, 30, 30), cv2.FILLED)

    # sector name → (grid_row, grid_col)
    _grid_pos = {"up": (0, 1), "center": (1, 1),
                 "down": (2, 1), "left": (1, 0), "right": (1, 2)}

    for sector, (gr, gc) in _grid_pos.items():
        risk  = result.sector_risks.get(sector, 0.0)
        # Colour: green (0 risk) → red (1 risk) via blue=0 channel
        r_val = int(np.clip(risk * 2, 0, 1) * 220)
        g_val = int(np.clip(2 - risk * 2, 0, 1) * 180)
        cell_color = (0, g_val, r_val)   # BGR

        x0 = gx + gc * cell
        y0 = gy + gr * cell
        cv2.rectangle(out, (x0 + 1, y0 + 1), (x0 + cell - 1, y0 + cell - 1),
                      cell_color, cv2.FILLED)

        # Highlight the safest sector
        if sector == result.safest:
            cv2.rectangle(out, (x0, y0), (x0 + cell, y0 + cell),
                          (255, 255, 255), 1)

        # Risk number inside cell
        txt = f"{risk:.2f}"
        ts  = max(0.25, cell / 80.0)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, ts, 1)
        tx = x0 + (cell - tw) // 2
        ty = y0 + (cell + th) // 2
        cv2.putText(out, txt, (tx, ty),
                    cv2.FONT_HERSHEY_PLAIN, ts, (230, 230, 230), 1, cv2.LINE_AA)

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
