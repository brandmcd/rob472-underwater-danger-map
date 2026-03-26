"""
Underwater danger map: fuse SUIM-Net segmentation + SPADE depth into a per-pixel risk score.

Core function
─────────────
    risk_map, overlay = danger_map(rgb, seg_logits, depth_m)

Inputs
    rgb          H×W×3 uint8  — original camera frame (any resolution)
    seg_logits   H_s×W_s×5 float32  — SUIM-Net sigmoid outputs, values in [0, 1]
                 Channel order (matches CLASS_ORDER in src/suimnet/run_infer.py):
                     0 = RO  Robot/Instrument
                     1 = FV  Fish/Vertebrate
                     2 = HD  Human Diver
                     3 = RI  Reef/Invertebrate
                     4 = WR  Wreck/Ruin
    depth_m      H_d×W_d float32  — dense depth in metres from SPADE
                 Zero / NaN pixels are treated as invalid (contribute no risk)

Outputs
    risk_map     H×W float32  — per-pixel risk in [0, 1]
    overlay      H×W×3 uint8  — original RGB blended with a colour-coded heatmap

Risk formula
────────────
    proximity(x,y) = clip(NEAR_M / depth(x,y), 0, 1)
        → 1.0 when depth ≤ NEAR_M (1 m by default), falls off for farther objects
    hazard(x,y)    = max hazard weight over all classes active at (x,y)
    risk(x,y)      = hazard(x,y) × proximity(x,y)

Hazard weights reflect collision risk for an AUV:
    HD (diver)  1.0  — highest priority to avoid
    WR (wreck)  0.9  — large hard structure
    RI (reef)   0.8  — hard structure, irregular shape
    FV (fish)   0.5  — mobile, softer collision
    RO (robot)  0.2  — another instrument, low natural hazard

The overlay uses OpenCV's JET colormap (blue = low risk → red = high risk)
alpha-blended over the original RGB for intuitive visualisation.
"""
from __future__ import annotations

import cv2
import numpy as np

# ── Hazard weights (per SUIM-Net class) ──────────────────────────────────────
# Channel order: RO, FV, HD, RI, WR  (must match CLASS_ORDER in run_infer.py)
_CLASS_ORDER = ["RO", "FV", "HD", "RI", "WR"]

HAZARD_WEIGHTS: dict[str, float] = {
    "RO": 0.2,   # Robot / Instrument   — not a natural collision hazard
    "FV": 0.5,   # Fish / Vertebrate    — mobile, moderate risk
    "HD": 1.0,   # Human Diver          — highest priority to avoid
    "RI": 0.8,   # Reef / Invertebrate  — hard structural hazard
    "WR": 0.9,   # Wreck / Ruin         — large hard structural hazard
}

# Pixels with sigmoid output above this threshold are considered "active" for a class
DEFAULT_SEG_THRESHOLD: float = 0.5

# Objects at this depth (metres) receive maximum proximity score (1.0).
# Objects farther away scale down as NEAR_M / depth.
NEAR_M: float = 1.0


# ── Public API ────────────────────────────────────────────────────────────────

def danger_map(
    rgb: np.ndarray,
    seg_logits: np.ndarray,
    depth_m: np.ndarray,
    *,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD,
    near_m: float = NEAR_M,
    overlay_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a per-pixel danger map by combining semantic segmentation and depth.

    Args:
        rgb:            Original camera frame, shape (H, W, 3), dtype uint8.
        seg_logits:     SUIM-Net sigmoid outputs, shape (H_s, W_s, 5), dtype float32.
                        Channel order: RO, FV, HD, RI, WR.
        depth_m:        Dense metric depth map in metres, shape (H_d, W_d), dtype float32.
                        Zeros and NaNs are treated as invalid (no depth → no proximity risk).
        seg_threshold:  Sigmoid cutoff to decide whether a class is present. Default 0.5.
        near_m:         Reference "close" distance in metres. Objects at this depth get
                        proximity = 1.0; objects farther away scale down proportionally.
        overlay_alpha:  Blend weight for the danger heatmap over the RGB image.
                        0.0 = only RGB, 1.0 = only heatmap.

    Returns:
        risk_map:  (H, W) float32 array, values in [0, 1].
                   High values indicate nearby, high-hazard pixels.
        overlay:   (H, W, 3) uint8 array.
                   Original RGB alpha-blended with a JET heatmap of risk_map.
                   Blue = low risk, red = high risk.
    """
    H, W = rgb.shape[:2]

    # Bring segmentation logits and depth to the same resolution as the RGB frame
    seg   = _resize_hwc(seg_logits.astype(np.float32), H, W)  # (H, W, 5)
    depth = _resize_hw(depth_m.astype(np.float32),     H, W)  # (H, W)

    # Per-pixel hazard: take the maximum weight among all classes active at each pixel
    hazard = np.zeros((H, W), dtype=np.float32)
    for ch, cls_name in enumerate(_CLASS_ORDER):
        active = seg[..., ch] > seg_threshold          # bool mask: class present?
        w = HAZARD_WEIGHTS[cls_name]
        hazard = np.where(active, np.maximum(hazard, w), hazard)

    # Per-pixel proximity: near_m / depth, clipped to [0, 1]
    # Invalid depth pixels (zero or NaN) contribute 0 proximity → 0 risk
    valid = (depth > 0) & np.isfinite(depth)
    proximity = np.zeros((H, W), dtype=np.float32)
    proximity[valid] = np.clip(near_m / depth[valid], 0.0, 1.0)

    # Risk = hazard × proximity, already in [0, 1]
    risk_map = hazard * proximity

    overlay = _colorize_risk(rgb, risk_map, overlay_alpha)

    return risk_map, overlay


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resize_hwc(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize a (H0, W0, C) array to (H, W, C) using bilinear interpolation."""
    if arr.shape[:2] == (H, W):
        return arr
    return cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)


def _resize_hw(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize a (H0, W0) array to (H, W) using nearest-neighbour to preserve depth values."""
    if arr.shape[:2] == (H, W):
        return arr
    return cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)


def _colorize_risk(rgb: np.ndarray, risk: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend a JET heatmap of `risk` over `rgb`.

    risk: (H, W) float32 in [0, 1]
    Returns (H, W, 3) uint8.
    """
    risk_u8 = (risk * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(risk_u8, cv2.COLORMAP_JET)
    heatmap_rgb = heatmap_bgr[..., ::-1]                        # BGR → RGB
    blended = (1 - alpha) * rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)
