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
    overlay      H×W×3 uint8  — grayscale-background image with a colour-coded
                               heatmap blended in + per-class contour labels

TUNING GUIDE  (change the constants at the top of this file)

1. HAZARD_WEIGHTS  — how dangerous is each class?
       0.0 = ignore completely,  1.0 = maximum danger
       Example: make reef the top priority →  "RI": 1.0,  "WR": 0.9

2. NEAR_M  — "danger zone" radius in metres
       Objects closer than NEAR_M get proximity = 1.0 (full risk).
       Objects at 2×NEAR_M get proximity = 0.5, at 10×NEAR_M → 0.1.
       Increase for faster AUVs that need more stopping distance.

3. PROXIMITY_POWER  — how fast does risk fall off with distance?
       1.0 = linear decay (1/depth)       ← default
       2.0 = quadratic (1/depth²)         — drops faster beyond NEAR_M
       0.5 = square-root decay            — drops more slowly

4. DEFAULT_SEG_THRESHOLD  — sigmoid cutoff for "class is present"
       0.5 = balanced  (default)
       0.3 = more sensitive (more pixels flagged)
       0.7 = stricter (fewer pixels flagged)

5. Custom formula via risk_fn parameter:
       Pass risk_fn=my_fn to danger_map() where
           my_fn(hazard, proximity) → risk
       Both hazard and proximity are (H,W) float32 arrays in [0, 1].
       Example — additive blend instead of multiplicative:
           danger_map(rgb, seg, depth, risk_fn=lambda h, p: 0.6*h + 0.4*p)

Default risk formula
────────────────────
    proximity(x,y) = clip( (near_m / depth(x,y))^power, 0, 1 )
    hazard(x,y)    = max hazard_weight over all SUIM-Net classes active at (x,y)
    risk(x,y)      = hazard(x,y) × proximity(x,y)   ∈ [0, 1]
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

# ── Class ordering (must match CLASS_ORDER in src/suimnet/run_infer.py) ────────
_CLASS_ORDER = ["RO", "FV", "HD", "RI", "WR"]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TUNING PARAMETERS — edit these to change how danger is calculated          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Per-class hazard weights: how dangerous is a collision with each class?
# Range: 0.0 (no risk) → 1.0 (maximum risk).
HAZARD_WEIGHTS: dict[str, float] = {
    "RO": 0.2,   # Robot / Instrument   — not a natural collision hazard
    "FV": 0.5,   # Fish / Vertebrate    — mobile, moderate risk
    "HD": 1.0,   # Human Diver          — highest priority to avoid
    "RI": 0.8,   # Reef / Invertebrate  — hard structural hazard
    "WR": 0.9,   # Wreck / Ruin         — large hard structural hazard
}

# Objects closer than NEAR_M metres get full proximity score (1.0).
# Increase this for faster vehicles that need earlier warnings.
NEAR_M: float = 1.0

# Proximity fall-off exponent.  Higher → danger drops faster with distance.
#   1.0 = linear (1/d),  2.0 = quadratic (1/d²),  0.5 = slow decay
PROXIMITY_POWER: float = 1.0

# Sigmoid cutoff for class activation.  Lower = more pixels flagged.
DEFAULT_SEG_THRESHOLD: float = 0.5

# ── Per-class annotation colours for the overlay (BGR for OpenCV) ─────────────
# Change these if you want different colours for the class contour labels.
_CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "RO": (255, 200,   0),   # cyan-ish   — robot
    "FV": ( 50, 220,  50),   # green      — fish
    "HD": (  0,   0, 255),   # red        — diver  (highest danger)
    "RI": (  0, 165, 255),   # orange     — reef
    "WR": (  0, 220, 220),   # yellow     — wreck
}


# ── Public API ────────────────────────────────────────────────────────────────

def danger_map(
    rgb: np.ndarray,
    seg_logits: np.ndarray,
    depth_m: np.ndarray,
    *,
    hazard_weights: dict[str, float] | None = None,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD,
    near_m: float = NEAR_M,
    proximity_power: float = PROXIMITY_POWER,
    overlay_alpha: float = 0.6,
    risk_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a per-pixel danger map by combining semantic segmentation and depth.

    Args:
        rgb:             Original camera frame, shape (H, W, 3), dtype uint8.
        seg_logits:      SUIM-Net sigmoid outputs, shape (H_s, W_s, 5), dtype float32.
                         Channel order: RO, FV, HD, RI, WR.
        depth_m:         Dense metric depth map in metres, shape (H_d, W_d), dtype float32.
                         Zeros and NaNs are treated as invalid (no depth → no proximity risk).
        hazard_weights:  Per-class collision severity override dict.
                         Unspecified classes use HAZARD_WEIGHTS defaults.
        seg_threshold:   Sigmoid cutoff to decide whether a class is active. Default 0.5.
        near_m:          Danger-zone radius in metres. See TUNING GUIDE above.
        proximity_power: Fall-off exponent for distance. See TUNING GUIDE above.
        overlay_alpha:   Blend weight for the heatmap (0=invisible, 1=fully opaque).
        risk_fn:         Custom risk formula: fn(hazard, proximity) → risk.
                         If None, uses  risk = hazard × proximity^power.

    Returns:
        risk_map:  (H, W) float32, values in [0, 1].
        overlay:   (H, W, 3) uint8.
                   Grayscale background with HOT heatmap and class contour labels.
                   Using grayscale avoids the colour-clash with underwater blue/green.
    """
    weights = {**HAZARD_WEIGHTS, **(hazard_weights or {})}

    H, W = rgb.shape[:2]

    # Resize segmentation and depth to match the RGB frame
    seg   = _resize_hwc(seg_logits.astype(np.float32), H, W)  # (H, W, 5)
    depth = _resize_hw(depth_m.astype(np.float32),     H, W)  # (H, W)

    # ── Hazard: max weight over all classes active at each pixel ──────────────
    hazard = np.zeros((H, W), dtype=np.float32)
    for ch, cls_name in enumerate(_CLASS_ORDER):
        active = seg[..., ch] > seg_threshold
        w = weights[cls_name]
        hazard = np.where(active, np.maximum(hazard, w), hazard)

    # ── Proximity: (near_m / depth)^power, clipped to [0, 1] ─────────────────
    # Invalid depth pixels (zero or NaN) → proximity = 0 → risk = 0
    valid = (depth > 0) & np.isfinite(depth)
    proximity = np.zeros((H, W), dtype=np.float32)
    proximity[valid] = np.clip(
        (near_m / depth[valid]) ** proximity_power, 0.0, 1.0
    )

    # ── Risk formula ──────────────────────────────────────────────────────────
    if risk_fn is not None:
        risk_map = np.clip(risk_fn(hazard, proximity).astype(np.float32), 0.0, 1.0)
    else:
        risk_map = hazard * proximity   # default: multiplicative

    overlay = _colorize_risk(rgb, risk_map, seg, seg_threshold, overlay_alpha)

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


def _colorize_risk(
    rgb: np.ndarray,
    risk: np.ndarray,
    seg: np.ndarray,
    seg_threshold: float,
    alpha: float,
) -> np.ndarray:
    """
    Render the danger overlay.

    Background is desaturated to grayscale so the HOT heatmap (black→red→yellow→white)
    stands out clearly without clashing with the blue/green underwater colours.
    Per-class contours and text labels are drawn on top.
    """
    # 1. Grayscale background — eliminates colour-on-colour clash
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    # 2. HOT heatmap blended by per-pixel risk weight
    risk_u8 = (risk * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(risk_u8, cv2.COLORMAP_HOT)
    heatmap_rgb = heatmap_bgr[..., ::-1].astype(np.float32)   # BGR → RGB

    w = (risk * alpha)[..., np.newaxis]                        # (H, W, 1)
    blended = ((1.0 - w) * bg + w * heatmap_rgb).clip(0, 255).astype(np.uint8)

    # 3. Class contours + labels
    blended = _draw_class_annotations(blended, seg, seg_threshold)

    return blended


def _draw_class_annotations(
    canvas: np.ndarray,
    seg: np.ndarray,
    seg_threshold: float,
    min_area: int = 200,
) -> np.ndarray:
    """
    Draw per-class segmentation contours and text labels on the canvas.

    Each active class gets its own contour colour (see _CLASS_COLORS_BGR) and
    a label placed at the centroid of the largest blob.  Tiny blobs smaller than
    `min_area` pixels are skipped to reduce visual clutter.

    Args:
        canvas:        (H, W, 3) uint8 image to annotate (modified in-place copy).
        seg:           (H, W, 5) float32 segmentation at full canvas resolution.
        seg_threshold: Class-active cutoff, same value used for the risk map.
        min_area:      Minimum blob area (pixels) to annotate. Increase to suppress
                       small false-positive blobs; decrease to catch faint detections.
    """
    H, W = canvas.shape[:2]
    out = canvas.copy()

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(W, H) / 800.0)   # scales with image size
    thickness  = max(1, int(font_scale * 2))

    for ch, cls_name in enumerate(_CLASS_ORDER):
        mask = (seg[..., ch] > seg_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        color = _CLASS_COLORS_BGR[cls_name]

        # Draw all contours for this class
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            cv2.drawContours(out, [cnt], -1, color, thickness + 1)

        # Label the largest blob
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            continue

        M = cv2.moments(largest)
        if M["m00"] <= 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        (tw, th), baseline = cv2.getTextSize(cls_name, font, font_scale, thickness)

        # Keep label inside frame
        lx = int(np.clip(cx - tw // 2, 1, W - tw - 2))
        ly = int(np.clip(cy, th + 3, H - baseline - 2))

        # Dark background box for readability
        cv2.rectangle(out,
                      (lx - 2, ly - th - 3),
                      (lx + tw + 2, ly + baseline + 1),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(out, cls_name, (lx, ly), font, font_scale,
                    color, thickness, cv2.LINE_AA)

    return out
