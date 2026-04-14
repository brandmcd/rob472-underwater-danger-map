"""
Generate a 4-panel pipeline walkthrough figure for slide presentation.

Output: figures/pipeline/pipeline_4step.png
        figures/pipeline/pipeline_4step_{scene}.png  (per scene)

Panels (left → right):
    1. Raw RGB        — original camera frame
    2. Segmentation   — SUIM-Net class mask (colour-coded, legend)
    3. Depth Map      — synthetic gradient depth, PLASMA colourmap
    4. Danger Map     — fused HOT heatmap overlay

Also generates a two-row version with titles for slides.

Usage
─────
    python scripts/make_pipeline_figure.py
    python scripts/make_pipeline_figure.py --scene d_r_598    # single scene
    python scripts/make_pipeline_figure.py --depth_m 2.0      # flat depth override
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import skimage.transform as sktf
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT    = Path(__file__).resolve().parent.parent
SUIMNET_ROOT = REPO_ROOT / "vendor" / "SUIM-Net"
SAMPLE_IMGS  = SUIMNET_ROOT / "sample_test" / "images"
WEIGHTS      = SUIMNET_ROOT / "sample_test" / "ckpt_seg_5obj.hdf5"
OUT_DIR      = REPO_ROOT / "figures" / "pipeline"

sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(SUIMNET_ROOT))

# ── Keras 2.13 compatibility shim ─────────────────────────────────────────────
import keras
import keras.models as _km
if not hasattr(_km, "Input"):
    _km.Input = keras.layers.Input
_OrigModel = _km.Model
class _ModelShim(_OrigModel):
    def __init__(self, *args, **kwargs):
        if "input" in kwargs and "inputs" not in kwargs:
            kwargs["inputs"] = kwargs.pop("input")
        if "output" in kwargs and "outputs" not in kwargs:
            kwargs["outputs"] = kwargs.pop("output")
        super().__init__(*args, **kwargs)
_km.Model = _ModelShim
# ─────────────────────────────────────────────────────────────────────────────

from model import SUIM_Net  # type: ignore
from src.danger_map import danger_map

SUIM_H, SUIM_W = 240, 320

# ── Class metadata ─────────────────────────────────────────────────────────────
_CLASS_ORDER = ["RO", "FV", "HD", "RI", "WR"]
_CLASS_LABELS = {
    "RO": "Robot",
    "FV": "Fish",
    "HD": "Diver",
    "RI": "Reef",
    "WR": "Wreck",
}
# RGB colours for the segmentation mask  (match _CLASS_COLORS_BGR from __init__)
_CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "RO": (  0, 200, 255),   # cyan
    "FV": ( 50, 220,  50),   # green
    "HD": (255,   0,   0),   # red   ← diver, highest danger
    "RI": (255, 165,   0),   # orange
    "WR": (220, 220,   0),   # yellow
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _run_suimnet(model, rgb: np.ndarray) -> np.ndarray:
    img_rs = sktf.resize(rgb, (SUIM_H, SUIM_W, 3),
                         preserve_range=False, anti_aliasing=True)
    x = np.expand_dims(img_rs, axis=0)
    return model.predict(x, verbose=0)[0]   # (240, 320, 5)


def _make_seg_panel(rgb: np.ndarray, seg_logits: np.ndarray,
                    threshold: float = 0.5) -> np.ndarray:
    """
    Render a class-coloured segmentation mask.

    Background is desaturated to ~40% so colours pop.
    Each active class is filled with its colour (additive blend,
    higher channels win when classes overlap).
    A compact legend is drawn in the lower-left corner.
    """
    H, W = rgb.shape[:2]
    seg = cv2.resize(seg_logits, (W, H), interpolation=cv2.INTER_LINEAR)

    # Desaturate background
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) * 0.45
    canvas = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    # Overlay each class
    for ch, cls_name in enumerate(_CLASS_ORDER):
        mask = seg[..., ch] > threshold
        if not mask.any():
            continue
        cr, cg, cb = _CLASS_COLORS_RGB[cls_name]
        # Semi-transparent fill
        canvas[mask, 0] = canvas[mask, 0] * 0.25 + cr * 0.75
        canvas[mask, 1] = canvas[mask, 1] * 0.25 + cg * 0.75
        canvas[mask, 2] = canvas[mask, 2] * 0.25 + cb * 0.75

    # Draw contours for crisp edges
    for ch, cls_name in enumerate(_CLASS_ORDER):
        mask_u8 = ((seg[..., ch] > threshold).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = _CLASS_COLORS_RGB[cls_name]
        # cv2 wants BGR
        color_bgr = (color[2], color[1], color[0])
        out_u8 = canvas.clip(0, 255).astype(np.uint8)
        cv2.drawContours(out_u8, contours, -1, color_bgr, 2)
        canvas = out_u8.astype(np.float32)

    out = canvas.clip(0, 255).astype(np.uint8)

    # Legend in lower-left
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.40
    th = 1
    lx, ly = 6, H - 6 - len(_CLASS_COLORS_RGB) * 16
    for cls_name, label in _CLASS_LABELS.items():
        cr, cg, cb = _CLASS_COLORS_RGB[cls_name]
        cv2.rectangle(out, (lx, ly - 10), (lx + 12, ly + 2), (cb, cg, cr), cv2.FILLED)
        cv2.putText(out, label, (lx + 16, ly), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        ly += 16

    return out


def _make_depth_panel(rgb: np.ndarray, depth_m: np.ndarray,
                      max_depth_m: float = 8.0) -> np.ndarray:
    """
    Render depth map as PLASMA colourmap panel.
    0 m → dark purple,  max_depth_m → bright yellow.
    """
    H, W = rgb.shape[:2]
    depth_rs = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST) \
               if depth_m.shape[:2] != (H, W) else depth_m.copy()

    depth_norm = np.clip(depth_rs / max_depth_m, 0.0, 1.0)
    depth_u8   = (depth_norm * 255).astype(np.uint8)
    depth_bgr  = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
    depth_rgb  = depth_bgr[..., ::-1].copy()
    return depth_rgb


def _make_semantic_depth(H: int, W: int,
                         seg_logits: np.ndarray,
                         threshold: float = 0.4) -> np.ndarray:
    """
    Synthetic depth driven by the segmentation mask — semantically correct.

    Background water (no class active) → far (5–8 m).
    Segmented objects (diver, reef, wreck, fish) → close (0.8–2.5 m).
    A Gaussian blur blends the boundaries so transitions look like a real
    dense depth map rather than hard binary regions.

    This correctly encodes the scene geometry:
      - Open water / background  →  far  →  yellow in PLASMA
      - Detected obstacles       →  close →  dark purple in PLASMA
    """
    from scipy.ndimage import gaussian_filter

    # Resize seg to image resolution
    seg_full = cv2.resize(seg_logits, (W, H), interpolation=cv2.INTER_LINEAR)

    # Per-class nominal depth (metres) for the closest-object heuristic
    #   HD (diver) and WR (wreck) are often 1–3 m away
    #   RI (reef/seabed) is often very close, 0.5–2 m
    #   FV (fish) float mid-range, 1.5–3 m
    #   RO (robot) typically close, 0.5–1.5 m
    class_depths = {
        0: 1.2,  # RO — robot / instrument
        1: 2.0,  # FV — fish
        2: 1.8,  # HD — human diver
        3: 1.0,  # RI — reef / invertebrate (seabed)
        4: 1.5,  # WR — wreck / ruin
    }

    # Start with uniform background depth (far water column)
    depth = np.full((H, W), 7.0, dtype=np.float32)

    # Build a "foreground proximity" map: how close should each pixel be?
    foreground = np.zeros((H, W), dtype=np.float32)
    for ch, close_d in class_depths.items():
        mask = seg_full[..., ch] > threshold
        if not mask.any():
            continue
        # Weight proportional to sigmoid confidence
        conf = seg_full[..., ch] * mask.astype(np.float32)
        # Convert to depth: high confidence → close depth
        pixel_depth = close_d + (1.0 - conf) * 1.5   # range: close_d … close_d+1.5
        # Update foreground map where this class is most confident
        better = mask & (pixel_depth < foreground + 7.0 * (foreground == 0))
        foreground = np.where(better, pixel_depth, np.where(foreground > 0, foreground, 0.0))

    # Where foreground detected, blend depth toward close_d
    fg_mask = foreground > 0
    depth[fg_mask] = foreground[fg_mask]

    # Gaussian blur for smooth transitions (sigma scales with image size)
    sigma = max(H, W) / 10.0
    depth = gaussian_filter(depth, sigma=sigma)

    # Mild vertical bias: bottom of frame is slightly closer (perspective effect)
    row_bias = np.linspace(0.5, -0.5, H, dtype=np.float32)[:, np.newaxis]
    depth = depth + row_bias

    return np.clip(depth, 0.3, 9.0)


def _colorbar_strip(width: int, cmap_id: int, label_left: str, label_right: str,
                    label_mid: str = "", bar_h: int = 22) -> np.ndarray:
    """Horizontal colourbar strip with left/mid/right labels."""
    grad = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :].repeat(bar_h, axis=0)
    bar_bgr = cv2.applyColorMap(grad, cmap_id)
    bar_rgb = bar_bgr[..., ::-1].copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar_rgb, label_left,  (4, bar_h - 5), font, 0.36, (220, 220, 220), 1)
    cv2.putText(bar_rgb, label_right, (width - 28, bar_h - 5), font, 0.36, (20, 20, 20), 1)
    if label_mid:
        cv2.putText(bar_rgb, label_mid, (width // 2 - 10, bar_h - 5), font, 0.36, (160, 160, 160), 1)
    return bar_rgb


def _assemble(panels: list[tuple[np.ndarray, str, np.ndarray | None]],
              title_h: int = 32, bar_h: int = 22) -> np.ndarray:
    """
    Assemble labelled panels into a single composite image.

    Each entry: (image HxWx3, panel_title, colorbar_strip | None)
    """
    H, W = panels[0][0].shape[:2]
    n = len(panels)
    total_h = title_h + H + bar_h
    canvas = np.zeros((total_h, W * n, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (img, label, bar) in enumerate(panels):
        x0 = i * W
        # Resize image if needed
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        canvas[title_h : title_h + H, x0 : x0 + W] = img
        cv2.putText(canvas, label, (x0 + 10, title_h - 8), font, 0.70,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if bar is not None:
            if bar.shape[1] != W:
                bar = cv2.resize(bar, (W, bar_h))
            canvas[title_h + H :, x0 : x0 + W] = bar

    # Dividers between panels
    for i in range(1, n):
        canvas[:, i * W - 1 : i * W + 1] = 60

    return canvas


def process_scene(model, img_path: Path, args) -> Path:
    rgb = _load_rgb(img_path)
    H, W = rgb.shape[:2]

    # 1. SUIM-Net segmentation
    seg_logits = _run_suimnet(model, rgb)   # (240, 320, 5)

    # 2. Depth map — semantic (objects close, water far) unless flat override
    if args.depth_m is not None:
        depth = np.full((H, W), args.depth_m, dtype=np.float32)
    else:
        depth = _make_semantic_depth(H, W, seg_logits)

    # 3. Panel 1 — raw RGB (no change)
    p_rgb = rgb.copy()

    # 4. Panel 2 — segmentation mask
    p_seg = _make_seg_panel(rgb, seg_logits)

    # 5. Panel 3 — depth map (PLASMA)
    max_depth = args.max_depth_m
    p_depth = _make_depth_panel(rgb, depth, max_depth_m=max_depth)

    # 6. Panel 4 — danger map fusion
    _risk_map, p_danger = danger_map(
        rgb, seg_logits, depth,
        near_m=args.near_m,
        display_gamma=args.display_gamma,
        overlay_alpha=args.overlay_alpha,
    )

    # Build colorbars
    bar_seg   = None   # legend is drawn on the seg panel itself
    bar_depth = _colorbar_strip(
        W, cv2.COLORMAP_PLASMA,
        "0 m (close)", f"{max_depth:.0f} m (far)",
        f"{max_depth / 2:.0f} m",
    )
    bar_risk  = _colorbar_strip(
        W, cv2.COLORMAP_HOT,
        "risk: 0", "1.0", "0.5",
    )

    panels = [
        (p_rgb,    "1. Raw RGB",      None),
        (p_seg,    "2. Segmentation", bar_seg),
        (p_depth,  "3. Depth Map",    bar_depth),
        (p_danger, "4. Danger Map",   bar_risk),
    ]

    composite = _assemble(panels)

    stem = img_path.stem.rstrip("_")
    out_path = OUT_DIR / f"pipeline_{stem}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out_path.relative_to(REPO_ROOT)}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 4-step pipeline composite figures.")
    ap.add_argument("--scene", default=None,
                    help="Process only this image stem (e.g. 'd_r_598'). "
                         "Default: all sample images.")
    ap.add_argument("--depth_m", type=float, default=None,
                    help="Use a flat synthetic depth (metres) instead of the gradient. "
                         "Default: gradient 0.8–7.0 m (top→bottom).")
    ap.add_argument("--max_depth_m", type=float, default=8.0,
                    help="Max depth for PLASMA colour scale. Default: 8.0")
    ap.add_argument("--near_m", type=float, default=2.5,
                    help="Danger-zone radius for risk formula. Default: 2.5")
    ap.add_argument("--display_gamma", type=float, default=0.5,
                    help="Gamma for danger map visualisation. Default: 0.5 (sqrt).")
    ap.add_argument("--overlay_alpha", type=float, default=0.6,
                    help="Danger overlay blend weight. Default: 0.6")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(SAMPLE_IMGS.glob("*.jpg")) + sorted(SAMPLE_IMGS.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No sample images found in {SAMPLE_IMGS}")

    if args.scene:
        image_paths = [p for p in image_paths if args.scene in p.stem]
        if not image_paths:
            raise RuntimeError(f"No image matching scene '{args.scene}' found.")

    print(f"Loading SUIM-Net weights …")
    model = SUIM_Net(im_res=(SUIM_H, SUIM_W), n_classes=5).model
    model.load_weights(str(WEIGHTS))
    print(f"Processing {len(image_paths)} scene(s) …\n")

    out_paths = []
    for p in image_paths:
        out_paths.append(process_scene(model, p, args))

    # If multiple scenes, also make a best-of composite (d_r_598 + w_r_147 + n_l_100)
    if len(out_paths) >= 3:
        best = ["d_r_598", "w_r_147", "n_l_100"]
        best_imgs = []
        for stem in best:
            candidate = [p for p in out_paths if stem in p.name]
            if candidate:
                best_imgs.append(candidate[0])

        if len(best_imgs) >= 2:
            rows = []
            for bp in best_imgs:
                img_bgr = cv2.imread(str(bp))
                rows.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            grid = np.vstack(rows)
            grid_path = OUT_DIR / "pipeline_grid.png"
            cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"\n  Grid saved: {grid_path.relative_to(REPO_ROOT)}")

    print(f"\nDone. Figures in: {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
