"""
Quick smoke-test for the danger map — no SPADE weights needed.

Runs SUIM-Net on the 8 bundled sample images, substitutes a flat synthetic
depth map (default 2.0 m), and writes side-by-side overlay PNGs so you can
visually verify the overlay looks correct.

Usage
─────
    python -m src.danger_map.quick_test

    # Change the synthetic depth (e.g. very close = high risk):
    python -m src.danger_map.quick_test --depth_m 0.5

    # Turn knobs:
    python -m src.danger_map.quick_test --near_m 2.0 --proximity_power 2.0

Outputs: reports/danger_map/quick_test/  (one PNG per sample image)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import skimage.transform as sktf
from PIL import Image

REPO_ROOT    = Path(__file__).resolve().parents[2]
SUIMNET_ROOT = REPO_ROOT / "vendor" / "SUIM-Net"
SAMPLE_IMGS  = SUIMNET_ROOT / "sample_test" / "images"
WEIGHTS      = SUIMNET_ROOT / "sample_test" / "ckpt_seg_5obj.hdf5"

sys.path.append(str(SUIMNET_ROOT))

# ── Keras 2.13+ compatibility shim (identical to src/suimnet/run_infer.py) ────
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

from model import SUIM_Net  # type: ignore  (vendor/SUIM-Net/model.py)
from src.danger_map import danger_map
from src.danger_map.navigate import nav_command, draw_nav_overlay
from src.danger_map.run_video import _make_side_by_side

SUIM_H, SUIM_W = 240, 320


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _run_suimnet(model, rgb: np.ndarray) -> np.ndarray:
    img_rs = sktf.resize(rgb, (SUIM_H, SUIM_W, 3),
                         preserve_range=False, anti_aliasing=True)
    x = np.expand_dims(img_rs, axis=0)
    return model.predict(x, verbose=0)[0]   # (240, 320, 5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Danger map quick smoke-test (no SPADE needed).")
    ap.add_argument("--depth_m", type=float, default=2.0,
                    help="Synthetic flat depth in metres applied to every pixel. "
                         "Try 0.5 for very high risk, 5.0 for low risk. Default: 2.0")
    ap.add_argument("--near_m", type=float, default=1.0,
                    help="Danger-zone radius in metres (see NEAR_M in __init__.py). Default: 1.0")
    ap.add_argument("--proximity_power", type=float, default=1.0,
                    help="Distance fall-off exponent (1=linear, 2=quadratic). Default: 1.0")
    ap.add_argument("--overlay_alpha", type=float, default=0.6,
                    help="Heatmap blend weight (0=invisible, 1=fully opaque). Default: 0.6")
    ap.add_argument("--out_dir", default="reports/danger_map/quick_test",
                    help="Output directory. Default: reports/danger_map/quick_test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(SAMPLE_IMGS.glob("*.jpg")) + sorted(SAMPLE_IMGS.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No images found in: {SAMPLE_IMGS}")

    print(f"Loading SUIM-Net weights from {WEIGHTS} …")
    model = SUIM_Net(im_res=(SUIM_H, SUIM_W), n_classes=5).model
    model.load_weights(str(WEIGHTS))
    print(f"Found {len(image_paths)} sample images.\n")

    for img_path in image_paths:
        rgb = _load_rgb(img_path)
        H, W = rgb.shape[:2]

        seg_logits = _run_suimnet(model, rgb)

        # Flat synthetic depth — every pixel at the same distance
        depth_flat = np.full((H, W), args.depth_m, dtype=np.float32)

        risk_map, overlay = danger_map(
            rgb, seg_logits, depth_flat,
            near_m=args.near_m,
            proximity_power=args.proximity_power,
            overlay_alpha=args.overlay_alpha,
        )

        nav = nav_command(risk_map, depth_flat)
        overlay = draw_nav_overlay(overlay, nav)

        panel = _make_side_by_side(rgb, overlay)
        out_path = out_dir / (img_path.stem + "_danger.png")
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"  {img_path.name:30s}  risk_max={risk_map.max():.3f}  "
              f"[{nav.risk_level}] → {nav.command}")

    print(f"\nDone. Overlays saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
