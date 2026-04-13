"""
Generate turbidity example images showing the effect on the *input RGB frame only*.

For each source image the script:
  1. Extracts the left ("Original") panel from a processed danger-map composite.
  2. Applies apply_turbidity() at each of the 5 levels to that panel.
  3. Saves a side-by-side strip:   [0.0 | 0.25 | 0.50 | 0.75 | 1.00]
  4. Also saves each level as an individual PNG.

Output layout
─────────────
  reports/turbidity/examples/
    strip_img_<N>.png                  ← 5-panel strip for each source
    level_0.00_img_<N>.png  }
    level_0.25_img_<N>.png  }  individual turbid input images
    ...                     }

Usage:
    python scripts/make_turbidity_examples.py [--images_dir <dir>] [--out_dir <dir>]

--images_dir  Directory containing processed danger-map PNG outputs
              (the 3-panel composites whose left panel is the original RGB).
              Default: reports/danger_map/quick_test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.augment.turbidity import apply_turbidity

LEVELS    = [0.0, 0.25, 0.5, 0.75, 1.0]
N_IMAGES  = 3     # use at most this many source images


_TITLE_H = 30   # pixels — matches title_h in run_video.py
_BAR_H   = 22   # pixels — matches bar_h in run_video.py


def _extract_original_panel(composite: np.ndarray) -> np.ndarray:
    """
    Extract the left ('Original') panel from a multi-panel danger-map composite.

    The composite has N equal-width panels side-by-side.  We assume a 2-, 3-, or
    4-panel layout and pick the leftmost panel, stripping the title bar and
    colorbar rows so we get only the raw RGB content.
    """
    H, W = composite.shape[:2]
    # Heuristic: 3-panel ≈ W/H > 2.5; 4-panel ≈ W/H > 3.5; 2-panel ≈ W/H > 1.4
    aspect = W / H
    if   aspect > 3.5:
        n_panels = 4
    elif aspect > 2.5:
        n_panels = 3
    elif aspect > 1.4:
        n_panels = 2
    else:
        n_panels = 1   # already a single panel

    panel_w = W // n_panels
    # Strip title bar (top) and colorbar (bottom) to keep only the RGB content
    return composite[_TITLE_H : H - _BAR_H, :panel_w].copy()


def _make_strip(images: list[np.ndarray], labels: list[str],
                label_h: int = 24) -> np.ndarray:
    """Stack images horizontally with a text label above each panel."""
    H, W = images[0].shape[:2]
    strip = np.zeros((label_h + H, W * len(images), 3), dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    for i, (img, label) in enumerate(zip(images, labels)):
        x0 = i * W
        strip[label_h:, x0: x0 + W] = img
        cv2.putText(strip, label, (x0 + 6, label_h - 6),
                    font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return strip


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate turbidity augmentation examples on the raw input image."
    )
    ap.add_argument("--images_dir",
                    default=str(REPO_ROOT / "reports/danger_map/quick_test"),
                    help="Directory of processed danger-map PNG composites "
                         "(left panel = original RGB). Default: reports/danger_map/quick_test")
    ap.add_argument("--out_dir",
                    default=str(REPO_ROOT / "reports/turbidity/examples"),
                    help="Output directory. Default: reports/turbidity/examples")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in exts)[:N_IMAGES]
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"Source images   : {images_dir}")
    print(f"Output directory: {out_dir}")
    print()

    for img_idx, src_path in enumerate(paths, start=1):
        composite = np.array(Image.open(src_path).convert("RGB"))
        original  = _extract_original_panel(composite)

        strip_panels: list[np.ndarray] = []
        strip_labels: list[str]        = []

        for level in LEVELS:
            if level == 0.0:
                turbid = original.copy()
            else:
                turbid = apply_turbidity(original, level)

            # Individual file
            out_path = out_dir / f"level_{level:.2f}_img_{img_idx}.png"
            Image.fromarray(turbid).save(str(out_path))
            print(f"  {out_path.name}")

            strip_panels.append(turbid)
            strip_labels.append(f"level {level:.2f}")

        # Side-by-side strip
        strip = _make_strip(strip_panels, strip_labels)
        strip_path = out_dir / f"strip_img_{img_idx}.png"
        Image.fromarray(strip).save(str(strip_path))
        print(f"  {strip_path.name}  ← 5-panel strip")
        print()

    total = len(LEVELS) * len(paths)
    strips = len(paths)
    print(f"Done — {total} individual images + {strips} strips saved to {out_dir}")


if __name__ == "__main__":
    main()
