"""
Generate 15 individual turbidity example images.

3 source images × 5 turbidity levels (0.0, 0.25, 0.5, 0.75, 1.0) = 15 PNGs.

Output: reports/turbidity/examples/level_<X.XX>_img_<N>.png

Usage:
    python scripts/make_turbidity_examples.py [--images_dir <dir>] [--out_dir <dir>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.augment.turbidity import apply_turbidity

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_IMAGES = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",
                    default=str(REPO_ROOT / "reports/danger_map/quick_test"))
    ap.add_argument("--out_dir",
                    default=str(REPO_ROOT / "reports/turbidity/examples"))
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)[:N_IMAGES]
    if not paths:
        raise FileNotFoundError(f"No images in {images_dir}")

    imgs = [np.array(Image.open(p).convert("RGB")) for p in paths]

    for level in LEVELS:
        for i, img in enumerate(imgs):
            aug = apply_turbidity(img, level)
            out_path = out_dir / f"level_{level:.2f}_img_{i+1}.png"
            Image.fromarray(aug).save(str(out_path))
            print(f"  {out_path.name}")

    print(f"\nDone — {len(LEVELS) * len(imgs)} images saved to {out_dir}")


if __name__ == "__main__":
    main()
