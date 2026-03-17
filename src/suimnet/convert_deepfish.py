"""
Prepare DeepFish ground-truth masks for use with metric_calc.py.

DeepFish ships binary fish-presence masks alongside each image.
Since all masks correspond to the FV (Fish / Vertebrate) SUIM class,
this script copies (or symlinks) them into the expected per-class
subdirectory layout:

    <out>/
        FV/<image_stem>.png

Only the FV class has ground truth in DeepFish. metric_calc.py will
skip RO, HD, RI, WR automatically when their subdirectories are absent.

DeepFish archive layout (after extraction):
    DeepFish/
        Fish/
            Segmentation/
                valid/
                    images/    ← RGB frames  (jpg)
                    masks/     ← binary masks (png, 255=fish)
                test/
                    images/
                    masks/

Usage:
    python -m src.suimnet.convert_deepfish \\
        --masks_src /path/to/DeepFish/Fish/Segmentation/valid/masks \\
        --out       /path/to/data/deepfish/masks

    # Great Lakes (set data_root in configs/profiles.yaml first):
    python -m src.suimnet.convert_deepfish \\
        --profile greatlakes --split valid

    python -m src.suimnet.convert_deepfish \\
        --profile greatlakes --split test
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.common.config import resolve_dataset_paths


def convert(masks_src: Path, out_dir: Path) -> None:
    fv_dir = out_dir / "FV"
    fv_dir.mkdir(parents=True, exist_ok=True)

    masks = sorted(masks_src.glob("*.png")) + sorted(masks_src.glob("*.bmp"))
    if not masks:
        print(f"ERROR: No PNG/BMP masks found in {masks_src}", file=sys.stderr)
        sys.exit(1)

    print(f"Source masks : {masks_src}  ({len(masks)} files)")
    print(f"Output FV/   : {fv_dir}")

    for src in masks:
        dst = fv_dir / (src.stem + ".png")
        shutil.copy2(src, dst)

    print(f"Done. {len(masks)} masks → {fv_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare DeepFish masks as SUIM-style per-class binary masks (FV only)."
    )
    ap.add_argument("--masks_src", default=None,
                    help="Path to the raw DeepFish masks directory")
    ap.add_argument("--out", default=None,
                    help="Output directory for SUIM-format masks")
    ap.add_argument("--profile", default=None,
                    help="Profile key (e.g. greatlakes) — used with --split")
    ap.add_argument("--split", default="valid", choices=["valid", "test"],
                    help="DeepFish split to convert (default: valid)")
    args = ap.parse_args()

    if args.profile:
        ds = resolve_dataset_paths(profile=args.profile, dataset="deepfish")
        data_root = ds.images_dir.parent.parent   # .../data/deepfish/
        masks_src = data_root / "Fish" / "Segmentation" / args.split / "masks"
        out_dir   = ds.labels_dir
        if out_dir is None:
            print("ERROR: labels_rel not configured for deepfish in datasets.yaml", file=sys.stderr)
            sys.exit(1)
    else:
        if not (args.masks_src and args.out):
            ap.error("Provide either --profile or both --masks_src and --out")
        masks_src = Path(args.masks_src)
        out_dir   = Path(args.out)

    if not masks_src.exists():
        print(f"ERROR: Masks directory not found: {masks_src}", file=sys.stderr)
        sys.exit(1)

    convert(masks_src, out_dir)


if __name__ == "__main__":
    main()
