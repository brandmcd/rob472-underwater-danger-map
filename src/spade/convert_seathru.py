"""Convert the SeaThru dataset (Kaggle) to SPADE input format.

SeaThru (Akkaynak & Treibitz, CVPR 2019) contains underwater RGB images
and SFM-reconstructed dense depth maps (metres).  This script:

  1. Scans the raw directory for paired RGB / depth files
  2. Saves depth as float32 TIFF (SPADE format)
  3. Generates a sparse depth CSV by sampling the dense depth at
     Shi-Tomasi corner locations — coordinates in 240 × 320 space
     (matches SPADE's sparse_feature_height / sparse_feature_width)
  4. Writes an absolute-path filenames list for run_eval.py

Expected raw directory layout (as delivered by the Kaggle download):
    <raw_dir>/
      D1/D1/linearPNG/*.png            ← RGB images
      D1/D1/depth/depthT_*.tif         ← depth, metres, float32
      D3/D3/linearPNG/*.png
      D3/D3/depth_resized/*.tif
      D4/ ...

RGB and depth are paired by numeric ID extracted from the filenames
(e.g. S03082.png ↔ depthT_S03082.tif) within the same scene directory.

Usage:
    python -m src.spade.convert_seathru \\
        --raw_dir  $DATA_ROOT/seathru/raw \\
        --out_dir  $DATA_ROOT/seathru/spade \\
        --filenames_out data/spade_lists/seathru_test.txt

    # Smoke-test with 10 images:
    python -m src.spade.convert_seathru \\
        --raw_dir  $DATA_ROOT/seathru/raw \\
        --out_dir  /tmp/seathru_spade \\
        --filenames_out /tmp/seathru_test.txt \\
        --max_images 10
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from ._spade_utils import load_depth, generate_sparse_csv


# ── Pairing helpers ───────────────────────────────────────────────────────────

def find_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Return (rgb_path, depth_path) pairs by scanning raw_dir recursively.

    Handles two common Kaggle SeaThru layouts:
      Nested:  raw/D1/D1/linearPNG/S03082.png  +  raw/D1/D1/depth/depthT_S03082.tif
      Flat:    raw/D1/linearPNG/S03082.png      +  raw/D1/depth/depthT_S03082.tif

    Pairing logic: extract the numeric ID from each filename and match RGB ↔ depth
    by numeric ID within the same ancestor directory (tried at two levels so both
    layouts are handled without knowing which one is present on disk).

    Both .tif and .tiff depth extensions are accepted.
    """
    # Collect all depth files (.tif and .tiff)
    depth_files = sorted(raw_dir.rglob("*.tif")) + sorted(raw_dir.rglob("*.tiff"))

    if not depth_files:
        print(f"  DEBUG: no .tif/.tiff files found anywhere under {raw_dir}")

    # Index depth files by (ancestor_dir, numeric_id) at two parent levels.
    # This handles both flat (D1/depth/…) and nested (D1/D1/depth/…) layouts.
    depth_index: dict[tuple[str, str], Path] = {}
    for tif in depth_files:
        m = re.search(r"(\d+)", tif.stem)
        if not m:
            continue
        numeric_id = m.group()
        # Level 1: parent of the depth/ folder  (flat layout: D1/)
        # Level 2: grandparent                   (nested layout: D1/D1/)
        for ancestor in (tif.parent.parent, tif.parent.parent.parent):
            key = (str(ancestor), numeric_id)
            if key not in depth_index:
                depth_index[key] = tif

    # Collect RGB PNGs (skip anything inside a depth/ folder)
    rgb_files = [
        p for p in sorted(raw_dir.rglob("*.png"))
        if "depth" not in p.parent.name.lower()
    ]

    if not rgb_files:
        print(f"  DEBUG: no .png RGB files found anywhere under {raw_dir}")

    pairs: list[tuple[Path, Path]] = []
    for png in rgb_files:
        m = re.search(r"(\d+)", png.stem)
        if not m:
            continue
        numeric_id = m.group()
        # Try matching at the same two ancestor levels as depth indexing
        for ancestor in (png.parent.parent, png.parent.parent.parent):
            key = (str(ancestor), numeric_id)
            if key in depth_index:
                pairs.append((png, depth_index[key]))
                break

    # Diagnostic output when pairing fails so it's easy to fix
    if not pairs and depth_files and rgb_files:
        print("  DEBUG: found depth files and RGB files but could not pair them.")
        print(f"  Sample depth paths : {[str(p) for p in depth_files[:3]]}")
        print(f"  Sample RGB paths   : {[str(p) for p in rgb_files[:3]]}")
        print("  Check that numeric IDs in filenames match across RGB/depth pairs.")

    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert SeaThru dataset to SPADE format."
    )
    ap.add_argument("--raw_dir", required=True,
                    help="Root directory of the Kaggle SeaThru download")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for converted SPADE data")
    ap.add_argument("--filenames_out", required=True,
                    help="Path to write the SPADE filenames list (absolute paths)")
    ap.add_argument("--max_images", type=int, default=None,
                    help="Cap the number of images converted (smoke-test)")
    args = ap.parse_args()

    raw_dir    = Path(args.raw_dir)
    out_dir    = Path(args.out_dir)
    rgb_out    = out_dir / "rgb"
    gt_out     = out_dir / "gt_depth"
    sparse_out = out_dir / "sparse_csv"
    for d in (rgb_out, gt_out, sparse_out):
        d.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(raw_dir)
    print(f"Found {len(pairs)} RGB/depth pairs in {raw_dir}")
    if not pairs:
        print("ERROR: no pairs found. Check --raw_dir and verify SeaThru layout.")
        raise SystemExit(1)

    if args.max_images:
        pairs = pairs[: args.max_images]
        print(f"Capped to {len(pairs)} images (--max_images {args.max_images})")

    filenames_lines: list[str] = []
    n_ok = 0
    for rgb_path, depth_path in pairs:
        stem  = rgb_path.stem
        depth = load_depth(depth_path)
        if depth is None:
            continue

        # ── RGB → TIFF ────────────────────────────────────────────────────────
        pil_img  = Image.open(rgb_path).convert("RGB")
        rgb_tiff = rgb_out / f"{stem}.tiff"
        pil_img.save(str(rgb_tiff))

        # ── Depth → float32 TIFF ──────────────────────────────────────────────
        gt_tiff = gt_out / f"{stem}.tif"
        tifffile.imwrite(str(gt_tiff), depth.astype(np.float32))

        # ── Sparse CSV (BGR numpy — avoids re-reading image from disk) ────────
        img_bgr    = np.array(pil_img)[:, :, ::-1]
        sparse_csv = sparse_out / f"{stem}_features.csv"
        n_pts = generate_sparse_csv(img_bgr, depth, sparse_csv)
        if n_pts == 0:
            print(f"  WARNING: no sparse points for {stem} — skipping")
            continue

        # Absolute paths; run_eval.py passes data_path_eval="/" so the data
        # loader strips the leading "/" and prepends "/" → original abs path.
        filenames_lines.append(
            f"{rgb_tiff.resolve()} {gt_tiff.resolve()} {sparse_csv.resolve()}"
        )
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  Converted {n_ok}/{len(pairs)} ...")

    # ── Filenames list ────────────────────────────────────────────────────────
    ff_path = Path(args.filenames_out)
    ff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ff_path, "w") as f:
        f.write("\n".join(filenames_lines) + "\n")

    print(f"\nDone. {n_ok}/{len(pairs)} images converted.")
    print(f"Filenames list → {ff_path}")
    print(f"\nNext step:")
    print(f"  python -m src.spade.run_eval \\")
    print(f"      --dataset seathru \\")
    print(f"      --weights $WEIGHTS_PATH \\")
    print(f"      --filenames_file {ff_path.resolve()} \\")
    print(f"      --save_image")


if __name__ == "__main__":
    main()
