"""Convert the Katherine Skinner (DROP Lab) HIMB stereo dataset to SPADE format.

Source: https://github.com/kskin/data
Images : http://www.umich.edu/~dropopen2/DROPUWStereo_HIMB1_docksite.tar.gz
GT depth: http://www.umich.edu/~dropopen2/DROPUWStereo_HIMB_ground.tar.gz

The HIMB datasets are BlueROV2 underwater stereo recordings from Hawaii.
Ground-truth depth comes from stereo matching and is stored per-image.

This script:
  1. Pairs image files with ground-truth depth files by shared stem / numeric ID
  2. Converts depth to float32 metres (handles NPY, TIFF, and 16-bit PNG)
  3. Generates sparse depth CSV via Shi-Tomasi corner sampling
     (coordinates in 240 × 320 space to match SPADE config)
  4. Writes an absolute-path filenames list for run_eval.py

Expected layout after extracting both archives:
    img_dir/   (DROPUWStereo_HIMB1_docksite/)
      <id>.jpg  (or .png)
    gt_dir/    (DROPUWStereo_HIMB_ground/)
      <id>_depth.npy  (or .tif / .png — format depends on archive version)

The numeric-ID matching handles common kskin archive naming conventions.

Usage:
    python -m src.spade.convert_kskin \\
        --img_dir $DATA_ROOT/kskin/HIMB1 \\
        --gt_dir  $DATA_ROOT/kskin/HIMB_ground \\
        --out_dir $DATA_ROOT/kskin/spade \\
        --filenames_out data/spade_lists/kskin_test.txt

    # Smoke-test with 10 images:
    python -m src.spade.convert_kskin \\
        --img_dir $DATA_ROOT/kskin/HIMB1 \\
        --gt_dir  $DATA_ROOT/kskin/HIMB_ground \\
        --out_dir /tmp/kskin_spade \\
        --filenames_out /tmp/kskin_test.txt \\
        --max_images 10
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from ._spade_utils import DEPTH_EXTS, load_depth, generate_sparse_csv

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


# ── Pairing helpers ───────────────────────────────────────────────────────────

def find_pairs(img_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    """Match image files to ground-truth depth files by stem or numeric ID.

    Tries exact stem match first, then falls back to sharing the same
    leading numeric sequence (common in HIMB archive naming).
    """
    # Build depth lookup: stem → path, numeric-id → path
    depth_map: dict[str, Path] = {}
    for p in sorted(gt_dir.rglob("*")):
        if p.suffix.lower() in DEPTH_EXTS:
            depth_map[p.stem] = p
            m = re.search(r"\d+", p.stem)
            if m:
                depth_map.setdefault(m.group(), p)

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue
        stem = img_path.stem
        depth_path = depth_map.get(stem)
        if depth_path is None:
            m = re.search(r"\d+", stem)
            if m:
                depth_path = depth_map.get(m.group())
        if depth_path is not None:
            pairs.append((img_path, depth_path))

    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert kskin DROP Lab stereo dataset to SPADE format."
    )
    ap.add_argument("--img_dir", required=True,
                    help="Directory of extracted HIMB image archive")
    ap.add_argument("--gt_dir", required=True,
                    help="Directory of extracted HIMB ground-truth depth archive")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for converted SPADE data")
    ap.add_argument("--filenames_out", required=True,
                    help="Path to write SPADE filenames list (absolute paths)")
    ap.add_argument("--max_images", type=int, default=None,
                    help="Cap the number of images converted (smoke-test)")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    gt_dir  = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    rgb_out = out_dir / "rgb"
    gt_out  = out_dir / "gt_depth"
    sp_out  = out_dir / "sparse_csv"
    for d in (rgb_out, gt_out, sp_out):
        d.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(img_dir, gt_dir)
    print(f"Found {len(pairs)} image/depth pairs")
    if not pairs:
        print("ERROR: no pairs found.  Check --img_dir and --gt_dir layouts.")
        return

    if args.max_images:
        pairs = pairs[: args.max_images]
        print(f"Capped to {len(pairs)} images (--max_images {args.max_images})")

    filenames_lines: list[str] = []
    n_ok = 0
    for img_path, depth_path in pairs:
        stem  = img_path.stem
        depth = load_depth(depth_path, squeeze=True)
        if depth is None:
            continue

        # RGB → TIFF (keep PIL image in memory for corner detection)
        pil_img  = Image.open(img_path).convert("RGB")
        rgb_tiff = rgb_out / f"{stem}.tiff"
        pil_img.save(str(rgb_tiff))

        # Depth → float32 TIFF
        gt_tiff = gt_out / f"{stem}.tif"
        tifffile.imwrite(str(gt_tiff), depth.astype(np.float32))

        # Sparse CSV (BGR numpy — avoids re-reading image; resize_depth for stereo)
        img_bgr    = np.array(pil_img)[:, :, ::-1]
        sparse_csv = sp_out / f"{stem}_features.csv"
        n_pts = generate_sparse_csv(img_bgr, depth, sparse_csv, resize_depth=True)
        if n_pts == 0:
            print(f"  WARNING: no sparse points for {stem} — skipping")
            continue

        filenames_lines.append(
            f"{rgb_tiff.resolve()} {gt_tiff.resolve()} {sparse_csv.resolve()}"
        )
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  Converted {n_ok}/{len(pairs)} ...")

    ff_path = Path(args.filenames_out)
    ff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ff_path, "w") as f:
        f.write("\n".join(filenames_lines) + "\n")

    print(f"\nDone. {n_ok}/{len(pairs)} images converted.")
    print(f"Filenames list → {ff_path}")
    print(f"\nNext step:")
    print(f"  python -m src.spade.run_eval \\")
    print(f"      --dataset kskin \\")
    print(f"      --weights $WEIGHTS_PATH \\")
    print(f"      --filenames_file {ff_path.resolve()} \\")
    print(f"      --save_image")


if __name__ == "__main__":
    main()
