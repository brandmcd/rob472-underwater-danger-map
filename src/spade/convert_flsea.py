"""Convert the FLSea-VI dataset (HuggingFace) to SPADE input format.

Source: bhowmikabhimanyu/flsea-vi  (CC-BY-NC-SA-4.0)
        https://huggingface.co/datasets/bhowmikabhimanyu/flsea-vi

The parquet files contain two columns:
  image : HuggingFace ImageObject  — RGB uint8 image bytes
  depth : HuggingFace ImageObject  — depth map image bytes

This script:
  1. Reads all validation-*.parquet files in raw_dir/data/
  2. Decodes each image/depth pair from parquet bytes
  3. Saves RGB as float32 TIFF (SPADE format)
  4. Saves depth as float32 TIFF in metres
     (16-bit PNG → mm → m; float images kept as-is)
  5. Generates sparse depth CSV via Shi-Tomasi corner sampling
     (coordinates in 240 × 320 space to match SPADE config)
  6. Writes an absolute-path filenames list for run_eval.py

Usage:
    python -m src.spade.convert_flsea \\
        --raw_dir  $DATA_ROOT/flsea/raw \\
        --out_dir  $DATA_ROOT/flsea/spade \\
        --filenames_out data/spade_lists/flsea_test.txt

    # Smoke-test with 20 images:
    python -m src.spade.convert_flsea \\
        --raw_dir  $DATA_ROOT/flsea/raw \\
        --out_dir  /tmp/flsea_spade \\
        --filenames_out /tmp/flsea_test.txt \\
        --max_images 20
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tifffile
from PIL import Image

from ._spade_utils import generate_sparse_csv

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def _decode_hf_image(cell) -> Image.Image | None:
    """Decode a HuggingFace ImageObject cell from a parquet row.

    HuggingFace stores images as a struct: {'bytes': <bytes>, 'path': <str>}
    or as a raw bytes object.
    """
    if cell is None:
        return None
    if isinstance(cell, dict):
        raw = cell.get("bytes")
    elif hasattr(cell, "as_py"):
        cell = cell.as_py()
        raw = cell.get("bytes") if isinstance(cell, dict) else cell
    else:
        raw = cell
    if raw is None:
        return None
    return Image.open(io.BytesIO(raw))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert FLSea-VI HuggingFace parquet dataset to SPADE format."
    )
    ap.add_argument("--raw_dir", required=True,
                    help="Directory containing data/validation-*.parquet files")
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

    parquet_files = sorted((raw_dir / "data").glob("validation-*.parquet"))
    if not parquet_files:
        # Fall back: accept any parquet in raw_dir
        parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: no parquet files found under {raw_dir}")
        print("Run: bash scripts/download_spade_data.sh")
        return

    print(f"Found {len(parquet_files)} parquet file(s) in {raw_dir}/data/")

    filenames_lines: list[str] = []
    n_ok  = 0
    n_err = 0
    done  = False

    for pq_path in parquet_files:
        if done:
            break
        table = pq.read_table(str(pq_path))
        n_rows = len(table)

        for row_idx in range(n_rows):
            if args.max_images and n_ok >= args.max_images:
                done = True
                break

            stem = f"flsea_{n_ok + n_err:06d}"

            # ── Decode image ──────────────────────────────────────────────
            img_cell   = table["image"][row_idx]
            depth_cell = table["depth"][row_idx]

            pil_img   = _decode_hf_image(img_cell)
            pil_depth = _decode_hf_image(depth_cell)

            if pil_img is None or pil_depth is None:
                n_err += 1
                continue

            pil_img = pil_img.convert("RGB")

            # ── Depth → float32 metres ────────────────────────────────────
            depth = np.array(pil_depth, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]   # take first channel if RGB-encoded
            # 16-bit PNG heuristic: values >> 200 suggest mm → convert to m
            if depth.max() > 200:
                depth = depth / 1000.0

            # ── RGB → TIFF ────────────────────────────────────────────────
            rgb_tiff = rgb_out / f"{stem}.tiff"
            pil_img.save(str(rgb_tiff))

            # ── Depth → float32 TIFF ──────────────────────────────────────
            gt_tiff = gt_out / f"{stem}.tif"
            tifffile.imwrite(str(gt_tiff), depth.astype(np.float32))

            # ── Sparse CSV ────────────────────────────────────────────────
            img_bgr    = np.array(pil_img)[:, :, ::-1]
            sparse_csv = sparse_out / f"{stem}_features.csv"
            n_pts = generate_sparse_csv(img_bgr, depth, sparse_csv)
            if n_pts == 0:
                print(f"  WARNING: no sparse points for {stem} — skipping")
                n_err += 1
                continue

            filenames_lines.append(
                f"{rgb_tiff.resolve()} {gt_tiff.resolve()} {sparse_csv.resolve()}"
            )
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  Converted {n_ok} images ...")

    ff_path = Path(args.filenames_out)
    ff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ff_path, "w") as f:
        f.write("\n".join(filenames_lines) + "\n")

    print(f"\nDone. {n_ok} images converted ({n_err} skipped).")
    print(f"Filenames list → {ff_path}")
    print(f"\nNext step:")
    print(f"  python -m src.spade.run_eval \\")
    print(f"      --dataset flsea \\")
    print(f"      --weights $WEIGHTS_PATH \\")
    print(f"      --filenames_file {ff_path.resolve()} \\")
    print(f"      --save_image")


if __name__ == "__main__":
    main()
