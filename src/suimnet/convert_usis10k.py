"""
Convert USIS10K COCO-format annotations to SUIM-style per-class binary PNG masks.

USIS10K has 7 classes that map onto the 5 SUIM-Net output classes:

    USIS10K category          → SUIM class
    ─────────────────────────────────────
    robots                    → RO  (Robot / Instrument)
    fish                      → FV  (Fish / Vertebrate)
    human_divers              → HD  (Human Diver)
    reefs / sea_floor         → RI  (Reef / Invertebrate)
    wrecks_ruins              → WR  (Wreck / Ruin)
    aquatic_plants            → RI  (mapped to Reef / Invertebrate)

Output structure mirrors what metric_calc.py expects:
    <masks_dir>/
        RO/<image_stem>.png   ← binary mask, 255 = foreground
        FV/<image_stem>.png
        HD/<image_stem>.png
        RI/<image_stem>.png
        WR/<image_stem>.png

Usage:
    python -m src.suimnet.convert_usis10k \\
        --ann  /path/to/usis10k/multi_class_annotations/multi_class_test_annotations.json \\
        --img_dir /path/to/usis10k/test \\
        --out  /path/to/usis10k/masks

    # Local shorthand (edit profiles.yaml first):
    python -m src.suimnet.convert_usis10k \\
        --profile local --dataset usis10k

    # Great Lakes:
    python -m src.suimnet.convert_usis10k \\
        --profile greatlakes --dataset usis10k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as coco_mask
    _HAS_COCO = True
except ImportError:
    _HAS_COCO = False

from src.common.config import resolve_dataset_paths

# ── Class mapping ─────────────────────────────────────────────────────────────
# USIS10K category name → SUIM class code
# Category names are lowercase and may vary slightly — we do a substring match.
CATEGORY_MAP: dict[str, str] = {
    "robot":         "RO",
    "auv":           "RO",
    "rov":           "RO",
    "fish":          "FV",
    "turtle":        "FV",
    "vertebrate":    "FV",
    "diver":         "HD",
    "human":         "HD",
    "reef":          "RI",
    "coral":         "RI",
    "sea_floor":     "RI",
    "seafloor":      "RI",
    "floor":         "RI",
    "plant":         "RI",
    "algae":         "RI",
    "wreck":         "WR",
    "ruin":          "WR",
    "artifact":      "WR",
}
SUIM_CLASSES = ["RO", "FV", "HD", "RI", "WR"]


def map_category(name: str) -> str | None:
    """Return the SUIM class code for a USIS10K category name, or None to skip."""
    n = name.lower().replace("-", "_").replace(" ", "_")
    for key, cls in CATEGORY_MAP.items():
        if key in n:
            return cls
    return None


def rle_to_mask(rle, h: int, w: int) -> np.ndarray:
    """Decode a COCO RLE annotation to a binary boolean array (H, W)."""
    if _HAS_COCO:
        return coco_mask.decode(rle).astype(bool)
    # ── Fallback pure-Python RLE decoder ──────────────────────────────────────
    if isinstance(rle["counts"], list):
        # Uncompressed RLE
        counts = rle["counts"]
    else:
        # Compressed RLE string — requires pycocotools; raise a clear error
        raise RuntimeError(
            "Compressed RLE detected but pycocotools is not installed.\n"
            "Install it with:  pip install pycocotools"
        )
    mask = np.zeros(h * w, dtype=bool)
    idx, val = 0, False
    for c in counts:
        mask[idx: idx + c] = val
        idx += c
        val = not val
    return mask.reshape(h, w, order="F")


def polygon_to_mask(segmentation: list, h: int, w: int) -> np.ndarray:
    """Rasterise a COCO polygon annotation to a binary boolean array (H, W)."""
    from PIL import ImageDraw
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for poly in segmentation:
        xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(xy, fill=255)
    return np.array(img) > 0


def convert(ann_path: Path, img_dir: Path, out_dir: Path) -> None:
    print(f"Annotation : {ann_path}")
    print(f"Images     : {img_dir}")
    print(f"Output     : {out_dir}")
    print()

    with ann_path.open() as f:
        coco = json.load(f)

    # Build lookup tables
    images   = {img["id"]: img for img in coco["images"]}
    cats     = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_suim = {cat_id: map_category(name) for cat_id, name in cats.items()}

    unmapped = {name for cid, name in cats.items() if cat_suim[cid] is None}
    if unmapped:
        print(f"WARNING: unmapped USIS10K categories (will be skipped): {unmapped}")

    # Group annotations by image
    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    # Create output class directories
    for cls in SUIM_CLASSES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    n_processed = 0
    for img_id, anns in by_image.items():
        meta = images[img_id]
        h, w = meta["height"], meta["width"]
        stem = Path(meta["file_name"]).stem

        # Accumulate per-class binary masks
        class_masks: dict[str, np.ndarray] = {cls: np.zeros((h, w), dtype=bool)
                                               for cls in SUIM_CLASSES}

        for ann in anns:
            suim_cls = cat_suim.get(ann["category_id"])
            if suim_cls is None:
                continue

            seg = ann.get("segmentation", {})
            if isinstance(seg, dict):                      # RLE
                m = rle_to_mask(seg, h, w)
            elif isinstance(seg, list) and len(seg) > 0:  # polygon
                m = polygon_to_mask(seg, h, w)
            else:
                continue

            class_masks[suim_cls] |= m

        # Save one binary PNG per class
        for cls, mask in class_masks.items():
            arr = (mask.astype(np.uint8)) * 255
            Image.fromarray(arr, mode="L").save(out_dir / cls / f"{stem}.png")

        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  {n_processed}/{len(by_image)} images processed...")

    print(f"\nDone. {n_processed} images → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert USIS10K COCO annotations to SUIM-style binary masks."
    )
    ap.add_argument("--ann", default=None,
                    help="Path to COCO JSON annotation file "
                         "(e.g. multi_class_test_annotations.json)")
    ap.add_argument("--img_dir", default=None,
                    help="Directory containing the test images")
    ap.add_argument("--out", default=None,
                    help="Output directory for binary masks")
    ap.add_argument("--profile", default=None,
                    help="Profile key (e.g. greatlakes) — used with --dataset")
    ap.add_argument("--dataset", default="usis10k",
                    help="Dataset key in configs/datasets.yaml (default: usis10k)")
    args = ap.parse_args()

    if args.profile:
        ds = resolve_dataset_paths(profile=args.profile, dataset=args.dataset)
        img_dir = ds.images_dir
        out_dir = ds.labels_dir
        if out_dir is None:
            print("ERROR: labels_rel not set for this dataset in datasets.yaml", file=sys.stderr)
            sys.exit(1)
        # Annotation file lives next to the images dir
        ann_path = img_dir.parent.parent / "multi_class_annotations" / \
                   "multi_class_test_annotations.json"
        if not ann_path.exists():
            ann_path = img_dir.parent / "multi_class_annotations" / \
                       "multi_class_test_annotations.json"
    else:
        if not (args.ann and args.img_dir and args.out):
            ap.error("Provide either --profile/--dataset or all of --ann --img_dir --out")
        ann_path = Path(args.ann)
        img_dir  = Path(args.img_dir)
        out_dir  = Path(args.out)

    if not ann_path.exists():
        print(f"ERROR: Annotation file not found: {ann_path}", file=sys.stderr)
        print("       Pass --ann explicitly if the JSON is at a non-standard path.")
        sys.exit(1)

    convert(ann_path, img_dir, out_dir)


if __name__ == "__main__":
    main()
