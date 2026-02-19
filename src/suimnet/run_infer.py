from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from skimage import io as skio
import skimage.transform as sktf

from src.common.config import resolve_dataset_paths

# Import SUIM-Net model from the submodule
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
SUIMNET_ROOT = REPO_ROOT / "vendor" / "SUIM-Net"
sys.path.append(str(SUIMNET_ROOT))

from model import SUIM_Net  # type: ignore


def list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])


def suimnet_rgb_from_logits(logits_hwk: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    SUIM-Net test.py uses 5 outputs in order: RO, FV, HD, RI, WR (based on their code).
    This replicates their bitwise-to-RGB composition so your results match their examples.
    """
    out = (logits_hwk > thr).astype(np.uint8)

    RO = out[..., 0].astype(bool)
    FV = out[..., 1].astype(bool)
    HD = out[..., 2].astype(bool)
    RI = out[..., 3].astype(bool)
    WR = out[..., 4].astype(bool)

    H, W = RO.shape
    r = np.zeros((H, W), dtype=bool)
    g = np.zeros((H, W), dtype=bool)
    b = np.zeros((H, W), dtype=bool)

    r |= RO
    b |= HD
    r |= RI
    r |= FV
    g |= WR
    g |= FV
    b |= WR
    b |= RI

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8) * 255
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SUIM-Net inference on a directory of images.")
    ap.add_argument("--profile", default=None, help="profile key in configs/profiles.yaml (e.g. local, greatlakes)")
    ap.add_argument("--dataset", default=None, help="dataset key in configs/datasets.yaml (e.g. suim, deepfish, lars)")
    ap.add_argument("--images_dir", default=None,
                    help="path to images directory (overrides --profile/--dataset resolution)")
    ap.add_argument("--out", default="outputs", help="output root (default: outputs)")
    ap.add_argument("--weights", default=str(SUIMNET_ROOT / "sample_test" / "ckpt_seg_5obj.hdf5"),
                    help="path to SUIM-Net weights (default: vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5)")
    ap.add_argument("--input_w", type=int, default=320)
    ap.add_argument("--input_h", type=int, default=240)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    # Resolve images directory
    if args.images_dir is not None:
        images_dir = Path(args.images_dir).resolve()
        dataset_name = args.dataset or images_dir.name
    elif args.profile is not None and args.dataset is not None:
        ds = resolve_dataset_paths(profile=args.profile, dataset=args.dataset)
        images_dir = ds.images_dir
        dataset_name = args.dataset
    else:
        ap.error("Provide either --images_dir OR both --profile and --dataset")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            f"Expected default at vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5\n"
            f"If you have it elsewhere, pass --weights /path/to/ckpt.hdf5"
        )

    out_root = Path(args.out) / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = list_images(images_dir)
    if not imgs:
        raise RuntimeError(f"No images found under: {images_dir}")

    print(f"Running SUIM-Net on {len(imgs)} images from: {images_dir}")

    model = SUIM_Net(im_res=(args.input_h, args.input_w), n_classes=5).model
    model.load_weights(str(weights_path))

    for i, p in enumerate(imgs, 1):
        img = skio.imread(str(p))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]

        img_rs = sktf.resize(
            img, (args.input_h, args.input_w, 3),
            preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

        x = np.expand_dims(img_rs, axis=0)
        y = model.predict(x, verbose=0)[0]  # H x W x 5
        rgb = suimnet_rgb_from_logits(y, thr=args.thr)

        rel = p.relative_to(images_dir)
        out_path = (out_root / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_path, rgb)

        if i % 10 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}] {rel}")

    print(f"Done. Wrote {len(imgs)} masks to: {out_root}")


if __name__ == "__main__":
    main()
