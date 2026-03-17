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

# ── Keras 2.13+ compatibility shim ───────────────────────────────────────────
# vendor/SUIM-Net/model.py uses `from keras.models import Input, Model` and
# `Model(input=..., output=...)` — APIs that were removed in Keras 2.13+.
# Patch the keras.models namespace before importing model.py.
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

# Channel order in SUIM-Net's 5-channel sigmoid output
CLASS_ORDER = ["RO", "FV", "HD", "RI", "WR"]


def parse_thresholds(thresholds_str: str | None, default: float,
                     base: dict[str, float] | None = None) -> dict[str, float]:
    """
    Build a per-class threshold dict.

    base     : starting values (e.g. from datasets.yaml)
    default  : fallback for any class not in base or thresholds_str
    thresholds_str : CLI override, e.g. "FV=0.3,RI=0.7,WR=0.65"
    """
    result = {cls: default for cls in CLASS_ORDER}
    if base:
        result.update(base)
    if thresholds_str:
        for part in thresholds_str.split(","):
            part = part.strip()
            if "=" not in part:
                continue
            cls, val = part.split("=", 1)
            result[cls.upper()] = float(val)
    return result


def list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])


def suimnet_rgb_from_logits(logits_hwk: np.ndarray,
                            thresholds: dict[str, float]) -> np.ndarray:
    """
    Apply per-class thresholds to raw sigmoid logits and compose an RGB mask.

    SUIM-Net test.py uses 5 outputs in order: RO, FV, HD, RI, WR.
    This replicates their bitwise-to-RGB composition so results match examples.
    """
    RO = logits_hwk[..., 0] > thresholds["RO"]
    FV = logits_hwk[..., 1] > thresholds["FV"]
    HD = logits_hwk[..., 2] > thresholds["HD"]
    RI = logits_hwk[..., 3] > thresholds["RI"]
    WR = logits_hwk[..., 4] > thresholds["WR"]

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
    ap.add_argument("--dataset", default=None, help="dataset key in configs/datasets.yaml (e.g. suim, deepfish, usis10k)")
    ap.add_argument("--images_dir", default=None,
                    help="path to images directory (overrides --profile/--dataset resolution)")
    ap.add_argument("--out", default="outputs", help="output root (default: outputs)")
    ap.add_argument("--weights", default=str(SUIMNET_ROOT / "sample_test" / "ckpt_seg_5obj.hdf5"),
                    help="path to SUIM-Net weights (default: vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5)")
    ap.add_argument("--input_w", type=int, default=320)
    ap.add_argument("--input_h", type=int, default=240)
    ap.add_argument("--thr", type=float, default=0.5,
                    help="Global sigmoid threshold for all classes (default: 0.5). "
                         "Per-class overrides via --thresholds take precedence.")
    ap.add_argument("--thresholds", default=None,
                    help="Per-class sigmoid threshold overrides, e.g. 'FV=0.3,RI=0.7,WR=0.65'. "
                         "Unspecified classes use --thr (or dataset default from datasets.yaml).")
    ap.add_argument("--save_logits", action="store_true",
                    help="Save raw sigmoid outputs as NPZ files alongside predictions. "
                         "Enables post-hoc threshold sweeping without re-running inference.")
    args = ap.parse_args()

    # Resolve images directory and dataset-level threshold defaults
    dataset_thresholds: dict[str, float] = {}
    if args.images_dir is not None:
        images_dir = Path(args.images_dir).resolve()
        dataset_name = args.dataset or images_dir.name
    elif args.profile is not None and args.dataset is not None:
        ds = resolve_dataset_paths(profile=args.profile, dataset=args.dataset)
        images_dir = ds.images_dir
        dataset_name = args.dataset
        dataset_thresholds = ds.thresholds          # from datasets.yaml
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

    # Build final per-class threshold dict: dataset defaults → CLI --thresholds
    thresholds = parse_thresholds(args.thresholds, args.thr, base=dataset_thresholds)

    out_root = Path(args.out) / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    logits_root = out_root / "logits" if args.save_logits else None
    if logits_root is not None:
        logits_root.mkdir(parents=True, exist_ok=True)

    imgs = list_images(images_dir)
    if not imgs:
        raise RuntimeError(f"No images found under: {images_dir}")

    print(f"Running SUIM-Net on {len(imgs)} images from: {images_dir}")
    print(f"Thresholds: { {k: thresholds[k] for k in CLASS_ORDER} }")
    if logits_root:
        print(f"Saving logits to: {logits_root}")
    print()

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
            preserve_range=False, anti_aliasing=True
        )  # preserve_range=False normalizes to [0,1] like the original

        x = np.expand_dims(img_rs, axis=0)
        y = model.predict(x, verbose=0)[0]  # H x W x 5

        rgb = suimnet_rgb_from_logits(y, thresholds)

        rel = p.relative_to(images_dir)
        out_path = (out_root / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_path, rgb)

        if logits_root is not None:
            logits_path = (logits_root / rel).with_suffix(".npz")
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(logits_path), logits=y.astype(np.float16))

        if i % 10 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}] {rel}")

    print(f"\nDone. Wrote {len(imgs)} masks to: {out_root}")
    if logits_root:
        print(f"     Logits saved to:  {logits_root}")
        print(f"     Run threshold sweep: python -m src.suimnet.threshold_sweep "
              f"--profile {args.profile or '<profile>'} --dataset {dataset_name} "
              f"--logits_dir {logits_root}")


if __name__ == "__main__":
    main()
