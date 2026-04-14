"""Generate SPADE depth visualisation demos for SUIM sample images.

SUIM images have no depth ground truth, so SPADE is run in monocular-only
mode (zero sparse-depth hints).  The output format matches the existing
flsea_samples demos: RGB on the left, predicted depth on the right.

Usage (from repo root, with the rob472-spade venv active):
    python scripts/make_suim_spade_demos.py \\
        --weights ~/Downloads/underwater_depth_pipeline.pt

    # Or override defaults:
    python scripts/make_suim_spade_demos.py \\
        --weights /path/to/underwater_depth_pipeline.pt \\
        --images_dir vendor/SUIM-Net/sample_test/images \\
        --out reports/spade/suim_samples
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT    = Path(__file__).resolve().parents[1]
VENDOR_SPADE = REPO_ROOT / "vendor" / "SPADE"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# SPADE's DPT decoder outputs (H//14)*14 × (W//14)*14, so the input must
# be a multiple of patch_size=14 to avoid a size mismatch with the sparse map.
# We pick the nearest multiples-of-14 below the config values (608 × 968).
_PATCH = 14
SPADE_H = (608 // _PATCH) * _PATCH  # 43 × 14 = 602
SPADE_W = (968 // _PATCH) * _PATCH  # 69 × 14 = 966

_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
_to_tensor = transforms.ToTensor()  # PIL → (C, H, W) float [0, 1]
_resize    = transforms.Resize((SPADE_H, SPADE_W), antialias=True)


def preprocess_image(
    img_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and normalise an RGB image for SPADE.

    Returns:
        tensor : (1, 3, SPADE_H, SPADE_W) float32, ImageNet-normalised, on *device*
        rgb_np : (H, W, 3) uint8 original resolution for matplotlib visualisation
    """
    img    = Image.open(img_path).convert("RGB")
    rgb_np = np.array(img)
    tensor = _normalize(_resize(_to_tensor(img))).unsqueeze(0).to(device)
    return tensor, rgb_np


def save_demo(
    rgb_np:   np.ndarray,
    depth_np: np.ndarray,
    out_path: Path,
    *,
    cmap: str = "viridis_r",
    dpi:  int = 200,
) -> None:
    """Save a side-by-side RGB / predicted-depth visualisation."""
    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(
        nrows=1, ncols=3,
        width_ratios=[1, 1, 0.05],
        wspace=0.05, hspace=0.05,
    )

    # Left: original RGB (no sparse points — SUIM has no real depth hints)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_np)
    ax1.set_title("RGB + sparse points")
    ax1.axis("off")

    # Middle: predicted depth
    vmin = float(np.nanpercentile(depth_np, 2))
    vmax = float(np.nanpercentile(depth_np, 98))
    ax2  = fig.add_subplot(gs[0, 1])
    im   = ax2.imshow(depth_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title("Predicted depth")
    ax2.axis("off")

    # Right: colorbar
    cax = fig.add_subplot(gs[0, 2])
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Depth (m)", fontsize=10, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def load_model(weights_path: Path, device: torch.device):
    """Build the SPADE model and load weights onto *device*."""
    sys.path.insert(0, str(VENDOR_SPADE))
    os.chdir(VENDOR_SPADE)

    from UnderwaterDepth.utils.config import get_config
    from UnderwaterDepth.models.builder import build_model

    config = get_config(
        "SPADE", "eval", "flsea_sparse_feature",
        pretrained_resource="",
        da_pretrained_resource="",
    )
    model = build_model(config)

    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate SPADE depth demos for SUIM sample images."
    )
    ap.add_argument(
        "--weights",
        default=str(Path.home() / "Downloads" / "underwater_depth_pipeline.pt"),
        help="Path to SPADE .pt weights file",
    )
    ap.add_argument(
        "--images_dir",
        default=str(REPO_ROOT / "vendor" / "SUIM-Net" / "sample_test" / "images"),
        help="Directory of input RGB images",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "reports" / "spade" / "suim_samples"),
        help="Output directory for demo images",
    )
    args = ap.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    images_dir   = Path(args.images_dir).resolve()
    out_dir      = Path(args.out).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Run scripts/build_spade_weights.py first, or provide --weights."
        )
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Weights: {weights_path}")
    print(f"Input  : {images_dir}")
    print(f"Output : {out_dir}\n")

    print("Loading SPADE model...")
    model = load_model(weights_path, device)
    print("Model ready.\n")

    exts   = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")
    print(f"Found {len(images)} images.\n")

    with torch.no_grad():
        for img_path in images:
            print(f"  {img_path.name}", end=" ... ", flush=True)

            img_tensor, rgb_np = preprocess_image(img_path, device)

            # Zero sparse-depth map — SUIM has no real depth measurements.
            # SPADE falls back to pure monocular (DA V2 backbone) prediction.
            B, C, H, W = img_tensor.shape
            sparse_map = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)

            pred       = model(img_tensor, prompt_depth=sparse_map, fx=None, cx=None)
            depth_pred = pred["metric_depth"].squeeze().cpu().numpy()  # (H, W)

            out_path = out_dir / (img_path.stem + ".png")
            save_demo(rgb_np, depth_pred, out_path)
            print(f"saved → {out_path.name}")

    print(f"\nDone. {len(images)} demo images in {out_dir}")


if __name__ == "__main__":
    main()
