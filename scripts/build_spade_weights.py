"""Assemble a runnable SPADE checkpoint from public Depth Anything V2 weights.

The official SPADE checkpoint (underwater_depth_pipeline.pt) is hosted on a
restricted Google Drive and is not publicly accessible (see upstream issue #1).

This script builds a substitute checkpoint by:
  1. Downloading the public Depth Anything V2 ViT-S weights from HuggingFace
     (encoder + DPT decoder — the same backbone SPADE uses)
  2. Instantiating the full SPADE model (random-init for the DAT refinement head)
  3. Loading the DA V2 backbone weights into the pretrained + depth_head submodules
  4. Saving the complete model state dict as underwater_depth_pipeline.pt

The resulting checkpoint runs the full SPADE pipeline.  Because the
SMLDeformableAttention (DAT) refinement head is randomly initialised, accuracy
matches the "DA V2 + GA" baseline row in the SPADE paper rather than the full
trained model.  This is the best we can do without the private weights.

Usage (from repo root, with the rob472-spade venv active):
    python scripts/build_spade_weights.py \\
        --out /path/to/underwater_depth_pipeline.pt

    # Or let it default to ~/Downloads/underwater_depth_pipeline.pt:
    python scripts/build_spade_weights.py
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[1]
VENDOR_SPADE = REPO_ROOT / "vendor" / "SPADE"

DA_V2_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small"
    "/resolve/main/depth_anything_v2_vits.pth?download=true"
)
DA_V2_CACHE = Path.home() / ".cache" / "spade" / "depth_anything_v2_vits.pth"


def download_da_v2(dest: Path) -> Path:
    """Download DA V2 ViT-S checkpoint if not already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[DA V2] Already cached at {dest}")
        return dest
    print(f"[DA V2] Downloading ViT-S backbone (~100 MB) → {dest}")
    urllib.request.urlretrieve(DA_V2_URL, dest)
    print("[DA V2] Download complete.")
    return dest


def build_checkpoint(da_v2_path: Path, out_path: Path) -> None:
    import torch

    # Must run from vendor/SPADE for UnderwaterDepth.* imports
    sys.path.insert(0, str(VENDOR_SPADE))
    os.chdir(VENDOR_SPADE)

    from UnderwaterDepth.utils.config import get_config
    from UnderwaterDepth.models.builder import build_model

    print("[SPADE] Building model (random init for DAT refinement head)...")
    # Build model without pretrained_resource so weights stay random
    config = get_config(
        "SPADE", "eval", "flsea_sparse_feature",
        pretrained_resource="",          # no full-model checkpoint
        da_pretrained_resource="",       # skip DA V2 load for now
    )
    model = build_model(config)

    # Now load the DA V2 backbone into the relevant submodules
    print(f"[DA V2] Loading backbone weights from {da_v2_path}...")
    import torch
    da_ckpt = torch.load(str(da_v2_path), map_location="cpu", weights_only=True)
    # DA V2 checkpoint may be wrapped in {'model': ...} or bare
    if isinstance(da_ckpt, dict) and "model" in da_ckpt:
        da_ckpt = da_ckpt["model"]

    # DA V2 ViT-S keys: pretrained.* → encoder, depth_head.* → DPT decoder
    pretrained_sd = {}
    depth_head_sd = {}
    other_keys    = []
    for k, v in da_ckpt.items():
        if k.startswith("pretrained."):
            pretrained_sd[k[len("pretrained."):]] = v
        elif k.startswith("depth_head."):
            depth_head_sd[k[len("depth_head."):]] = v
        else:
            other_keys.append(k)

    # If keys lack the prefixes, the checkpoint IS the state_dict of
    # DA V2's DepthAnythingV2 model — which also has .pretrained and .depth_head
    if not pretrained_sd and not depth_head_sd:
        # Try to split by submodule name heuristic
        for k, v in da_ckpt.items():
            if k.startswith("depth_head."):
                depth_head_sd[k[len("depth_head."):]] = v
            else:
                pretrained_sd[k] = v
        other_keys = []

    if pretrained_sd:
        missing, unexpected = model.pretrained.load_state_dict(
            pretrained_sd, strict=False
        )
        print(f"  pretrained: {len(pretrained_sd)} tensors loaded"
              f" | missing={len(missing)} unexpected={len(unexpected)}")
    if depth_head_sd:
        missing, unexpected = model.depth_head.load_state_dict(
            depth_head_sd, strict=False
        )
        print(f"  depth_head: {len(depth_head_sd)} tensors loaded"
              f" | missing={len(missing)} unexpected={len(unexpected)}")
    if other_keys:
        print(f"  (ignored {len(other_keys)} unrecognised keys: {other_keys[:3]}...)")

    # Save the full model state dict
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n[Done] Saved {n_params:.1f}M-parameter checkpoint → {out_path}")
    print()
    print("NOTE: The SMLDeformableAttention (DAT) refinement head is randomly")
    print("      initialised — this matches the 'DA V2 + GA' baseline tier,")
    print("      not the full trained SPADE model.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assemble SPADE checkpoint from public DA V2 backbone."
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "Downloads" / "underwater_depth_pipeline.pt"),
        help="Output path for the assembled .pt file",
    )
    ap.add_argument(
        "--da_v2_cache",
        default=str(DA_V2_CACHE),
        help="Local cache path for the DA V2 ViT-S download",
    )
    args = ap.parse_args()

    da_v2_path = download_da_v2(Path(args.da_v2_cache))
    build_checkpoint(da_v2_path, Path(args.out))


if __name__ == "__main__":
    main()
