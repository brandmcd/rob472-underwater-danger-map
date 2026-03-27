"""
Measure per-frame latency of SUIM-Net and SPADE on a GPU node.

What this script does
─────────────────────
For each frame in --frames_dir (up to --n_frames):
  1. Time SUIM-Net inference  → segmentation logits
  2. Time SPADE inference     → depth map
  3. Time danger_map fusion   → risk map
Writes results to --out_csv with one row per frame.

Usage (on ARC Great Lakes — run inside a GPU srun session)
──────────────────────────────────────────────────────────
    srun --account=rob572w26_class --partition=gpu --qos=class \\
         --gpus=1 --cpus-per-task=4 --mem=24G --time=00:30:00 --pty bash

    source /scratch/.../venvs/rob472-spade/bin/activate
    cd ~/rob472-underwater-danger-map

    python -m src.danger_map.profile_latency \\
        --frames_dir $DATA_ROOT/flsea/spade/rgb \\
        --depth_dir  $DATA_ROOT/flsea/spade/depth \\
        --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \\
        --spade_weights   $SPADE_WEIGHTS \\
        --n_frames 50 \\
        --out_csv  reports/latency.csv

Output CSV columns
──────────────────
    frame, suimnet_ms, spade_ms, fusion_ms, total_ms
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

# ── Repo-relative imports ─────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
SUIMNET_ROOT = REPO_ROOT / "vendor" / "SUIM-Net"
VENDOR_SPADE = REPO_ROOT / "vendor" / "SPADE"
sys.path.append(str(SUIMNET_ROOT))

# Keras 2.13+ compatibility shim
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

from model import SUIM_Net  # type: ignore  (vendor/SUIM-Net/model.py)

import skimage.transform as sktf

from src.danger_map import danger_map
from src.spade._spade_utils import load_depth, generate_sparse_csv

SUIM_H, SUIM_W = 240, 320
SPADE_H, SPADE_W = 352, 480      # SPADE native input resolution


def _load_frames(frames_dir: Path, depth_dir: Path | None, n: int):
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in exts])[:n]
    frames = []
    for p in paths:
        ext = p.suffix.lower()
        if ext in (".tif", ".tiff"):
            arr = tifffile.imread(str(p))
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / (arr.ptp() + 1e-6) * 255).astype(np.uint8)
        else:
            arr = np.array(Image.open(p).convert("RGB"))

        depth = None
        if depth_dir is not None:
            dp = depth_dir / p.name
            if not dp.exists():
                for ext2 in (".tif", ".tiff", ".npy", ".png"):
                    alt = depth_dir / (p.stem + ext2)
                    if alt.exists():
                        dp = alt
                        break
            if dp.exists():
                depth = load_depth(dp)

        frames.append((p.name, arr, depth))
    return frames


def _load_suimnet(weights: Path):
    model = SUIM_Net(im_res=(SUIM_H, SUIM_W), n_classes=5).model
    model.load_weights(str(weights))
    return model


def _load_spade(weights: Path):
    _orig_cwd = os.getcwd()
    os.chdir(str(VENDOR_SPADE))
    sys.path.insert(0, str(VENDOR_SPADE))
    try:
        import torch
        from UnderwaterDepth.models.spade_model import SpadeModel  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(weights), map_location=device)
        model = SpadeModel(device=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
    finally:
        os.chdir(_orig_cwd)
    return model, device


def _infer_suimnet(model, rgb: np.ndarray) -> np.ndarray:
    """Return (H_s, W_s, 5) float32 logits."""
    img_rs = sktf.resize(rgb, (SUIM_H, SUIM_W, 3),
                         preserve_range=False, anti_aliasing=True).astype(np.float32)
    x = np.expand_dims(img_rs, axis=0)
    return model.predict(x, verbose=0)[0]


def _infer_spade(model, device, rgb: np.ndarray, depth: np.ndarray | None) -> np.ndarray:
    """Return (H_spade, W_spade) float32 depth map in metres."""
    import torch

    rgb_rs = cv2.resize(rgb, (SPADE_W, SPADE_H))
    t = torch.from_numpy(rgb_rs.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    t = t.to(device)

    if depth is not None:
        import tempfile, csv as _csv
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        generate_sparse_csv(bgr, depth, tmp_path)
        sparse_pts = []
        with tmp_path.open() as f:
            for row in _csv.DictReader(f):
                sparse_pts.append([float(row["row"]), float(row["column"]), float(row["depth"])])
        tmp_path.unlink(missing_ok=True)
        if sparse_pts:
            sp = torch.tensor(sparse_pts, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            sp = torch.zeros((1, 1, 3), dtype=torch.float32).to(device)
    else:
        sp = torch.zeros((1, 1, 3), dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(t, sp)
    return out.squeeze().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Profile per-frame latency of SUIM-Net, SPADE, and danger_map fusion."
    )
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--depth_dir", default=None)
    ap.add_argument("--suimnet_weights",
                    default=str(SUIMNET_ROOT / "sample_test" / "ckpt_seg_5obj.hdf5"))
    ap.add_argument("--spade_weights", required=True)
    ap.add_argument("--n_frames", type=int, default=50,
                    help="Number of frames to profile. Default: 50")
    ap.add_argument("--out_csv", default="reports/latency.csv")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    depth_dir  = Path(args.depth_dir) if args.depth_dir else None
    out_csv    = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Loading frames…")
    frames = _load_frames(frames_dir, depth_dir, args.n_frames)
    print(f"  {len(frames)} frames loaded")

    print("Loading SUIM-Net…")
    suimnet = _load_suimnet(Path(args.suimnet_weights))

    print("Loading SPADE…")
    spade, device = _load_spade(Path(args.spade_weights))
    print(f"  Device: {device}")
    print()

    rows = []
    for i, (name, rgb, depth) in enumerate(frames, 1):
        t0 = time.perf_counter()
        logits = _infer_suimnet(suimnet, rgb)
        t1 = time.perf_counter()
        depth_pred = _infer_spade(spade, device, rgb, depth)
        t2 = time.perf_counter()
        _, _ = danger_map(rgb, logits, depth_pred)
        t3 = time.perf_counter()

        suimnet_ms = (t1 - t0) * 1000
        spade_ms   = (t2 - t1) * 1000
        fusion_ms  = (t3 - t2) * 1000
        total_ms   = (t3 - t0) * 1000

        rows.append({
            "frame":       name,
            "suimnet_ms":  f"{suimnet_ms:.1f}",
            "spade_ms":    f"{spade_ms:.1f}",
            "fusion_ms":   f"{fusion_ms:.1f}",
            "total_ms":    f"{total_ms:.1f}",
        })

        if i % 10 == 0 or i == len(frames):
            print(f"  [{i:3d}/{len(frames)}] {name}: "
                  f"SUIM={suimnet_ms:.0f}ms  SPADE={spade_ms:.0f}ms  "
                  f"fusion={fusion_ms:.0f}ms  total={total_ms:.0f}ms")

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","suimnet_ms","spade_ms","fusion_ms","total_ms"])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary stats
    totals = [float(r["total_ms"]) for r in rows]
    suimnets = [float(r["suimnet_ms"]) for r in rows]
    spades   = [float(r["spade_ms"])   for r in rows]
    print()
    print("=== Latency Summary ===")
    print(f"  SUIM-Net : mean={np.mean(suimnets):.1f}ms  p50={np.median(suimnets):.1f}ms  p90={np.percentile(suimnets,90):.1f}ms")
    print(f"  SPADE    : mean={np.mean(spades):.1f}ms    p50={np.median(spades):.1f}ms    p90={np.percentile(spades,90):.1f}ms")
    print(f"  Total    : mean={np.mean(totals):.1f}ms    p50={np.median(totals):.1f}ms    p90={np.percentile(totals,90):.1f}ms")
    print(f"  Effective FPS (mean): {1000/np.mean(totals):.1f}")
    print(f"Output → {out_csv}")


if __name__ == "__main__":
    main()
