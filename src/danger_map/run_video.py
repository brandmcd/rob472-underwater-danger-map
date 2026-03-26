"""
Run the underwater danger map pipeline on a folder of frames and produce overlay videos.

What this script does
─────────────────────
For each frame in --frames_dir:
  1. Run SUIM-Net → segmentation logits  (5-channel sigmoid, classes RO/FV/HD/RI/WR)
  2. Run SPADE    → metric depth map     (float32, metres)
  3. Call danger_map()  → per-pixel risk score in [0, 1]
  4. Write a side-by-side overlay image:  original RGB | danger map heatmap
After all frames, stitch the overlays into an .mp4 video.

Data layout expected
────────────────────
--frames_dir   A flat folder of RGB images (.png / .tif / .jpg).
               For FLSea-VI, this is the `rgb/` sub-directory produced by
               src/spade/convert_flsea.py  (float32 TIFF files).

--depth_dir    (Optional) Folder of depth images with the same filenames as frames.
               Depth files must be float32 TIFF (.tif) or 16-bit PNG (.png) in mm.
               If omitted, SPADE runs in zero-hint mode (Depth-Anything V2 backbone
               only, no sparse-depth guidance) — results are still plausible but
               may not be metric-scale.

Usage (on ARC Great Lakes)
──────────────────────────
    cd /path/to/rob472-underwater-danger-map

    python -m src.danger_map.run_video \\
        --frames_dir  $DATA_ROOT/flsea/spade/rgb \\
        --depth_dir   $DATA_ROOT/flsea/spade/depth \\
        --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \\
        --spade_weights   /path/to/underwater_depth_pipeline.pt \\
        --out_dir         figures/danger_map_videos \\
        --max_frames      300

    # Produces:
    #   figures/danger_map_videos/frames/000001_overlay.png  (per-frame overlay)
    #   figures/danger_map_videos/danger_map.mp4             (stitched video)

Notes
─────
- Frames are processed in sorted filename order so the video is temporally consistent.
- Default video FPS is 10. Override with --fps.
- If opencv cannot write .mp4 on the cluster, try --video_ext avi.
"""
from __future__ import annotations

import argparse
import os
import sys
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

# Keras 2.13+ compatibility shim (same patch used in src/suimnet/run_infer.py)
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

from src.danger_map import danger_map

# ── Constants ─────────────────────────────────────────────────────────────────
# SUIM-Net native input resolution (must match the weights)
SUIM_H, SUIM_W = 240, 320

# SPADE model input resolution (matches generate_feature_map_for_ga defaults in
# vendor/SPADE/UnderwaterDepth/data/data_mono.py)
SPADE_H, SPADE_W = 336, 448

# Sparse depth hint parameters (must match src/spade/_spade_utils.py)
MAX_CORNERS   = 500
CORNER_QUAL   = 0.01
CORNER_DIST   = 5

# ImageNet normalisation constants used by SPADE's data pipeline
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Image / depth I/O ─────────────────────────────────────────────────────────

def _list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB regardless of source format."""
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))
        # Float TIFFs from convert_flsea.py are in [0, 1] — scale to [0, 255]
        if arr.dtype in (np.float32, np.float64):
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    else:
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    if arr.ndim == 2:                              # grayscale → RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr


def _load_depth(path: Path) -> np.ndarray:
    """Load depth as float32 metres. Handles float32 TIFF and 16-bit PNG (mm)."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return tifffile.imread(str(path)).astype(np.float32)
    if ext == ".png":
        arr = np.array(Image.open(path), dtype=np.float32)
        if arr.max() > 200:      # 16-bit PNG heuristic: values in mm → convert to m
            arr /= 1000.0
        return arr
    raise ValueError(f"Unsupported depth format: {path}")


# ── SUIM-Net inference ────────────────────────────────────────────────────────

def _run_suimnet(model, rgb: np.ndarray) -> np.ndarray:
    """
    Run SUIM-Net on one RGB frame.

    Args:
        model: Loaded Keras SUIM_Net model.
        rgb:   (H, W, 3) uint8 array.

    Returns:
        (SUIM_H, SUIM_W, 5) float32 sigmoid logits.
    """
    import skimage.transform as sktf
    img_rs = sktf.resize(rgb, (SUIM_H, SUIM_W, 3),
                         preserve_range=False, anti_aliasing=True)
    x = np.expand_dims(img_rs, axis=0)            # (1, H, W, 3)
    return model.predict(x, verbose=0)[0]          # (H, W, 5)


# ── SPADE inference ───────────────────────────────────────────────────────────

def _build_sparse_map(rgb: np.ndarray, depth_m: np.ndarray | None) -> np.ndarray:
    """
    Build a (SPADE_H, SPADE_W, 1) sparse depth map for SPADE.

    If depth_m is provided, Shi-Tomasi corners are sampled from the image and
    their GT depth values are used as hints — the same strategy as the SPADE
    evaluation protocol (see src/spade/_spade_utils.py).

    If depth_m is None, an all-zero map is returned and SPADE falls back to
    Depth-Anything V2 global-alignment mode (no metric-scale hints).
    """
    sparse = np.zeros((SPADE_H, SPADE_W, 1), dtype=np.float32)
    if depth_m is None:
        return sparse

    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUAL,
        minDistance=CORNER_DIST,
    )
    if corners is None:
        return sparse

    # Resize depth to match the original image if needed
    if depth_m.shape[:2] != (H, W):
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

    for c in corners:
        x, y = c[0]
        row = int(np.clip(round(y), 0, H - 1))
        col = int(np.clip(round(x), 0, W - 1))
        d = float(depth_m[row, col])
        if d <= 0 or not np.isfinite(d):
            continue
        # Scale coordinates from original image space to SPADE model input space
        r_s = int(np.clip(row * SPADE_H / H, 0, SPADE_H - 1))
        c_s = int(np.clip(col * SPADE_W / W, 0, SPADE_W - 1))
        sparse[r_s, c_s, 0] = d

    return sparse


def _run_spade(model, rgb: np.ndarray, depth_m: np.ndarray | None) -> np.ndarray:
    """
    Run SPADE on one RGB frame and return a (H, W) float32 depth map in metres.

    Args:
        model:   Loaded SPADE torch model (on CUDA).
        rgb:     (H, W, 3) uint8 array.
        depth_m: (H_d, W_d) float32 GT depth in metres, or None for hint-free mode.

    Returns:
        (SPADE_H, SPADE_W) float32 depth map in metres.
    """
    import torch

    # Normalise image and resize to SPADE's expected input resolution
    img_f = rgb.astype(np.float32) / 255.0
    img_rs = cv2.resize(img_f, (SPADE_W, SPADE_H), interpolation=cv2.INTER_LINEAR)
    img_norm = (img_rs - _IMAGENET_MEAN) / _IMAGENET_STD       # (H, W, 3)
    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1))       # (3, H, W)
    img_t = img_t.unsqueeze(0).float().cuda()                   # (1, 3, H, W)

    # Build sparse hint map
    sparse_np = _build_sparse_map(rgb, depth_m)                 # (H, W, 1)
    sparse_t  = torch.from_numpy(sparse_np.transpose(2, 0, 1))  # (1, H, W)
    sparse_t  = sparse_t.unsqueeze(0).float().cuda()            # (1, 1, H, W)

    with torch.no_grad():
        out = model(img_t, prompt_depth=sparse_t, fx=None, cx=None)

    depth_pred = out["metric_depth"].squeeze().cpu().numpy()    # (H, W) float32
    return depth_pred


# ── SPADE model loading ───────────────────────────────────────────────────────

def _load_spade(weights_path: Path):
    """
    Load the SPADE depth model.

    Follows the same sys.path / os.chdir pattern as src/spade/run_eval.py so that
    all SPADE-internal imports resolve correctly.  The original working directory
    is restored afterwards.
    """
    orig_cwd = os.getcwd()
    try:
        sys.path.insert(0, str(VENDOR_SPADE))
        os.chdir(VENDOR_SPADE)

        from UnderwaterDepth.utils.config  import get_config
        from UnderwaterDepth.models.builder import build_model

        config = get_config(
            "SPADE", "eval", "flsea_sparse_feature",
            pretrained_resource=f"local::{weights_path}",
        )
        model = build_model(config).cuda()
        model.eval()
        return model
    finally:
        os.chdir(orig_cwd)


# ── Overlay frame assembly ────────────────────────────────────────────────────

def _risk_colorbar(width: int, bar_h: int = 22) -> np.ndarray:
    """Return a (bar_h, width, 3) uint8 RGB colorbar for the HOT colormap."""
    gradient = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :].repeat(bar_h, axis=0)
    bar_bgr = cv2.applyColorMap(gradient, cv2.COLORMAP_HOT)
    bar_rgb = bar_bgr[..., ::-1].copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar_rgb, "risk: 0",  (4,        bar_h - 5), font, 0.38, (180, 180, 180), 1)
    cv2.putText(bar_rgb, "1.0",      (width - 28, bar_h - 5), font, 0.38, ( 30,  30,  30), 1)
    # Mid-point tick
    mid = width // 2
    cv2.putText(bar_rgb, "0.5", (mid - 10, bar_h - 5), font, 0.38, (120, 120, 120), 1)
    return bar_rgb


def _make_side_by_side(rgb: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Assemble a labelled side-by-side frame:
        [title bar]
        [ Original RGB  |  Danger Map overlay ]
        [    risk colorbar (right panel only)  ]
    """
    H, W = rgb.shape[:2]
    title_h  = 30
    bar_h    = 22
    total_h  = title_h + H + bar_h

    canvas = np.zeros((total_h, W * 2, 3), dtype=np.uint8)

    # Panels
    canvas[title_h : title_h + H, :W]  = rgb
    canvas[title_h : title_h + H, W:]  = overlay

    # Colorbar under the right (danger map) panel
    bar = _risk_colorbar(W, bar_h)
    canvas[title_h + H :, W:] = bar

    # Title labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original",   (10,      24), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Danger Map", (W + 10,  24), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run danger map pipeline on a folder of frames and produce overlay videos."
    )
    ap.add_argument("--frames_dir", required=True,
                    help="Folder of RGB image frames (.png / .tif / .jpg), sorted alphabetically.")
    ap.add_argument("--depth_dir", default=None,
                    help="(Optional) Folder of depth images matching the frame filenames. "
                         "If omitted, SPADE runs without sparse depth hints.")
    ap.add_argument("--suimnet_weights",
                    default=str(SUIMNET_ROOT / "sample_test" / "ckpt_seg_5obj.hdf5"),
                    help="Path to SUIM-Net .hdf5 weights.")
    ap.add_argument("--spade_weights", required=True,
                    help="Path to SPADE .pt weights file.")
    ap.add_argument("--out_dir", default="figures/danger_map_videos",
                    help="Output directory for overlay frames and video. Default: figures/danger_map_videos")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Stop after this many frames (useful for a quick test run).")
    ap.add_argument("--fps", type=int, default=10,
                    help="Frames per second for the output video. Default: 10")
    ap.add_argument("--video_ext", default="mp4", choices=["mp4", "avi"],
                    help="Output video container. Use 'avi' if mp4 fails on the cluster.")
    ap.add_argument("--overlay_alpha", type=float, default=0.5,
                    help="Blend weight for the danger heatmap (0=RGB only, 1=heatmap only). Default: 0.5")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    depth_dir  = Path(args.depth_dir).resolve() if args.depth_dir else None
    out_dir    = Path(args.out_dir).resolve()
    frames_out = out_dir / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    frame_paths = _list_images(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No images found in: {frames_dir}")
    if args.max_frames:
        frame_paths = frame_paths[:args.max_frames]

    print(f"Found {len(frame_paths)} frames in: {frames_dir}")
    print(f"Depth hints: {'from ' + str(depth_dir) if depth_dir else 'DISABLED (zero-hint mode)'}")
    print(f"Output dir : {out_dir}")
    print()

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading SUIM-Net…")
    suimnet = SUIM_Net(im_res=(SUIM_H, SUIM_W), n_classes=5).model
    suimnet.load_weights(str(args.suimnet_weights))

    print("Loading SPADE…")
    spade = _load_spade(Path(args.spade_weights).resolve())
    print()

    # ── Process frames ────────────────────────────────────────────────────────
    video_writer = None
    n_written = 0

    for i, frame_path in enumerate(frame_paths, 1):
        rgb = _load_rgb(frame_path)

        # Load matching depth if available
        depth_m: np.ndarray | None = None
        if depth_dir is not None:
            for ext in (".tif", ".tiff", ".png"):
                dp = depth_dir / (frame_path.stem + ext)
                if dp.exists():
                    depth_m = _load_depth(dp)
                    break

        # Run both models
        seg_logits = _run_suimnet(suimnet, rgb)            # (240, 320, 5)
        depth_pred = _run_spade(spade, rgb, depth_m)       # (336, 448)

        # Fuse into danger map
        risk_map, overlay = danger_map(
            rgb, seg_logits, depth_pred,
            overlay_alpha=args.overlay_alpha,
        )

        # Build side-by-side frame
        side_by_side = _make_side_by_side(rgb, overlay)

        # Save individual frame
        out_path = frames_out / f"{i:06d}_overlay.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

        # Initialise video writer on first frame (needs frame dimensions)
        if video_writer is None:
            fh, fw = side_by_side.shape[:2]
            codec = cv2.VideoWriter_fourcc(*("mp4v" if args.video_ext == "mp4" else "XVID"))
            video_path = out_dir / f"danger_map.{args.video_ext}"
            video_writer = cv2.VideoWriter(str(video_path), codec, args.fps, (fw, fh))

        video_writer.write(cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        n_written += 1

        if i % 10 == 0 or i == len(frame_paths):
            print(f"  [{i}/{len(frame_paths)}]  risk_max={risk_map.max():.3f}  {frame_path.name}")

    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved → {video_path}")

    print(f"Overlay frames saved → {frames_out}  ({n_written} frames)")


if __name__ == "__main__":
    main()
