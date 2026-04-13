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

--video_file   Alternative to --frames_dir: a single video file (.mp4 / .avi / .mov).
               Frames are decoded directly from the video using OpenCV; no pre-extraction
               step is needed. Mutually exclusive with --frames_dir.

--depth_dir    (Optional) Folder of depth images with the same filenames as frames.
               Depth files must be float32 TIFF (.tif) or 16-bit PNG (.png) in mm.
               If omitted, SPADE runs in zero-hint mode (Depth-Anything V2 backbone
               only, no sparse-depth guidance) — results are still plausible but
               may not be metric-scale.
               Not used when --video_file is supplied (no GT depth available).

Usage (on ARC Great Lakes)
──────────────────────────
    cd /path/to/rob472-underwater-danger-map

    # Frame-folder mode (FLSea / SeaThru):
    python -m src.danger_map.run_video \\
        --frames_dir  $DATA_ROOT/flsea/spade/rgb \\
        --depth_dir   $DATA_ROOT/flsea/spade/depth \\
        --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \\
        --spade_weights   /path/to/underwater_depth_pipeline.pt \\
        --out_dir         figures/danger_map_videos \\
        --max_frames      300

    # Video-file mode (DRUVA):
    python -m src.danger_map.run_video \\
        --video_file  $DATA_ROOT/druva/artifact_01.mp4 \\
        --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \\
        --spade_weights   /path/to/underwater_depth_pipeline.pt \\
        --out_dir         reports/danger_map/druva/artifact_01

    # Produces:
    #   <out_dir>/danger_map.mp4             (output video)

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
from src.danger_map.navigate import nav_command, draw_nav_overlay

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
    device = next(model.parameters()).device
    img_t = img_t.unsqueeze(0).float().to(device)               # (1, 3, H, W)

    # Build sparse hint map
    sparse_np = _build_sparse_map(rgb, depth_m)                 # (H, W, 1)
    sparse_t  = torch.from_numpy(sparse_np.transpose(2, 0, 1))  # (1, H, W)
    sparse_t  = sparse_t.unsqueeze(0).float().to(device)        # (1, 1, H, W)

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
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(config)
        model = model.to(device)
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


def _depth_colorbar(width: int, max_depth_m: float = 12.0, bar_h: int = 22) -> np.ndarray:
    """Return a (bar_h, width, 3) uint8 RGB colorbar for the PLASMA depth colormap.

    Convention: dark-purple = 0 m (close), bright-yellow = max_depth_m (far),
    matching the colour scale used in vendor/SPADE evaluation figures.
    """
    gradient = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :].repeat(bar_h, axis=0)
    bar_bgr = cv2.applyColorMap(gradient, cv2.COLORMAP_PLASMA)
    bar_rgb = bar_bgr[..., ::-1].copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar_rgb, "depth: 0m",        (4,           bar_h - 5), font, 0.38, (220, 220, 220), 1)
    cv2.putText(bar_rgb, f"{max_depth_m:.0f}m", (width - 30, bar_h - 5), font, 0.38, ( 20,  20,  20), 1)
    mid = width // 2
    cv2.putText(bar_rgb, f"{max_depth_m / 2:.0f}m", (mid - 8, bar_h - 5), font, 0.38, (160, 160, 160), 1)
    return bar_rgb


def _render_depth_panel(depth: np.ndarray, H: int, W: int,
                        max_depth_m: float = 12.0) -> np.ndarray:
    """Render a SPADE depth map as a PLASMA-colourmap panel of size (H, W, 3) uint8.

    Normalisation: 0 m → dark purple, max_depth_m → bright yellow.
    Invalid pixels (depth ≤ 0 or NaN) are rendered black.
    """
    depth_rs = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST) \
               if depth.shape[:2] != (H, W) else depth.copy()

    depth_norm = np.clip(depth_rs / max_depth_m, 0.0, 1.0)
    depth_u8   = (depth_norm * 255).astype(np.uint8)

    depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
    depth_rgb = depth_bgr[..., ::-1].copy()

    # Black-out invalid pixels
    invalid = (depth_rs <= 0) | ~np.isfinite(depth_rs)
    depth_rgb[invalid] = 0

    return depth_rgb


def _make_side_by_side(
    rgb: np.ndarray,
    overlay: np.ndarray,
    nav_panel: np.ndarray | None = None,
    depth_panel: np.ndarray | None = None,
    max_depth_m: float = 12.0,
) -> np.ndarray:
    """
    Assemble a labelled multi-panel frame.

    Panel order (left → right):
        Original  |  [Depth Map]  |  Danger Map  |  [Navigation]

    Each panel is W×H pixels.  A title bar (30 px) sits above the panels and a
    per-panel colorbar (22 px) sits below each panel that has one.
    """
    H, W = rgb.shape[:2]
    title_h = 30
    bar_h   = 22

    # Build ordered list of (image, label, colorbar | None)
    panels: list[tuple[np.ndarray, str, np.ndarray | None]] = [
        (rgb, "Original", None),
    ]
    if depth_panel is not None:
        dp = depth_panel if depth_panel.shape[:2] == (H, W) \
             else cv2.resize(depth_panel, (W, H))
        panels.append((dp, "Depth Map", _depth_colorbar(W, max_depth_m, bar_h)))
    panels.append((overlay, "Danger Map", _risk_colorbar(W, bar_h)))
    if nav_panel is not None:
        np_rs = nav_panel if nav_panel.shape[:2] == (H, W) \
                else cv2.resize(nav_panel, (W, H))
        panels.append((np_rs, "Navigation", _risk_colorbar(W, bar_h)))

    n = len(panels)
    canvas = np.zeros((title_h + H + bar_h, W * n, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (img, label, bar) in enumerate(panels):
        x0 = i * W
        canvas[title_h : title_h + H, x0 : x0 + W] = img
        cv2.putText(canvas, label, (x0 + 10, 24), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if bar is not None:
            canvas[title_h + H :, x0 : x0 + W] = bar

    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run danger map pipeline on a folder of frames and produce overlay videos."
    )
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--frames_dir",
                    help="Folder of RGB image frames (.png / .tif / .jpg), sorted alphabetically.")
    src_group.add_argument("--video_file",
                    help="Input video file (.mp4 / .avi / .mov). Frames are read directly; "
                         "no pre-extraction needed. Use this for DRUVA videos.")
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
    ap.add_argument("--near_m", type=float, default=2.5,
                    help="Danger-zone radius in metres: objects closer than this get full proximity risk (1.0). "
                         "Increase for larger vehicles or to surface risk at typical underwater distances. "
                         "Default: 2.5 m  (use 1.0 m when GT depth hints are available).")
    ap.add_argument("--display_gamma", type=float, default=0.5,
                    help="Gamma compression for the danger-map visualisation only (risk_map itself is unaffected). "
                         "Values < 1 boost low-to-mid risk areas so they show as visible colour rather than near-black. "
                         "0.5 = sqrt compression.  1.0 = no compression (linear).  Default: 0.5")
    ap.add_argument("--show_depth", action="store_true",
                    help="Add a Depth Map panel (PLASMA colourmap, 0→dark-purple, max_depth→bright-yellow) "
                         "between the Original and Danger Map panels.")
    ap.add_argument("--max_depth_m", type=float, default=12.0,
                    help="Maximum depth (m) for the depth-panel colour scale. Default: 12.0")
    ap.add_argument("--save_ply", action="store_true",
                    help="Write a coloured risk .ply point cloud for every Nth frame to <out_dir>/clouds/.")
    ap.add_argument("--ply_every", type=int, default=10,
                    help="Save a PLY every N frames when --save_ply is set. Default: 10")
    args = ap.parse_args()

    depth_dir  = Path(args.depth_dir).resolve() if args.depth_dir else None
    out_dir    = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clouds_out = out_dir / "clouds"
    if args.save_ply:
        clouds_out.mkdir(parents=True, exist_ok=True)

    # ── Source: video file or frame folder ────────────────────────────────────
    video_cap: cv2.VideoCapture | None = None
    frame_paths: list[Path] | None = None

    if args.video_file:
        video_path_in = Path(args.video_file).resolve()
        if not video_path_in.exists():
            raise RuntimeError(f"Video file not found: {video_path_in}")
        video_cap = cv2.VideoCapture(str(video_path_in))
        if not video_cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path_in}")
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames:
            total_frames = min(total_frames, args.max_frames)
        print(f"Video input : {video_path_in}  ({int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))} total frames)")
        depth_dir = None  # no GT depth for raw video
    else:
        frames_dir = Path(args.frames_dir).resolve()
        frame_paths = _list_images(frames_dir)
        if not frame_paths:
            raise RuntimeError(f"No images found in: {frames_dir}")
        if args.max_frames:
            frame_paths = frame_paths[:args.max_frames]
        total_frames = len(frame_paths)
        print(f"Found {total_frames} frames in: {frames_dir}")

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

    def _frame_iter():
        """Yield (frame_index, rgb_array, depth_or_None, label) tuples."""
        if video_cap is not None:
            i = 0
            while True:
                ret, bgr = video_cap.read()
                if not ret:
                    break
                i += 1
                if args.max_frames and i > args.max_frames:
                    break
                rgb = bgr[..., ::-1].copy()  # BGR → RGB
                yield i, rgb, None, f"frame {i}"
        else:
            for i, frame_path in enumerate(frame_paths, 1):
                rgb = _load_rgb(frame_path)
                depth_m: np.ndarray | None = None
                if depth_dir is not None:
                    for ext in (".tif", ".tiff", ".png"):
                        dp = depth_dir / (frame_path.stem + ext)
                        if dp.exists():
                            depth_m = _load_depth(dp)
                            break
                yield i, rgb, depth_m, frame_path.name

    for i, rgb, depth_m, label in _frame_iter():

        # Run both models
        seg_logits = _run_suimnet(suimnet, rgb)            # (240, 320, 5)
        depth_pred = _run_spade(spade, rgb, depth_m)       # (336, 448)

        # Fuse into danger map
        risk_map, overlay = danger_map(
            rgb, seg_logits, depth_pred,
            overlay_alpha=args.overlay_alpha,
            near_m=args.near_m,
            display_gamma=args.display_gamma,
        )

        # Navigation command + optional PLY
        ply_path = None
        if args.save_ply and (i % args.ply_every == 0 or i == 1):
            ply_path = clouds_out / f"{i:06d}_cloud.ply"
        nav_result = nav_command(risk_map, depth_pred, ply_path=ply_path)

        # Optional depth visualisation panel
        depth_panel = _render_depth_panel(depth_pred, *rgb.shape[:2], args.max_depth_m) \
                      if args.show_depth else None

        # Build multi-panel frame (Original | [Depth] | Danger Map | Navigation)
        nav_panel    = draw_nav_overlay(overlay, nav_result)
        side_by_side = _make_side_by_side(rgb, overlay, nav_panel, depth_panel, args.max_depth_m)

        # Initialise video writer on first frame (needs frame dimensions)
        if video_writer is None:
            fh, fw = side_by_side.shape[:2]
            codec = cv2.VideoWriter_fourcc(*("mp4v" if args.video_ext == "mp4" else "XVID"))
            video_path = out_dir / f"danger_map.{args.video_ext}"
            video_writer = cv2.VideoWriter(str(video_path), codec, args.fps, (fw, fh))

        video_writer.write(cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        n_written += 1

        if i % 10 == 0:
            print(f"  [{i}/{total_frames}]  risk_max={risk_map.max():.3f}  {label}")

    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved → {video_path}")
    if video_cap is not None:
        video_cap.release()

    print(f"Frames written: {n_written}")


if __name__ == "__main__":
    main()
