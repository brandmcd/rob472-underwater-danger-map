"""
Process clips from a raw video through the full pipeline (SUIM-Net + SPADE) and
save as animated GIFs.

SPADE depth mode for raw video (no sensor depth available):
    A flat synthetic depth is used to seed the sparse hint map — Shi-Tomasi
    corners are detected in each frame and stamped at `--hint_depth_m` metres.
    SPADE uses those hints to anchor its metric scale and produces a real
    spatially-varying depth map scaled to that operating range.

Usage
─────
    # All three BlueROV2 clips:
    python scripts/make_danger_gif.py videos/raw/bluerov2.mp4 \\
        --clips "33.0:36.0:bluerov2_139_148" "60.4:64.7:bluerov2_301_314" "114.1:119.5:bluerov2_542_558" \\
        --out_dir figures/pipeline

    # Single clip:
    python scripts/make_danger_gif.py videos/raw/bluerov2.mp4 \\
        --start_s 33.0 --end_s 36.0 --out figures/pipeline/bluerov2_139.gif
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import skimage.transform as sktf
from PIL import Image

REPO_ROOT    = Path(__file__).resolve().parent.parent
SUIMNET_ROOT = REPO_ROOT / "vendor" / "SUIM-Net"
WEIGHTS      = SUIMNET_ROOT / "sample_test" / "ckpt_seg_5obj.hdf5"

sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(SUIMNET_ROOT))

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

from model import SUIM_Net  # type: ignore
from src.danger_map import danger_map
from src.danger_map.navigate import nav_command, draw_nav_overlay
from src.danger_map.run_video import (
    _run_suimnet, _run_spade, _load_spade,
    _make_side_by_side, _render_depth_panel,
    SUIM_H, SUIM_W, SPADE_H, SPADE_W,
    MAX_CORNERS, CORNER_QUAL, CORNER_DIST,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_synthetic_sparse(rgb: np.ndarray, hint_depth_m: float) -> np.ndarray:
    """
    Build a sparse hint map using Shi-Tomasi corners, all stamped at hint_depth_m.

    This mirrors the FLSea evaluation protocol in _build_sparse_map(), but uses
    a fixed metric depth rather than GT sensor depth.  SPADE uses these hints
    to anchor its metric scale — it still produces a spatially-varying map
    driven by the Depth-Anything V2 backbone.
    """
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUAL, minDistance=CORNER_DIST,
    )
    sparse = np.zeros((SPADE_H, SPADE_W, 1), dtype=np.float32)
    if corners is None:
        return sparse
    for c in corners:
        x, y = c[0]
        row = int(np.clip(round(y), 0, H - 1))
        col = int(np.clip(round(x), 0, W - 1))
        r_s = int(np.clip(row * SPADE_H / H, 0, SPADE_H - 1))
        c_s = int(np.clip(col * SPADE_W / W, 0, SPADE_W - 1))
        sparse[r_s, c_s, 0] = hint_depth_m
    return sparse


def _run_spade_synthetic(spade_model, rgb: np.ndarray, hint_depth_m: float) -> np.ndarray:
    """Run SPADE with synthetic sparse hints stamped at hint_depth_m."""
    import torch
    img_f  = rgb.astype(np.float32) / 255.0
    img_rs = cv2.resize(img_f, (SPADE_W, SPADE_H), interpolation=cv2.INTER_LINEAR)
    img_n  = (img_rs - _IMAGENET_MEAN) / _IMAGENET_STD
    img_t  = torch.from_numpy(img_n.transpose(2, 0, 1)).unsqueeze(0).float()

    sparse_np = _build_synthetic_sparse(rgb, hint_depth_m)
    sparse_t  = torch.from_numpy(sparse_np.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        out = spade_model(img_t, prompt_depth=sparse_t, fx=None, cx=None)

    return out["metric_depth"].squeeze().cpu().numpy()


def process_clip(
    suimnet,
    spade_model,
    video_path: Path,
    start_s: float,
    end_s:   float,
    out_path: Path,
    hint_depth_m:  float = 2.0,
    near_m:        float = 2.5,
    display_gamma: float = 0.5,
    overlay_alpha: float = 0.6,
    max_depth_m:   float = 8.0,
    show_depth:    bool  = True,
    gif_fps:   int   = 10,
    gif_scale: float = 0.45,
    gif_step:  int   = 2,
    palette:   int   = 128,
) -> None:
    cap     = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_s * src_fps))
    end_frame   = min(total, int(end_s * src_fps))

    print(f"  Clip {start_s:.1f}–{end_s:.1f}s  "
          f"frames {start_frame}–{end_frame}  "
          f"({end_frame - start_frame} src frames)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_out: list[np.ndarray] = []
    frame_i = start_frame

    while frame_i < end_frame:
        ret, bgr = cap.read()
        if not ret:
            break

        if (frame_i - start_frame) % gif_step == 0:
            rgb = bgr[..., ::-1].copy()
            H, W = rgb.shape[:2]

            seg_logits = _run_suimnet(suimnet, rgb)
            depth_pred = _run_spade_synthetic(spade_model, rgb, hint_depth_m)

            risk_map, overlay = danger_map(
                rgb, seg_logits, depth_pred,
                near_m=near_m,
                display_gamma=display_gamma,
                overlay_alpha=overlay_alpha,
            )

            nav       = nav_command(risk_map, depth_pred)
            nav_panel = draw_nav_overlay(overlay, nav)

            depth_panel = _render_depth_panel(depth_pred, H, W, max_depth_m) \
                          if show_depth else None

            panel = _make_side_by_side(rgb, overlay, nav_panel, depth_panel, max_depth_m)

            if gif_scale != 1.0:
                gh = int(panel.shape[0] * gif_scale)
                gw = int(panel.shape[1] * gif_scale)
                panel = cv2.resize(panel, (gw, gh), interpolation=cv2.INTER_AREA)

            frames_out.append(panel)

            if len(frames_out) % 10 == 0:
                r_max = risk_map.max()
                print(f"    frame {frame_i}  risk_max={r_max:.3f}  "
                      f"depth {depth_pred.min():.2f}–{depth_pred.max():.2f}m  "
                      f"nav={nav.command} [{nav.risk_level}]")

        frame_i += 1

    cap.release()

    if not frames_out:
        raise RuntimeError(f"No frames extracted for {start_s}–{end_s}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(f) for f in frames_out]
    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(1000 / gif_fps),
        optimize=False,
    )
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved → {out_path}  ({len(frames_out)} frames, {mb:.1f} MB)\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_file")
    ap.add_argument("--spade_weights",
                    default="/home/bamcd/Downloads/underwater_depth_pipeline.pt")
    ap.add_argument("--start_s",  type=float, default=None)
    ap.add_argument("--end_s",    type=float, default=None)
    ap.add_argument("--out",      default=None)
    ap.add_argument("--clips",    nargs="+", default=None,
                    help="'start:end:stem'  e.g.  '33.0:36.0:bluerov2_139'")
    ap.add_argument("--out_dir",  default="figures/pipeline")
    ap.add_argument("--hint_depth_m", type=float, default=2.0,
                    help="Synthetic depth used as sparse hints for SPADE. "
                         "Sets the metric scale anchor. Default: 2.0 m")
    ap.add_argument("--near_m",        type=float, default=2.5)
    ap.add_argument("--display_gamma", type=float, default=0.5)
    ap.add_argument("--overlay_alpha", type=float, default=0.6)
    ap.add_argument("--max_depth_m",   type=float, default=8.0)
    ap.add_argument("--show_depth",    action="store_true", default=True)
    ap.add_argument("--fps",    type=int,   default=10)
    ap.add_argument("--scale",  type=float, default=0.40)
    ap.add_argument("--step",   type=int,   default=3)
    ap.add_argument("--palette", type=int,  default=128, choices=[64, 128, 256])
    args = ap.parse_args()

    print("Loading SUIM-Net…")
    suimnet = SUIM_Net(im_res=(SUIM_H, SUIM_W), n_classes=5).model
    suimnet.load_weights(str(WEIGHTS))

    print("Loading SPADE…")
    spade = _load_spade(Path(args.spade_weights).resolve())
    print()

    kw = dict(
        hint_depth_m=args.hint_depth_m,
        near_m=args.near_m,
        display_gamma=args.display_gamma,
        overlay_alpha=args.overlay_alpha,
        max_depth_m=args.max_depth_m,
        show_depth=args.show_depth,
        gif_fps=args.fps,
        gif_scale=args.scale,
        gif_step=args.step,
        palette=args.palette,
    )

    vp = Path(args.video_file)

    if args.clips:
        out_dir = Path(args.out_dir)
        for spec in args.clips:
            parts = spec.split(":")
            s, e, stem = float(parts[0]), float(parts[1]), parts[2]
            process_clip(suimnet, spade, vp, s, e, out_dir / f"{stem}.gif", **kw)
    else:
        if None in (args.start_s, args.end_s, args.out):
            ap.error("Need --start_s, --end_s, and --out (or use --clips)")
        process_clip(suimnet, spade, vp,
                     args.start_s, args.end_s, Path(args.out), **kw)

    print("Done.")


if __name__ == "__main__":
    main()
