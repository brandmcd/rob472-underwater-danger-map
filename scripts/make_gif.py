#!/usr/bin/env python3
"""
Extract a clip from a processed danger-map video and save it as an animated GIF.

Usage examples
──────────────
# Full-video GIF at 1/3 speed, every 3rd frame, scaled to 50 %:
    python scripts/make_gif.py \
        videos/processed/bluerov1/danger_map.mp4 \
        --out figures/bluerov1_danger.gif \
        --fps 8 --step 3 --scale 0.4

# 10-second clip starting at frame 300:
    python scripts/make_gif.py \
        videos/processed/bluerov2/danger_map.mp4 \
        --out figures/bluerov2_clip.gif \
        --start 300 --end 400 --fps 10 --scale 0.5

# Just extract high-quality PNG stills (no GIF) for figures:
    python scripts/make_gif.py \
        videos/processed/bluerov1/danger_map.mp4 \
        --stills_dir figures/stills/bluerov1 \
        --start 100 --end 600 --step 50 --scale 1.0

Arguments
─────────
  video_file    Path to processed danger_map.mp4 (or any .mp4/.avi)
  --out         Output .gif path (default: <video_stem>.gif beside the video)
  --start       First frame index (0-based, default: 0)
  --end         Last frame index exclusive (default: all frames)
  --step        Sample every Nth frame (default: 2 — halves GIF size)
  --fps         GIF playback speed in frames per second (default: 10)
  --scale       Resize factor for the GIF frames (default: 0.5 — halves dimensions)
  --stills_dir  If given, save each sampled frame as a PNG here (no GIF written)
  --no_gif      Skip GIF writing even when --out is set (useful with --stills_dir)
  --palette     GIF colour depth: 256 | 128 | 64 (default: 256)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _read_frames(
    video_path: Path,
    start: int,
    end: int | None,
    step: int,
    scale: float,
) -> list[np.ndarray]:
    """Return a list of (H, W, 3) uint8 RGB frames sampled from the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop  = min(end, total) if end is not None else total
    print(f"Video: {video_path.name}  ({total} frames total)")
    print(f"Sampling frames [{start} … {stop}) every {step}  →  "
          f"{len(range(start, stop, step))} frames")

    frames: list[np.ndarray] = []
    idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx >= stop:
            break
        if idx >= start and (idx - start) % step == 0:
            rgb = bgr[..., ::-1].copy()
            if scale != 1.0:
                h, w = rgb.shape[:2]
                nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        idx += 1

    cap.release()
    return frames


def _save_gif(frames: list[np.ndarray], out_path: Path, fps: float,
              palette_size: int = 256) -> None:
    """Write a list of RGB uint8 frames to an animated GIF using Pillow."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed.  Run: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]

    if palette_size < 256:
        pil_frames = [f.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
                      for f in pil_frames]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"GIF saved → {out_path}  ({len(frames)} frames, {size_mb:.1f} MB)")


def _save_stills(frames: list[np.ndarray], stills_dir: Path,
                 start: int, step: int) -> None:
    """Save each frame as a numbered PNG in stills_dir."""
    stills_dir.mkdir(parents=True, exist_ok=True)
    for j, frame in enumerate(frames):
        frame_idx = start + j * step
        out = stills_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(out), frame[..., ::-1])  # RGB → BGR for OpenCV
    print(f"Stills saved → {stills_dir}/  ({len(frames)} PNGs)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract frames from a processed danger-map video and save as GIF or PNGs."
    )
    ap.add_argument("video_file", help="Path to the processed danger_map.mp4 (or any video file).")
    ap.add_argument("--out",         default=None,
                    help="Output .gif path.  Defaults to <video_stem>.gif next to the video.")
    ap.add_argument("--start",       type=int, default=0,    help="First frame index (0-based).")
    ap.add_argument("--end",         type=int, default=None, help="Last frame index (exclusive).")
    ap.add_argument("--step",        type=int, default=2,    help="Sample every Nth frame (default: 2).")
    ap.add_argument("--fps",         type=float, default=10, help="GIF playback FPS (default: 10).")
    ap.add_argument("--scale",       type=float, default=0.5,
                    help="Resize factor for GIF frames (default: 0.5).")
    ap.add_argument("--stills_dir",  default=None,
                    help="If given, save each sampled frame as PNG here.")
    ap.add_argument("--no_gif",      action="store_true",
                    help="Skip GIF writing (useful when --stills_dir is the main goal).")
    ap.add_argument("--palette",     type=int, default=256, choices=[64, 128, 256],
                    help="GIF colour depth (default: 256).")
    args = ap.parse_args()

    video_path = Path(args.video_file).resolve()
    if not video_path.exists():
        print(f"ERROR: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    frames = _read_frames(video_path, args.start, args.end, args.step, args.scale)
    if not frames:
        print("ERROR: no frames extracted — check --start / --end values.", file=sys.stderr)
        sys.exit(1)

    if args.stills_dir:
        _save_stills(frames, Path(args.stills_dir).resolve(), args.start, args.step)

    if not args.no_gif:
        out_path = Path(args.out).resolve() if args.out \
                   else video_path.parent / (video_path.stem + ".gif")
        _save_gif(frames, out_path, args.fps, args.palette)


if __name__ == "__main__":
    main()
