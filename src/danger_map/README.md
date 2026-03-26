# Danger Map — Goal 3: Fused Perception Risk Map

> **Project context:** This is Goal 3 of the ROB 472 Underwater Danger Map project.
> It combines the SUIM-Net segmentation masks (Goal 1) and SPADE depth estimates (Goal 2)
> into a single per-pixel **collision risk score** for AUV obstacle avoidance.

The danger map assigns each pixel a risk score in \[0, 1\] based on:
1. **What** is at that pixel (semantic class from SUIM-Net)
2. **How close** it is (metric depth from SPADE)

---

## Quick-start (no SPADE weights needed)

The quickest way to see the danger map working uses only the 8 bundled SUIM sample images and a synthetic flat depth map.

```bash
# From the repo root, activate the main venv
source .venv/bin/activate

# Run with default settings (all objects at 2.0 m depth)
python -m src.danger_map.quick_test

# Simulate objects very close by — expect high risk scores
python -m src.danger_map.quick_test --depth_m 0.5

# Tune the formula on the fly
python -m src.danger_map.quick_test --depth_m 1.5 --near_m 2.0 --proximity_power 2.0
```

Outputs saved to `reports/danger_map/quick_test/` — one side-by-side PNG per image:
`[ Original RGB ] | [ Danger Map overlay ]`

---

## Full pipeline (with SUIM-Net + SPADE on real data)

Run on a folder of real underwater frames. Requires SPADE weights
(see [src/spade/README.md](../spade/README.md) Step 2 to build them).

```bash
python -m src.danger_map.run_video \
    --frames_dir  $DATA_ROOT/flsea/spade/rgb \
    --depth_dir   $DATA_ROOT/flsea/spade/depth \
    --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \
    --spade_weights   ~/Downloads/underwater_depth_pipeline.pt \
    --out_dir         reports/danger_map/videos \
    --max_frames      50
```

Where `$DATA_ROOT` = `/scratch/rob572w26_class_root/rob572w26_class/$USER/data` on Great Lakes (set automatically inside SLURM jobs; use the full path on login nodes).

Produces:
- `reports/danger_map/videos/frames/000001_overlay.png` — one PNG per frame
- `reports/danger_map/videos/danger_map.mp4` — stitched video

### `run_video.py` CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--frames_dir` | *(required)* | Folder of RGB frames (`.png` / `.tif` / `.jpg`) |
| `--depth_dir` | None | Folder of ground-truth depth TIFFs (metres). Omit for hint-free SPADE. |
| `--suimnet_weights` | `vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5` | SUIM-Net `.hdf5` weights |
| `--spade_weights` | *(required)* | SPADE `.pt` weights |
| `--out_dir` | `figures/danger_map_videos` | Output directory |
| `--max_frames` | all | Stop after N frames (useful for quick tests) |
| `--fps` | 10 | Output video frame rate |
| `--overlay_alpha` | 0.5 | Heatmap blend strength (0 = invisible, 1 = opaque) |
| `--video_ext` | `mp4` | `mp4` or `avi` (use `avi` if mp4 fails on the cluster) |

---

## Running on ARC Great Lakes

The danger map pipeline requires both SUIM-Net (`.venv`) and SPADE (`.venv-spade`).
Run it interactively via `srun` or in the SPADE GPU job after evaluation completes.

### One-off run on a GPU node (interactive)

```bash
srun --account=rob572w26_class --partition=gpu --qos=class --gpus=1 \
     --cpus-per-task=4 --mem=24G --time=01:00:00 --pty bash

cd ~/rob472-underwater-danger-map
source /scratch/rob572w26_class_root/rob572w26_class/${USER}/venvs/rob472-spade/bin/activate

python -m src.danger_map.run_video \
    --frames_dir  /scratch/rob572w26_class_root/rob572w26_class/${USER}/data/flsea/spade/rgb \
    --depth_dir   /scratch/rob572w26_class_root/rob572w26_class/${USER}/data/flsea/spade/depth \
    --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \
    --spade_weights   /scratch/rob572w26_class_root/rob572w26_class/${USER}/spade_weights/underwater_depth_pipeline.pt \
    --out_dir         reports/danger_map/flsea \
    --max_frames      100
```

Note: `run_video.py` loads SUIM-Net (TensorFlow) and SPADE (PyTorch) in the same process using `sys.path` injection. Activate the `rob472-spade` venv which has PyTorch; TensorFlow is loaded via the repo's `.venv` path automatically.

### Pull results locally

```bash
scp -r "brandmcd@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/reports/danger_map/" ./reports/
```

---

## Risk formula

```
proximity(x,y) = clip( (near_m / depth(x,y))^power,  0, 1 )

hazard(x,y)    = max hazard_weight  over all SUIM-Net classes active at (x,y)
                 "active" = sigmoid output > seg_threshold

risk(x,y)      = hazard(x,y)  ×  proximity(x,y)       ∈ [0, 1]
```

**Proximity intuition:**
- `depth ≤ near_m` → proximity = **1.0** (full danger zone)
- `depth = 2×near_m` → proximity = **0.5**
- `depth = 10×near_m` → proximity = **0.1**

**Hazard intuition:**
- Only the *highest-weight active class* at each pixel contributes — a diver in front of
  a reef reads as `HD = 1.0`, not `HD + RI`.
- Invalid depth pixels (zero or NaN) → proximity = 0 → risk = 0.

---

## Tuning guide

All tunable constants are at the **top of `src/danger_map/__init__.py`** in a clearly
labelled block. Edit them once to change the defaults for all runs, or pass keyword
arguments to `danger_map()` to override per-call.

### 1. Hazard weights — how dangerous is each class?

```python
# src/danger_map/__init__.py
HAZARD_WEIGHTS: dict[str, float] = {
    "RO": 0.2,   # Robot / Instrument
    "FV": 0.5,   # Fish / Vertebrate
    "HD": 1.0,   # Human Diver          ← highest priority
    "RI": 0.8,   # Reef / Invertebrate
    "WR": 0.9,   # Wreck / Ruin
}
```

Range: `0.0` (no risk) → `1.0` (maximum risk). Example overrides:

```python
# Treat reef as the top hazard for a coral survey mission
risk_map, overlay = danger_map(rgb, seg, depth, hazard_weights={"RI": 1.0, "WR": 0.7})

# Ignore robots entirely
risk_map, overlay = danger_map(rgb, seg, depth, hazard_weights={"RO": 0.0})
```

### 2. `NEAR_M` — danger-zone radius

```python
NEAR_M: float = 1.0   # metres
```

Objects closer than `NEAR_M` receive **full proximity score (1.0)**.  Objects at
`2×NEAR_M` get 0.5; at `10×NEAR_M` get 0.1.

- Slow AUV / tight space → `NEAR_M = 0.5` (smaller bubble)
- Fast AUV / open water  → `NEAR_M = 3.0` (larger bubble, earlier warning)

### 3. `PROXIMITY_POWER` — how fast risk falls off with distance

```python
PROXIMITY_POWER: float = 1.0
```

| Value | Behaviour |
|-------|-----------|
| `0.5` | Slow decay — risk stays elevated at mid-range |
| `1.0` | Linear decay ← default |
| `2.0` | Quadratic — risk drops steeply beyond `near_m` |

Pass as `proximity_power=2.0` to `danger_map()` or change the constant.

### 4. `DEFAULT_SEG_THRESHOLD` — class activation cutoff

```python
DEFAULT_SEG_THRESHOLD: float = 0.5
```

SUIM-Net outputs sigmoid probabilities per class. A pixel is considered "active" for
a class only when its probability exceeds this threshold.

- `0.3` — more pixels flagged (higher recall, more false positives)
- `0.5` — balanced ← default
- `0.7` — strict (fewer pixels, higher confidence required)

### 5. Custom risk formula

Pass `risk_fn` to completely replace the formula:

```python
# Additive blend (risk even when object is far away)
danger_map(rgb, seg, depth,
           risk_fn=lambda hazard, proximity: 0.6 * hazard + 0.4 * proximity)

# Proximity only — ignore class weights
danger_map(rgb, seg, depth,
           risk_fn=lambda hazard, proximity: proximity)

# Sharper threshold: zero below 0.3, then steep rise
danger_map(rgb, seg, depth,
           risk_fn=lambda h, p: np.clip((h * p - 0.3) / 0.7, 0, 1))
```

Both `hazard` and `proximity` are `(H, W)` float32 arrays with values in `[0, 1]`.

---

## Overlay design

The danger map overlay (second panel in the side-by-side output) is built in three
layers, bottom to top:

| Layer | What it is | Why |
|-------|-----------|-----|
| **Grayscale background** | Original RGB desaturated to grey | Eliminates blue/green underwater colour clash with the heatmap |
| **HOT heatmap** | OpenCV `COLORMAP_HOT` (black→red→yellow→white) blended with per-pixel alpha = `risk × overlay_alpha` | Background pixels (risk≈0) stay grey; high-risk pixels turn red/yellow/white |
| **Class contours + labels** | Coloured outlines of each active segmentation class; text label at the centroid of the largest blob | Shows *which* class is causing the risk at each location |

### Per-class contour colours

| Class | Colour | Description |
|-------|--------|-------------|
| HD | Red | Human Diver |
| WR | Yellow | Wreck / Ruin |
| RI | Orange | Reef / Invertebrate |
| FV | Green | Fish / Vertebrate |
| RO | Cyan | Robot / Instrument |

To change colours, edit `_CLASS_COLORS_BGR` in `src/danger_map/__init__.py`.

---

## Project structure

```
src/danger_map/
  __init__.py      # Core danger_map() function + all tuning constants
  run_video.py     # Full pipeline: SUIM-Net + SPADE + danger_map → video
  quick_test.py    # Smoke-test: SUIM-Net only + synthetic depth → overlay PNGs
```

### `__init__.py` — public API

```python
from src.danger_map import danger_map

risk_map, overlay = danger_map(
    rgb,           # (H, W, 3) uint8
    seg_logits,    # (H_s, W_s, 5) float32  — SUIM-Net sigmoid output
    depth_m,       # (H_d, W_d) float32     — depth in metres

    # All keyword-only, all optional:
    hazard_weights  = {"HD": 1.0, "WR": 0.9, ...},  # per-class override
    seg_threshold   = 0.5,      # class activation cutoff
    near_m          = 1.0,      # danger-zone radius (metres)
    proximity_power = 1.0,      # distance fall-off exponent
    overlay_alpha   = 0.6,      # heatmap blend strength
    risk_fn         = None,     # custom formula: fn(hazard, proximity) → risk
)
# risk_map : (H, W) float32, values in [0, 1]
# overlay  : (H, W, 3) uint8, grayscale bg + heatmap + class labels
```

---

## Inputs and outputs reference

### Inputs

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `rgb` | `(H, W, 3)` | uint8 | Original camera frame, any resolution |
| `seg_logits` | `(H_s, W_s, 5)` | float32 | SUIM-Net sigmoid outputs, values in [0, 1]. Channel order: RO, FV, HD, RI, WR |
| `depth_m` | `(H_d, W_d)` | float32 | Dense depth in **metres**. Zero/NaN = invalid → no risk contribution |

Input arrays do **not** need to share a resolution — the module resizes everything
to match `rgb` internally.

### Outputs

| Name | Shape | dtype | Description |
|------|-------|-------|-------------|
| `risk_map` | `(H, W)` | float32 | Per-pixel risk score in [0, 1] |
| `overlay` | `(H, W, 3)` | uint8 | Grayscale background + HOT heatmap + class contour labels |

---

## Technical notes

### Resolution handling

Segmentation and depth are resized to the RGB frame resolution before fusion.
Segmentation uses bilinear interpolation (smooth probability maps).
Depth uses nearest-neighbour (preserves depth values, avoids interpolated artefacts at boundaries).

### Multi-class pixels

A pixel can belong to multiple SUIM-Net classes simultaneously (e.g. a diver holding a robot).
The **maximum** hazard weight over all active classes is used — so a diver-robot pixel gets
`risk = HD_weight × proximity`, not the sum.

### Zero/invalid depth

Any pixel where `depth ≤ 0` or `depth = NaN` gets `proximity = 0`, meaning `risk = 0`.
This prevents invalid depth estimates from generating false alarms.

### Venv dependency

The quick test (`quick_test.py`) only needs the main `.venv` (TensorFlow / SUIM-Net).

The full pipeline (`run_video.py`) needs both SUIM-Net (`.venv`) and SPADE (`.venv-spade` / ARC).
Because SPADE uses PyTorch and SUIM-Net uses TensorFlow, they cannot share a single environment.
`run_video.py` handles this by loading SPADE dynamically using `sys.path` manipulation —
run it from inside either venv (the SPADE imports only activate when SPADE is actually loaded).
