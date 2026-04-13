# Technical Approach — Underwater Danger Map Pipeline

**Brandon McDonald · Caitlin Roberts · Sydney Ragla**  
University of Michigan · ROB 472 Winter 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Full Pipeline Flowchart](#2-full-pipeline-flowchart)
3. [Depth Map System Flowchart](#3-depth-map-system-flowchart)
4. [Component 1 — SUIM-Net Semantic Segmentation](#4-component-1--suim-net-semantic-segmentation)
5. [Component 2 — SPADE Metric Depth Estimation](#5-component-2--spade-metric-depth-estimation)
6. [Component 3 — Danger Map Fusion](#6-component-3--danger-map-fusion)
7. [Component 4 — Navigation Command](#7-component-4--navigation-command)
8. [Component 5 — Turbidity Robustness](#8-component-5--turbidity-robustness)
9. [Baselining Methodology](#9-baselining-methodology)
10. [Video Pipeline (End-to-End)](#10-video-pipeline-end-to-end)
11. [Script Reference Guide](#11-script-reference-guide)
12. [Configuration Reference](#12-configuration-reference)
13. [Cluster Job Reference](#13-cluster-job-reference)

---

## 1. System Overview

The underwater danger map is an end-to-end perception pipeline for autonomous underwater vehicles (AUVs). Given a single RGB camera frame it produces a **per-pixel collision risk score** in [0, 1] by fusing two complementary models:

| Model | Question answered | Output |
|-------|------------------|--------|
| **SUIM-Net** (Islam et al., 2020) | *What is here?* — semantic class of each pixel | 5-channel sigmoid logits (RO, FV, HD, RI, WR) |
| **SPADE** (Zhang et al., 2025) | *How close is it?* — metric depth to each pixel | Dense float32 depth map (metres) |

These are fused by the core `danger_map()` function:

```
risk(x, y)  =  hazard(x, y)  ×  proximity(x, y)
```

where *hazard* encodes how dangerous a collision with the identified class would be, and *proximity* encodes how soon the AUV would reach it. The risk map feeds a 3×3 sector analysis that outputs a plain-English steering command ("PROCEED", "ASCEND LEFT", "STOP", etc.).

**Key design choices:**
- Using *multiplication* (not addition) means distant harmless objects score near zero automatically — no manual distance gate needed.
- Grayscale background in the overlay eliminates color-on-color clash with underwater blue/green.
- All processing happens at the original frame resolution; SUIM-Net and SPADE operate at their own native resolutions and are resized into the risk map frame.

---

## 2. Full Pipeline Flowchart

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                                    │
│                                                                          │
│   Raw video (.mp4)          Frame folder            Depth folder         │
│   bluerov1/2/multimedia     (PNG / TIF / JPG)       (optional GT depth)  │
└───────────┬──────────────────────┬──────────────────────┬───────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌───────────────────────┐  ┌───────────────┐  ┌──────────────────────────┐
│  cv2.VideoCapture     │  │  _list_images │  │  _load_depth             │
│  (frame-by-frame BGR) │  │  sorted paths │  │  (.tif float32 / 16-bit  │
│  → RGB uint8          │  │  → RGB uint8  │  │   PNG mm→m)              │
└──────────┬────────────┘  └───────┬───────┘  └────────────┬─────────────┘
           └──────────────┬────────┘                       │
                          ▼                                 │
          ┌───────────────────────────┐                    │
          │        RGB frame          │ (H × W × 3 uint8)  │
          └─────────┬─────────┬───────┘                    │
                    │         │                             │
          ┌─────────▼──┐  ┌───▼──────────────────────────┐ │
          │ SUIM-Net   │  │           SPADE               │◄┘
          │            │  │                               │
          │ resize →   │  │  resize RGB → (336×448)       │
          │ (240×320)  │  │  build sparse hint map        │
          │ keras      │  │  (Shi-Tomasi corners +        │
          │ predict()  │  │   GT depth or all-zero)       │
          │            │  │  torch model.forward()        │
          └────┬───────┘  └────────────┬──────────────────┘
               │                       │
               ▼                       ▼
     (240,320,5) float32       (336,448) float32
     sigmoid logits            depth map (metres)
     [RO, FV, HD, RI, WR]
               │                       │
               └──────────┬────────────┘
                          ▼
            ┌─────────────────────────────┐
            │        danger_map()         │
            │                             │
            │  1. resize both to (H,W)    │
            │  2. hazard = max weight     │
            │     over active classes     │
            │  3. proximity = clip(       │
            │     (near_m/depth)^power)   │
            │  4. risk = hazard × prox    │
            │  5. colorize overlay        │
            │     (grayscale bg + HOT     │
            │      heatmap + contours)    │
            └────────────┬────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    (H,W) float32 risk_map    (H,W,3) uint8 overlay
              │
              ▼
     ┌─────────────────┐
     │  nav_command()  │
     │                 │
     │  divide into    │
     │  3×3 sectors    │
     │  mean risk/sec  │
     │  → safest sec   │
     │  → command str  │
     └────────┬────────┘
              │
              ▼
     ┌──────────────────────────┐
     │    draw_nav_overlay()    │
     │    sector grid + banner  │
     └────────┬─────────────────┘
              │
              ▼
     ┌────────────────────────────────────────────────────┐
     │             _make_side_by_side()                   │
     │                                                    │
     │  [ Original ]  [ Depth* ]  [ Danger Map ] [ Nav ] │
     │  *optional — rendered if --show_depth is set       │
     │                                                    │
     │  title bar 30 px │ image H px │ colorbar 22 px     │
     └────────────────────────────────────────────────────┘
              │
              ▼
     cv2.VideoWriter → danger_map.mp4
     (+ optional PLY point clouds every N frames)
```

---

## 3. Depth Map System Flowchart

```
                         RGB FRAME  (H × W × 3 uint8)
                               │
                ┌──────────────┼──────────────────────────────────┐
                │              │                                   │
                │     OPTIONAL: GT depth available?                │
                │              │                                   │
                │       ┌──────┴──────┐                           │
                │       │ YES         │ NO                         │
                │       │             │                            │
                │       ▼             ▼                            │
                │  _load_depth()   depth_m = None                  │
                │  .tif / 16-bit               │                   │
                │  PNG (mm→m)                  │                   │
                │       │              ┌───────┘                   │
                │       ▼              ▼                           │
                │  _build_sparse_map(rgb, depth_m)                 │
                │                                                  │
                │  1. Shi-Tomasi corner detection                  │
                │     cv2.goodFeaturesToTrack(gray,                │
                │       maxCorners=500, quality=0.01,              │
                │       minDist=5)                                 │
                │     → up to 500 corner points                    │
                │                                                  │
                │  2. For each corner (u, v):                      │
                │     d = depth_m[row, col]   (if valid)           │
                │     scale → SPADE sparse space (240×320)         │
                │     place d at (row_s, col_s, 0) of sparse map   │
                │                                                  │
                │  3. Returns (336, 448, 1) float32                │
                │     sparse map (zeros if no GT depth)            │
                │                           │                      │
                └───────────────────────────┘                      │
                                            │                      │
                                            ▼                      │
                            ┌──────────────────────────────┐       │
                            │         SPADE MODEL          │       │
                            │   vendor/SPADE/              │       │
                            │   UnderwaterDepth/           │◄──────┘
                            │                              │
                            │  Config: flsea_sparse_feature│
                            │  Weights: underwater_depth_  │
                            │           pipeline.pt        │
                            │                              │
                            │  ┌─────────────────────────┐ │
                            │  │ Depth-Anything V2 (ViT-S)│ │
                            │  │  monocular backbone      │ │
                            │  │  pretrained on mixed     │ │
                            │  │  in-air + underwater     │ │
                            │  │  data → relative depth   │ │
                            │  └──────────┬──────────────┘ │
                            │             │                 │
                            │  ┌──────────▼──────────────┐ │
                            │  │  Sparse depth fusion     │ │
                            │  │  module                  │ │
                            │  │  conditions on hint map  │ │
                            │  │  → aligns to metric scale│ │
                            │  └──────────┬──────────────┘ │
                            │             │                 │
                            │  ┌──────────▼──────────────┐ │
                            │  │  Anisotropic bilateral   │ │
                            │  │  decoder                 │ │
                            │  │  sharpens depth edges    │ │
                            │  │  along intensity contours│ │
                            │  └──────────┬──────────────┘ │
                            └─────────────┼────────────────┘
                                          │
                                          ▼
                           (336, 448) float32 depth_pred
                              min_pred=0.1 m, max_pred=12 m
                                          │
                               ┌──────────┴───────────┐
                               │                      │
                    ┌──────────▼───────┐   ┌──────────▼──────────────┐
                    │  danger_map()    │   │  _render_depth_panel()  │
                    │                  │   │                         │
                    │  resize to (H,W) │   │  resize to (H,W)        │
                    │  via INTER_NEAR  │   │  clip(depth/12, 0, 1)   │
                    │                  │   │  × 255 → COLORMAP_PLASMA│
                    │  proximity(x,y)  │   │                         │
                    │  = clip(         │   │  PLASMA scale:          │
                    │    (near_m/d)^p, │   │  dark purple = 0m close │
                    │    0, 1)         │   │  bright yellow = 12m far│
                    │                  │   └─────────────────────────┘
                    │  invalid pixels  │
                    │  (d≤0, NaN)      │
                    │  → proximity=0   │
                    └──────────────────┘
```

### Zero-Hint Mode vs Hint-Guided Mode

| Mode | When | Sparse map | Scale accuracy |
|------|------|------------|----------------|
| **Hint-guided** | Frame folder + `--depth_dir` | Corners sampled from GT depth | Metric-accurate (trained regime) |
| **Zero-hint** | Raw video input | All-zeros (336×448×1) | Relies on DA V2 global alignment; may drift ±20-30% from true metric scale |

In zero-hint mode the model falls back to Depth-Anything V2's global-alignment mode. Results are still plausible metric depths (the backbone has metric supervision from its training), but are less constrained than hint-guided mode. Using `--near_m 2.5` (instead of the default 1.0 m) compensates for the slightly looser scale by ensuring objects at typical AUV operating distances (2-4 m) trigger visible risk.

---

## 4. Component 1 — SUIM-Net Semantic Segmentation

### Architecture

SUIM-Net is a fully-convolutional encoder-decoder network (Islam et al., 2020). The model accepts an RGB image resized to **240×320** pixels and predicts per-pixel class probabilities via independent sigmoid activations (not softmax — multiple classes can co-occur at the same pixel).

```
Input RGB (240, 320, 3)
       │
  VGG-style encoder (5 conv blocks, progressively downsample)
       │
  Bottleneck + skip connections
       │
  Decoder (transposed convolutions, upsample back to 240×320)
       │
  5 sigmoid heads (one per class):
       ├── RO  — Robot / Instrument     [0.0, 1.0]
       ├── FV  — Fish / Vertebrate      [0.0, 1.0]
       ├── HD  — Human Diver            [0.0, 1.0]
       ├── RI  — Reef / Invertebrate    [0.0, 1.0]
       └── WR  — Wreck / Ruin           [0.0, 1.0]
Output: (240, 320, 5) float32 sigmoid logits
```

The original SUIM-Net paper defines 8 classes including background, plants, and seafloor, but our pipeline uses only the 5 obstacle-relevant classes above.

### Keras 2.13 Compatibility Shim

The vendor weights (`ckpt_seg_5obj.hdf5`) were trained with an older Keras API that used `input=` / `output=` in `Model()`. Keras ≥2.13 renamed these to `inputs=` / `outputs=`. A shim in `run_video.py` and `run_infer.py` patches `keras.models.Model` to silently remap the old keyword arguments:

```python
_OrigModel = _km.Model
class _ModelShim(_OrigModel):
    def __init__(self, *args, **kwargs):
        if "input" in kwargs and "inputs" not in kwargs:
            kwargs["inputs"] = kwargs.pop("input")
        if "output" in kwargs and "outputs" not in kwargs:
            kwargs["outputs"] = kwargs.pop("output")
        super().__init__(*args, **kwargs)
_km.Model = _ModelShim
```

### Class Encoding (Overlay Colors)

| Class | Channel | Overlay contour color (BGR) |
|-------|---------|----------------------------|
| RO — Robot | 0 | Cyan (255, 200, 0) |
| FV — Fish | 1 | Green (50, 220, 50) |
| HD — Diver | 2 | Red (0, 0, 255) — highest danger |
| RI — Reef | 3 | Orange (0, 165, 255) |
| WR — Wreck | 4 | Yellow (0, 220, 220) |

---

## 5. Component 2 — SPADE Metric Depth Estimation

### Architecture

SPADE is a two-stage depth estimator. The checkpoint used in this project (`underwater_depth_pipeline.pt`) is the **DA V2 + Global Alignment** configuration — the Depth-Anything V2 ViT-S backbone with global scale alignment, without the full Deformable Attention Transformer (DAT) refinement head (the official fine-tuned SPADE weights are restricted-access; see note below).

```
Input: RGB (336, 448, 3) normalized with ImageNet mean/std
       Sparse depth map (336, 448, 1) float32  [zeros in video mode]
       ┌──────────────────────────────────────────────────────────┐
       │  Stage 1: Depth-Anything V2 (ViT-S backbone)             │
       │    Patch embed + positional encoding                      │
       │    12 Transformer blocks (self-attention on patches)      │
       │    → relative depth feature F ∈ ℝ^(H'×W'×C)             │
       └────────────────────────────┬─────────────────────────────┘
                                    │
       ┌────────────────────────────▼─────────────────────────────┐
       │  Stage 2: Sparse-depth fusion + Global Alignment         │
       │    Sparse hint map → scale the relative depth feature    │
       │    Anisotropic bilateral decoder (edge-aware smoothing)   │
       │    → metric depth ∈ [0.1, 12.0] metres                   │
       └──────────────────────────────────────────────────────────┘
Output: (336, 448) float32 depth map in metres
```

> **Note on weights.** The official SPADE fine-tuned checkpoint is hosted on a restricted Google Drive account and is not publicly accessible. We assembled `underwater_depth_pipeline.pt` from the publicly available DA V2 ViT-S backbone with the DAT head initialized but not fine-tuned on underwater data. This corresponds to the **DA V2 + GA** baseline row in the SPADE paper (Table 2), which the paper reports at MAE ≈ 0.277 m and AbsRel ≈ 0.081 on FLSea ≤10 m. Our measured results (MAE = 0.387 m, AbsRel = 0.111 at ≤10 m) are consistent given that we simulate sparse hints from dense GT depth using Shi-Tomasi corners, while the paper uses a real sparse sensor.

### Sparse Depth Hint Simulation

When ground-truth depth is available (frame-folder mode with `--depth_dir`), sparse hints are simulated to match the SPADE evaluation protocol:

```
1. Convert RGB → grayscale
2. cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=5)
   → up to 500 Shi-Tomasi corner points (u_i, v_i)
3. For each corner:
   d_i = depth_m[v_i, u_i]   (skip if d_i ≤ 0 or NaN)
   row_s = v_i × 240 / H     (scale to SPADE sparse space 240×320)
   col_s = u_i × 320 / W
   sparse[row_s, col_s] = d_i
4. Output: (336, 448, 1) sparse map (resized from 240×320 → 336×448 internally)
```

### Model Loading

The `_load_spade()` function must temporarily change the working directory to `vendor/SPADE/` so that SPADE's internal relative imports resolve correctly. The original CWD is restored in a `finally` block:

```python
def _load_spade(weights_path):
    orig_cwd = os.getcwd()
    try:
        sys.path.insert(0, str(VENDOR_SPADE))
        os.chdir(VENDOR_SPADE)
        from UnderwaterDepth.utils.config import get_config
        from UnderwaterDepth.models.builder import build_model
        config = get_config("SPADE", "eval", "flsea_sparse_feature",
                            pretrained_resource=f"local::{weights_path}")
        model = build_model(config).cuda().eval()
        return model
    finally:
        os.chdir(orig_cwd)
```

The `flsea_sparse_feature` dataset config sets `min_depth=0.1`, `max_depth=18.0`, `sparse_feature_height=240`, `sparse_feature_width=320`.

---

## 6. Component 3 — Danger Map Fusion

**File:** `src/danger_map/__init__.py`

### Risk Formula

```
proximity(x, y)  =  clip( (near_m / depth(x, y))^power, 0.0, 1.0 )

hazard(x, y)     =  max{ weight[c]  :  seg[x, y, c] > seg_threshold }

risk(x, y)       =  hazard(x, y)  ×  proximity(x, y)   ∈ [0, 1]
```

### Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `near_m` | 2.5 m (video) / 1.0 m (test) | Objects closer than this get full proximity (1.0). Increase for faster vehicles. |
| `proximity_power` | 1.0 | Fall-off exponent. 1.0 = linear 1/d, 2.0 = quadratic, 0.5 = slow decay. |
| `seg_threshold` | 0.5 | Sigmoid cutoff for "class is active". Lower = more pixels flagged. |
| `display_gamma` | 0.5 (video) / 1.0 (test) | Gamma compression on the VISUALISATION only. `risk_map` always returns raw values. |
| `overlay_alpha` | 0.5 | Blend weight of the heatmap (0 = transparent, 1 = fully opaque). |

### Hazard Weights

| Class | Weight | Rationale |
|-------|--------|-----------|
| HD — Human Diver | 1.0 | Highest priority — diver injury is unacceptable |
| WR — Wreck / Ruin | 0.9 | Large rigid structure, high collision damage |
| RI — Reef / Invertebrate | 0.8 | Hard structural hazard, ecologically sensitive |
| FV — Fish / Vertebrate | 0.5 | Mobile, moderate impact risk |
| RO — Robot / Instrument | 0.2 | Not a natural obstacle for the AUV |

### Display Gamma

Raw risk values in typical underwater footage fall in 0.1–0.4, which maps to near-black in the HOT colormap. `display_gamma < 1.0` applies a power-law stretch to make these visible without changing the underlying risk score:

```
vis_risk = risk ^ display_gamma   (only used for the overlay image)
```

| Raw risk | gamma=1.0 | gamma=0.5 | HOT colour at 0.5 |
|----------|-----------|-----------|-------------------|
| 0.04 | 0.04 (black) | 0.20 | dark red |
| 0.10 | 0.10 (dark) | 0.32 | red |
| 0.25 | 0.25 (red) | 0.50 | orange |
| 0.50 | 0.50 (orange) | 0.71 | yellow |
| 1.00 | 1.00 (white) | 1.00 | white |

### Overlay Assembly

1. **Grayscale background** — desaturate RGB to eliminate blue/green color clash
2. **HOT heatmap** — apply `cv2.COLORMAP_HOT` to `vis_risk × 255`
3. **Alpha blend** — `blended = (1 - w) × bg + w × heatmap` where `w = vis_risk × alpha`
4. **Class contours** — find connected components per class, draw contours + centroid label

---

## 7. Component 4 — Navigation Command

**File:** `src/danger_map/navigate.py`

### Sector Grid

The frame is divided into a **3×3 grid** of equal sectors:

```
┌──────────┬──────────┬──────────┐
│  top-l   │  top-c   │  top-r   │
│  (TL)    │  (TC)    │  (TR)    │
├──────────┼──────────┼──────────┤
│  mid-l   │  center  │  mid-r   │
│  (ML)    │  (C)     │  (MR)    │
├──────────┼──────────┼──────────┤
│  bot-l   │  bot-c   │  bot-r   │
│  (BL)    │  (BC)    │  (BR)    │
└──────────┴──────────┴──────────┘
```

For each sector the mean risk across all pixels is computed. The AUV steers **toward** the safest (lowest mean-risk) sector:

| Safest sector | Command | Safest sector | Command |
|---------------|---------|---------------|---------|
| top-l | ASCEND LEFT | top-c | ASCEND |
| top-r | ASCEND RIGHT | mid-l | GO LEFT |
| center | PROCEED | mid-r | GO RIGHT |
| bot-l | DESCEND LEFT | bot-c | DESCEND |
| bot-r | DESCEND RIGHT | — | — |

**STOP:** If overall risk > `DANGER_THRESHOLD` (0.55) AND even the safest sector has risk above `CLEAR_THRESHOLD` (0.20), the system commands STOP regardless.

### Risk Levels

| Level | Overall mean risk | Banner colour |
|-------|-------------------|---------------|
| CLEAR | < 0.20 | Green |
| CAUTION | 0.20 – 0.55 | Orange |
| DANGER | > 0.55 | Red |

### Optional 3-D Risk Point Cloud

When metric depth and camera intrinsics are available, each pixel is unprojected into 3-D camera-frame coordinates:

```
X = (u - cx) × depth / fx
Y = (v - cy) × depth / fy
Z = depth
```

Each 3-D point is colour-coded green (risk=0) → red (risk=1) and written as an ASCII PLY file. PLY files can be viewed in MeshLab or CloudCompare and are generated every `--ply_every` frames when `--save_ply` is passed.

---

## 8. Component 5 — Turbidity Robustness

**Files:** `src/augment/turbidity.py`, `src/augment/run_sweep.py`

### Turbidity Model

Simulates three physical mechanisms of underwater optical degradation:

```
apply_turbidity(img, level):

  1. Gaussian blur (particle light scattering)
     sigma = 4.0 × level  [px]
     cv2.GaussianBlur(img, ksize=0, sigmaX=sigma)

  2. Backscatter veil (suspended particle haze)
     veil_color = [140, 178, 153]  (greenish-white)
     veil_blend  = 0.45 × level
     img = (1 - veil_blend) × img + veil_blend × veil_color

  3. Red-channel attenuation (water absorbs red wavelengths)
     red_atten = 0.35 × level
     img[:, :, 0] *= (1 - red_atten)  [R channel]

  All effects scale linearly with level ∈ [0.0, 1.0].
```

### Robustness Sweep

`run_sweep.py` evaluates SUIM-Net at five turbidity levels:

```
for level in [0.0, 0.25, 0.50, 0.75, 1.0]:
    for image in test_set:
        augmented = apply_turbidity(image, level)
        logits    = suimnet.predict(augmented)
        metrics   = compute_iou_dice_prec_recall(logits, gt_masks)
    write_csv(level, per_class_metrics)
write_summary_csv()   # one row per level × class
```

**Results:**

| Turbidity level | mIoU | Δ from baseline |
|----------------|------|-----------------|
| 0.0 (clean) | 0.779 | — |
| 0.25 | ~0.74 | ~−5% |
| 0.50 | 0.670 | −14% |
| 0.75 | ~0.55 | ~−29% |
| 1.00 | 0.390 | −54% |

---

## 9. Baselining Methodology

### SUIM-Net Baselines

#### Step 1 — In-Domain Baseline (SUIM TEST)
```
dataset : SUIM test split (110 images)
script  : python -m src.suimnet.run_infer
          --images_dir  data/suim/TEST/images/
          --output_dir  outputs/suimnet/predictions/
          --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5
          --save_logits    ← saves .npz logit files for threshold sweep

python -m src.suimnet.metric_calc
          --pred_dir   outputs/suimnet/predictions/
          --gt_dir     data/suim/TEST/masks/
          --out_csv    reports/suimnet/suim_metrics.csv
```

Produces per-image, per-class IoU/Dice/Precision/Recall in a CSV. The 6 publication-ready charts are then generated by `src/suimnet/chart_metrics.py`.

#### Step 2 — Threshold Optimisation (post-hoc, no re-inference)
```
python -m src.suimnet.threshold_sweep
          --logits_dir outputs/suimnet/logits/
          --gt_dir     data/suim/TEST/masks/
          --out_dir    reports/suimnet/threshold_sweep/
```
Sweeps thresholds [0.05 … 0.95] per class, reports optimal threshold and IoU gain. Results are printed as a YAML block that can be pasted into `configs/datasets.yaml`.

#### Step 3 — Cross-Dataset Evaluation (DeepFish, USIS10K)

Same `run_infer.py` + `metric_calc.py` workflow, with converted dataset layouts:

| Dataset | Converter script | Conversion output |
|---------|-----------------|-------------------|
| DeepFish | `src/suimnet/convert_deepfish.py` | Binary FV/ masks |
| USIS10K | `src/suimnet/convert_usis10k.py` | RO/, FV/, HD/, RI/, WR/ masks from COCO JSON |

The cross-dataset runs use **no threshold tuning** — the default 0.5 threshold is used, so the IoU drop reflects genuine generalisation failure rather than threshold mismatch.

#### Step 4 — Turbidity Sweep
```
sbatch --export=DATASET=suim cluster/turbidity_sweep.sbat
# → reports/turbidity/level_*/metrics.csv
# → reports/turbidity/turbidity_summary.csv
python scripts/make_report_figures.py  # regenerates turbidity charts
```

### SPADE Baselines

#### Step 1 — Download and Convert Datasets
```
bash scripts/download_spade_data.sh   # downloads FLSea and SeaThru

# Convert FLSea (HuggingFace parquet → TIFF + sparse CSV):
sbatch --export=DATASET=flsea cluster/spade_convert.sbat
# → data/spade_lists/flsea_test.txt  (absolute paths)
# → data/flsea/rgb/*.tif + data/flsea/depth/*.tif + data/flsea/sparse/*.csv

# Convert SeaThru (Kaggle ZIP → matched RGB/depth pairs):
sbatch --export=DATASET=seathru cluster/spade_convert.sbat
# → data/spade_lists/seathru_test.txt
```

#### Step 2 — Run SPADE Evaluation
```
sbatch --export=DATASET=flsea  cluster/spade_metrics.sbat
sbatch --export=DATASET=seathru cluster/spade_metrics.sbat
# → reports/spade/flsea_metrics.csv
# → reports/spade/seathru_metrics.csv
```

Each job runs `src/spade/run_eval.py` which wraps `vendor/SPADE/evaluate.py`'s `eval_model()`. Metrics are computed at three depth ranges (≤10 m, ≤5 m, ≤2 m) using: MAE, RMSE, AbsRel, SILog, δ<1.25, δ<1.25², δ<1.25³.

#### Step 3 — Chart Generation
```
python scripts/make_report_figures.py   # or src/spade/chart_metrics.py
```

### Latency Profiling (CPU Baseline)
```
python -m src.danger_map.profile_latency \
    --frames_dir data/flsea/rgb/ \
    --depth_dir  data/flsea/depth/ \
    --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \
    --spade_weights   /path/to/underwater_depth_pipeline.pt \
    --n_frames 50 \
    --out_csv  reports/latency_local.csv
```

On GPU (ARC A40) the pipeline runs at ~13 FPS. CPU latency is reported in `reports/latency_local.csv`.

---

## 10. Video Pipeline (End-to-End)

The video pipeline (`run_video.py`) connects all components and is the primary deliverable for demonstrating the system on real BlueROV footage.

### Execution Flow

```
1. Load models
   ├── SUIM-Net: SUIM_Net(im_res=(240,320), n_classes=5).model
   │             .load_weights(suimnet_weights)
   └── SPADE:    _load_spade(spade_weights)
                 (chdir to vendor/SPADE/, build model, move to CUDA, eval mode)

2. Open source
   ├── Video mode:  cv2.VideoCapture(video_file)
   └── Frame mode:  sorted list of PNG/TIF/JPG in frames_dir

3. Per-frame loop:
   a. Read frame (BGR→RGB)
   b. _run_suimnet(model, rgb) → (240,320,5) logits
   c. _run_spade(model, rgb, depth_m) → (336,448) depth
   d. danger_map(rgb, logits, depth,
                 near_m=args.near_m,
                 display_gamma=args.display_gamma) → risk_map, overlay
   e. nav_command(risk_map, depth) → NavResult
   f. draw_nav_overlay(overlay, nav_result) → nav_panel
   g. [optional] _render_depth_panel(depth) → depth_panel
   h. _make_side_by_side(rgb, overlay, nav_panel, depth_panel) → frame
   i. VideoWriter.write(frame)
   j. [optional every ply_every frames] write PLY point cloud

4. Release video writer
```

### Output Video Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Original           Depth Map*         Danger Map  Navigation │  ← 30 px title bar
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   [Original RGB]  [PLASMA depth]  [HOT heatmap]  [Sectors]  │  ← H px image rows
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  (no bar)         [depth 0m…12m]  [risk 0…1.0]  [risk 0…1.0]│  ← 22 px colorbars
└──────────────────────────────────────────────────────────────┘
* Only present when --show_depth is passed (default: ON in cluster jobs)
```

### Cluster Submission

```bash
# Re-run all three videos with improved parameters:
sbatch --export=VIDEO=videos/raw/bluerov1.mp4 \
       cluster/video_danger_map.sbat

sbatch --export=VIDEO=videos/raw/bluerov2.mp4 \
       cluster/video_danger_map.sbat

sbatch --export=VIDEO=videos/raw/multimedia-unexpected-1920x1080-1.mp4 \
       cluster/video_danger_map.sbat

# Monitor:
squeue -u $USER

# Retrieve results:
scp bamcd@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/videos/processed/bluerov1/danger_map.mp4 videos/processed/bluerov1/
```

---

## 11. Script Reference Guide

### `src/danger_map/` — Core Pipeline

| File | Purpose | Key inputs | Key outputs |
|------|---------|-----------|------------|
| [`__init__.py`](src/danger_map/__init__.py) | Core risk map generation | `rgb`, `seg_logits`, `depth_m` | `risk_map` (H×W float32), `overlay` (H×W×3 uint8) |
| [`navigate.py`](src/danger_map/navigate.py) | Navigation command from risk map | `risk_map`, optional `depth_m` | `NavResult` (command, sector_risks, risk_level); optional PLY |
| [`run_video.py`](src/danger_map/run_video.py) | Full end-to-end video/frame pipeline | `--video_file` OR `--frames_dir`; `--spade_weights` | `danger_map.mp4`, optional PLY point clouds |
| [`quick_test.py`](src/danger_map/quick_test.py) | CPU smoke test, no GPU needed | 8 bundled SUIM images; synthetic flat depth | `reports/danger_map/quick_test/*.png` |
| [`profile_latency.py`](src/danger_map/profile_latency.py) | Benchmark per-frame inference latency | Frame dir, N frames to time | CSV with per-frame suimnet/spade/fusion/total latency in ms |

**How they work together:** `quick_test.py` is the development smoke-test (run this to verify your environment is working before submitting cluster jobs). `run_video.py` is the production entry point — it calls both models, then `danger_map()` from `__init__.py`, then `nav_command()` and `draw_nav_overlay()` from `navigate.py`.

---

### `src/suimnet/` — Segmentation Evaluation

| File | Purpose | Key inputs | Key outputs |
|------|---------|-----------|------------|
| [`run_infer.py`](src/suimnet/run_infer.py) | SUIM-Net inference on image directories | `--images_dir`, `--suimnet_weights` | RGB prediction masks (PNG); optional logit NPZ files |
| [`metric_calc.py`](src/suimnet/metric_calc.py) | Compute IoU/Dice/Precision/Recall per class | Prediction masks dir, GT masks dir | Per-image per-class CSV |
| [`threshold_sweep.py`](src/suimnet/threshold_sweep.py) | Post-hoc threshold optimisation | Saved logit NPZ files, GT masks | Threshold-vs-IoU CSV; optimal threshold table |
| [`chart_metrics.py`](src/suimnet/chart_metrics.py) | Generate 6 publication charts | Per-image metrics CSV | 6 PNGs: grouped bars, boxplots, macro bar, heatmap, PR scatter, breakdown |
| [`convert_usis10k.py`](src/suimnet/convert_usis10k.py) | Convert USIS10K COCO → SUIM mask layout | COCO JSON + images dir | Per-class binary PNG masks (RO/, FV/, HD/, RI/, WR/ subdirs) |
| [`convert_deepfish.py`](src/suimnet/convert_deepfish.py) | Prepare DeepFish masks for metric_calc | DeepFish binary masks | FV/ subdirectory with binary fish masks |

**How they work together:** `run_infer.py` → `metric_calc.py` → `chart_metrics.py` is the standard evaluation chain. `threshold_sweep.py` is run after `run_infer.py --save_logits` to optimise per-class thresholds without rerunning inference. The `convert_*` scripts are run once per dataset to produce the expected directory layout.

**Quick workflow:**
```bash
# 1. Run inference (save logits for threshold sweep)
python -m src.suimnet.run_infer --images_dir data/suim/TEST/images/ \
    --output_dir outputs/suimnet/suim/ --save_logits

# 2. Compute metrics
python -m src.suimnet.metric_calc \
    --pred_dir outputs/suimnet/suim/ \
    --gt_dir data/suim/TEST/masks/ \
    --out_csv reports/suimnet/suim_metrics.csv

# 3. (Optional) Find optimal thresholds
python -m src.suimnet.threshold_sweep \
    --logits_dir outputs/suimnet/suim/logits/ \
    --gt_dir data/suim/TEST/masks/

# 4. Generate charts
python -m src.suimnet.chart_metrics \
    --metrics_csv reports/suimnet/suim_metrics.csv \
    --out_dir figures/charts/
```

---

### `src/spade/` — Depth Evaluation

| File | Purpose | Key inputs | Key outputs |
|------|---------|-----------|------------|
| [`run_eval.py`](src/spade/run_eval.py) | Evaluate SPADE on benchmark datasets | `--dataset` tag, `--weights` path | CSV with MAE/RMSE/AbsRel/SILog/δ-accuracy at three depth ranges |
| [`_spade_utils.py`](src/spade/_spade_utils.py) | Shared depth loading + sparse hint CSV generation | RGB image, dense depth map | Sparse CSV files (row, col, depth) at 240×320 resolution |
| [`chart_metrics.py`](src/spade/chart_metrics.py) | Generate SPADE accuracy charts | Metrics CSV from run_eval | Error-by-range and accuracy-by-range PNG charts |
| [`convert_flsea.py`](src/spade/convert_flsea.py) | Convert FLSea-VI HuggingFace parquet → SPADE format | validation-*.parquet files | Float32 TIFF RGB + depth, sparse CSV, filenames list |
| [`convert_seathru.py`](src/spade/convert_seathru.py) | Convert SeaThru Kaggle dataset → SPADE format | SeaThru ZIP/directory | Matched RGB/depth TIFFs, filenames list |

**How they work together:** `convert_flsea.py` or `convert_seathru.py` prepares the dataset once (run via `cluster/spade_convert.sbat`). `run_eval.py` then loads the filenames list from `configs/spade_datasets.yaml` and runs `vendor/SPADE/evaluate.py`'s `eval_model()`. Results feed `chart_metrics.py` and `scripts/make_report_figures.py`.

---

### `src/augment/` — Turbidity Robustness

| File | Purpose | Key inputs | Key outputs |
|------|---------|-----------|------------|
| [`turbidity.py`](src/augment/turbidity.py) | Apply simulated turbidity to an RGB image | `(H,W,3) uint8` image, level ∈ [0,1] | Augmented `(H,W,3) uint8` image (copy) |
| [`run_sweep.py`](src/augment/run_sweep.py) | Evaluate SUIM-Net robustness across turbidity levels | Images dir, GT masks dir | Per-level CSV + turbidity_summary.csv |

**How they work together:** `run_sweep.py` imports `apply_turbidity()` from `turbidity.py`. For each level it applies the augmentation to each test image, runs SUIM-Net, and computes metrics. The summary CSV is read by `scripts/make_report_figures.py` to produce the turbidity mIoU chart and heatmap.

---

### `src/common/` — Shared Utilities

| File | Purpose | Key inputs | Key outputs |
|------|---------|-----------|------------|
| [`config.py`](src/common/config.py) | Load dataset paths from YAML configs | `profile` name, `dataset` name | `DatasetPaths` (images_dir, labels_dir, thresholds) |

**Usage:** Cluster batch scripts (e.g., `suimnet_infer.sbat`) read the profile (`greatlakes`) and dataset (`suim`) from environment variables, then call `resolve_dataset_paths()` to get machine-specific absolute paths without hardcoding them in the scripts.

---

### `scripts/` — Report Generation and Utilities

| File | Purpose |
|------|---------|
| [`make_report_figures.py`](scripts/make_report_figures.py) | Regenerate ALL report figures from CSVs. Reads `reports/suimnet/`, `reports/spade/`, `reports/turbidity/`. Outputs `figures/charts/`. Run after any evaluation jobs complete. |
| [`make_gif.py`](scripts/make_gif.py) | Extract a frame range from a processed danger-map video and save as animated GIF. `--start/--end/--step/--fps/--scale`. Also supports `--stills_dir` to save individual PNGs. |
| [`make_turbidity_examples.py`](scripts/make_turbidity_examples.py) | Generate 5-level turbidity strips from existing danger-map outputs. Extracts the **original RGB panel only** (strips title bar and colorbar) before applying turbidity — the danger map and nav panels are NOT affected. |
| [`render_ply.py`](scripts/render_ply.py) | Render a risk PLY point cloud to perspective + top-down PNG figures. |
| [`compare_pointcloud.py`](scripts/compare_pointcloud.py) | Side-by-side figure comparing danger map overlay with 3-D point cloud. |
| [`build_spade_weights.py`](scripts/build_spade_weights.py) | Assemble the `underwater_depth_pipeline.pt` checkpoint from the publicly available DA V2 ViT-S weights (used because the official SPADE weights are restricted-access). |
| [`download_spade_data.sh`](scripts/download_spade_data.sh) | Download FLSea-VI (HuggingFace) and SeaThru (Kaggle) datasets. |
| [`download_suimnet_data.sh`](scripts/download_suimnet_data.sh) | Download SUIM, DeepFish, and USIS10K datasets. |
| [`launch_danger_pipeline.sh`](scripts/launch_danger_pipeline.sh) | Local end-to-end launch script (CPU, for testing). |
| [`launch_spade.sh`](scripts/launch_spade.sh) | Local SPADE evaluation launch script. |
| [`launch_all.sh`](scripts/launch_all.sh) | Submit all ARC cluster jobs in sequence. |

---

## 12. Configuration Reference

### `configs/profiles.yaml`

Defines machine-specific data root paths. The `greatlakes` profile is used by all cluster jobs.

```yaml
profiles:
  local:
    data_root: ~/data/rob472
    outputs_root: outputs/
  greatlakes:
    data_root: /scratch/rob572w26_class_root/rob572w26_class/${USER}
    outputs_root: outputs/
```

### `configs/datasets.yaml`

Dataset definitions with relative paths from `data_root`, GT mask layout, and optionally optimised per-class thresholds from `threshold_sweep.py`.

```yaml
datasets:
  suim:
    images_rel: suim/TEST/images
    labels_rel:  suim/TEST/masks
    has_labels:  true
    thresholds:  {FV: 0.5, HD: 0.5, RI: 0.5, RO: 0.5, WR: 0.5}
  deepfish:
    images_rel: deepfish/TEST/images
    labels_rel:  deepfish/TEST/masks/FV
    has_labels:  true
```

### `configs/spade_datasets.yaml`

SPADE evaluation dataset paths, used by `src/spade/run_eval.py` and `cluster/spade_metrics.sbat`. Each dataset entry specifies the filenames list file (written by `convert_*.py`) and the data/GT root paths. `eval_ranges` defines the depth cutoffs at which metrics are reported.

```yaml
datasets:
  flsea:
    filenames_file_eval: ""   # filled by convert_flsea.py → data/spade_lists/flsea_test.txt
    data_path_eval: "/"
    gt_path_eval:   "/"
    min_depth: 0.1
    max_depth: 18.0
    eval_ranges: [10, 5, 2]   # metres
```

---

## 13. Cluster Job Reference

All cluster scripts live in `cluster/` and share a common pattern: they lock a virtualenv directory with `flock` to prevent concurrent pip installs, then activate it and run the Python module.

| Script | GPU? | Key env vars | What it runs |
|--------|------|-------------|--------------|
| `video_danger_map.sbat` | Yes (1 GPU) | `VIDEO`, `MAX_FRAMES`, `FPS`, `NEAR_M`, `DISPLAY_GAMMA`, `SHOW_DEPTH`, `PLY_EVERY` | `python -m src.danger_map.run_video` on a raw .mp4 |
| `danger_map.sbat` | Yes (1 GPU) | `FRAMES_DIR`, `DEPTH_DIR`, `OUT_DIR` | `python -m src.danger_map.run_video --frames_dir` on frame folders |
| `druva_danger_map.sbat` | Yes (1 GPU) | `VIDEO` | Same as `video_danger_map.sbat` but tuned for DRUVA dataset videos |
| `turbidity_sweep.sbat` | No | `DATASET` | `python -m src.augment.run_sweep` |
| `suimnet_infer.sbat` | No | `DATASET`, `PROFILE` | `python -m src.suimnet.run_infer` |
| `suimnet_infer_simple.sbat` | No | `IMAGES_DIR`, `OUT_DIR` | `python -m src.suimnet.run_infer` with explicit paths (no config system) |
| `suimnet_metrics.sbat` | No | `DATASET`, `PROFILE` | `metric_calc.py` → `chart_metrics.py` |
| `suimnet_sweep.sbat` | No | `DATASET` | `python -m src.suimnet.threshold_sweep` |
| `suimnet_quick.sbat` | No | — | `python -m src.danger_map.quick_test` |
| `spade_convert.sbat` | No | `DATASET` | `python -m src.spade.convert_flsea` or `convert_seathru` |
| `spade_infer.sbat` | Yes (1 GPU) | `DATASET` | Sparse hint generation for a full dataset |
| `spade_metrics.sbat` | Yes (1 GPU) | `DATASET` | `python -m src.spade.run_eval` → metrics CSV |

### Virtual Environment Strategy

Because PyTorch and TensorFlow have conflicting dependency requirements (TF downgrades `typing_extensions`), a **single combined venv** `rob472-spade` is used on the cluster that installs `requirements_spade.txt` (PyTorch + TF together with `typing_extensions` re-pinned afterward). The venv is created once per user in scratch and reused across all jobs.

```bash
# Location of shared cluster venv:
/scratch/rob572w26_class_root/rob572w26_class/${USER}/venvs/rob472-spade

# Location of SPADE weights:
/scratch/rob572w26_class_root/rob572w26_class/${USER}/spade_weights/underwater_depth_pipeline.pt
```

---

## References

1. Islam et al., "Semantic Segmentation of Underwater Imagery: Dataset and Benchmark," IROS 2020. [arXiv:2004.01241](https://arxiv.org/abs/2004.01241)
2. Zhang et al., "SPADE: Sparsity Adaptive Depth Estimator for Zero-Shot, Real-Time, Monocular Depth Estimation in Underwater Environments," 2025. [arXiv:2510.25463](https://arxiv.org/abs/2510.25463)
3. Ebner et al., "Metrically Scaled Monocular Depth Estimation through Sparse Priors for Underwater Robots," 2023. [arXiv:2310.16750](https://arxiv.org/abs/2310.16750)
4. Yang et al., "Depth Anything V2," 2024. [arXiv:2406.09414](https://arxiv.org/abs/2406.09414)
5. Randall & Treibitz, "FLSea-VI: Forward-Looking Stereo and Inertial Dataset," IROS 2023.
6. Akkaynak & Treibitz, "Sea-thru: A Method for Removing Water from Underwater Images," CVPR 2019.
7. Saleh et al., "DeepFish: A Realistic Fish-Habitat Dataset," Scientific Reports 2020.
8. Lian et al., "USIS10K: Underwater Salient Instance Segmentation Dataset," 2024. [arXiv:2406.06039](https://arxiv.org/abs/2406.06039)
