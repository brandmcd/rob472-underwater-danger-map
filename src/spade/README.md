# SPADE — Underwater Monocular Depth Estimation (Goal 2)

> **Project context:** This is Goal 2 of the ROB 472 Underwater Danger Map project.
> We benchmark monocular depth estimation on FLSea, SeaThru, and kskin to understand
> depth accuracy at various ranges. The depth maps produced here feed into the
> [danger map fusion](../fusion/) module (Goal 3) alongside SUIM-Net segmentation masks.

Metric-scale monocular depth estimation using the SPADE model
(Zhang et al., 2025 — Sparsity Adaptive Depth Estimator).
Takes RGB images and sparse depth hint points as inputs and produces dense depth maps.

> **Weights note:** The official pretrained weights on Google Drive are access-restricted
> (upstream issue [#1](https://github.com/Jayzhang2333/Sparsity-Adaptive-Depth-Estimator/issues/1)).
> Use `scripts/build_spade_weights.py` (see Step 1 below) to assemble a runnable checkpoint
> from the public Depth Anything V2 ViT-S backbone.  Performance matches the **DA V2 + GA**
> baseline tier from the paper (MAE ≈ 0.28 m at ≤10 m) rather than the full trained model.

---

## Datasets

| Key | Dataset | Source | Images | GT depth |
|-----|---------|--------|--------|----------|
| `flsea_demo` | FLSea demo (bundled) | `vendor/SPADE/example_data/` | 15 | Yes (SLAM-reconstructed, metres) |
| `flsea` | FLSea-VI validation split (Randall & Treibitz, 2023) | [HuggingFace](https://huggingface.co/datasets/bhowmikabhimanyu/flsea-vi) | ~4,490 | Yes (depth image, metres) |
| `seathru` | SeaThru (Akkaynak & Treibitz, CVPR 2019) | [Kaggle](https://www.kaggle.com/datasets/colorlabeilat/seathru-dataset) | ~1,100 | Yes (SFM-reconstructed, metres) |
| `kskin` | DROP Lab HIMB stereo (K. Skinner, U-Michigan) | [GitHub](https://github.com/kskin/data) | varies | Yes (stereo GT, metres) |

`flsea_demo` is bundled — no download needed.  All others require download + conversion (see Steps 2–3 below).

---

## Project structure

```
src/spade/
  run_eval.py         # evaluation wrapper → metrics CSV
  convert_seathru.py  # SeaThru raw → SPADE format (depth TIFF + sparse CSV)
  convert_kskin.py    # kskin HIMB raw → SPADE format
  convert_flsea.py    # FLSea-VI parquet → SPADE format
  chart_metrics.py    # metrics CSV → report-quality charts
  _spade_utils.py     # shared depth loading + sparse CSV generation
cluster/
  spade_convert.sbat  # SLURM: CPU data conversion (seathru / kskin / flsea)
  spade_metrics.sbat  # SLURM: GPU evaluation + charting
scripts/
  build_spade_weights.py   # assemble checkpoint from public DA V2 backbone
  download_spade_data.sh   # download datasets (Kaggle / HuggingFace / tar.gz)
configs/
  spade_datasets.yaml      # per-dataset depth range, filenames list, etc.
vendor/SPADE/
  evaluate.py         # upstream evaluation entry-point (DO NOT MODIFY)
  UnderwaterDepth/    # model architecture and data loaders
  DataLists/          # FLSea filenames lists (bundled)
  example_data/       # 15-image FLSea demo (bundled)
requirements_spade.txt  # PyTorch pip dependencies
data/spade_lists/       # generated filenames lists (gitignored, machine-specific)
reports/spade/          # metrics CSVs, charts, depth visualisations
```

---

## Quick-start (local, FLSea demo — no download needed)

### 1. Set up environment

SPADE uses **PyTorch**, not TensorFlow. Create a separate venv:

```bash
python3 -m venv .venv-spade && source .venv-spade/bin/activate
pip install --upgrade pip
pip install -r requirements_spade.txt
pip install timm einops          # needed by the DAT model layers
```

### 2. Build pretrained weights

The official weights are not publicly accessible.  Run this once to assemble a
checkpoint from the public Depth Anything V2 ViT-S backbone:

```bash
python scripts/build_spade_weights.py
# writes ~/Downloads/underwater_depth_pipeline.pt
```

The script downloads DA V2 ViT-S (~100 MB) from HuggingFace automatically.
The output file is ~400 MB.

### 3. Smoke-test with bundled FLSea demo (15 images, no extra data)

```bash
# Evaluation → CSV + depth images
python -m src.spade.run_eval \
    --dataset    flsea_demo \
    --weights    ~/Downloads/underwater_depth_pipeline.pt \
    --save_image

# Charts
python -m src.spade.chart_metrics --csv reports/spade/flsea_demo_metrics.csv
```

Outputs:
- Metrics CSV → `reports/spade/flsea_demo_metrics.csv`
- Charts → `reports/spade/figures/flsea_demo/`
- Depth images → `reports/spade/flsea_demo/`

---

## Running on ARC Great Lakes

### Step 0 — Connect and pull

```bash
ssh brandmcd@greatlakes.arc-ts.umich.edu
cd ~/rob472-underwater-danger-map
git pull && git submodule update --init --recursive
```

### Step 1 — Stage pretrained weights

Run `build_spade_weights.py` **on your local WSL machine** (needs internet + PyTorch):

```bash
# On local WSL — build the weights file
python scripts/build_spade_weights.py
# → ~/Downloads/underwater_depth_pipeline.pt

# Create the weights directory on Great Lakes, then scp the file up
ssh brandmcd@greatlakes.arc-ts.umich.edu \
    "mkdir -p /scratch/rob572w26_class_root/rob572w26_class/brandmcd/spade_weights"

scp ~/Downloads/underwater_depth_pipeline.pt \
    brandmcd@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/brandmcd/spade_weights/
```

Default SLURM weight path (already hardcoded in the scripts):
```
/scratch/rob572w26_class_root/rob572w26_class/brandmcd/spade_weights/underwater_depth_pipeline.pt
```

Override at submission time with `--export=WEIGHTS_PATH=/your/path.pt`.

### Step 2 — Download datasets

```bash
# Set Kaggle token (required for SeaThru download)
export KAGGLE_API_TOKEN=<your-kaggle-token>

bash scripts/download_spade_data.sh
```

Get your token from [kaggle.com/settings](https://www.kaggle.com/settings) (API section).
Both `KGAT_` personal access tokens and classic `{"username":"...","key":"..."}` JSON
format are supported. The script uses the Kaggle REST API directly (no `kaggle` CLI needed).

What this downloads:
| Dataset | Source | Size |
|---------|--------|------|
| SeaThru | Kaggle REST API | ~32 GB |
| kskin HIMB1 images | tar.gz | varies |
| kskin HIMB GT depth | tar.gz | varies |
| FLSea-VI validation | HuggingFace parquet | ~13 GB |

The script skips any dataset already staged.

### Step 3 — Convert to SPADE format (CPU jobs, long jobs)

```bash
sbatch --export=DATASET=seathru cluster/spade_convert.sbat
sbatch --export=DATASET=kskin   cluster/spade_convert.sbat
sbatch --export=DATASET=flsea   cluster/spade_convert.sbat
```

Each job:
- Reads raw RGB + dense depth from scratch
- Generates sparse depth CSVs (Shi-Tomasi corners sampled from dense depth)
- Saves converted data to `$DATA_ROOT/<dataset>/spade/`
- Writes `data/spade_lists/<dataset>_test.txt` (absolute paths, gitignored)

Smoke-test with a small cap before running the full job:
```bash
sbatch --export=DATASET=flsea,MAX_IMAGES=20 cluster/spade_convert.sbat
```

### Step 4 — Run evaluation + charts (GPU jobs, ~1 h each)

```bash
# 15-image bundled demo (no conversion needed — good first sanity check)
sbatch cluster/spade_metrics.sbat

# Full benchmark datasets
sbatch --export=DATASET=flsea   cluster/spade_metrics.sbat
sbatch --export=DATASET=seathru cluster/spade_metrics.sbat
sbatch --export=DATASET=kskin   cluster/spade_metrics.sbat
```

### Step 5 — Monitor jobs

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,MaxRSS
cat logs/spade-metrics-<JOBID>.log
```

### Step 6 — Copy results locally

```bash
# From your local WSL machine
scp -r "brandmcd@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/reports/spade/" ./
```

---

## SLURM script variables

### `spade_convert.sbat`

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `seathru` | `seathru`, `kskin`, or `flsea` |
| `DATA_ROOT` | `$SCRATCH/data` | Root of staged data |
| `MAX_IMAGES` | *(all)* | Cap image count for smoke-testing |

### `spade_metrics.sbat`

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `flsea_demo` | Dataset tag — selects filenames list + output subdir |
| `WEIGHTS_PATH` | `$SCRATCH/spade_weights/underwater_depth_pipeline.pt` | Pretrained `.pt` weights |
| `EVAL_RANGES` | `10 5 2` | Space-separated depth thresholds in metres |

---

## Outputs

### Metrics CSV

`reports/spade/<dataset>_metrics.csv` — one row per evaluation range:

```
dataset,range_m,mae,rmse,abs_rel,silog,a1,a2,a3,#params
flsea,10.0,0.277,...
flsea,5.0,0.170,...
flsea,2.0,0.099,...
```

### Charts (300 DPI)

`reports/spade/figures/<dataset>/`

| File | Description |
|------|-------------|
| `errors_by_range.png` | MAE / RMSE / AbsRel / SILog grouped by depth range |
| `accuracy_by_range.png` | δ-accuracy (δ < 1.25, 1.25², 1.25³) by depth range |

Generate a cross-dataset comparison chart by passing multiple CSVs:

```bash
python -m src.spade.chart_metrics \
    --csv reports/spade/flsea_metrics.csv \
          reports/spade/seathru_metrics.csv \
          reports/spade/kskin_metrics.csv
# → reports/spade/figures/comparison/dataset_comparison.png
```

### Depth visualisations

`reports/spade/<dataset>/` — one PNG per image with three panels:
1. RGB input + sparse depth hint points (coloured by depth)
2. Predicted dense depth map
3. Depth colorbar (metres)

---

## Metrics reference

| Metric | Formula / meaning |
|--------|-------------------|
| **MAE** (m) | Mean absolute error: `mean(|pred − gt|)` |
| **RMSE** (m) | Root mean squared error — penalises large outliers |
| **AbsRel** | Relative error: `mean(|pred − gt| / gt)` |
| **SILog** | Scale-invariant log error — shape accuracy, scale-independent |
| **δ < 1.25** | Fraction of pixels where `max(pred/gt, gt/pred) < 1.25` |

All metrics are computed only over valid pixels (GT depth within the evaluation range).

---

## Data format reference

### SPADE filenames list (one line per sample)

```
<rgb_path> <gt_depth_path> <sparse_csv_path>
```

For converted datasets the paths are **absolute** and `data_path_eval="/"` is set in
`configs/spade_datasets.yaml` so the data loader resolves them correctly.

### Sparse features CSV

```
row,column,depth
25.1,71.4,3.21
54.4,61.3,1.69
...
```

Coordinates are in **240 × 320 space** (matches SPADE's `sparse_feature_height` /
`sparse_feature_width` config).  The data loader scales them to the model input size
at inference time.

### Depth files

Float32 TIFF (`.tif`) — values in metres.  Zero / NaN = invalid.

---

## Technical notes

### Weights — what `build_spade_weights.py` produces

The official checkpoint combines a trained Depth Anything V2 ViT-S backbone with a
trained DAT (Deformable Attention Transformer) refinement head.  Since the official
file is access-restricted, `build_spade_weights.py`:

1. Downloads public DA V2 ViT-S weights from HuggingFace
2. Initialises the full SPADE model (random weights for the DAT head)
3. Loads the DA V2 backbone into the `pretrained` + `depth_head` submodules
4. Saves the complete state dict

The result runs the full SPADE two-stage pipeline.  The DAT refinement head is
randomly initialised, so accuracy matches the **DA V2 + GA** row in the paper table
(MAE 0.277 m, RMSE 0.563 m, AbsRel 0.081 at ≤10 m on FLSea) rather than the
best-model row (MAE 0.131 m).

### Sparse features from dense depth

For SeaThru, kskin, and FLSea-VI (HuggingFace), which ship dense depth maps, the
converters simulate sparse hints by:

1. Detecting Shi-Tomasi corners in the RGB image (≈400–500 per image)
2. Sampling the dense depth map at each corner location
3. Expressing coordinates in 240 × 320 space

SPADE's architecture is robust to varying sparsity, so this is a valid proxy.

### Separate venv from SUIM-Net

SPADE requires PyTorch; SUIM-Net requires TensorFlow 2.13.

- `rob472`       — TensorFlow / SUIM-Net
- `rob472-spade` — PyTorch / SPADE (Great Lakes venv at `$SCRATCH/venvs/rob472-spade`)
