# SPADE — Underwater Monocular Depth Estimation

Metric-scale monocular depth estimation using the SPADE model
(Zhang et al., 2025 — Sparsity Adaptive Depth Estimator).
Takes RGB images and sparse depth hint points as inputs and produces dense depth maps.

---

## Project structure

```
src/spade/              # this module (wrappers / docs)
cluster/
  spade_infer.sbat      # SLURM: GPU inference + evaluation
vendor/SPADE/
  evaluate.py           # upstream evaluation entry-point
  UnderwaterDepth/      # model architecture and data loaders
  DataLists/            # filenames lists for datasets
  example_data/         # 15-image FLSea demo subset (bundled)
requirements_spade.txt  # PyTorch pip dependencies
reports/spade/          # output depth visualisations
```

---

## Quick-start (local)

### 1. Set up environment

SPADE uses **PyTorch**, not TensorFlow. Create a separate venv:

```bash
python3 -m venv .venv-spade && source .venv-spade/bin/activate
pip install --upgrade pip
pip install -r requirements_spade.txt
```

Or use the conda environment bundled with the vendor code:

```bash
conda env create -f vendor/SPADE/environment.yml
conda activate spade
```

### 2. Download pretrained weights

Weights are too large for git. Download from the link in `vendor/SPADE/README.md`
(Google Drive) and place the `.pt` file at a location of your choosing, e.g.:

```
/path/to/weights/underwater_depth_pipeline.pt
```

### 3. Smoke-test with bundled FLSea demo data

The repo ships 15 FLSea images + sparse CSV features + ground-truth depth in
`vendor/SPADE/example_data/`. No extra download needed:

```bash
cd vendor/SPADE

python evaluate.py \
    -m SPADE \
    --pretrained_resource "local::/path/to/weights/underwater_depth_pipeline.pt" \
    -d flsea_sparse_feature \
    -r 10 5 2 \
    --save-image \
    --output-image-path ../../reports/spade/example
```

Depth visualisations saved to `reports/spade/example/`. Metrics printed to terminal.

### 4. Run on any dataset

**Step 1 — Prepare a filenames file** (relative to `vendor/SPADE/`):

Each line lists three paths: RGB image · ground-truth depth · sparse features CSV.

```
./data/my_dataset/rgb/img001.tiff  ./data/my_dataset/gt/img001_depth.tif  ./data/my_dataset/sparse/img001.csv
./data/my_dataset/rgb/img002.tiff  ./data/my_dataset/gt/img002_depth.tif  ./data/my_dataset/sparse/img002.csv
```

See `vendor/SPADE/DataLists/flease_testing/flsea_demo.txt` as a reference.

**Step 2 — Run evaluation**, overriding the filenames file:

```bash
cd vendor/SPADE

python evaluate.py \
    -m SPADE \
    --pretrained_resource "local::/path/to/weights/underwater_depth_pipeline.pt" \
    -d flsea_sparse_feature \
    -r 10 5 2 \
    --save-image \
    --output-image-path ../../reports/spade/my_dataset \
    filenames_file_eval=./DataLists/my_dataset/test.txt
```

The `filenames_file_eval=<value>` trailing argument overrides the config key directly.
Additional config keys (`data_path_eval`, `gt_path_eval`, etc.) can be overridden the same way.

---

## Running on ARC Great Lakes

### Prerequisites

**1. GitHub SSH key** — same as SUIM-Net; see `src/suimnet/README.md`.

**2. Download and stage weights**

```bash
# On Great Lakes (after downloading via browser locally and scp-ing up):
mkdir -p /scratch/rob572w26_class_root/rob572w26_class/$USER/spade_weights/
# scp weights file to the above path
```

Default SLURM script expects weights at:
```
/scratch/rob572w26_class_root/rob572w26_class/$USER/spade_weights/underwater_depth_pipeline.pt
```

Override with `--export=WEIGHTS_PATH=/your/path.pt` at submission time.

**3. (For full FLSea) Stage dataset in scratch**

```
/scratch/rob572w26_class_root/rob572w26_class/$USER/data/flsea/
  rgb/              ← TIFF images
  gt_depth/         ← ground truth depth TIFFs
  sparse_csv/       ← per-image sparse feature CSVs
```

Then create a filenames file in `vendor/SPADE/DataLists/` pointing to those paths.

### Step-by-step

**0. Connect and pull**

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
cd ~/rob472-underwater-danger-map
git pull && git submodule update --init --recursive
```

**1. Run inference + evaluation (GPU)**

```bash
# FLSea bundled demo data (default, no extra data needed)
sbatch cluster/spade_infer.sbat

# Custom dataset
sbatch --export=DATASET=flsea_full,FILENAMES_FILE=./DataLists/flsea_testing/full.txt \
       cluster/spade_infer.sbat
```

The script creates a dedicated PyTorch venv at `/scratch/.../venvs/rob472-spade`.

**2. Monitor jobs**

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,MaxRSS
cat logs/spade-infer-<JOBID>.log
```

**3. Copy results locally**

```bash
# From your local machine
scp -r "brandmcd@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/reports/spade/" ./
```

---

## SLURM script variables

All options can be set via `sbatch --export=VAR=value`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `example` | Tag for the output subdirectory under `reports/spade/` |
| `WEIGHTS_PATH` | `$SCRATCH/spade_weights/underwater_depth_pipeline.pt` | Absolute path to `.pt` weights |
| `FILENAMES_FILE` | *(bundled demo list)* | Custom filenames file, relative to `vendor/SPADE/` |
| `EVAL_RANGES` | `10 5 2` | Space-separated depth thresholds in metres |

---

## Outputs

### Depth visualisations

`reports/spade/<dataset>/` — one PNG per image showing three panels:
1. RGB input with sparse depth hint points overlaid (colored by depth)
2. Predicted dense depth map
3. Colorbar (depth in metres)

### Terminal metrics

Printed to the SLURM log (`logs/spade-infer-<JOBID>.log`) for each evaluation range:

| Metric | What it measures |
|--------|-----------------|
| **MAE** (m) | Mean absolute depth error |
| **RMSE** (m) | Root mean squared depth error — penalises large outliers more |
| **AbsRel** | Absolute relative error: `|pred - gt| / gt` averaged over valid pixels |
| **SILog** | Scale-invariant log error — measures shape accuracy independent of global scale |

Reported separately for the ≤ 10 m, ≤ 5 m, and ≤ 2 m depth ranges.

---

## Data format reference

### FLSea dataset structure

```
rgb/<timestamp>.tiff             ← input RGB image
gt_depth/<timestamp>_SeaErra_abs_depth.tif ← absolute depth (metres)
sparse_csv/<timestamp>_features.csv        ← sparse depth hints
```

### Sparse features CSV format

```
row,column,depth
142,387,3.21
...
```

Coordinates are in the original image pixel space. The data loader scales them to the
model's internal feature resolution.

### Filenames list format (one line per sample)

```
<rgb_path> <gt_depth_path> <sparse_csv_path>
```

All paths relative to `vendor/SPADE/`.

---

## Technical notes

### Two-stage pipeline

1. **Global alignment** — a relative depth backbone (Depth Anything V2) produces a
   shape-accurate but scale-ambiguous depth map. Sparse depth hints align it to metric scale.
2. **Cascade Conv-Deformable Transformer** — refines pixel-wise depth using the aligned
   map and sparse hints, producing the final dense metric depth.

### Inference without sparse hints

Set all sparse features to zero (or create a CSV with no rows). The model degrades
gracefully to a pure monocular depth estimator in this case.

### CUDA requirements

Requires a CUDA-capable GPU. On Great Lakes, `cuda/11.8.0` is loaded by the SLURM script.
PyTorch >= 2.1 with CUDA 11.8 is installed via `requirements_spade.txt`.

### Separate venv from SUIM-Net

SPADE requires PyTorch; SUIM-Net requires TensorFlow 2.13. Keep them in separate venvs:
- `rob472` — TensorFlow / SUIM-Net
- `rob472-spade` — PyTorch / SPADE
