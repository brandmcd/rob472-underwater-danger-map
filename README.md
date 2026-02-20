# Underwater Danger Map

Underwater semantic segmentation pipeline using **SUIM-Net** to identify objects of interest (robots, fish, divers, reefs, wrecks) in underwater imagery. Designed to run locally (CPU) or on **U-M Great Lakes HPC** (GPU via SLURM).

---

## Table of Contents

1. [Versions & Requirements](#versions--requirements)
2. [Project Structure](#project-structure)
3. [Local Quick-Start (WSL / Linux)](#local-quick-start-wsl--linux)
4. [Great Lakes HPC — Full Walkthrough](#great-lakes-hpc--full-walkthrough)
   - [SSH Access](#1-ssh-into-great-lakes)
   - [Clone Repo](#2-clone-repo-on-great-lakes)
   - [Stage Data](#3-stage-data-on-scratch)
   - [Run Inference](#4-run-inference-slurm)
   - [Run Metrics](#5-run-metrics-benchmark)
   - [Monitor Jobs](#6-monitor--check-results)
5. [Adding a New Dataset](#adding-a-new-dataset)
6. [Benchmarking a New Dataset End-to-End](#benchmarking-a-new-dataset-end-to-end)
7. [Metrics Script Reference](#metrics-script-reference)
8. [Output Format & Class Table](#output-format--class-table)
9. [Technical Notes](#technical-notes)

---

## Versions & Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | **3.10.4** | Great Lakes module: `python/3.10.4` |
| TensorFlow | **2.13.x** | CPU or GPU; GPU needs CUDA 11.8 |
| Keras | **2.13.x** | Must match TensorFlow version |
| CUDA | **11.8.0** | Great Lakes module: `cuda/11.8.0` |
| cuDNN | **8.7.0** | Great Lakes module: `cudnn/11.8-v8.7.0` |
| NumPy | latest | via `requirements.txt` |
| Pillow | latest | used by `metric_calc.py` |
| scikit-image | latest | image I/O and resize |
| OpenCV | headless | `opencv-python-headless` |
| PyYAML | latest | config loading |
| imageio | latest | mask writing |
| h5py | latest | model weight loading |

Full list in `requirements.txt`:

```
pyyaml
numpy
imageio
scikit-image
opencv-python-headless
h5py
tensorflow==2.13.*
keras==2.13.*
pillow
```

---

## Project Structure

```
configs/
  profiles.yaml           # data_root / outputs_root per environment (local, greatlakes)
  datasets.yaml            # dataset-specific relative paths (images, masks)
src/
  common/config.py         # resolves profile + dataset → absolute paths
  suimnet/run_infer.py     # SUIM-Net inference wrapper
  suimnet/metric_calc.py   # IoU / Dice / Precision / Recall evaluation script
vendor/
  SUIM-Net/                # upstream model code (DO NOT MODIFY)
cluster/
  suimnet_infer.sbat       # SLURM batch script — inference (GPU)
  suimnet_metrics.sbat     # SLURM batch script — metrics  (CPU-only)
logs/                      # SLURM job logs (auto-created)
outputs/                   # inference output masks
reports/                   # CSV metric reports
```

---

## Local Quick-Start (WSL / Linux)

### 1. Clone & init submodules

```bash
git clone git@github.com:brandmcd/rob472-underwater-danger-map.git
cd rob472-underwater-danger-map
git submodule update --init --recursive
```

### 2. Create a virtual environment & install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run SUIM-Net on the bundled sample images

The repo ships 8 sample images + pretrained weights in `vendor/SUIM-Net/sample_test/`.
No external data download needed:

```bash
python -m src.suimnet.run_infer \
    --images_dir vendor/SUIM-Net/sample_test/images \
    --dataset sample
```

Output masks → `outputs/sample/`.

### 4. Run metrics on sample data

```bash
python -m src.suimnet.metric_calc \
    --preds_dir outputs/sample \
    --masks_dir vendor/SUIM-Net/sample_test/masks \
    --out_csv reports/sample_metrics.csv
```

This prints per-class mIoU, Dice, Precision, Recall and writes per-image results to CSV.

### 5. Run on a full dataset (e.g. SUIM TEST split)

Download the SUIM dataset from <https://drive.google.com/drive/folders/10KMK0rNB43V2g30NcA1RYipL535DuZ-h>
and place it so that images live at `<data_root>/suim/TEST/images/` and masks at `<data_root>/suim/TEST/masks/`.

Edit `configs/profiles.yaml` so `local.data_root` points to your data directory, then:

```bash
python -m src.suimnet.run_infer --profile local --dataset suim
python -m src.suimnet.metric_calc --profile local --dataset suim \
    --preds_dir outputs/suim --out_csv reports/suim_metrics.csv
```

---

## Great Lakes HPC — Full Walkthrough

> **Tested and working** — successfully ran 110 SUIM test images in ~60 s on a Tesla V100-PCIE-16GB.

### 1. SSH into Great Lakes

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
```

Replace `<uniqname>` with your UMich uniquename. You need:
- An active UMich account with Great Lakes access
- Membership in the `rob572w26_class` Slurm account (for `--qos=class`)
- An SSH key added to GitHub (Settings → SSH Keys) so you can clone/pull

### 2. Clone repo on Great Lakes

```bash
cd ~
git clone git@github.com:brandmcd/rob472-underwater-danger-map.git
cd rob472-underwater-danger-map
git submodule update --init --recursive
```

If the repo already exists, just pull latest changes:

```bash
cd ~/rob472-underwater-danger-map
git pull
git submodule update --init --recursive
```

### 3. Stage data on scratch

Great Lakes home directories have limited quota. Store datasets on scratch:

```bash
# Create the directory structure
mkdir -p /scratch/rob572w26_class_root/rob572w26_class/$USER/data/suim/TEST/images
mkdir -p /scratch/rob572w26_class_root/rob572w26_class/$USER/data/suim/TEST/masks
```

**Option A — Upload from your local machine (WSL/Mac/Linux):**

```bash
# Run these on your LOCAL machine, not on Great Lakes
scp -r /path/to/suim/TEST/images/* \
    <uniqname>@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/<uniqname>/data/suim/TEST/images/

scp -r /path/to/suim/TEST/masks/* \
    <uniqname>@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/<uniqname>/data/suim/TEST/masks/
```

**Option B — Download directly on Great Lakes** (if dataset is from a public URL):

```bash
# On Great Lakes login node
cd /scratch/rob572w26_class_root/rob572w26_class/$USER/data
# Use wget, curl, or gdown for Google Drive links
pip install --user gdown
gdown --folder <google-drive-folder-id> -O suim/
```

**Verify** the masks directory has per-class sub-folders:

```bash
ls /scratch/rob572w26_class_root/rob572w26_class/$USER/data/suim/TEST/masks/
# Expected: FV/  HD/  RI/  RO/  WR/  (plus composite .bmp files)
```

The path layout must match `images_rel` and `labels_rel` in `configs/datasets.yaml`.

### 4. Run inference (SLURM)

From the repo root on Great Lakes:

```bash
cd ~/rob472-underwater-danger-map

# Submit inference job (GPU, ~1 min for 110 images)
sbatch cluster/suimnet_infer.sbat
```

The SLURM script automatically:
1. Loads modules: `python/3.10.4`, `cuda/11.8.0`, `cudnn/11.8-v8.7.0`
2. Creates/reuses a Python venv at `/scratch/.../venvs/rob472`
3. Installs dependencies from `requirements.txt`
4. Runs inference with the `greatlakes` profile
5. Writes output masks to `outputs/suim/`

To use a different dataset:

```bash
sbatch --export=DATASET=deepfish cluster/suimnet_infer.sbat
```

### 5. Run metrics (benchmark)

After inference completes:

```bash
# Submit metrics job (CPU-only, ~2 min)
sbatch cluster/suimnet_metrics.sbat
```

Or run interactively (quick for small datasets):

```bash
# Activate the existing venv
source /scratch/rob572w26_class_root/rob572w26_class/$USER/venvs/rob472/bin/activate

python -m src.suimnet.metric_calc \
    --profile greatlakes --dataset suim \
    --preds_dir outputs/suim \
    --out_csv reports/suim_metrics.csv
```

Example output:

```
========================================================================
Class      mIoU    mDice    mPrec     mRec   Images
------------------------------------------------------------------------
RO       0.4521   0.6229   0.7813   0.5392      110
FV       0.3812   0.5521   0.6145   0.5012      110
HD       0.5234   0.6873   0.7321   0.6489      110
RI       0.4198   0.5913   0.6512   0.5421      110
WR       0.3956   0.5667   0.6234   0.5198      110
------------------------------------------------------------------------
MEAN     0.4344
========================================================================
```

### 6. Monitor & check results

```bash
# Check job status (R=running, PD=pending)
squeue -u $USER

# Detailed job stats after completion
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,MaxRSS

# Read logs
cat logs/suimnet-infer-<JOBID>.log
cat logs/suimnet-metrics-<JOBID>.log

# Copy results back to your local machine (run on LOCAL)
scp <uniqname>@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/reports/suim_metrics.csv .
```

### SLURM resource allocation

| Script | Partition | GPUs | CPUs | Memory | Wall Time |
|---|---|---|---|---|---|
| `suimnet_infer.sbat` | `gpu` | 1 | 4 | 16 GB | 1 hour |
| `suimnet_metrics.sbat` | `standard` | 0 | 4 | 8 GB | 30 min |

Both use `--account=rob572w26_class` and `--qos=class`.

---

## Adding a New Dataset

Follow these steps to add any new underwater image dataset and benchmark it on Great Lakes.

### Step 1: Register the dataset in configs

Edit `configs/datasets.yaml` and add an entry:

```yaml
datasets:
  suim:
    images_rel: suim/TEST/images
    labels_rel: suim/TEST/masks
    has_labels: true

  # ---- ADD YOUR NEW DATASET HERE ----
  mydataset:
    images_rel: mydataset/images       # relative to data_root
    labels_rel: mydataset/masks        # set to null if no GT masks
    has_labels: true                   # false if no GT available
```

### Step 2: Organize the data

Your dataset directory (under `data_root`) must follow this layout:

```
mydataset/
├── images/          # input images (.jpg, .png, .bmp)
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── masks/           # ground truth (only needed for metrics)
    ├── RO/          # per-class binary masks
    │   ├── img001.bmp
    │   └── ...
    ├── FV/
    ├── HD/
    ├── RI/
    └── WR/
```

**Ground truth masks** must be:
- Binary images (white = class present, black = absent)
- Same filename stem as the input image (extension can differ)
- Organized into per-class sub-directories: `RO/`, `FV/`, `HD/`, `RI/`, `WR/`

If your GT is in a different format (e.g. composite RGB masks only), you'll need to split it into per-class binaries first. See the SUIM paper for the RGB encoding.

### Step 3: Upload to Great Lakes scratch

```bash
# On your LOCAL machine
scp -r /path/to/mydataset \
    <uniqname>@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/<uniqname>/data/mydataset/
```

Or directly on Great Lakes:

```bash
mkdir -p /scratch/rob572w26_class_root/rob572w26_class/$USER/data/mydataset/{images,masks}
# Then copy/download your files there
```

### Step 4: Push config changes

On your development machine (WSL):

```bash
cd ~/rob472/rob472-underwater-danger-map
# Edit configs/datasets.yaml as shown above
git add configs/datasets.yaml
git commit -m "Add mydataset to datasets.yaml"
git push
```

### Step 5: Pull on Great Lakes and run

```bash
# On Great Lakes
cd ~/rob472-underwater-danger-map
git pull

# Run inference
sbatch --export=DATASET=mydataset cluster/suimnet_infer.sbat

# After inference completes, run metrics (if you have GT masks)
sbatch --export=DATASET=mydataset cluster/suimnet_metrics.sbat
```

---

## Benchmarking a New Dataset End-to-End

Complete checklist for benchmarking a new dataset from scratch:

```bash
# ============================================================
# ON WSL (your development machine)
# ============================================================

# 1. Add dataset to configs/datasets.yaml (see "Adding a New Dataset")
# 2. Commit and push
git add configs/datasets.yaml
git commit -m "Add <datasetname>"
git push

# ============================================================
# ON GREAT LAKES
# ============================================================

# 3. SSH in
ssh <uniqname>@greatlakes.arc-ts.umich.edu

# 4. Stage data
DNAME=mydataset
mkdir -p /scratch/rob572w26_class_root/rob572w26_class/$USER/data/$DNAME/{images,masks/{RO,FV,HD,RI,WR}}
# ... copy images and per-class GT masks into those dirs ...

# 5. Pull latest code
cd ~/rob472-underwater-danger-map
git pull
git submodule update --init --recursive

# 6. Run inference
sbatch --export=DATASET=$DNAME cluster/suimnet_infer.sbat
# Wait for job to finish:
squeue -u $USER

# 7. Run metrics
sbatch --export=DATASET=$DNAME cluster/suimnet_metrics.sbat
# Wait for job to finish:
squeue -u $USER

# 8. Check results
cat logs/suimnet-metrics-*.log | tail -20
cat reports/${DNAME}_metrics.csv

# 9. Copy CSV to local machine (run on LOCAL WSL)
scp <uniqname>@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/reports/${DNAME}_metrics.csv .
```

---

## Metrics Script Reference

### `src/suimnet/metric_calc.py`

Computes pixel-level metrics by comparing RGB prediction masks against per-class binary ground truth.

**Metrics computed:**
| Metric | Description |
|---|---|
| **IoU** (Intersection over Union) | `TP / (TP + FP + FN)` — standard segmentation metric |
| **Dice** (F1) | `2·TP / (2·TP + FP + FN)` — harmonic mean of precision & recall |
| **Precision** | `TP / (TP + FP)` — what fraction of predicted pixels are correct |
| **Recall** | `TP / (TP + FN)` — what fraction of GT pixels are detected |

**Usage:**

```bash
# With explicit directories
python -m src.suimnet.metric_calc \
    --preds_dir outputs/suim \
    --masks_dir /path/to/suim/TEST/masks \
    --out_csv reports/suim_metrics.csv

# With profile/dataset config resolution
python -m src.suimnet.metric_calc \
    --profile greatlakes --dataset suim \
    --preds_dir outputs/suim \
    --out_csv reports/suim_metrics.csv

# Evaluate only specific classes
python -m src.suimnet.metric_calc \
    --preds_dir outputs/suim \
    --masks_dir /path/to/masks \
    --classes RO HD WR
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--preds_dir` | Yes | Directory with prediction PNG masks from `run_infer.py` |
| `--masks_dir` | No* | Root GT masks directory (must have `RO/`, `FV/`, etc. sub-dirs) |
| `--profile` | No* | Profile key from `configs/profiles.yaml` |
| `--dataset` | No* | Dataset key from `configs/datasets.yaml` |
| `--out_csv` | No | Path to write per-image CSV results |
| `--classes` | No | Subset of classes to evaluate (default: all 5) |

\* Must provide either `--masks_dir` or both `--profile` and `--dataset`.

---

## Output Format & Class Table

Inference produces **RGB-encoded PNG masks** in `outputs/<dataset>/`, one per input image.

| Channel | Class | Description | RGB Color |
|---|---|---|---|
| 0 | **RO** | Robot/instrument | Red `(255, 0, 0)` |
| 1 | **FV** | Fish/vertebrate | Yellow `(255, 255, 0)` |
| 2 | **HD** | Human diver | Blue `(0, 0, 255)` |
| 3 | **RI** | Reef/invertebrate | Magenta `(255, 0, 255)` |
| 4 | **WR** | Wreck/ruin | Cyan `(0, 255, 255)` |

The RGB composition uses logical OR across classes, so multi-class pixels produce combined colors.

---

## Technical Notes

### Keras Compatibility Shim

The vendor SUIM-Net code uses old Keras APIs (`from keras.models import Input`, `Model(input=, output=)`)
that changed in Keras 2.13+. Rather than modifying vendor code, `src/suimnet/run_infer.py` includes a
monkey-patch shim that:
1. Re-exports `keras.layers.Input` as `keras.models.Input`
2. Wraps `keras.models.Model` to accept old-style `input`/`output` kwargs

### SUIM-Net Preprocessing

**Critical:** The SUIM-Net model expects input images normalized to `[0, 1]`. Preprocessing uses
`skimage.transform.resize(..., preserve_range=False)` which handles this automatically.

### Config System

Two YAML files drive path resolution:

- `configs/profiles.yaml` — per-environment roots (local WSL path vs. Great Lakes scratch)
- `configs/datasets.yaml` — per-dataset relative paths (images, masks)

`src/common/config.py` combines them: `images_dir = data_root / images_rel`.

### Development Workflow

All code changes happen on **WSL** (or your local dev machine), then get pushed to GitHub and pulled on Great Lakes:

```
WSL (edit code) → git push → Great Lakes (git pull → sbatch)
```

Never edit code directly on Great Lakes. Keep the cluster copy read-only except for data and outputs.
