# ROB 472 — Underwater Danger Map

**Brandon McDonald · Caitlin Roberts · Sydney Ragla**
University of Michigan · ROB 472 Winter 2026

Combines underwater semantic segmentation (SUIM-Net) with monocular depth estimation
(SPADE) to construct a depth-aware **danger map** for autonomous underwater vehicle
(AUV) perception. The danger map assigns each detected object a collision-risk score
based on its semantic class and estimated proximity to the robot.

---

## Project goals

| # | Goal | Module | Description |
|---|------|--------|-------------|
| 1 | Semantic segmentation baseline | [SUIM-Net](src/suimnet/README.md) | Segment underwater scenes into obstacle-relevant classes (fish, reef, wreck, diver, robot). Evaluate on SUIM, DeepFish, and USIS10K to measure cross-dataset generalization. |
| 2 | Monocular depth estimation | [SPADE](src/spade/README.md) | Estimate metric-scale depth from a single RGB image + sparse depth hints. Benchmark on FLSea and SeaThru datasets. |
| 3 | Danger map fusion | `src/fusion/` | Combine segmentation masks and depth maps into a spatial risk map that highlights nearby obstacles by type and distance. |

---

## Repository structure

```
src/
  suimnet/            # Goal 1: segmentation inference, metrics, charts
  spade/              # Goal 2: depth estimation, evaluation, charts
  fusion/             # Goal 3: danger map (in progress)
  common/config.py    # shared config: profile + dataset path resolution
configs/
  profiles.yaml       # data_root per environment (local, greatlakes)
  datasets.yaml       # SUIM-Net dataset paths + threshold overrides
  spade_datasets.yaml # SPADE dataset depth ranges + filenames lists
cluster/
  suimnet_convert.sbat  # SLURM: label format conversion (CPU)
  suimnet_infer.sbat    # SLURM: segmentation inference (GPU)
  suimnet_metrics.sbat  # SLURM: segmentation metrics + charts (CPU)
  spade_convert.sbat    # SLURM: depth data conversion (CPU)
  spade_metrics.sbat    # SLURM: depth evaluation + charts (GPU)
scripts/
  download_suimnet_data.sh  # download SUIM, DeepFish, USIS10K
  download_spade_data.sh    # download SeaThru, FLSea-VI
  build_spade_weights.py    # assemble SPADE checkpoint from DA V2 backbone
  launch_all.sh             # submit all SLURM jobs with dependency chaining
vendor/
  SUIM-Net/           # upstream 5-class segmentation model (submodule)
  SPADE/              # upstream monocular depth estimator (submodule)
outputs/              # inference masks (gitignored)
reports/
  suimnet/            # segmentation metrics CSVs + figures
  spade/              # depth metrics CSVs, charts, depth visualisations
logs/                 # SLURM job logs (gitignored)
```

---

## Datasets

### Semantic segmentation (Goal 1)

| Key | Dataset | Images | Use |
|-----|---------|--------|-----|
| `suim` | [SUIM TEST split](https://drive.google.com/file/d/1uEnlqKrlt6lITc_i80NTtb7iHGcO47sU) | 110 | Quantitative baseline (IoU, Dice, Precision, Recall) |
| `deepfish` | [DeepFish](https://alzayats.github.io/DeepFish/) | ~40k | Cross-dataset generalization (fish class only) |
| `usis10k` | [USIS10K TEST split](https://drive.google.com/file/d/1LdjLPaieWA4m8vLV6hEeMvt5wHnLg9gV) | ~1,595 | Cross-dataset generalization (7 classes) |

### Monocular depth estimation (Goal 2)

| Key | Dataset | Images | Use |
|-----|---------|--------|-----|
| `flsea_demo` | FLSea demo (bundled) | 15 | Smoke-test, no download needed |
| `flsea` | [FLSea-VI validation](https://huggingface.co/datasets/bhowmikabhimanyu/flsea-vi) | ~4,490 | Quantitative benchmark (MAE, RMSE, delta-accuracy) |
| `seathru` | [SeaThru](https://www.kaggle.com/datasets/colorlabeilat/seathru-dataset) | ~1,100 | Quantitative benchmark |

---

## Setup

```bash
git clone <repo_url>
cd rob472-underwater-danger-map
git submodule update --init --recursive
```

The two models use different frameworks and require **separate virtual environments**:

| Model | Requirements | Framework | Great Lakes venv |
|-------|-------------|-----------|------------------|
| SUIM-Net | `requirements.txt` | TensorFlow 2.13 | `$SCRATCH/venvs/rob472` |
| SPADE | `requirements_spade.txt` | PyTorch >= 2.1 | `$SCRATCH/venvs/rob472-spade` |

Venvs are created automatically by the SLURM scripts on first run.

---

## Running on ARC Great Lakes

### 1. Connect and pull

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
cd ~/rob472-underwater-danger-map
git pull && git submodule update --init --recursive
```

### 2. Download all datasets

```bash
# SUIM-Net datasets (SUIM, DeepFish, USIS10K)
bash scripts/download_suimnet_data.sh

# SPADE datasets (SeaThru, FLSea-VI)
export KAGGLE_API_TOKEN=<your-kaggle-token>
bash scripts/download_spade_data.sh
```

### 3. Build + stage SPADE weights

Run locally (needs internet + PyTorch), then scp to Great Lakes:

```bash
# Local
python scripts/build_spade_weights.py
# -> ~/Downloads/underwater_depth_pipeline.pt

# Upload to cluster
scp ~/Downloads/underwater_depth_pipeline.pt \
    <uniqname>@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/<uniqname>/spade_weights/
```

### 4. Submit all jobs (one command, then go to sleep)

```bash
bash scripts/launch_all.sh
```

This submits every conversion, inference, and metrics job with SLURM
`--dependency=afterok` so they run in the correct order automatically.
Monitor with `squeue -u $USER`.

### 5. Copy results locally

```bash
scp -r "<uniqname>@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/reports/" ./
```

---

## Per-model documentation

- **[SUIM-Net guide](src/suimnet/README.md)** — segmentation datasets, metrics, charts, threshold tuning, SLURM variables
- **[SPADE guide](src/spade/README.md)** — depth datasets, data conversion, evaluation, weights, SLURM variables

---

## References

1. Islam et al., "Semantic Segmentation of Underwater Imagery: Dataset and Benchmark," 2020. [arXiv:2004.01241](https://arxiv.org/abs/2004.01241)
2. Zhang et al., "SPADE: Sparsity Adaptive Depth Estimator for Zero-Shot, Real-Time, Monocular Depth Estimation in Underwater Environments," 2025. [arXiv:2510.25463](https://arxiv.org/abs/2510.25463)
3. Ebner et al., "Metrically Scaled Monocular Depth Estimation through Sparse Priors for Underwater Robots," 2023. [arXiv:2310.16750](https://arxiv.org/abs/2310.16750)
4. DeepFish — [alzayats.github.io/DeepFish](https://alzayats.github.io/DeepFish/)
5. USIS10K — [Google Drive](https://drive.google.com/file/d/1LdjLPaieWA4m8vLV6hEeMvt5wHnLg9gV)
6. FLSea-VI — Randall & Treibitz, 2023. [HuggingFace](https://huggingface.co/datasets/bhowmikabhimanyu/flsea-vi)
7. SeaThru — Akkaynak & Treibitz, CVPR 2019. [Kaggle](https://www.kaggle.com/datasets/colorlabeilat/seathru-dataset)
