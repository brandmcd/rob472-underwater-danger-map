# ROB 472 — Underwater Danger Map

**Brandon McDonald · Caitlin Roberts · Sydney Ragla**
University of Michigan · ROB 472 Winter 2026

Explores combining underwater semantic segmentation (SUIM-Net) and monocular depth estimation
(SPADE) to construct a depth-aware danger map for autonomous underwater vehicle (AUV) perception.

---

## Project overview

| Goal | Model | Status |
|------|-------|--------|
| Underwater semantic segmentation | [SUIM-Net](src/suimnet/README.md) | Baselining |
| Monocular depth estimation | [SPADE](src/spade/README.md) | Baselining |
| Depth-aware danger map | Fusion module | In progress |

---

## Repository structure

```
src/
  suimnet/            # segmentation inference, metrics, charts  ← see src/suimnet/README.md
  spade/              # depth estimation inference               ← see src/spade/README.md
  fusion/             # danger map fusion (in progress)
  common/config.py    # profile + dataset path resolution
configs/
  profiles.yaml       # data_root per environment (local, greatlakes)
  datasets.yaml       # dataset relative paths (suim, deepfish, lars)
cluster/
  suimnet_infer.sbat  # SLURM: SUIM-Net inference (GPU)
  suimnet_metrics.sbat# SLURM: SUIM-Net metrics (CPU)
  spade_infer.sbat    # SLURM: SPADE inference (GPU)
vendor/
  SUIM-Net/           # pretrained 5-class segmentation model
  SPADE/              # pretrained monocular depth estimator
outputs/              # inference masks and depth maps
reports/
  suimnet/            # segmentation metrics CSVs + figures
  spade/              # depth estimation output images
logs/                 # SLURM job logs
```

---

## Setup

```bash
git clone <repo_url>
cd rob472-underwater-danger-map
git submodule update --init --recursive
```

Dependencies are split by model because they use different frameworks:

| Model | Requirements | Framework |
|-------|-------------|-----------|
| SUIM-Net | `requirements.txt` | TensorFlow 2.13 |
| SPADE | `requirements_spade.txt` | PyTorch ≥ 2.1 |

---

## Per-model guides

Full setup, local and Great Lakes instructions, output formats, and metrics explanations:

- **[SUIM-Net guide →](src/suimnet/README.md)**
- **[SPADE guide →](src/spade/README.md)**

---

## Datasets

| Key | Dataset | Labels | Use |
|-----|---------|--------|-----|
| `suim` | [SUIM TEST split](https://drive.google.com/drive/folders/10KMK0rNB43V2g30NcA1RYipL535DuZ-h) | Per-class binary masks | SUIM-Net quantitative eval |
| `deepfish` | [DeepFish](https://alzayats.github.io/DeepFish/) | None | SUIM-Net qualitative eval |
| `flsea` | [FLSea](https://arxiv.org/abs/2302.12772) | Metric depth + sparse pts | SPADE quantitative eval |

To add a new dataset, add an entry to `configs/datasets.yaml` (SUIM-Net) or create
a filenames list file under `vendor/SPADE/DataLists/` (SPADE).

---

## Key results (SUIM-Net baseline)

Evaluated on SUIM TEST split (110 images), Tesla V100-PCIE-16GB, ~20 s inference.
Charts in `reports/suimnet/figures/`.

---

## References

1. Islam et al., "Semantic Segmentation of Underwater Imagery: Dataset and Benchmark," 2020.
2. Zhang et al., "SPADE: Sparsity Adaptive Depth Estimator," 2025.
3. DeepFish dataset — Alzayats et al.
