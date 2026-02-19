# ROB472 — Underwater Danger Map

Semantic segmentation + monocular depth estimation for underwater scenes.
This repo currently baselines **SUIM-Net** (5-class underwater segmentation) and is structured to scale to additional models (SPADE depth) and datasets (DeepFish, LaRS).

---

## Repository layout

```
configs/            # YAML configs for dataset paths & cluster profiles
  datasets.yaml     # dataset names → relative image/label paths
  profiles.yaml     # local / greatlakes data-root paths
cluster/            # SLURM batch scripts for Great Lakes HPC
src/
  common/config.py  # shared path-resolution logic
  suimnet/run_infer.py   # SUIM-Net inference entry-point
  spade/            # (future) SPADE depth estimation
  fusion/           # (future) danger-map fusion
vendor/
  SUIM-Net/         # upstream SUIM-Net (model + pretrained weights)
  SPADE/            # upstream SPADE depth estimator
outputs/            # inference results (gitignored)
logs/               # SLURM logs (gitignored)
```

---

## Quick-start (local, CPU)

### 1. Clone & init submodules

```bash
git clone <repo-url> rob472-underwater-danger-map
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
No external data download needed for this step:

```bash
python -m src.suimnet.run_infer \
    --images_dir vendor/SUIM-Net/sample_test/images \
    --dataset sample
```

Output masks are written to `outputs/sample/`.

### 4. Run on a full dataset (e.g. SUIM TEST split)

Download the SUIM dataset from <https://drive.google.com/drive/folders/10KMK0rNB43V2g30NcA1RYipL535DuZ-h>
and place it so that the images live at `<data_root>/suim/TEST/images/`.

Edit `configs/profiles.yaml` so `local.data_root` points to your data directory, then:

```bash
python -m src.suimnet.run_infer --profile local --dataset suim
```

To point at any arbitrary image folder instead:

```bash
python -m src.suimnet.run_infer --images_dir /path/to/my/images --dataset my_run
```

---

## Running on Great Lakes (UMich HPC)

### 1. Stage data on scratch

Copy your dataset(s) to the class scratch space:

```
/scratch/rob572w26_class_root/rob572w26_class/$USER/data/suim/TEST/images/
```

The path layout must match the `images_rel` entries in `configs/datasets.yaml`.

### 2. Submit a SLURM job

From the repo root on a Great Lakes login node:

```bash
# SUIM dataset (default)
sbatch cluster/suimnet_infer.sbat

# DeepFish
sbatch --export=DATASET=deepfish cluster/suimnet_infer.sbat

# LaRS
sbatch --export=DATASET=lars cluster/suimnet_infer.sbat
```

Logs appear in `logs/suimnet-infer-<JOBID>.log`.
Outputs go to the scratch `outputs/` directory (see `profiles.yaml → greatlakes.outputs_root`).

### 3. Monitor

```bash
squeue -u $USER          # job status
cat logs/suimnet-infer-*.log   # stdout / stderr
```

---

## CLI reference

```
python -m src.suimnet.run_infer [OPTIONS]

Required (pick one):
  --images_dir DIR           Direct path to an images folder
  --profile P --dataset D    Resolve paths via configs/*.yaml

Optional:
  --out DIR        Output root directory (default: outputs)
  --weights FILE   Path to .hdf5 weights (default: vendor weights)
  --input_w INT    Model input width  (default: 320)
  --input_h INT    Model input height (default: 240)
  --thr FLOAT      Sigmoid threshold  (default: 0.5)
```

---

## Adding a new dataset

1. Add an entry to `configs/datasets.yaml`:

   ```yaml
   datasets:
     my_dataset:
       images_rel: my_dataset/images
       labels_rel: my_dataset/masks   # or null
       has_labels: true                # or false
   ```

2. Place the data under `<data_root>/my_dataset/images/`.

3. Run:

   ```bash
   python -m src.suimnet.run_infer --profile local --dataset my_dataset
   ```

---

## SUIM-Net output classes

The pretrained weights produce 5-channel sigmoid outputs corresponding to:

| Channel | Class | Color in RGB mask |
|---------|-------|-------------------|
| 0 | Robot/instrument (RO) | Red |
| 1 | Fish/vertebrate (FV) | Yellow (R+G) |
| 2 | Human diver (HD) | Blue |
| 3 | Reef/invertebrate (RI) | Magenta (R+B) |
| 4 | Wreck/ruin (WR) | Cyan (G+B) |

---

## Requirements

- Python 3.10+
- TensorFlow 2.13.x (CPU or GPU)
- See `requirements.txt` for full list
