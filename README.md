## Quick-start (local, CPU)

### 1. Clone & init submodules

```bash
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

## Running on Great Lakes (UMich HPC) - UNTESTED

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
```

Logs appear in `logs/suimnet-infer-<JOBID>.log`.
Outputs go to the scratch `outputs/` directory (see `profiles.yaml → greatlakes.outputs_root`).

### 3. Monitor

```bash
squeue -u $USER          # job status
cat logs/suimnet-infer-*.log   # stdout / stderr
```

---

## Requirements

- Python 3.10+
- TensorFlow 2.13.x (CPU or GPU)
- See `requirements.txt` for full list
