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

---

## Technical Notes

### SUIM-Net Preprocessing

**Critical:** The SUIM-Net model expects input images normalized to [0,1]. Our preprocessing automatically handles this using `skimage.transform.resize(..., preserve_range=False)`.

### SUIM-Net Output Classes

The pretrained weights produce 5-channel sigmoid outputs corresponding to:

| Channel | Class | Description | Color in RGB mask |
|---------|-------|-------------|-------------------|
| 0 | RO | Robot/instrument | Red component |
| 1 | FV | Fish/vertebrate | Yellow (Red + Green) |
| 2 | HD | Human diver | Blue component |
| 3 | RI | Reef/invertebrate | Magenta (Red + Blue) |
| 4 | WR | Wreck/ruin | Cyan (Green + Blue) |

The final RGB visualization combines these channels using logical OR operations to produce colored segmentation masks that match the original SUIM-Net paper.

---

## Great Lakes Data Setup

### Copying SUIM Dataset to Great Lakes

Your username is `brandmcd`, so you need to stage data at:
```
/scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST/images/
```

#### Option 1: Transfer from local machine

1. **Download SUIM dataset locally** from <https://drive.google.com/drive/folders/10KMK0rNB43V2g30NcA1RYipL535DuZ-h>

2. **Extract and organize locally** so you have:
   ```
   /path/to/local/suim/TEST/images/
   /path/to/local/suim/TEST/masks/
   ```

3. **Copy to Great Lakes** (run from your local machine):
   ```bash
   # Create directory structure
   ssh brandmcd@greatlakes.arc-ts.umich.edu "mkdir -p /scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST"
   
   # Copy images
   scp -r /path/to/local/suim/TEST/images/ \
       brandmcd@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST/
   
   # Copy masks (optional, for evaluation)
   scp -r /path/to/local/suim/TEST/masks/ \
       brandmcd@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST/
   ```

#### Option 2: Download directly on Great Lakes

1. **SSH to Great Lakes:**
   ```bash
   ssh brandmcd@greatlakes.arc-ts.umich.edu
   ```

2. **Create directory and download:**
   ```bash
   mkdir -p /scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST
   cd /scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST
   
   # Download from Google Drive (you'll need the direct download URL)
   # Or use wget/curl if you have a direct link
   # Alternative: use gdown if available
   module load python
   pip install --user gdown
   gdown <google-drive-file-id>
   
   # Extract the downloaded archive
   unzip *.zip  # or tar -xzf *.tar.gz
   ```

#### Verify the setup:

```bash
# On Great Lakes, check the structure:
ls -la /scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST/
# Should show: images/ and masks/ directories

ls /scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST/images/ | wc -l
# Should show the number of test images
```

### Quick test on Great Lakes:

```bash
# Clone this repo on Great Lakes
git clone <your-repo-url> rob472-underwater-danger-map
cd rob472-underwater-danger-map
git submodule update --init --recursive

# Submit a test job
sbatch cluster/suimnet_infer.sbat
```

