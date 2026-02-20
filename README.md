## Project Structure

```
configs/
  profiles.yaml          # data_root / outputs_root per environment (local, greatlakes)
  datasets.yaml           # dataset-specific relative paths (images, masks)
src/
  common/config.py        # resolves profile + dataset → absolute paths
  suimnet/run_infer.py    # SUIM-Net inference wrapper (includes Keras compat shim)
vendor/
  SUIM-Net/               # upstream model code (DO NOT MODIFY)
cluster/
  suimnet_infer.sbat      # SLURM batch script for Great Lakes
logs/                     # SLURM job logs (suimnet-infer-<JOBID>.log)
outputs/                  # inference output masks
```

---

## Quick-start (local, CPU)

### 1. Clone & init submodules

```bash
git clone 
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

### 4. Run metrics & generate CSV

```bash
python -m src.suimnet.metric_calc \
    --preds_dir outputs/sample \
    --masks_dir vendor/SUIM-Net/sample_test/masks \
    --out_csv reports/sample_metrics.csv
```

This prints per-class mIoU, Dice, Precision, Recall to the terminal and writes per-image results to `reports/sample_metrics.csv`.

### 5. Chart the results

```bash
python -m src.suimnet.chart_metrics --csv reports/suim_metrics.csv
```

Saves 5 PNG charts to `reports/figures/`.

### 6. Run on a full dataset (e.g. SUIM TEST split)

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

## Running on Great Lakes

> **Tested and working**, successfully ran 110 SUIM test images in ~20 seconds on a Tesla V100-PCIE-16GB.

### 0. Connect to Great Lakes & pull latest code

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
cd ~/rob472-underwater-danger-map
git pull
git submodule update --init --recursive
```

> **GitHub SSH on Great Lakes:** You need an SSH key on Great Lakes linked to your GitHub account.
> If you haven't set this up, run `ssh-keygen -t ed25519` on Great Lakes, then copy
> `~/.ssh/id_ed25519.pub` and add it at <https://github.com/settings/keys>.
> Test with `ssh -T git@github.com` — you should see *"Hi \<user\>! You've been authenticated"*.

### 1. Stage data on scratch

Copy the dataset to the class scratch space:

```
/scratch/rob572w26_class_root/rob572w26_class/$USER/data/suim/TEST/images/
/scratch/rob572w26_class_root/rob572w26_class/$USER/data/suim/TEST/masks/   # for evaluation
```

The path layout must match the `images_rel` entries in `configs/datasets.yaml`.

### 2. Submit a SLURM job

From the repo root on a Great Lakes login node:

```bash
# SUIM dataset (default)
sbatch cluster/suimnet_infer.sbat

# Ex) with a different dataset
sbatch --export=DATASET=deepfish cluster/suimnet_infer.sbat
```

The SLURM script (`cluster/suimnet_infer.sbat`) handles everything automatically:
- Loads modules: `python/3.10.4`, `cuda/11.8.0`, `cudnn/11.8-v8.7.0`
- Creates/reuses a Python venv at `/scratch/.../venvs/rob472`
- Installs dependencies from `requirements.txt`
- Runs inference with the `greatlakes` profile

### 3. Monitor & check results

```bash
squeue -u $USER                        # job status (R=running, PD=pending)
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,MaxRSS  # final stats
cat logs/suimnet-infer-<JOBID>.log     # full stdout/stderr
```

Output masks are written to `outputs/suim/` (within the repo directory).

## Evaluating Output

### Output format

Inference produces **RGB-encoded PNG masks** in `outputs/<dataset>/`, one per input image.
Colors encode the segmented classes (see table below), matching the SUIM paper's visualization.

### Ground truth

The SUIM TEST split provides ground truth in two formats (both under `masks/`):
- **Per-image RGB BMP masks**: composite color masks at the top level (e.g. `masks/d_r_122_.bmp`)
- **Per-class binary masks**: in subdirectories `masks/FV/`, `masks/HD/`, `masks/RI/`, `masks/RO/`, `masks/WR/`

### Visual inspection

To quickly compare predictions vs ground truth, copy a few samples locally:

```bash
# From your local machine (WSL / Mac / Linux)
scp "brandmcd@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/outputs/suim/d_r_122_.png" ./pred.png
scp "brandmcd@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST/masks/d_r_122_.bmp" ./gt.bmp
scp "brandmcd@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/brandmcd/data/suim/TEST/images/d_r_122_.jpg" ./input.jpg
```

Then open them in any image viewer.

### Quantitative metrics

We compute four pixel-level metrics per class:

| Metric | Formula | What it measures |
|--------|---------|------------------|
| **IoU** (Intersection over Union) | `TP / (TP + FP + FN)` | How well the predicted region overlaps with the ground truth. Penalises both missed pixels (FN) and false alarms (FP) equally. Also called Jaccard Index. |
| **Dice** (F1 score) | `2·TP / (2·TP + FP + FN)` | Harmonic mean of precision and recall. Similar to IoU but gives a higher number for the same prediction. |
| **Precision** | `TP / (TP + FP)` | Of everything the model *predicted* as this class, what fraction was actually correct? High precision = few false positives. |
| **Recall** | `TP / (TP + FN)` | Of everything that *actually is* this class, what fraction did the model detect? High recall = few missed detections. |

Where TP = true positive pixels, FP = false positive pixels, FN = false negative pixels.

> **Rule of thumb:** IoU > 0.5 is "acceptable", > 0.7 is "good", > 0.85 is "excellent".

To compute metrics, compare the per-class binary GT masks (e.g. `masks/RO/d_r_122_.bmp`)
against thresholded per-channel predictions. The prediction masks use this RGB encoding:

| Channel | Class | Description | RGB Color |
|---------|-------|-------------|-----------|
| 0 | RO | Robot/instrument | Red (255, 0, 0) |
| 1 | FV | Fish/vertebrate | Yellow (255, 255, 0) |
| 2 | HD | Human diver | Blue (0, 0, 255) |
| 3 | RI | Reef/invertebrate | Magenta (255, 0, 255) |
| 4 | WR | Wreck/ruin | Cyan (0, 255, 0)* |

\* Note: The RGB composition uses logical OR across classes, so multi-class pixels produce combined colors.

### Quick evaluation on Great Lakes (interactive)

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
cd ~/rob472-underwater-danger-map

# Activate the existing venv
source /scratch/rob572w26_class_root/rob572w26_class/$USER/venvs/rob472/bin/activate

# Run metrics (IoU, Dice, Precision, Recall per class)
python -m src.suimnet.metric_calc \
    --profile greatlakes --dataset suim \
    --preds_dir outputs/suim \
    --out_csv reports/suim_metrics.csv
```

Or submit as a SLURM job:

```bash
sbatch cluster/suimnet_metrics.sbat
```

See `src/suimnet/metric_calc.py` for all options (`--classes`, `--masks_dir`, etc.).

### Charting results

After generating the metrics CSV, create visualisation charts:

```bash
python -m src.suimnet.chart_metrics --csv reports/suim_metrics.csv
```

This produces 5 PNG charts in `reports/figures/`:

| Chart | File | Description |
|-------|------|-------------|
| Per-class means | `class_means.png` | Grouped bar chart of mIoU, Dice, Precision, Recall for each class |
| IoU box plots | `iou_boxplots.png` | Distribution of per-image IoU scores per class (median, quartiles, outliers) |
| Overall summary | `overall_summary.png` | Horizontal bars showing macro-averaged metrics across all classes |
| IoU heatmap | `iou_heatmap.png` | Classes × images heatmap, sorted so worst-performing images are on the left |
| Precision vs Recall | `precision_recall.png` | Scatter plot of every image, coloured by class — shows the precision/recall trade-off |

---

## Requirements

- Python 3.10+
- TensorFlow 2.13.x (CPU or GPU)
- See `requirements.txt` for full list

---

## Technical Notes

### Keras Compatibility Shim

The vendor SUIM-Net code uses old Keras APIs (`from keras.models import Input`, `Model(input=, output=)`)
that changed in Keras 2.13+. Rather than modifying vendor code, `src/suimnet/run_infer.py` includes a
monkey-patch shim that:
1. Re-exports `keras.layers.Input` as `keras.models.Input`
2. Wraps `keras.models.Model` to accept old-style singular `input`/`output` kwargs

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

The final RGB visualization combines these channels using logical OR operations to produce colored segmentation masks that match the original SUIM-Net paper

