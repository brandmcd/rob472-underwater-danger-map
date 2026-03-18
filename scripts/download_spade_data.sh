#!/usr/bin/env bash
# Download SPADE depth-estimation benchmark datasets into scratch / local data root.
#
# Datasets:
#   1. SeaThru — Akkaynak & Treibitz CVPR 2019 (Kaggle, ~32 GB)
#   2. kskin HIMB1 — DROP Lab BlueROV2 stereo images (U-Michigan, tar.gz)
#   3. kskin HIMB ground truth depth (U-Michigan, tar.gz)
#   4. FLSea-VI — validation split, HuggingFace (~13 GB parquet)
#
# Usage:
#   bash scripts/download_spade_data.sh                          # Great Lakes default
#   DATA_ROOT=/mnt/g/rob472/data bash scripts/download_spade_data.sh
#
# Kaggle authentication (for SeaThru) — set ONE of:
#   export KAGGLE_API_TOKEN=KGAT_xxxxx          ← new-style personal token
#   export KAGGLE_API_TOKEN='{"username":"...","key":"..."}' ← JSON format
#   ~/.kaggle/kaggle.json                       ← classic file-based auth
#
# After this script, convert each dataset to SPADE format:
#   sbatch --export=DATASET=seathru  cluster/spade_convert.sbat
#   sbatch --export=DATASET=kskin    cluster/spade_convert.sbat
#   sbatch --export=DATASET=flsea   cluster/spade_convert.sbat

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/scratch/rob572w26_class_root/rob572w26_class/${USER}/data}"
mkdir -p "$DATA_ROOT"
echo "Data root: $DATA_ROOT"
echo ""

# Load Python 3.10 on Great Lakes — system python3 is 3.6 (too old for kaggle/pip).
# module load is a no-op outside SLURM/Great Lakes environments.
module load python/3.10.4 2>/dev/null || true

# ── Kaggle download helper ────────────────────────────────────────────────────
# Bypasses the kaggle CLI entirely — uses the REST API directly via wget.
# Supports:
#   KAGGLE_API_TOKEN=KGAT_xxx          → Bearer auth  (new personal access token)
#   KAGGLE_API_TOKEN='{"username":...}' → HTTP Basic   (old key JSON in env var)
#   ~/.kaggle/kaggle.json               → HTTP Basic   (classic file-based auth)
#
# Usage: _kaggle_download <owner/dataset-slug> <output-zip-path>
_kaggle_download() {
    local slug="$1"
    local out="$2"
    local url="https://www.kaggle.com/api/v1/datasets/download/${slug}"

    if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
        echo "  ERROR: no Kaggle credentials."
        echo "  Set KAGGLE_API_TOKEN=KGAT_xxx or place kaggle.json at ~/.kaggle/kaggle.json"
        return 1
    fi

    if [[ "${KAGGLE_API_TOKEN:-}" == KGAT_* ]]; then
        # New-style personal access token — Bearer auth
        wget -q --show-progress \
            --header="Authorization: Bearer ${KAGGLE_API_TOKEN}" \
            "$url" -O "$out"
    else
        # Classic username+key — HTTP Basic auth
        # Parse from KAGGLE_API_TOKEN JSON env var, or fall back to kaggle.json
        local json_src
        if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
            json_src="$KAGGLE_API_TOKEN"
        else
            json_src="$(cat "$HOME/.kaggle/kaggle.json")"
        fi
        local kg_user kg_key
        kg_user=$(printf '%s' "$json_src" | python3 -c "import sys,json; print(json.load(sys.stdin)['username'])")
        kg_key=$(printf  '%s' "$json_src" | python3 -c "import sys,json; print(json.load(sys.stdin)['key'])")
        wget -q --show-progress \
            --user="$kg_user" --password="$kg_key" \
            "$url" -O "$out"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. SeaThru  (Kaggle dataset — requires Kaggle API token)
# ─────────────────────────────────────────────────────────────────────────────
SEATHRU_RAW="$DATA_ROOT/seathru/raw"

# Check for both RGB (.png) AND depth (.tif) files — an empty dir from a failed download won't pass
if [[ -d "$SEATHRU_RAW" \
      && -n "$(find "$SEATHRU_RAW" -name '*.png' 2>/dev/null | head -1)" \
      && -n "$(find "$SEATHRU_RAW" \( -name '*.tif' -o -name '*.tiff' \) 2>/dev/null | head -1)" ]]; then
    echo "[SeaThru] Already staged at $SEATHRU_RAW — skipping."
else
    echo "[SeaThru] Downloading via Kaggle REST API (~32 GB)..."
    mkdir -p "$SEATHRU_RAW"
    _kaggle_download "colorlabeilat/seathru-dataset" "$SEATHRU_RAW/seathru-dataset.zip"
    echo "[SeaThru] Extracting..."
    unzip -q "$SEATHRU_RAW/seathru-dataset.zip" -d "$SEATHRU_RAW"
    rm "$SEATHRU_RAW/seathru-dataset.zip"
    echo "[SeaThru] Done → $SEATHRU_RAW"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. kskin HIMB1 images (DROP Lab, BlueROV2 docksite)
# ─────────────────────────────────────────────────────────────────────────────
HIMB1_DIR="$DATA_ROOT/kskin/HIMB1"
HIMB1_URL="http://www.umich.edu/~dropopen2/DROPUWStereo_HIMB1_docksite.tar.gz"

if [[ -d "$HIMB1_DIR" && -n "$(find "$HIMB1_DIR" \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | head -1)" ]]; then
    echo "[kskin HIMB1 images] Already staged — skipping."
else
    echo "[kskin HIMB1 images] Downloading..."
    mkdir -p "$HIMB1_DIR"
    wget -q --show-progress -O "$HIMB1_DIR/HIMB1.tar.gz" "$HIMB1_URL"
    echo "[kskin HIMB1 images] Extracting..."
    tar -xf "$HIMB1_DIR/HIMB1.tar.gz" -C "$HIMB1_DIR" --strip-components=1
    rm "$HIMB1_DIR/HIMB1.tar.gz"
    echo "[kskin HIMB1 images] Done → $HIMB1_DIR"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 3. kskin HIMB ground truth depth
# ─────────────────────────────────────────────────────────────────────────────
HIMB_GT_DIR="$DATA_ROOT/kskin/HIMB_ground"
HIMB_GT_URL="http://www.umich.edu/~dropopen2/DROPUWStereo_HIMB_ground.tar.gz"

if [[ -d "$HIMB_GT_DIR" && -n "$(find "$HIMB_GT_DIR" \( -name '*.npy' -o -name '*.tif' -o -name '*.png' \) 2>/dev/null | head -1)" ]]; then
    echo "[kskin GT depth]     Already staged — skipping."
else
    echo "[kskin GT depth]     Downloading..."
    mkdir -p "$HIMB_GT_DIR"
    wget -q --show-progress -O "$HIMB_GT_DIR/HIMB_ground.tar.gz" "$HIMB_GT_URL"
    echo "[kskin GT depth]     Extracting..."
    tar -xf "$HIMB_GT_DIR/HIMB_ground.tar.gz" -C "$HIMB_GT_DIR" --strip-components=1
    rm "$HIMB_GT_DIR/HIMB_ground.tar.gz"
    echo "[kskin GT depth]     Done → $HIMB_GT_DIR"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 4. FLSea-VI — validation split (HuggingFace, ~13 GB parquet)
# ─────────────────────────────────────────────────────────────────────────────
export FLSEA_RAW="$DATA_ROOT/flsea/raw"

if [[ -d "$FLSEA_RAW" && -n "$(find "$FLSEA_RAW" -name '*.parquet' 2>/dev/null | head -1)" ]]; then
    echo "[FLSea-VI]           Already staged at $FLSEA_RAW — skipping."
else
    echo "[FLSea-VI]           Downloading validation split via HuggingFace (~13 GB)..."
    python3 - <<'PYEOF'
import os, sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface-hub"])
    from huggingface_hub import snapshot_download

dest = os.environ["FLSEA_RAW"]
print(f"  Downloading to {dest} ...")
snapshot_download(
    "bhowmikabhimanyu/flsea-vi",
    repo_type="dataset",
    local_dir=dest,
    allow_patterns=["data/validation-*.parquet"],
    ignore_patterns=["*.lock"],
)
print("  Done.")
PYEOF
    echo "[FLSea-VI]           Done → $FLSEA_RAW"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
echo "=== All downloads complete ==="
echo ""
echo "Next — convert each dataset to SPADE format (CPU jobs, ~2h each):"
echo ""
echo "  sbatch --export=DATASET=seathru cluster/spade_convert.sbat"
echo "  sbatch --export=DATASET=kskin   cluster/spade_convert.sbat"
echo "  sbatch --export=DATASET=flsea   cluster/spade_convert.sbat"
echo ""
echo "Then run evaluation + charting (GPU jobs):"
echo ""
echo "  sbatch --export=DATASET=seathru cluster/spade_metrics.sbat"
echo "  sbatch --export=DATASET=kskin   cluster/spade_metrics.sbat"
echo "  sbatch --export=DATASET=flsea   cluster/spade_metrics.sbat"
