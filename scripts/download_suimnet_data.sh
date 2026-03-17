#!/usr/bin/env bash
# Download and stage all three SUIM-Net benchmark datasets into the
# Great Lakes scratch space (or a local directory).
#
# Usage:
#   bash scripts/download_suimnet_data.sh                  # uses $DATA_ROOT default
#   DATA_ROOT=/mnt/g/rob472/data bash scripts/download_suimnet_data.sh   # local override
#
# Requires:
#   pip install gdown          (Google Drive downloads)
#   wget                       (DeepFish direct download)
#   tar / unzip                (extraction)
#
# After this script completes, run the label converters:
#   python -m src.suimnet.convert_deepfish --profile greatlakes
#   python -m src.suimnet.convert_usis10k  --profile greatlakes

set -euo pipefail

# ── Data root ─────────────────────────────────────────────────────────────────
# On Great Lakes this must be under scratch (home has a small quota).
DATA_ROOT="${DATA_ROOT:-/scratch/rob572w26_class_root/rob572w26_class/${USER}/data}"
mkdir -p "$DATA_ROOT"
echo "Data root: $DATA_ROOT"
echo ""

# ── Google Drive file IDs ─────────────────────────────────────────────────────
SUIM_ID="1uEnlqKrlt6lITc_i80NTtb7iHGcO47sU"
USIS10K_ID="1LdjLPaieWA4m8vLV6hEeMvt5wHnLg9gV"

# ── Helpers ───────────────────────────────────────────────────────────────────
check_gdown() {
    if ! command -v gdown &> /dev/null; then
        echo "gdown not found. Installing..."
        pip install --quiet gdown
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. SUIM TEST split
# ─────────────────────────────────────────────────────────────────────────────
SUIM_DIR="$DATA_ROOT/suim"
if [[ -d "$SUIM_DIR/TEST/images" ]]; then
    echo "[SUIM] Already staged at $SUIM_DIR — skipping."
else
    echo "[SUIM] Downloading..."
    check_gdown
    mkdir -p "$SUIM_DIR"
    gdown --id "$SUIM_ID" -O "$SUIM_DIR/suim.zip"

    echo "[SUIM] Extracting..."
    unzip -q "$SUIM_DIR/suim.zip" -d "$SUIM_DIR"
    rm "$SUIM_DIR/suim.zip"

    # Normalise layout if needed: expect TEST/images and TEST/masks
    # The archive may unzip to a subdirectory — move contents up if so
    INNER=$(find "$SUIM_DIR" -maxdepth 1 -mindepth 1 -type d | head -1)
    if [[ -n "$INNER" && ! -d "$SUIM_DIR/TEST" ]]; then
        mv "$INNER"/* "$SUIM_DIR/" 2>/dev/null || true
        rmdir "$INNER" 2>/dev/null || true
    fi
    echo "[SUIM] Done → $SUIM_DIR"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. DeepFish
# ─────────────────────────────────────────────────────────────────────────────
DEEPFISH_DIR="$DATA_ROOT/deepfish"
DEEPFISH_URL="http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar"

if [[ -d "$DEEPFISH_DIR/Segmentation/images/valid" ]]; then
    echo "[DeepFish] Already staged at $DEEPFISH_DIR — skipping."
else
    echo "[DeepFish] Downloading (~7 GB — this may take a while)..."
    mkdir -p "$DEEPFISH_DIR"
    wget -q --show-progress -O "$DEEPFISH_DIR/DeepFish.tar" "$DEEPFISH_URL"

    echo "[DeepFish] Extracting..."
    tar -xf "$DEEPFISH_DIR/DeepFish.tar" -C "$DEEPFISH_DIR"
    rm "$DEEPFISH_DIR/DeepFish.tar"

    # Clean up the empty images/ dir created by a prior failed run, if present
    rmdir "$DEEPFISH_DIR/images" 2>/dev/null || true
    echo "[DeepFish] Done → $DEEPFISH_DIR"
    echo "           Segmentation images : $DEEPFISH_DIR/Segmentation/images/valid"
    echo "           Segmentation masks  : $DEEPFISH_DIR/Segmentation/masks/valid"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 3. USIS10K TEST split
# ─────────────────────────────────────────────────────────────────────────────
USIS_DIR="$DATA_ROOT/usis10k"
if [[ -d "$USIS_DIR/test" ]]; then
    echo "[USIS10K] Already staged at $USIS_DIR — skipping."
else
    echo "[USIS10K] Downloading..."
    check_gdown
    mkdir -p "$USIS_DIR"
    gdown --id "$USIS10K_ID" -O "$USIS_DIR/usis10k.zip"

    echo "[USIS10K] Extracting..."
    unzip -q "$USIS_DIR/usis10k.zip" -d "$USIS_DIR"
    rm "$USIS_DIR/usis10k.zip"

    # Normalise layout if needed
    INNER=$(find "$USIS_DIR" -maxdepth 1 -mindepth 1 -type d | head -1)
    if [[ -n "$INNER" && ! -d "$USIS_DIR/test" ]]; then
        mv "$INNER"/* "$USIS_DIR/" 2>/dev/null || true
        rmdir "$INNER" 2>/dev/null || true
    fi
    echo "[USIS10K] Done → $USIS_DIR"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
echo "=== All downloads complete ==="
echo ""
echo "Next steps — convert labels to SUIM binary mask format:"
echo "  python -m src.suimnet.convert_deepfish --profile greatlakes --split valid"
echo "  python -m src.suimnet.convert_usis10k  --profile greatlakes"
echo ""
echo "Then run inference + metrics for each dataset:"
echo "  sbatch --export=DATASET=suim     cluster/suimnet_infer.sbat"
echo "  sbatch --export=DATASET=deepfish cluster/suimnet_infer.sbat"
echo "  sbatch --export=DATASET=usis10k  cluster/suimnet_infer.sbat"
