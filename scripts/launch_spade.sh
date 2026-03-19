#!/usr/bin/env bash
# End-to-end SPADE pipeline: download data, set up venv, convert, evaluate.
# Run once on the Great Lakes login node, then go to sleep.
# Jobs chain automatically and requeue on node failure.
#
# Prerequisites:
#   export KAGGLE_API_TOKEN=<your-kaggle-token>   (for SeaThru)
#   Weights are downloaded automatically by download_spade_data.sh
#
# Usage:
#   export KAGGLE_API_TOKEN=KGAT_xxxxx
#   bash scripts/launch_spade.sh
#
# Safe to re-run — downloads, venv, and completed conversions are skipped.

set -euo pipefail

echo "=== SPADE pipeline ==="
echo ""

# ── Step 1: Download datasets (login node) ───────────────────────────────────
echo "── Downloading datasets ──"
bash scripts/download_spade_data.sh
echo ""

# ── Step 2: Set up venv (login node — avoids race in SLURM jobs) ─────────────
echo "── Setting up Python venv ──"
module load python/3.10.4 2>/dev/null || true

VENV_DIR="/scratch/rob572w26_class_root/rob572w26_class/${USER}/venvs/rob472-spade"
mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -f "$VENV_DIR/.deps_ok" ]]; then
    echo "  Creating virtualenv at $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements_spade.txt
    touch "$VENV_DIR/.deps_ok"
    deactivate 2>/dev/null || true
    echo "  Venv ready."
else
    echo "  Venv already set up — skipping."
fi
echo ""

# ── Step 3: Build weights from public DA V2 backbone (if not present) ────────
WEIGHTS_PATH="/scratch/rob572w26_class_root/rob572w26_class/${USER}/spade_weights/underwater_depth_pipeline.pt"
if [[ -f "$WEIGHTS_PATH" ]]; then
    echo "── Weights already at $WEIGHTS_PATH — skipping build."
else
    echo "── Building SPADE weights from public Depth Anything V2 backbone ──"
    source "$VENV_DIR/bin/activate"
    python scripts/build_spade_weights.py --out "$WEIGHTS_PATH"
    deactivate 2>/dev/null || true
fi
echo ""

# ── Step 4: Submit SLURM jobs with dependency chaining ───────────────────────
echo "── Submitting SLURM jobs ──"

# --requeue: auto-resubmit on node/preemption failure
# --dependency=afterany: attempt metrics even if convert had issues
RQ="--requeue"

# FLSea demo — bundled, no conversion needed
DEMO_MET=$(sbatch --parsable $RQ cluster/spade_metrics.sbat)
echo "  metrics  flsea_demo → $DEMO_MET"

# Convert raw datasets to SPADE format (CPU)
ST_CONV=$(sbatch --parsable $RQ --export=DATASET=seathru cluster/spade_convert.sbat)
FL_CONV=$(sbatch --parsable $RQ --export=DATASET=flsea   cluster/spade_convert.sbat)
echo "  convert  seathru    → $ST_CONV"
echo "  convert  flsea      → $FL_CONV"

# Evaluation + charts (GPU) — run after convert finishes (success or fail)
ST_MET=$(sbatch --parsable $RQ --dependency=afterany:${ST_CONV} --export=DATASET=seathru cluster/spade_metrics.sbat)
FL_MET=$(sbatch --parsable $RQ --dependency=afterany:${FL_CONV} --export=DATASET=flsea   cluster/spade_metrics.sbat)
echo "  metrics  seathru    → $ST_MET  (after $ST_CONV)"
echo "  metrics  flsea      → $FL_MET  (after $FL_CONV)"

echo ""
echo "=== 5 jobs submitted. Monitor with: squeue -u \$USER ==="
echo ""
echo "If anything fails, just re-run this script — downloads and"
echo "completed conversions are skipped. Only remaining work runs."
