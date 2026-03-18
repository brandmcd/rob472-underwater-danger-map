#!/usr/bin/env bash
# End-to-end SPADE pipeline: download data, convert, evaluate, charts.
# Run once on the Great Lakes login node, then go to sleep.
# Jobs chain automatically and requeue on node failure.
#
# Prerequisites:
#   export KAGGLE_API_TOKEN=<your-kaggle-token>   (for SeaThru)
#   Weights at $SCRATCH/spade_weights/underwater_depth_pipeline.pt
#
# Usage:
#   export KAGGLE_API_TOKEN=KGAT_xxxxx
#   bash scripts/launch_spade.sh
#
# Safe to re-run — downloads and completed conversions are skipped.

set -euo pipefail

echo "=== SPADE pipeline ==="
echo ""

# ── Step 1: Download datasets (runs on login node) ──────────────────────────
echo "── Downloading datasets ──"
bash scripts/download_spade_data.sh
echo ""

# ── Step 2: Submit SLURM jobs with dependency chaining ───────────────────────
echo "── Submitting SLURM jobs ──"

# --requeue: auto-resubmit on node/preemption failure
# --dependency=afterany: attempt metrics even if convert had issues
#   (the eval script checks for missing data and exits cleanly;
#    re-run this script to retry — completed work is skipped)
RQ="--requeue"

# FLSea demo — bundled, no conversion needed
DEMO_MET=$(sbatch --parsable $RQ cluster/spade_metrics.sbat)
echo "  metrics  flsea_demo → $DEMO_MET"

# Convert raw datasets to SPADE format (CPU)
ST_CONV=$(sbatch --parsable $RQ --export=DATASET=seathru cluster/spade_convert.sbat)
KS_CONV=$(sbatch --parsable $RQ --export=DATASET=kskin   cluster/spade_convert.sbat)
FL_CONV=$(sbatch --parsable $RQ --export=DATASET=flsea   cluster/spade_convert.sbat)
echo "  convert  seathru    → $ST_CONV"
echo "  convert  kskin      → $KS_CONV"
echo "  convert  flsea      → $FL_CONV"

# Evaluation + charts (GPU) — run after convert finishes (success or fail)
ST_MET=$(sbatch --parsable $RQ --dependency=afterany:${ST_CONV} --export=DATASET=seathru cluster/spade_metrics.sbat)
KS_MET=$(sbatch --parsable $RQ --dependency=afterany:${KS_CONV} --export=DATASET=kskin   cluster/spade_metrics.sbat)
FL_MET=$(sbatch --parsable $RQ --dependency=afterany:${FL_CONV} --export=DATASET=flsea   cluster/spade_metrics.sbat)
echo "  metrics  seathru    → $ST_MET  (after $ST_CONV)"
echo "  metrics  kskin      → $KS_MET  (after $KS_CONV)"
echo "  metrics  flsea      → $FL_MET  (after $FL_CONV)"

echo ""
echo "=== 7 jobs submitted. Monitor with: squeue -u \$USER ==="
echo ""
echo "If anything fails, just re-run this script — downloads and"
echo "completed conversions are skipped. Only remaining work runs."
