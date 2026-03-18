#!/usr/bin/env bash
# Submit all SPADE SLURM jobs with dependency chaining.
# Run this once, then go to sleep — jobs chain automatically.
#
# Prerequisites:
#   export KAGGLE_API_TOKEN=<your-token>
#   bash scripts/download_spade_data.sh
#   scp weights to $SCRATCH/spade_weights/underwater_depth_pipeline.pt
#
# Usage:
#   bash scripts/launch_spade.sh

set -euo pipefail

echo "=== SPADE pipeline — submitting SLURM jobs ==="
echo ""

# FLSea demo — bundled, no conversion needed
DEMO_MET=$(sbatch --parsable cluster/spade_metrics.sbat)
echo "  metrics  flsea_demo → $DEMO_MET"

# Convert raw datasets to SPADE format (CPU)
ST_CONV=$(sbatch --parsable --export=DATASET=seathru cluster/spade_convert.sbat)
KS_CONV=$(sbatch --parsable --export=DATASET=kskin   cluster/spade_convert.sbat)
FL_CONV=$(sbatch --parsable --export=DATASET=flsea   cluster/spade_convert.sbat)
echo "  convert  seathru    → $ST_CONV"
echo "  convert  kskin      → $KS_CONV"
echo "  convert  flsea      → $FL_CONV"

# Evaluation + charts (GPU) — wait for conversion
ST_MET=$(sbatch --parsable --dependency=afterok:${ST_CONV} --export=DATASET=seathru cluster/spade_metrics.sbat)
KS_MET=$(sbatch --parsable --dependency=afterok:${KS_CONV} --export=DATASET=kskin   cluster/spade_metrics.sbat)
FL_MET=$(sbatch --parsable --dependency=afterok:${FL_CONV} --export=DATASET=flsea   cluster/spade_metrics.sbat)
echo "  metrics  seathru    → $ST_MET  (after $ST_CONV)"
echo "  metrics  kskin      → $KS_MET  (after $KS_CONV)"
echo "  metrics  flsea      → $FL_MET  (after $FL_CONV)"

echo ""
echo "=== 7 jobs submitted. Monitor with: squeue -u \$USER ==="
