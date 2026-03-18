#!/usr/bin/env bash
# Submit all SLURM jobs for the ROB 472 Underwater Danger Map project.
# Jobs are chained with --dependency=afterok so they run in the correct order.
#
# Prerequisites (run manually before this script):
#   bash scripts/download_suimnet_data.sh
#   export KAGGLE_API_TOKEN=<your-token>
#   bash scripts/download_spade_data.sh
#   scp weights to $SCRATCH/spade_weights/underwater_depth_pipeline.pt
#
# Usage:
#   bash scripts/launch_all.sh          # submit everything
#   bash scripts/launch_all.sh suimnet  # SUIM-Net pipeline only
#   bash scripts/launch_all.sh spade    # SPADE pipeline only

set -euo pipefail

PIPELINE="${1:-all}"

echo "=== ROB 472 — Launching SLURM jobs ==="
echo "Pipeline : $PIPELINE"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# SUIM-Net pipeline: convert → infer → metrics
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$PIPELINE" == "all" || "$PIPELINE" == "suimnet" ]]; then
    echo "── SUIM-Net ──"

    # Convert labels (DeepFish + USIS10K; SUIM is already in correct format)
    DF_CONV=$(sbatch --parsable --export=DATASET=deepfish cluster/suimnet_convert.sbat)
    US_CONV=$(sbatch --parsable --export=DATASET=usis10k  cluster/suimnet_convert.sbat)
    echo "  convert  deepfish  → $DF_CONV"
    echo "  convert  usis10k   → $US_CONV"

    # Inference (GPU) — SUIM needs no conversion, deepfish/usis10k wait for convert
    SUIM_INF=$(sbatch --parsable --export=DATASET=suim cluster/suimnet_infer.sbat)
    DF_INF=$(sbatch --parsable --dependency=afterok:${DF_CONV} --export=DATASET=deepfish cluster/suimnet_infer.sbat)
    US_INF=$(sbatch --parsable --dependency=afterok:${US_CONV} --export=DATASET=usis10k  cluster/suimnet_infer.sbat)
    echo "  infer    suim      → $SUIM_INF"
    echo "  infer    deepfish  → $DF_INF  (after $DF_CONV)"
    echo "  infer    usis10k   → $US_INF  (after $US_CONV)"

    # Metrics + charts (CPU) — wait for inference
    SUIM_MET=$(sbatch --parsable --dependency=afterok:${SUIM_INF} --export=DATASET=suim     cluster/suimnet_metrics.sbat)
    DF_MET=$(sbatch --parsable   --dependency=afterok:${DF_INF}   --export=DATASET=deepfish cluster/suimnet_metrics.sbat)
    US_MET=$(sbatch --parsable   --dependency=afterok:${US_INF}   --export=DATASET=usis10k  cluster/suimnet_metrics.sbat)
    echo "  metrics  suim      → $SUIM_MET  (after $SUIM_INF)"
    echo "  metrics  deepfish  → $DF_MET  (after $DF_INF)"
    echo "  metrics  usis10k   → $US_MET  (after $US_INF)"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# SPADE pipeline: convert → metrics
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$PIPELINE" == "all" || "$PIPELINE" == "spade" ]]; then
    echo "── SPADE ──"

    # FLSea demo — bundled, no conversion needed
    DEMO_MET=$(sbatch --parsable cluster/spade_metrics.sbat)
    echo "  metrics  flsea_demo → $DEMO_MET"

    # Convert raw datasets to SPADE format (CPU)
    ST_CONV=$(sbatch --parsable --export=DATASET=seathru cluster/spade_convert.sbat)
    FL_CONV=$(sbatch --parsable --export=DATASET=flsea   cluster/spade_convert.sbat)
    echo "  convert  seathru    → $ST_CONV"
    echo "  convert  flsea      → $FL_CONV"

    # Evaluation + charts (GPU) — wait for conversion
    ST_MET=$(sbatch --parsable --dependency=afterok:${ST_CONV} --export=DATASET=seathru cluster/spade_metrics.sbat)
    FL_MET=$(sbatch --parsable --dependency=afterok:${FL_CONV} --export=DATASET=flsea   cluster/spade_metrics.sbat)
    echo "  metrics  seathru    → $ST_MET  (after $ST_CONV)"
    echo "  metrics  flsea      → $FL_MET  (after $FL_CONV)"
    echo ""
fi

echo "=== All jobs submitted. Monitor with: squeue -u \$USER ==="
