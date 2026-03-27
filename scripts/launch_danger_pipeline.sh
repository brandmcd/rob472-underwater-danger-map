#!/usr/bin/env bash
# Submit all remaining danger map pipeline jobs with SLURM dependency chaining.
# Run once on the Great Lakes login node — jobs execute automatically in order.
#
# Dependency chain:
#
#   [already running] spade-metrics seathru (46232045)
#                          │
#                          ▼
#   turbidity-sweep ─── (independent, runs immediately)
#   danger-map flsea ─── (independent, runs immediately)
#   danger-map seathru ─ afterok: seathru metrics
#   latency ──────────── afterok: danger-map flsea
#
# Usage:
#   bash scripts/launch_danger_pipeline.sh           # full run (all frames)
#   bash scripts/launch_danger_pipeline.sh --preview # 50-frame preview

set -euo pipefail

PREVIEW="${1:-}"
MAX_FRAMES_EXPORT=""
[[ "$PREVIEW" == "--preview" ]] && MAX_FRAMES_EXPORT=",MAX_FRAMES=50"

RQ="--requeue"

echo "=== Danger map pipeline ==="
echo ""

# ── Find the currently running SeaThru metrics job ───────────────────────────
ST_METRICS_RUNNING=$(squeue -u "$USER" -h --format="%i %j" \
    | awk '/spade-me/ {print $1; exit}')

if [[ -n "$ST_METRICS_RUNNING" ]]; then
    echo "  Detected running SeaThru metrics job: $ST_METRICS_RUNNING"
    ST_DEP="--dependency=afterok:${ST_METRICS_RUNNING}"
else
    echo "  No SeaThru metrics job detected — submitting SeaThru danger map immediately."
    ST_DEP=""
fi
echo ""

# ── Turbidity robustness sweep (independent) ─────────────────────────────────
TURB=$(sbatch --parsable $RQ cluster/turbidity_sweep.sbat)
echo "  turbidity-sweep          → $TURB  (starts immediately)"

# ── FLSea danger map (independent — rgb/ already exists) ─────────────────────
DM_FL=$(sbatch --parsable $RQ \
    --export="DATASET=flsea${MAX_FRAMES_EXPORT}" \
    cluster/danger_map.sbat)
echo "  danger-map  flsea        → $DM_FL  (starts immediately)"

# ── SeaThru danger map (waits for SeaThru metrics if still running) ──────────
DM_ST=$(sbatch --parsable $RQ \
    $ST_DEP \
    --export="DATASET=seathru${MAX_FRAMES_EXPORT}" \
    cluster/danger_map.sbat)
if [[ -n "$ST_METRICS_RUNNING" ]]; then
    echo "  danger-map  seathru      → $DM_ST  (after $ST_METRICS_RUNNING)"
else
    echo "  danger-map  seathru      → $DM_ST  (starts immediately)"
fi

# ── Latency profiler (waits for FLSea danger map — shares GPU resources) ─────
LAT=$(sbatch --parsable $RQ \
    --dependency=afterok:${DM_FL} \
    --export=ALL \
    --wrap="
        source /scratch/rob572w26_class_root/rob572w26_class/\${USER}/venvs/rob472-spade/bin/activate
        DATA_ROOT=/scratch/rob572w26_class_root/rob572w26_class/\${USER}/data
        SPADE_WEIGHTS=/scratch/rob572w26_class_root/rob572w26_class/\${USER}/spade_weights/underwater_depth_pipeline.pt
        python -m src.danger_map.profile_latency \
            --frames_dir  \$DATA_ROOT/flsea/spade/rgb \
            --depth_dir   \$DATA_ROOT/flsea/spade/depth \
            --spade_weights \$SPADE_WEIGHTS \
            --n_frames 50 \
            --out_csv reports/latency.csv
    " \
    --job-name=latency-profile \
    --account=rob572w26_class \
    --partition=gpu \
    --qos=class \
    --gpus=1 \
    --cpus-per-task=4 \
    --mem=24G \
    --time=00:30:00 \
    --output=logs/latency-profile-%j.log \
    --chdir="$(pwd)")
echo "  latency-profile          → $LAT  (after $DM_FL)"

echo ""
echo "=== $(( 4 )) jobs submitted ==="
echo ""
echo "  squeue -u \$USER                     — watch status"
echo "  tail -f logs/danger-map-${DM_FL}.log  — FLSea danger map log"
echo "  tail -f logs/turbidity-sweep-${TURB}.log"
echo ""
echo "Once done, pull results locally:"
echo "  scp -r brandmcd@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/reports/ ./reports/"
