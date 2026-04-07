# ROB 472 — Underwater Danger Map
**Presentation: April 12, 2026**

---

## Project Overview

End-to-end pipeline that takes underwater imagery, segments hazards with SUIM-Net, estimates metric depth with SPADE, fuses them into a per-pixel danger map, and outputs an actionable AUV navigation command.

```
RGB frame → SUIM-Net → class logits ─┐
                                      ├→ danger_map() → risk_map → nav_command()
           → SPADE   → depth (m) ────┘                             ↓
                                                         command + sector risks
```

---

## Results at a Glance

| Metric | Value | Source |
|--------|-------|--------|
| SUIM-Net mIoU (SUIM test set) | **0.78** | `reports/suimnet/suim_metrics.csv` |
| SUIM-Net mIoU at turbidity 0.5 | **0.67** | `reports/turbidity/turbidity_summary.csv` |
| SUIM-Net mIoU at turbidity 1.0 | **0.39** (−54%) | `reports/turbidity/turbidity_summary.csv` |
| SPADE δ<1.25 (FLSea, 0–5m) | **0.928** | `reports/spade/flsea_metrics.csv` |
| SPADE abs_rel (FLSea, 0–5m) | **0.102** | `reports/spade/flsea_metrics.csv` |
| SPADE δ<1.25 (SeaThru, 0–5m) | **0.879** | `reports/spade/seathru_metrics.csv` |
| Pipeline FPS — CPU (WSL2) | **0.3** | `reports/latency_local.csv` |
| Pipeline FPS — GPU est. (A40) | **~13** | projected from component times |

---

## Where to Find Everything

### Code
| Component | File |
|-----------|------|
| Danger map fusion | [src/danger_map/__init__.py](src/danger_map/__init__.py) |
| Navigation + PLY writer | [src/danger_map/navigate.py](src/danger_map/navigate.py) |
| Video pipeline | [src/danger_map/run_video.py](src/danger_map/run_video.py) |
| Quick test (8 samples) | [src/danger_map/quick_test.py](src/danger_map/quick_test.py) |
| Latency profiler | [src/danger_map/profile_latency.py](src/danger_map/profile_latency.py) |
| Turbidity sweep | [src/augment/run_sweep.py](src/augment/run_sweep.py) |
| SPADE depth eval | `src/spade/` |
| SUIM-Net eval | `src/suimnet/` |

### Scripts (run from repo root)
| Script | What it does |
|--------|-------------|
| [scripts/make_report_figures.py](scripts/make_report_figures.py) | Regenerate all report charts from CSVs |
| [scripts/compare_pointcloud.py](scripts/compare_pointcloud.py) | Danger map + 3D point cloud comparison figure |
| [scripts/render_ply.py](scripts/render_ply.py) | Render a .ply file to perspective + top-down PNGs |

### Figures (report-ready)
| Figure | Path |
|--------|------|
| **System summary card** | [figures/charts/system_summary.png](figures/charts/system_summary.png) |
| Turbidity mIoU line chart | [figures/charts/turbidity_miou.png](figures/charts/turbidity_miou.png) |
| Turbidity heatmap (class × level) | [figures/charts/turbidity_heatmap.png](figures/charts/turbidity_heatmap.png) |
| Latency breakdown (CPU vs GPU) | [figures/charts/latency_breakdown.png](figures/charts/latency_breakdown.png) |
| SPADE depth accuracy | [figures/charts/spade_accuracy.png](figures/charts/spade_accuracy.png) |
| SUIM-Net per-class IoU | [figures/charts/suimnet_iou.png](figures/charts/suimnet_iou.png) |
| 3-panel nav overlay (wreck) | [figures/danger_map/w_r_147__danger.png](figures/danger_map/w_r_147__danger.png) |
| 3-panel nav overlay (diver) | [figures/danger_map/d_r_598__danger.png](figures/danger_map/d_r_598__danger.png) |
| Point cloud comparison | [figures/pointcloud/comparison.png](figures/pointcloud/comparison.png) |
| Navigation explanation | [NAVIGATION.md](NAVIGATION.md) |

### Raw Data / CSVs
| Data | Path |
|------|------|
| SUIM-Net metrics (SUIM test) | `reports/suimnet/suim_metrics.csv` |
| SUIM-Net metrics (DeepFish) | `reports/suimnet/deepfish_metrics.csv` |
| SPADE metrics (FLSea) | `reports/spade/flsea_metrics.csv` |
| SPADE metrics (SeaThru) | `reports/spade/seathru_metrics.csv` |
| Turbidity sweep summary | `reports/turbidity/turbidity_summary.csv` |
| Latency (local CPU) | `reports/latency_local.csv` |
| Latency (ARC GPU) | `reports/latency.csv` ← pending ARC job |
| Quick test overlays | `reports/danger_map/quick_test/` |

---

## Remaining Before Presentation

### Must-do (before April 12)

- [ ] **`git push` and submit ARC jobs** to get GPU results:

  ```bash
  # Danger map video (FLSea)
  sbatch cluster/danger_map.sbat

  # Danger map video (SeaThru)
  sbatch --export=DATASET=seathru cluster/danger_map.sbat

  # Turbidity sweep on full SUIM test set (currently only 8 sample images)
  sbatch cluster/turbidity_sweep.sbat

  # GPU latency profiler
  srun --account=rob572w26_class --partition=gpu --qos=class \
       --gpus=1 --cpus-per-task=4 --mem=24G --time=00:30:00 --pty bash
  python -m src.danger_map.profile_latency \
      --frames_dir $DATA_ROOT/flsea/spade/rgb \
      --depth_dir  $DATA_ROOT/flsea/spade/depth \
      --spade_weights $SPADE_WEIGHTS \
      --n_frames 50 \
      --out_csv reports/latency.csv
  ```

- [ ] **Regenerate figures after ARC results land:**
  ```bash
  python scripts/make_report_figures.py
  ```

- [ ] **Tune thresholds** in [navigate.py](src/danger_map/navigate.py) once real SPADE depth is confirmed:
  - `CLEAR_THRESHOLD = 0.20` — below this → CLEAR (currently most frames hit this on flat depth)
  - `DANGER_THRESHOLD = 0.55` — above this → DANGER

- [ ] **Test `run_video.py` end-to-end** with real data on ARC:
  ```bash
  python -m src.danger_map.run_video \
      --frames_dir $DATA_ROOT/flsea/spade/rgb \
      --depth_dir  $DATA_ROOT/flsea/spade/depth \
      --spade_weights $SPADE_WEIGHTS \
      --out_dir reports/danger_map/flsea \
      --save_ply --ply_every 30
  ```

### Nice-to-have

- [ ] Point cloud comparison figure from real SPADE depth (current one uses synthetic flat depth)
  ```bash
  python scripts/compare_pointcloud.py \
      --danger_png reports/danger_map/flsea/frames/000001_overlay.png \
      --ply reports/danger_map/flsea/clouds/000001_cloud.ply \
      --out figures/pointcloud/comparison_real.png
  ```

- [ ] Add turbidity + latency results to `PROGRESS_REPORT.md`

---

## Completed

- [x] SUIM-Net evaluation on SUIM test set, DeepFish, USIS10K
- [x] SPADE depth estimation on FLSea and SeaThru
- [x] Danger map fusion (`risk = hazard_weight × proximity`)
- [x] Navigation system — 3x3 sector grid, nav command, 3-panel overlay
- [x] `NAVIGATION.md` with images explaining the system
- [x] Turbidity robustness sweep (local — 8 samples; ARC — full test set pending)
- [x] Latency profiler (local CPU baseline working; ARC GPU pending)
- [x] All report figures generated: `figures/charts/`
- [x] `--save_ply` + `--video_file` flags added to `run_video.py`
- [x] `scripts/render_ply.py`, `scripts/compare_pointcloud.py`
