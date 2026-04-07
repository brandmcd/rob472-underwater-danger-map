# ROB 472 Underwater Danger Map — TODO
**Presentation: April 12, 2026** (5 days away — focus on results + polish)

---

## Immediate coding tasks

- [x] Install CPU torch + timm + einops into `.venv` so latency profiler works locally
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install timm einops
  ```
- [x] Run latency profiler locally (CPU baseline — confirms script works before ARC):
  ```
  python -m src.danger_map.profile_latency \
      --frames_dir vendor/SUIM-Net/sample_test/images \
      --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \
      --spade_weights ~/Downloads/underwater_depth_pipeline.pt \
      --n_frames 8 \
      --out_csv reports/latency_local.csv
  ```
  **Local CPU results (for reference, NOT for report):**
  SUIM-Net p50=666ms, SPADE p50=2012ms, total p50=2665ms @ 0.3 FPS

- [x] Wire `nav_command()` + 3-panel layout into `run_video.py`
  - `--save_ply` flag writes one PLY per `--ply_every` frames to `<out_dir>/clouds/`
  - `--video_file` flag added (direct video input, no frame extraction needed)
- [x] Fix navigation overlay — replaced tiny corner grid with full 3-panel:
  `[ Original | Danger Map | Sector Risk + Nav Command ]`
- [x] Run turbidity sweep locally on SUIM sample data → `reports/turbidity/`
- [x] Write `NAVIGATION.md` with images explaining how the navigation system works
- [x] Point cloud scripts:
  - `scripts/render_ply.py` — PLY → PNG (perspective + top-down)
  - `scripts/compare_pointcloud.py` — danger map + point cloud comparison figure

- [ ] Test `run_video.py` locally end-to-end (needs SPADE + SUIM-Net together):
  ```bash
  python -m src.danger_map.run_video \
      --frames_dir vendor/SUIM-Net/sample_test/images \
      --spade_weights ~/Downloads/underwater_depth_pipeline.pt \
      --out_dir reports/danger_map/local_test \
      --max_frames 8
  ```
- [ ] Tune `CLEAR_THRESHOLD` and `DANGER_THRESHOLD` in `navigate.py` once real SPADE depth
      is available — current flat synthetic depth makes most scenes appear "CLEAR"

---

## ARC jobs to submit (after `git push`)

Run these on ARC Great Lakes. Make sure to `git pull` on the cluster first.

- [ ] `sbatch cluster/danger_map.sbat` — FLSea danger map video (3-panel with nav)
- [ ] `sbatch --export=DATASET=seathru cluster/danger_map.sbat` — SeaThru danger map video
- [ ] `sbatch cluster/turbidity_sweep.sbat` — turbidity robustness sweep on full SUIM test set
- [ ] Run GPU latency profiler on ARC:
  ```bash
  srun --account=rob572w26_class --partition=gpu --qos=class \
       --gpus=1 --cpus-per-task=4 --mem=24G --time=00:30:00 --pty bash
  source /scratch/.../venvs/rob472-spade/bin/activate
  cd ~/rob472-underwater-danger-map

  python -m src.danger_map.profile_latency \
      --frames_dir $DATA_ROOT/flsea/spade/rgb \
      --depth_dir  $DATA_ROOT/flsea/spade/depth \
      --spade_weights $SPADE_WEIGHTS \
      --n_frames 50 \
      --out_csv reports/latency.csv
  ```
  **Target numbers (GPU A40):** SUIM-Net ~15ms, SPADE ~50ms, total ~70ms → ~14 FPS

---

## Navigation system (done)

- [x] `navigate.py` — `nav_command()`, `draw_nav_overlay()`, PLY writer
- [x] 3-panel layout in `quick_test.py` and `run_video.py`
- [x] `NAVIGATION.md` — full explanation with images (how sectors work, commands, point cloud)
- [x] `scripts/render_ply.py` — renders PLY to perspective + top-down PNG
- [x] `scripts/compare_pointcloud.py` — side-by-side comparison figure (danger map + 3D cloud)
- [x] `--save_ply` flag in `run_video.py` — writes PLY every N frames

---

## Turbidity sweep (done locally — ARC for full results)

Local results (8 sample images, 5 turbidity levels):

| Turbidity | mIoU |
|-----------|------|
| 0.00 | 0.878 |
| 0.25 | 0.809 |
| 0.50 | 0.665 |
| 0.75 | 0.483 |
| 1.00 | 0.393 |

→ Summary CSV: `reports/turbidity/turbidity_summary.csv`
→ Full SUIM test set results pending ARC job

---

## Collect and add to report (once ARC jobs land)

- [ ] SeaThru SPADE metrics table + charts → PROGRESS_REPORT.md
- [ ] 2–3 danger map overlay frames from real SPADE depth → PROGRESS_REPORT.md
- [ ] Latency table (GPU): SUIM-Net ms / SPADE ms / total ms / FPS
- [ ] Turbidity sweep figure on full SUIM test set → `figures/charts/turbidity_sweep.png`
- [ ] Point cloud comparison figure from real SPADE depth → `figures/pointcloud/comparison.png`

---

## Report / presentation writing

- [ ] Discussion: turbidity degradation table, actionable risk vs raw accuracy, nav system
- [ ] Explain 3-panel layout in report — use `figures/danger_map/w_r_147__danger.png` as example
- [ ] Future work: SLAM integration, Jetson Orin deployment, fine-tuning on DeepFish
- [ ] Limitations: DA V2 + GA only (no full DAT head), SUIM misses seafloor/cables
- [ ] Final abstract + intro once all results are in

---

## Quick reference

| What | Path |
|------|------|
| Danger map core | `src/danger_map/__init__.py` |
| Navigation commands | `src/danger_map/navigate.py` |
| Video pipeline | `src/danger_map/run_video.py` |
| Latency profiler | `src/danger_map/profile_latency.py` |
| Quick test | `src/danger_map/quick_test.py` |
| Turbidity sweep | `src/augment/run_sweep.py` |
| PLY renderer | `scripts/render_ply.py` |
| Comparison figure | `scripts/compare_pointcloud.py` |
| Navigation doc | `NAVIGATION.md` |
| ARC — danger map | `cluster/danger_map.sbat` |
| ARC — turbidity | `cluster/turbidity_sweep.sbat` |
| Report | `PROGRESS_REPORT.md` |
| SPADE weights (local) | `~/Downloads/underwater_depth_pipeline.pt` |
| Local latency CSV | `reports/latency_local.csv` |
| Turbidity summary | `reports/turbidity/turbidity_summary.csv` |
