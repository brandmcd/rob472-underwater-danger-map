# ROB 472 Underwater Danger Map — TODO
**Due: April 12, 2026**

---

## Immediate coding tasks

- [ ] Install CPU torch into `.venv` so `run_video.py` works locally without ARC:
  ```
  source .venv/bin/activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- [ ] Test `run_video.py` locally end-to-end using SUIM sample images + local weights:
  ```
  python -m src.danger_map.run_video \
      --frames_dir vendor/SUIM-Net/sample_test/images \
      --suimnet_weights vendor/SUIM-Net/sample_test/ckpt_seg_5obj.hdf5 \
      --spade_weights ~/Downloads/underwater_depth_pipeline.pt \
      --out_dir reports/danger_map/local_test \
      --max_frames 8
  ```
- [ ] Run latency profiler locally (CPU, just to verify it works before ARC):
  ```
  python -m src.danger_map.profile_latency \
      --frames_dir vendor/SUIM-Net/sample_test/images \
      --spade_weights ~/Downloads/underwater_depth_pipeline.pt \
      --n_frames 8 \
      --out_csv reports/latency_local.csv
  ```
- [ ] Wire `nav_command()` + PLY output into `run_video.py` — add `--save_ply` flag that writes a point cloud for each frame (or every Nth frame) to `out_dir/clouds/`
- [ ] Tune `CLEAR_THRESHOLD` and `DANGER_THRESHOLD` in `navigate.py` once you see real SPADE depth (flat synthetic depth may make every scene look "CLEAR")

---

## ARC jobs to submit (after `git push`)

- [ ] `sbatch cluster/danger_map.sbat` — FLSea danger map video
- [ ] `sbatch --export=DATASET=seathru cluster/danger_map.sbat` — SeaThru danger map video
- [ ] `sbatch cluster/turbidity_sweep.sbat` — turbidity robustness sweep
- [ ] Run latency profiler on GPU node:
  ```
  python -m src.danger_map.profile_latency \
      --frames_dir $DATA_ROOT/flsea/spade/rgb \
      --depth_dir  $DATA_ROOT/flsea/spade/depth \
      --spade_weights $SPADE_WEIGHTS \
      --n_frames 50 \
      --out_csv reports/latency.csv
  ```

---

## Stretch goal — point cloud navigation (in progress)

- [x] `navigate.py` — nav_command(), draw_nav_overlay(), PLY writer
- [x] Integrated into quick_test.py — "^^ ASCEND" HUD working
- [ ] Add `--save_ply` to `run_video.py` to write one PLY per N frames
- [ ] Render a top-down screenshot of a PLY in MeshLab / CloudCompare → figure for report
- [ ] Write 2-sentence report blurb: "the nav command can feed directly into an AUV's planning stack; the coloured point cloud gives a 3D map operators can inspect after a mission"

---

## Collect and add to report (once ARC jobs land)

- [ ] SeaThru SPADE metrics table + charts → PROGRESS_REPORT.md
- [ ] 2–3 danger map overlay frames from real SPADE depth → PROGRESS_REPORT.md
- [ ] Latency table (SUIM-Net ms / SPADE ms / total ms / FPS)
- [ ] Turbidity sweep figure → `figures/charts/turbidity_sweep.png`
- [ ] Color histogram SUIM vs DeepFish → `figures/charts/color_histogram_suim_vs_deepfish.png`

---

## Report writing (pending data)

- [ ] Discussion: domain gap (color histogram), turbidity degradation, actionable risk vs raw accuracy
- [ ] Future work: SLAM integration, Jetson Orin deployment, fine-tuning on DeepFish
- [ ] Limitations: DA V2 + GA only (no full DAT head), SUIM misses seafloor/cables
- [ ] Final abstract + intro updated once all results are in

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
| ARC — danger map | `cluster/danger_map.sbat` |
| ARC — turbidity | `cluster/turbidity_sweep.sbat` |
| ARC — SPADE metrics | `cluster/spade_metrics.sbat` |
| Report | `PROGRESS_REPORT.md` |
| SPADE weights (local) | `~/Downloads/underwater_depth_pipeline.pt` |
| FLSea metrics CSV | `reports/spade/flsea_metrics.csv` |
| Latency CSV (ARC) | `reports/latency.csv` |
