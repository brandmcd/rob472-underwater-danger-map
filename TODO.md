# TODO
**Presentation: April 12, 2026**

---

## Must do before presentation

- [ ] `git push` then submit video jobs on ARC — see [ARC_VIDEO_PIPELINE.md](ARC_VIDEO_PIPELINE.md)
  ```bash
  sbatch --export=VIDEO=videos/raw/bluerov1.mp4 cluster/video_danger_map.sbat
  sbatch --export=VIDEO=videos/raw/bluerov2.mp4 cluster/video_danger_map.sbat
  sbatch --export=VIDEO=videos/raw/multimedia-unexpected-1920x1080-1.mp4 cluster/video_danger_map.sbat
  ```
- [ ] GPU latency profiler on ARC (need real FPS number, not CPU estimate)
  ```bash
  srun --account=rob572w26_class --partition=gpu --qos=class \
       --gpus=1 --cpus-per-task=4 --mem=24G --time=00:30:00 --pty bash
  python -m src.danger_map.profile_latency \
      --frames_dir $DATA_ROOT/flsea/spade/rgb \
      --depth_dir  $DATA_ROOT/flsea/spade/depth \
      --spade_weights $SPADE_WEIGHTS \
      --n_frames 50 --out_csv reports/latency.csv
  ```
- [ ] Turbidity sweep on full SUIM test set (current results are 8-image sample only)
  ```bash
  sbatch cluster/turbidity_sweep.sbat
  ```
- [ ] Regenerate charts after ARC results land: `python scripts/make_report_figures.py`

---

## Nice to have

- [ ] Point cloud figure from real SPADE depth (current one uses synthetic flat depth)
- [ ] Add results to PROGRESS_REPORT.md once ARC jobs land
