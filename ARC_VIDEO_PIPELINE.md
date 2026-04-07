# Running the Video Danger Map Pipeline on ARC

Running the full pipeline on all 3 videos locally takes ~14 hours on CPU.
On ARC (A40 GPU) it runs at ~13 FPS, finishing each video in a few minutes.

---

## Prerequisites

### 1. Push your code and copy SPADE weights to ARC

```bash
# From your local machine:
git push

# Copy SPADE weights to ARC scratch (only needed once)
UNIQNAME=bamcd   # change to your uniqname
SCRATCH_WEIGHTS="/scratch/rob572w26_class_root/rob572w26_class/${UNIQNAME}/spade_weights/underwater_depth_pipeline.pt"

ssh ${UNIQNAME}@greatlakes.arc-ts.umich.edu "mkdir -p $(dirname $SCRATCH_WEIGHTS)"
scp ~/Downloads/underwater_depth_pipeline.pt \
    ${UNIQNAME}@greatlakes.arc-ts.umich.edu:${SCRATCH_WEIGHTS}
```

### 2. Pull on ARC

```bash
ssh bamcd@greatlakes.arc-ts.umich.edu
cd ~/rob472-underwater-danger-map
git pull
git submodule update --init --recursive
```

---

## Run all 3 videos (submit one job per video)

```bash
cd ~/rob472-underwater-danger-map

sbatch --export=VIDEO=videos/raw/bluerov1.mp4 \
       cluster/video_danger_map.sbat

sbatch --export=VIDEO=videos/raw/bluerov2.mp4 \
       cluster/video_danger_map.sbat

sbatch --export=VIDEO=videos/raw/multimedia-unexpected-1920x1080-1.mp4 \
       cluster/video_danger_map.sbat
```

Monitor with:
```bash
squeue -u $USER
```

Each job writes to `videos/processed/<stem>/`:
```
videos/processed/bluerov1/
    danger_map.mp4          ← 3-panel output video  [ Original | Danger Map | Navigation ]
    frames/                 ← per-frame PNGs
    clouds/                 ← risk PLY point clouds (every 30 frames)
```

---

## Quick preview (30 frames only — confirm it looks right before full run)

```bash
sbatch --export=VIDEO=videos/raw/bluerov1.mp4,MAX_FRAMES=30 \
       cluster/video_danger_map.sbat
```

---

## Copy results back locally

```bash
UNIQNAME=bamcd
scp -r ${UNIQNAME}@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/videos/processed/ \
    videos/
```

Or just the videos (not the per-frame PNGs, which are large):
```bash
scp ${UNIQNAME}@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/videos/processed/bluerov1/danger_map.mp4 \
    videos/processed/bluerov1/
scp ${UNIQNAME}@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/videos/processed/bluerov2/danger_map.mp4 \
    videos/processed/bluerov2/
scp ${UNIQNAME}@greatlakes.arc-ts.umich.edu:~/rob472-underwater-danger-map/videos/processed/multimedia-unexpected-1920x1080-1/danger_map.mp4 \
    "videos/processed/multimedia-unexpected-1920x1080-1/"
```

---

## Video details

| Video | Frames | Duration | Resolution | Est. ARC time |
|-------|--------|----------|------------|----------------|
| bluerov1.mp4 | 7,036 | ~4 min | 640×360 | ~9 min |
| bluerov2.mp4 | 4,329 | ~2.5 min | 640×360 | ~6 min |
| multimedia-unexpected-1920x1080-1.mp4 | 4,850 | ~2.7 min | 1920×1080 | ~7 min |

*(GPU estimates based on ~13 FPS throughput on A40)*

---

## Troubleshooting

**"SPADE weights not found"**
```bash
# Re-copy weights:
scp ~/Downloads/underwater_depth_pipeline.pt \
    bamcd@greatlakes.arc-ts.umich.edu:/scratch/rob572w26_class_root/rob572w26_class/bamcd/spade_weights/underwater_depth_pipeline.pt
```

**"venv not found / module errors"**
The job script auto-creates the venv on first run. If it fails mid-setup, delete the partial venv and resubmit:
```bash
rm -rf /scratch/rob572w26_class_root/rob572w26_class/$USER/venvs/rob472-spade
```

**Check job logs:**
```bash
cat logs/video-danger-map-<JOBID>.log
```
