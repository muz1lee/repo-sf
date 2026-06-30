# real2sim_scene_foundry Scripts

These scripts are server-oriented wrappers around the `rsf` CLI. They assume the
project virtual environment exists at `.venv` unless `RSF_PYTHON` is set.

## Phone RGB Video Prep

```bash
scripts/rsf_video_prep.sh \
  --video /mnt/workspace/wenqian/real2sim_scene_foundry/inputs/videos/desk_cups_001.mp4 \
  --out runs/desk_cups_001 \
  --frame-stride 10 \
  --reference-frame-index 0
```

This prepares sampled frames and `video/video_manifest.json`. It does not train
3DGS yet; camera poses and BG-only frame generation are the next M7 steps.

## M7 Video Front End

```bash
scripts/rsf_video_m7.sh \
  --video /mnt/workspace/wenqian/real2sim_scene_foundry/inputs/videos/desk_cups_001.mp4 \
  --out runs/desk_cups_001_m7 \
  --frame-stride 10 \
  --reference-frame-index 0
```

This runs video prep and then tries COLMAP. If COLMAP is not installed it writes
`video/colmap_status.json` with `status=missing_colmap`.

## Stereo Full Scene

```bash
scripts/rsf_stereo_scene.sh \
  --left /mnt/workspace/wenqian/scene_edit_v0/test_data/041_L.png \
  --right /mnt/workspace/wenqian/scene_edit_v0/test_data/041_R.png \
  --calib /mnt/workspace/wenqian/scene_edit_v0/test_data/041_calib.json \
  --label "red cup" \
  --out runs/041_stereo_full
```

By default this uses `http://localhost:5092/inpaint` for foreground removal and
then runs Genesis no-viewer settle through the knowin-world venv.

## Genesis Settle Only

```bash
scripts/rsf_genesis_settle.sh --run-dir runs/041_stereo_full --settle-steps 5
```
