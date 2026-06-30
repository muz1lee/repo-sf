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

This runs video prep, COLMAP camera reconstruction, MoGe dense reference
geometry, Qwen/SAM foreground removal, and the 3DGS runner. Missing optional
runtimes are recorded as status JSON files so the earlier artifacts remain
usable.

Set `RSF_VIDEO_BG_ONLY_SKIP=1` or `RSF_VIDEO_3DGS_SKIP=1` to stop before those
heavier stages.

## MoGe Dense Reference Geometry

```bash
scripts/rsf_video_moge_background.sh --run-dir runs/desk_cups_001_m7
```

This is the current dense geometry bridge for the video background branch:
`video/reference.png` -> dense point map/depth/PLY. It uses
`/mnt/workspace/wenqian/hawor_runtime/moge_venv/bin/python` and keeps torch/model
dependencies out of this project's `.venv`.

## BG-only Video Frames

```bash
scripts/rsf_video_bg_only.sh --run-dir runs/desk_cups_001_m7
```

This uses Qwen proposals before SAM masks, then writes inpainted foreground-free
frames to `video/bg_only/frames`, masks to `video/bg_only/masks`, and status to
`video/bg_only_status.json`.

## 3DGS Background Runner

```bash
scripts/rsf_video_3dgs.sh --run-dir runs/desk_cups_001_m7 --max-steps 3000
```

This prepares a Nerfstudio-style dataset from `video/bg_only/frames` and
`video/colmap/sparse/0`, then runs `ns-train splatfacto` when a 3DGS runtime is
available. The heavy training environment belongs in `.venv_3dgs`, not the
project `.venv`. On completion, `video/3dgs_status.json` records the latest
`config.yml` and checkpoint.

```bash
scripts/rsf_video_view_3dgs.sh --run-dir runs/desk_cups_001_m7 --websocket-port 7007
```

This starts Nerfstudio's interactive viewer from the latest trained 3DGS config.

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
