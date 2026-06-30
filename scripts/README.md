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

## Video Object Scene + 3DGS Background

```bash
scripts/rsf_video_scene.sh \
  --run-dir runs/desk_cups_001_m7 \
  --target-label cup \
  --target-label bottle \
  --max-objects 4 \
  --settle-steps 5
```

This uses the MoGe reference frame and `points.exr` as metric RGB-D input,
builds foreground object meshes through Qwen/SAM3/SAM3D/alignment, attaches the
trained video 3DGS background to `scene_manifest.json`, exports the Genesis
launcher, and optionally runs no-viewer Genesis settle. The Genesis launcher
currently loads physics objects and a plane; the 3DGS background is inspected in
Nerfstudio via `rsf_video_view_3dgs.sh`.

Existing reconstructed runs can be normalized without rerunning Qwen/SAM/SAM3D:

```bash
source .venv/bin/activate
rsf support-plane --run-dir runs/desk_cups_001_m7
```

This estimates the support plane from `xyz.npy` and object-mask background rings,
falling back to mesh bottoms when dense points are unavailable. It shifts object
world poses so the Genesis/Isaac support plane is `z=0` and writes the result to
`scene_manifest.json`, `exports/scene.usda`, object `pose.json`, and QA.
Use `--force` to recompute a run that already has a `support_plane` entry.

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

This wrapper first runs `rsf support-plane`, then regenerates the interactive
launcher and executes Genesis through the knowin-world virtual environment.
