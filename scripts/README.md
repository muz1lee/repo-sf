# real2sim_scene_foundry 脚本说明

这些脚本是 `rsf` CLI 的服务器侧封装。默认项目虚拟环境位于 `.venv`；如果设置了 `RSF_PYTHON`，脚本会使用该 Python。

## 手机 RGB 视频预处理

```bash
scripts/rsf_video_prep.sh \
  --video /mnt/workspace/wenqian/real2sim_scene_foundry/inputs/videos/desk_cups_001.mp4 \
  --out runs/desk_cups_001 \
  --frame-stride 10 \
  --reference-frame-index 0
```

该命令会生成 sampled frames 和 `video/video_manifest.json`。它不会训练 3DGS；camera pose 和 BG-only frame generation 是后续 M7 步骤。

## M7 视频前端

```bash
scripts/rsf_video_m7.sh \
  --video /mnt/workspace/wenqian/real2sim_scene_foundry/inputs/videos/desk_cups_001.mp4 \
  --out runs/desk_cups_001_m7 \
  --frame-stride 10 \
  --reference-frame-index 0
```

该脚本串起 video prep、COLMAP camera reconstruction、MoGe dense reference geometry、Qwen/SAM foreground removal 和 3DGS runner。缺失的可选 runtime 会写入 status JSON，不会破坏已经生成的早期 artifacts。

如果想停在重阶段之前，可以设置：

```bash
RSF_VIDEO_BG_ONLY_SKIP=1
RSF_VIDEO_3DGS_SKIP=1
```

## MoGe Dense Reference Geometry

```bash
scripts/rsf_video_moge_background.sh --run-dir runs/desk_cups_001_m7
```

这是 video background 分支当前使用的 dense geometry bridge：`video/reference.png` -> dense point map / depth / PLY。它使用 `/mnt/workspace/wenqian/hawor_runtime/moge_venv/bin/python`，避免把 torch/model 依赖塞进项目 `.venv`。

## BG-only 视频帧

```bash
scripts/rsf_video_bg_only.sh --run-dir runs/desk_cups_001_m7
```

该脚本先用 Qwen proposals，再用 SAM masks，最后把 foreground-free frames 写入 `video/bg_only/frames`，把 masks 写入 `video/bg_only/masks`，把状态写入 `video/bg_only_status.json`。

## 3DGS 背景训练

```bash
scripts/rsf_video_3dgs.sh --run-dir runs/desk_cups_001_m7 --max-steps 3000
```

该脚本会用 `video/bg_only/frames` 和 `video/colmap/sparse/0` 准备 Nerfstudio 风格数据集，并在 3DGS runtime 可用时运行 `ns-train splatfacto`。重训练环境应放在 `.venv_3dgs`，不要放进项目 `.venv`。完成后，`video/3dgs_status.json` 会记录最新 `config.yml` 和 checkpoint。

查看训练好的 3DGS：

```bash
scripts/rsf_video_view_3dgs.sh --run-dir runs/desk_cups_001_m7 --websocket-port 7007
```

这会从最新 3DGS config 启动 Nerfstudio interactive viewer。

## Video Object Scene + 3DGS Background

```bash
scripts/rsf_video_scene.sh \
  --run-dir runs/desk_cups_001_m7 \
  --target-label cup \
  --target-label bottle \
  --max-objects 4 \
  --settle-steps 5
```

该脚本使用 MoGe reference frame 和 `points.exr` 作为 metric RGB-D 输入，通过 Qwen/SAM3/SAM3D/alignment 构建前景 object meshes，把训练好的 video 3DGS 背景写入 `scene_manifest.json`，导出 Genesis launcher，并可选运行 no-viewer Genesis settle。当前 Genesis launcher 加载物理 object 和 plane；3DGS 背景仍通过 `rsf_video_view_3dgs.sh` 在 Nerfstudio 中检查。

已有 reconstructed run 可以不重跑 Qwen/SAM/SAM3D，直接标准化支撑面：

```bash
source .venv/bin/activate
rsf support-plane --run-dir runs/desk_cups_001_m7
```

该命令从 `xyz.npy` 和 object-mask background rings 估计 support plane；dense points 不可用时会 fallback 到 mesh bottom。它会把 object world poses 平移到 Genesis/Isaac support plane `z=0`，并写回 `scene_manifest.json`、`exports/scene.usda`、object `pose.json` 和 QA。已有 `support_plane` 时使用 `--force` 重新计算。

## 双目完整场景

```bash
scripts/rsf_stereo_scene.sh \
  --left /mnt/workspace/wenqian/scene_edit_v0/test_data/041_L.png \
  --right /mnt/workspace/wenqian/scene_edit_v0/test_data/041_R.png \
  --calib /mnt/workspace/wenqian/scene_edit_v0/test_data/041_calib.json \
  --label "red cup" \
  --out runs/041_stereo_full
```

默认会使用 `http://localhost:5092/inpaint` 做 foreground removal，然后通过 knowin-world venv 运行 Genesis no-viewer settle。
