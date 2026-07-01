# real2sim-scene-foundry

`real2sim-scene-foundry` 是一个面向 SimFoundry 风格 real-to-sim 场景重建的工程仓库。它把双目图片或 RGB 视频输入整理成物体资产、metric scene manifest、碰撞几何、仿真导出、QA 报告和可交互检查 viewer。

这个仓库是工程复现工作台，不是“已经完整复现 SimFoundry 论文效果”的声明。当前 canonical run 已经采用 fail-closed 策略：object mesh、table collision、Genesis settle、USD/Genesis/Isaac export 骨架和 viewer provenance 已经存在；native/live 3DGS 背景注册仍是主要 blocker。

## 当前状态

- 物体视觉资产：最终视觉路径是 `objects/*/visual.glb`；legacy `mesh_aligned.glb` 只允许作为 debug/proxy。
- 背景：可以存在训练后的 3DGS sidecar 资产，但 viewer 当前仍使用外部 reference-view PNG sidecar，除非后续接入 native splat runtime。
- 物理：object collision mesh 和 support surface 与 visual mesh 分离导出。
- 导出：`sim_export_manifest.json` 和 `qa/sim_export_report.json` 应在背景未注册、native runtime 缺失、table QA 过弱或 Isaac evidence 过期时 fail closed。
- Viewer：`rsf composite-viewer` 是检查 UI，会明确报告背景是 live 3DGS、external sidecar 还是 diagnostic point cloud。

## 仓库结构

```text
src/real2sim_scene_foundry/   Python package 和 CLI 实现
tests/                        单元测试和回归测试
scripts/                      服务器侧 video / 3DGS 辅助脚本
docs/                         gap ledger、runtime probe、发布说明
AGENTS.md                     后续 agent 的项目规则
AI_START_HERE.md              新 AI/Codex 会话的启动事实
PLAN.md                       开发计划和当前 milestone
```

生成产物默认不进 Git：`runs/`、`outputs/`、虚拟环境、视频、点云、mesh、checkpoint、USD 导出和渲染图都应被忽略。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m pytest tests -q
```

CLI 入口：

```bash
rsf --help
```

## 常用命令

服务器上有 smoke 数据时，可以跑双目 smoke：

```bash
rsf smoke \
  --left /mnt/workspace/wenqian/scene_edit_v0/test_data/041_L.png \
  --right /mnt/workspace/wenqian/scene_edit_v0/test_data/041_R.png \
  --calib /mnt/workspace/wenqian/scene_edit_v0/test_data/041_calib.json \
  --out runs/smoke_041
```

刷新 canonical artifacts：

```bash
rsf support-plane --run-dir runs/my_table_m7_20260630_185332 --force
rsf refine-pose --run-dir runs/my_table_m7_20260630_185332
rsf render-3dgs-background --run-dir runs/my_table_m7_20260630_185332
rsf export-manifest --run-dir runs/my_table_m7_20260630_185332
rsf export-sim --run-dir runs/my_table_m7_20260630_185332 --backend genesis --backend usd --backend isaac
rsf qa-sim --run-dir runs/my_table_m7_20260630_185332
rsf composite-viewer --run-dir runs/my_table_m7_20260630_185332 --export-only
```

启动 viewer：

```bash
rsf composite-viewer --run-dir runs/my_table_m7_20260630_185332 --host 127.0.0.1 --port 7010
```

如果 viewer 跑在远端服务器上，先做端口转发：

```bash
ssh -fN -o ExitOnForwardFailure=yes -L 7010:127.0.0.1:7010 wenqian_h200
```

## 环境变量

参考 `.env.example`。SAM3D 没有硬编码默认 endpoint；只有在当前服务注册表确认后，才设置 `SAM3D_PROCESS_URL`。

项目可以调用外部 segmentation、depth、inpainting、3D asset generation、3DGS training、Genesis 和 Isaac validation 服务。测试应能在没有这些服务的情况下运行。

## GitHub 发布说明

发布前先读 `docs/GITHUB_PUBLISHING.md`。尤其注意：

- 不要提交 `runs/`、`.venv*/`、模型权重、视频、GLB/PLY/USD 输出、checkpoint 或截图。
- 如果 repo 需要公开，先脱敏内部服务器路径、SSH alias、服务地址和 run 名称。
- 公开仓库前先选择 license。

## License

当前尚未选择 license。添加 license 前，请把仓库视为 private/internal 或 all-rights-reserved。
