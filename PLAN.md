# SimFoundry 风格 Real-to-Sim 场景复现计划

## Summary

- 工作线：**项目落地先行、论文后置**。
- 目标：不是只做 object mesh，而是复现 SimFoundry 图示主链路：真实双目/视频输入 -> extraction -> foreground removal -> object digital twins -> background reconstruction -> automatic alignment -> 可交互 sim scene。
- M10+ simulator export 计划见 `docs/superpowers/plans/2026-06-30-simfoundry-simulator-export-reproduction.md`。后续不以 fallback、服务 smoke 或最小单测作为目标，而以 artifact-level Genesis/USD/Isaac export 和 QA 报告作为验收口径。
- 当前状态（2026-07-01）：object visual path 已完成；外部 Nerfstudio 3DGS renderer 可输出 `qa/background_3dgs_render.png`；但 full SimFoundry visual reconstruction 仍 blocked 在 native/live 3DGS runtime 和 `T_3dgs_world_to_sim_world` 注册缺口。
- 当前不能把 `sim_export_manifest.json` 或 `qa/sim_export_report.json` 的工程 contract 当论文效果复现；背景仍是 external 3DGS PNG sidecar 时，整体只能标成 partial。
- 论文依据：SimFoundry 将流程拆成 Extraction、Generation、Augmentation 三段，从单个真实视频重建 sim-ready digital twin，并可扩展 object/scene/task cousins。当前先做到 simulator export，不做 cousins/policy。
- 服务器位置：`wenqian_h200`；权威代码目录：`/mnt/workspace/wenqian/real2sim_scene_foundry`。
- 不改 `ego_hand_pipeline`、`ego_eef`、`knowin-world`、`scene_edit_v0` 这些旧仓库，除非用户明确要求。

## Key Changes

- 独立 Python package + CLI：`rsf`。
- RGB 视频入口：`rsf video-prep --video phone.mp4 --out runs/<id> --frame-stride 10`，先输出 sampled frames、reference frame 和 `video_manifest.json`。
- 服务适配器：
  - SAM3：`http://101.132.143.105:5081/segment`
  - SAM3D：当前未登记可用 HTTP endpoint；除非当前服务注册表显式给出，否则不要调用任何假定的 SAM3D HTTP 地址。
  - MoGe focal：`http://101.132.143.105:5014/api/focal`
  - S2M2：`http://10.10.4.244:5060-5067/api/process`
  - Gemini/Qwen：通过环境变量读取 API key；无 key 时允许 YAML 手写 object list。
- 坐标约定：
  - 相机系：OpenCV `x right, y down, z forward`，单位 meter。
  - 仿真世界系：`z up`，单位 meter。
- artifact policy：所有 run 产物保存在 `runs/<id>`，不进 Git。

## Milestones

### M1-M4：双目 object reconstruction baseline

- 输入 left/right/calib。
- 输出 extraction manifest、object mask/crop/cloud、scene cloud、object pose/scale、USD stub、QA overlay。
- 当前状态：基础能力已存在，但不等于完整 SimFoundry reproduction。

### M5-M7：视频背景和 3DGS

- 支持 RGB video prep、COLMAP、MoGe dense reference geometry、BG-only frames、Nerfstudio/splatfacto 3DGS training。
- 当前状态：canonical run 有 completed 3DGS training 和 external 3DGS render sidecar。
- 主要缺口：native/live 3DGS runtime 未集成；3DGS-to-sim transform 未完成。

### M8-M9：object scene composition + support/physics QA

- 用 reference frame 和 metric point map 生成 foreground object digital twins。
- 支撑面从 background/tabletop evidence 重建。
- object visual asset 与 collision asset 分离。
- Genesis settle 可检查物理稳定性。
- 当前状态：table collision 已升级到 tabletop mask polygon slab；Genesis settle 可通过，但 settled pose writeback 仍 partial。

### M10：Simulator export gate

- 输出 `sim_export_manifest.json`、`exports/scene.usda`、`exports/genesis_scene.py`、`exports/isaac_scene.py`、`qa/sim_export_report.json`。
- export gate 必须 fail closed：背景未注册、native 3DGS runtime 缺失、物理属性来源不清、table QA 弱通过、Isaac preserved report 被冒充 fresh validation，都不能 passed。
- 当前状态：export bundle 存在，但 full reproduction 被 `background_unregistered` / native runtime 缺口阻塞。

## Acceptance Criteria

一个 run 只有同时满足这些条件，才可以称为 export-grade reproduction：

1. 每个 object 有 final visual asset，不能是 bbox/proxy。
2. 每个 object 有 explicit collision asset 和 `physics.json`。
3. 6D pose 有 source report，并通过 mask/depth/support QA。
4. background branch 输出 foreground mask、BG-only image/cloud、3DGS artifact 和 registration report。
5. `T_3dgs_world_to_sim_world` 不是 placeholder；缺失时必须 blocked。
6. table/support collision 来自 tabletop mask/depth polygon，不是大 box 冒充。
7. Genesis/Isaac export 有 loader/report；Isaac preserved report 必须标明 caveat。
8. viewer 明确区分 live 3DGS、external PNG sidecar、BG-only diagnostic cloud 和 full-scene debug cloud。
9. `python -m pytest tests -q` 通过。
10. `qa/sim_export_report.json` 不能用 fallback/proxy 掩盖 blocker。

## Test Plan

常规测试：

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate
python -m pytest tests -q
```

canonical export 链路：

```bash
rsf support-plane --run-dir runs/my_table_m7_20260630_185332 --force
rsf refine-pose --run-dir runs/my_table_m7_20260630_185332
rsf render-3dgs-background --run-dir runs/my_table_m7_20260630_185332
rsf export-manifest --run-dir runs/my_table_m7_20260630_185332
rsf export-sim --run-dir runs/my_table_m7_20260630_185332 --backend genesis --backend usd --backend isaac
rsf qa-sim --run-dir runs/my_table_m7_20260630_185332
rsf composite-viewer --run-dir runs/my_table_m7_20260630_185332 --export-only
```

## Assumptions

- 用户要求完整复现图示主流程；单帧双目只能生成 proxy background，真正 3DGS 需要视频/多帧。
- SAM3D 物体几何服务当前没有项目内确认 endpoint；V1 仍把“对齐修正”作为核心任务，不能把外部服务原始 pose/scale 当 metric truth。
- 所有持久代码都在 `/mnt/workspace/wenqian/real2sim_scene_foundry` 下。
- 现有 dirty state 不回滚；新项目通过服务 API 和渲染脚本复用能力，不直接迁移旧仓库代码。
