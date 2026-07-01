# SimFoundry Gap Ledger

审计日期：2026-07-01
审计对象：服务器权威仓库 `/mnt/workspace/wenqian/real2sim_scene_foundry`
canonical evidence run：`runs/my_table_m7_20260630_185332`

## 状态定义

实现状态：

- `implemented`：代码和 canonical artifacts 已存在，满足项目工程 contract。
- `partial`：有可用实现，但弱于论文目标，或依赖明确记录的替代方案。
- `missing`：没有有意义的实现或 artifact。
- `blocked`：工程路径存在，但缺外部服务、runtime 或必要输入，无法完成。

复现状态更严格：

- `reproduced`：canonical evidence 证明该项达到 paper-level effect。
- `partial`：有实用工程替代，但未达到完整论文效果。
- `blocked`：paper-level effect 被外部 runtime/service/data 缺口阻塞。

`pytest passed`、`sim_export_manifest.json passed` 或 `qa passed` 都不能单独证明 SimFoundry paper effect reproduced。

## 快照摘要

- canonical objects：`bottle`、`cup`、`tissue_pack`。
- BG-only video：`video/bg_only_status.json` 为 `completed`，131 frames，其中 36 frames 在 proposal failure 后复用了上一帧 mask。
- 3DGS training：`video/3dgs_status.json` 为 `completed`，方法是 Nerfstudio `splatfacto`。
- 3DGS native candidate：`background/3dgs_native/splat_rgb.ply` 存在且 header 字段完整，`gaussian_count=179536`。
- 3DGS registration：`background/registration.json` 为 `partial_external_render_only`；`T_3dgs_world_to_sim_world=null`；`registrations.3dgs.status=blocked_missing_camera_pose_scale_evidence`。
- 背景渲染：`qa/background_3dgs_render_report.json` 为 `rendered`，backend 是 `external_3dgs_renderer`，`simulator_native=false`。
- simulator export：最终 `sim_export_manifest.json` 和 `qa/sim_export_report.json` 应在 `background_unregistered` 上 fail closed。
- Genesis settle：fresh CPU settle 可通过，无 fall-out / NaN；但 `qa/settled_pose_delta_report.json` 仍为 `partial`，因为 moved pose 未写回。
- Isaac：`qa/isaac_load_report.json` 的 loaded 证据来自 `preserved_existing_worker_report`，`validation_reused=true`，不是 fresh direct validation。
- object visual：`objects/*/visual.glb` 是高拓扑 visual mesh；legacy `mesh_aligned.glb` 仍是 8-vertex / 12-face debug-era box。
- collision：object collision assets 是 convex hull，不是 CoACD。
- table collision：`qa/table_collision_report.json` 为 `passed`，`source_backend=tabletop_mask_polygon_slab`，`projection_iou=0.531253837651971`，threshold `0.5`，真实 `table_collision.glb` 是 190 vertices / 348 faces，不是 bbox proxy。

## 对照表

| 项目 | 实现状态 | 复现状态 | 当前证据 | 剩余差距 |
| --- | --- | --- | --- | --- |
| extraction | partial | partial | 有 video manifest、MoGe/COLMAP、Qwen/SAM proposals、masks/crops/object clouds、scene manifest | 不是完整论文式 iterative RGB-D inpainting foreground decomposition |
| foreground removal | partial | partial | 有 `background/foreground_mask.png`、`bg_only.png`、`bg_only_cloud.ply` 和 video BG-only frames | 主要是 RGB inpaint + metric cloud masking；depth inpaint/iteration provenance 不完整 |
| BG-only video | partial | partial | 131 frames 完成，36 frames 复用 previous mask | 不是 SAM2-style propagation，stale mask 不能 silent pass |
| 3DGS training | implemented | partial | `splatfacto` 训练完成，有 config/checkpoint 和 PLY candidate | 不是 Appendix E.5 的完整 depth-supervised + pose optimization 路径 |
| 3DGS registration | partial | blocked | `T_3dgs_world_to_sim_world=null`，3DGS registration blocked | 缺 shared camera pose / metric scale evidence 或手动 SE(3)+scale editor 输出 |
| object mesh | implemented | partial | `objects/*/visual.glb` 已作为 final visual path，bbox 只作 debug/proxy | backend 和质量评估不等同 Hunyuan/TRELLIS 论文路径 |
| pose alignment | partial | partial | 有 manual/refined pose report、support alignment、projection overlay | 不是 service-backed FoundationPose-equivalent refinement |
| collision geometry | partial | partial | 有独立 `objects/*/collision.glb` | 当前是 convex hull，不是 CoACD/VHACD |
| physics annotation | implemented | partial | 有 `objects/*/physics.json`，mass/friction 被 export 使用 | 来源是 scene manifest/heuristic，不是完整 VLM inference |
| physics stability | implemented | partial | Genesis settle 通过，无 fall-out/NaN | moved pose 未回写；settle runtime 主要用 collision assets |
| simulator export | partial | blocked | `scene.usda`、`genesis_scene.py`、`isaac_scene.py` 存在 | full export blocked by `background_unregistered` 和 Isaac preserved-report caveat |
| interactive viewer | partial | blocked | viewer 显示 object visual、collision/table toggles、external 3DGS sidecar provenance | 不是 live/native 3DGS；orbit 时必须承认 reference-view sidecar |

## 仍需解决的核心 gap

1. 用 temporally grounded mask track 替换 per-frame BG-only mask reuse。
2. 如果要 paper-level automatic background reproduction，需要补 depth-supervised 3DGS 或明确记录替代路径。
3. 用 derived bridge 或 manual editor 输出替换 `T_3dgs_world_to_sim_world=null`。
4. 接入 browser/simulator native 3DGS runtime，或继续把 sidecar 标成 `simulator_native=false`。
5. 增加 service-backed 6D pose refinement，或继续把 manual refinement 标成 partial。
6. 用 CoACD/VHACD 替换 convex hull collision。
7. 让 Isaac validation 能从服务器 workflow 可重复执行，或继续保留 `preserved_existing_worker_report` caveat。
8. 如果 physics stability 要达到 Appendix E.4，必须缓存 settled pose 或在 viewer/export 中明确展示 initial/settled 差异。
