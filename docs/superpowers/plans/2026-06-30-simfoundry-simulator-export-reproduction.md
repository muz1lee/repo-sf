# SimFoundry Simulator Export 复现计划

> **给 agentic worker：**实现本计划时必须逐项执行 checklist。推荐使用 subagent-driven-development 或 executing-plans。每项完成后写清文件、artifact、验证命令、状态和剩余 gap。

**目标：**把 `real2sim_scene_foundry` 从可运行原型推进到 SimFoundry Figure-2-style pipeline 的 simulator export 阶段：真实采集 -> 前景/背景提取 -> metric 6D object pose -> 注册背景 -> 稳定物理资产 -> Genesis/USD/Isaac 兼容导出和 QA 证据。

**架构：**保留 artifact-first pipeline，但把 fallback-oriented artifacts 升级成严格 export contract。pose、background registration、collision、physics、simulator export 都必须有 explicit manifest 和 acceptance report。单元测试只覆盖纯逻辑；最终成功由 end-to-end exported scene、视觉 QA 和物理 QA 判定。

**技术栈：**Python package `real2sim_scene_foundry`；SAM3、SAM3D、Qwen/Gemini-compatible VLM、S2M2、MoGe 等外部服务；Nerfstudio/splatfacto 训练 3DGS；Genesis 和 USD/OpenUSD 做 simulator export；IsaacLab/Isaac loader 在外部 runtime 可用时验证。

## 全局约束

- 工作线：**项目落地先行、论文后置**。
- 权威仓库：`/mnt/workspace/wenqian/real2sim_scene_foundry`。
- 不要修改 `knowin-world`、`ego_hand_pipeline`、`ego_eef`、`scene_edit_v0`，除非用户明确要求。
- 当前目标不是 fallback、service smoke 或最小单测，而是 artifact-level simulator export。
- 本计划覆盖 SimFoundry Figure 2 上半部分到 **Simulator export**；object/scene/task cousins、policy training 和 policy evaluation 不在当前范围。
- run 只有产出可检查 artifacts 才可接受：object visual/collision assets、6D pose、background assets、scene manifest、USD/GLB export、Genesis/Isaac loading entrypoint、render-vs-input QA、physics stability report。
- fallback 只允许做诊断；fallback-backed output 必须标成 `diagnostic_only`，不能满足 final acceptance gate。
- 生产依赖必须写进 `pyproject.toml`，或隔离到 `.venv_3dgs` 这类 runtime 环境；不要装进 system Python。

## 强制执行队列

1. **分类当前资产并阻止 proxy-as-visual success。** 8-vertex box GLB 只能是 debug/collision proxy，不能当 final visual。
2. **先修 visual object assets，再修 viewer 外观。** 每个 object 必须有 `visual.glb` 或明确标注的 visual candidate。
3. **分离 visual geometry 和 physical geometry。** `collision.glb` 和 `physics.json` 必须独立于 visual asset。
4. **停止把 full-scene point cloud 当 background visual。** `scene_cloud.ply` 只能 debug；`bg_only_cloud.ply` 也只能作 proxy/diagnostic。
5. **注册背景到 simulator world。** 必须写出 `background/registration.json` 和真实 `T_3dgs_world_to_sim_world`，否则 blocked。
6. **从 center/scale 升级到 6D pose。** manual refinement 必须记录为 `manual_refined_from_auto`。
7. **构建 simulator export bundle。** USD/Genesis/Isaac 都要写出 visual、collision、physics、support surface 和 reference camera provenance。
8. **跑 export-level QA，再判断是否 complete。** `qa/sim_export_report.json` 是最终 gate。

## 为什么有这个计划

项目已经能产出很多 artifact，但这不等于完整 SimFoundry simulator-export reproduction。下一步目标是让输出能作为 coherent simulator scene 被加载和检查：geometry、background、object transform、collision、physical properties 必须在同一个 contract 下自洽。

canonical evidence run：

```text
/mnt/workspace/wenqian/real2sim_scene_foundry/runs/my_table_m7_20260630_185332
```

关键事实：

- objects：`bottle`、`cup`、`tissue_pack`。
- 当前 visual path 使用 `objects/*/visual.glb`；`mesh_aligned.glb` 是 debug/proxy。
- 背景 3DGS training 已完成，但 native runtime 未打通。
- `qa/background_3dgs_render.png` 是 external 3DGS renderer sidecar，`simulator_native=false`。
- table collision 已升级为 tabletop mask polygon slab。
- Genesis no-viewer settle 可通过，但 settled pose 未全部写回。
- Isaac validation 依赖 preserved worker report，repeatability 仍是 caveat。

## 论文上下文

SimFoundry reconstruction path：

1. **Extraction**：RGB video -> representative frame -> depth/intrinsics -> RGB-D point cloud -> scene VLM + SAM3 foreground segmentation -> RGB-D inpainting -> object crops/masks + background assets。
2. **Generation**：upsample object images -> 2D-to-3D mesh -> RGB-D/mask/scene geometry pose refinement -> collision mesh -> physical properties -> physics depenetration -> simulator export。
3. **Background Reconstruction**：BG-only video -> depth/camera poses -> depth-supervised 3DGS -> simulator-world rigid bridge；或 second foreground-free video -> manual SE(3)+scale alignment。
4. **Augmentation**：object/scene/task cousins；当前计划不做。

工程替代可以存在，例如 Qwen 替代 Gemini、SAM3D 替代 Hunyuan/TRELLIS、MoGe+COLMAP 替代 DepthAnything3、Genesis 先于 IsaacLab。但替代必须写 provenance，不能冒充 paper-equivalent。

## 当前主要 gap

1. **Foreground video segmentation 不等价论文。** 当前有 per-frame Qwen/SAM 和 previous-mask reuse；需要 temporally grounded mask track。
2. **Metric camera/background bridge 不完整。** `T_3dgs_world_to_sim_world` 仍缺真实推导或手动 editor 输出。
3. **3DGS native runtime 未集成。** Genesis/Isaac/browser 当前不原生加载 splat；external PNG sidecar 只能 partial。
4. **6D pose refinement 不等价 FoundationPose。** 当前 explicit manual refinement 可用于工程，但 paper-level automatic pose refinement 仍 partial。
5. **Collision decomposition 不等价 CoACD。** 当前主要是 convex hull。
6. **Physics annotation 不等价 VLM inference。** mass/friction 来源必须明确，heuristic 不能叫 inferred。
7. **Settled pose contract 未完成。** moved pose 要么写回，要么 viewer/export 明确显示 initial/settled 差异。
8. **Isaac repeatability 有 caveat。** preserved report 不能冒充 fresh direct validation。

## 最终判断规则

- 背景仍是 external PNG sidecar：整体不能叫 full SimFoundry visual reconstruction reproduced。
- `T_3dgs_world_to_sim_world` 缺失或是 placeholder：background registration blocked。
- physics property 仍是 heuristic/scene_manifest：physics annotation partial。
- table projection IoU 低但阈值放水：support surface blocked。
- settled pose 未回写且 viewer 不展示差异：interactive physics scene partial。
- 全量测试不绿：整体不能 complete。

真正交付的是 gap ledger、分模块复现状态、fail-closed QA 和 canonical artifacts，不是单句 “passed”。
