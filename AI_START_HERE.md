# AI_START_HERE - real2sim_scene_foundry

本文件是新 AI/Codex 会话接手本项目时必须先读的启动卡。新会话不能依赖旧聊天记忆；先读本文件，再读 `AGENTS.md`，然后读当前执行计划。

## 必须先做的启动步骤

先在服务器上运行：

```bash
ssh wenqian_h200
cd /mnt/workspace/wenqian/real2sim_scene_foundry
git status --short
sed -n '1,260p' AI_START_HERE.md
sed -n '1,260p' AGENTS.md
sed -n '1,220p' docs/superpowers/plans/2026-06-30-simfoundry-simulator-export-reproduction.md
```

Python 工作前激活项目环境：

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate
```

声称完成任何任务前，至少运行：

```bash
python -m pytest tests -q
```

仿真/export 工作不能只看测试结果，还必须产出并检查 `runs/<run_id>/` 下的 artifacts。

## 项目身份

- 工作线：**项目落地先行、论文后置**。
- 目标：复现 SimFoundry Figure 2 上半部分，直到 simulator export。
- 权威服务器：`ssh wenqian_h200`。
- 权威仓库：`/mnt/workspace/wenqian/real2sim_scene_foundry`。
- 本地镜像：`/Users/knowin-wenqian/Claude/Projects/07_simulation/real2sim_scene_foundry`，除非用户明确要求，否则只作参考。
- 不要把持久项目代码放在 `/tmp`。
- 不要修改这些仓库，除非用户明确要求：
  - `/mnt/workspace/wenqian/knowin-world`
  - `/mnt/workspace/wenqian/ego_hand_pipeline`
  - `/mnt/workspace/wenqian/ego_eef`
  - `/mnt/workspace/wenqian/scene_edit_v0`

## 当前目标

完成 SimFoundry 风格重建到 **Simulator export**：

```text
real capture
-> foreground/background extraction
-> real visual object assets
-> 6D object poses
-> registered BG-only / 3DGS background
-> separated visual/collision/physics assets
-> Genesis/USD/Isaac export bundle
-> export-level QA
```

当前不做：object cousins、scene cousins、task cousins、policy training、policy evaluation、完整 articulated object pipeline，除非用户重新打开这些范围。

## 不要漂移

以下都不算完成：

- 用 8-vertex / 12-face bbox mesh 冒充 visual asset。
- 用 `scene_cloud.ply` 冒充 final background visual。
- 用 `table_collision.glb` 冒充 visual tabletop。
- 用 box geometry 掩盖 identity rotation，声称完成 6D pose。
- 没有 provenance 的 `Genesis convexify=True` 冒充 final collision decomposition。
- browser viewer 成功加载 proxy assets。
- 只有 service smoke，没有 exported scene artifacts。
- unit tests passed，但 export artifacts 缺失。

fallback 只能用于诊断。任何 fallback-backed artifact 必须标成 `diagnostic_only`、`debug_proxy`、`collision_proxy`、`runtime_unavailable` 或其他明确非 final 状态。如果真实实现被外部 runtime/service 阻塞，要写 blocked reason 和证据，不能静默简化。

## 当前必读计划

```text
docs/superpowers/plans/2026-06-30-simfoundry-simulator-export-reproduction.md
```

该计划约束：已知 gap、执行队列、asset provenance、simulator export contract、blocked status 和最终验收定义。后续 agent 必须按该计划建立 checklist，完成一项更新一项，并报告 exact artifacts 和 verification commands。

## Canonical Evidence Run

当前 reference run：

```text
runs/my_table_m7_20260630_185332
```

已知事实：

- Objects：`bottle`、`cup`、`tissue_pack`。
- `objects/*/mesh_aligned.glb` 是 box proxy，每个 8 vertices / 12 faces。
- 当前 canonical export 使用 `objects/*/visual.glb` 作为 visual path；bbox mesh 只能作为 debug/collision/proxy。
- `background/3dgs_native/splat_rgb.ply` 是候选 native-side 3DGS 背景资产，不证明 Genesis/USD/Isaac 已经 native render 3DGS。
- `qa/background_3dgs_render.png` 是当前 B 路线的 final visual background；报告写明 `backend=external_3dgs_renderer`、`simulator_native=false`。
- Isaac validation 当前依赖 preserved loaded worker report；`wenqian_h200 -> worker` 直接执行仍不可靠。
- Genesis no-viewer settle 已通过 foreground/collision physics；Genesis 当前加载 foreground objects 和 support plane，不原生渲染 3DGS。
- 最终 full SimFoundry visual reconstruction 仍 blocked 在 `background_unregistered` / native 3DGS runtime。

## 服务和 Runtime 注册表

服务状态可能变化。服务失败时先 health check，再判断代码问题。

| 能力 | 地址 / 路径 | 用途 | 备注 |
| --- | --- | --- | --- |
| 主服务器 | `ssh wenqian_h200` | 开发和运行 | 权威工作环境 |
| 项目仓库 | `/mnt/workspace/wenqian/real2sim_scene_foundry` | 当前项目 | 持久代码和文档放这里 |
| SAM3 segmentation | `http://101.132.143.105:5081/segment` | 文本/框提示生成 object mask | 根路径 404 不代表 `/segment` 不可用 |
| SAM3D object | 未登记 | 物体几何 / 初始 pose scale | 本项目没有确认的 SAM3D HTTP endpoint；不要假设 endpoint |
| MoGe focal | `http://101.132.143.105:5014/api/focal` | 单图 focal / intrinsics | local health 可用 `http://localhost:5014/healthz` |
| HaWoR | `http://101.132.143.105:5012/api/recon` | 手部重建参考 | 不属于 V1 主线 |
| S2M2 stereo depth | `http://10.10.4.244:5060-5067/api/process` | 双目 metric XYZ map | 服务器侧可达，本地 Mac 不通 |
| scene-edit inpaint | `http://101.132.143.105:5091/inpaint` / `:5092/inpaint` | foreground removal / BG-only RGB | 优先 HTTP；fallback 不能隐藏 |
| Genesis/USD runtime | `/mnt/workspace/wenqian/knowin-world/.venv/bin/python` | render/export/settle runtime | subprocess 调用，不要改 `knowin-world` |
| Isaac worker | `ssh -p 1024 root@101.132.143.105` + `/isaac-sim/python.sh` | Isaac/Omniverse USD load validation | run bundle 需传到 worker；权威代码仍在本项目 |
| 3DGS runtime | 项目下 `.venv_3dgs` | Nerfstudio/splatfacto | 与 `.venv` 分离 |
| Smoke data | `/mnt/workspace/wenqian/scene_edit_v0/test_data` | 只读双目 smoke 输入 | 不要往旧仓库写新代码 |

2026-07-01 备注：`wenqian_h200 -> Isaac worker` 直接 SSH 目前会失败。canonical Isaac validation 是通过本地 operator bridge 传输最小 run bundle、在 worker 执行 `/isaac-sim/python.sh exports/isaac_scene.py --run-dir <worker_run_dir>` 后，把 `qa/isaac_load_report.json` 拷回权威 run。

## 主命令

测试：

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate
python -m pytest tests -q
```

export / QA：

```bash
rsf support-plane --run-dir runs/my_table_m7_20260630_185332 --force
rsf refine-pose --run-dir runs/my_table_m7_20260630_185332
rsf render-3dgs-background --run-dir runs/my_table_m7_20260630_185332
rsf export-manifest --run-dir runs/my_table_m7_20260630_185332
rsf export-sim --run-dir runs/my_table_m7_20260630_185332 --backend genesis --backend usd --backend isaac
rsf qa-sim --run-dir runs/my_table_m7_20260630_185332
rsf composite-viewer --run-dir runs/my_table_m7_20260630_185332 --export-only
```

## 当前真实状态

- `pytest` 通过不等于 SimFoundry effect reproduced。
- manifest/QA 应 fail closed：背景 3DGS 未注册或 native runtime 缺失时，full reconstruction 不能写 passed。
- 如果背景仍是 external PNG sidecar，整体只能叫 `partial: external_3dgs_render_sidecar`。
- 如果 `T_3dgs_world_to_sim_world` 缺失或是 placeholder，background registration 必须 blocked。
- 如果 physics property 仍来自 heuristic/scene_manifest，只能 partial。
- 如果 settled pose 没回写或没在 viewer 显示差异，interactive physics scene 只能 partial。
