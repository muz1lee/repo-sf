# AGENTS.md - real2sim_scene_foundry

本文件是 `real2sim_scene_foundry` 的项目级规则。任何 agent 在本项目继续写代码、跑脚本、同步服务器或生成产物前，必须先读本文件。若用户当前明确指令与本文冲突，以用户当前明确指令为准；若涉及其他仓库，还必须先读对方仓库自己的 `AGENTS.md`。

## 项目定位

- 工作线：**项目落地先行、论文后置**。
- 目标：复现 SimFoundry 图示里的 real-to-sim 主流程：Physical Scene Extraction -> Foreground Removal / Object RGB-D Masks -> Mesh + Property Generation -> Background Reconstruction -> Automatic Alignment -> Sim Scene Generation。
- 当前主线输入优先支持双目图片；完整背景 3DGS 需要 BG-only 视频或多帧输入，单个双目 pair 只能生成 inpainted background + metric background point cloud + proxy scene。
- 不把服务 smoke 当验收点。验收必须看完整产物：object mesh/pose、background artifacts、scene manifest、USD/GLB、Genesis/Isaac 交互入口、QA overlay/report。
- 论文里的 object/scene/task cousins、policy training 和完整 articulated object pipeline 是后续扩展，不是当前图示复现的第一验收线。

## 代码与路径边界

- 本地镜像目录：`/Users/knowin-wenqian/Claude/Projects/07_simulation/real2sim_scene_foundry`
- 服务器权威目录：`/mnt/workspace/wenqian/real2sim_scene_foundry`
- SSH 入口：`ssh wenqian_h200`，对应阿里云 DSW，公网主机 `101.132.143.105`，端口 `1023` 由本机 SSH alias 处理。
- 所有持久代码、测试和项目文档必须放在服务器 `/mnt/workspace/wenqian/real2sim_scene_foundry` 下；不要把持久代码放到 `/tmp`。
- 不修改这些既有仓库，除非用户明确要求：
  - `/mnt/workspace/wenqian/knowin-world`
  - `/mnt/workspace/wenqian/ego_hand_pipeline`
  - `/mnt/workspace/wenqian/ego_eef`
  - `/mnt/workspace/wenqian/scene_edit_v0`
- 可以通过 HTTP API、CLI subprocess 或只读参考复用旧仓库能力，但不要把新项目代码塞进旧仓库。

## 虚拟环境规则

- **任何项目都必须有独立虚拟环境。不要使用系统 Python 安装项目依赖。**
- 本项目默认虚拟环境：`<project>/.venv`
- 推荐初始化命令：

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

- 新增 Python 依赖时，先更新 `pyproject.toml`，再在 `.venv` 中安装；不要临时 `pip install` 后忘记写入依赖声明。
- 渲染阶段可以调用 `knowin-world` 的现有环境：`/mnt/workspace/wenqian/knowin-world/.venv/bin/python`，因为该环境已有 `genesis`、`pxr`、`trimesh`、`cv2` 等渲染依赖。调用它只用于运行渲染脚本，不允许修改 `knowin-world` 代码。

## 服务器与服务注册表

以下状态来自 2026-06-30 的探测，服务状态可能变化；调用失败时先重新 health check，再判断代码问题。

| 能力 | 地址 / 路径 | 用途 | 备注 |
| --- | --- | --- | --- |
| 计算服务器 | `ssh wenqian_h200` | 主要开发与运行环境 | 4 x NVIDIA L20X |
| 持久工作区 | `/mnt/workspace/wenqian/` | 所有本项目持久文件 | `/tmp` 易失 |
| SAM3 segmentation | `http://101.132.143.105:5081/segment` | object mask / text prompt / box prompt | 根路径 404 不代表 `/segment` 不可用 |
| SAM3D object | `http://101.132.143.105:5077/api/process` | 单图物体几何、初始 pose/scale | 已知 pose 投影可能偏，需要本项目做对齐修正 |
| MoGe focal | `http://101.132.143.105:5014/api/focal` | 单目 RGB 的 focal / K 估计 | local health path: `http://localhost:5014/healthz` |
| HaWoR | `http://101.132.143.105:5012/api/recon` | 手部重建参考，不属于本项目 V1 主线 | local health path: `http://localhost:5012/healthz` |
| S2M2 stereo depth | `http://10.10.4.244:5060-5067/api/process` | 双目 metric XYZ map | 只在服务器侧可达；本地 Mac 不通 |
| scene-edit inpaint | `http://101.132.143.105:5091/inpaint` / `:5092/inpaint` | foreground removal / BG-only RGB | 可用时走 HTTP；不可用时本项目用 OpenCV inpaint fallback |
| SAM3D weights | `/mnt/workspace/SAM3D/object/checkpoints` | 服务端模型资产 | 不要移动 |
| knowin-world render env | `/mnt/workspace/wenqian/knowin-world/.venv/bin/python` | Genesis/USD 渲染 | 不作为本项目依赖环境 |
| scene edit smoke data | `/mnt/workspace/wenqian/scene_edit_v0/test_data` | V1 双目 smoke 输入 | 可只读复用 |

## 工程纪律

- 开始修改前先检查目标项目状态：`git status --short`。如果不是本项目目录，先确认是否误进旧仓库。
- 默认最小实现：只写当前 SimFoundry 图示复现需要的代码；背景 3DGS、交互 viewer、物理 settle 都要可插拔、可替换，不提前做 cousins 或 policy。
- 生产代码采用 TDD：先写失败测试，再写最小实现，再跑测试。文档和纯配置变更可不走 TDD，但要做基本校验。
- 生成文件不进 Git：`runs/`、`outputs/`、`cache/`、模型权重、视频、渲染图、`.npy`、`.ply`、`.glb` 大产物默认忽略。
- 不覆盖用户已有改动；不要在旧仓库里 `git reset --hard`、`git checkout --` 或批量删除。
- 服务可能过期或重启，遇到失败先区分：网络/服务不可用、输入格式不匹配、代码 bug。

## 坐标与数据约定

- 相机系：OpenCV convention，`x` 向右，`y` 向下，`z` 向前，单位 meters。
- 仿真世界系：ground-plane aligned，`z` 向上，单位 meters。
- 每个物体必须保存：
  - `object_id`
  - `label`
  - `mesh_path`
  - `mask_path`
  - `crop_path`
  - `T_object_to_camera`
  - `T_object_to_world`
  - `scale_m`
  - `mass_kg`
  - `friction`
  - `source_backend`
  - `confidence`
  - `needs_manual_refine`
- SAM3D 输出不能直接当 metric scene truth；必须经过相机系转换和 metric point cloud / mask 对齐检查。
- QA 必须至少输出 render-vs-input overlay、投影误差或 mask IoU、depth residual、physics settle 状态。
- Extraction 必须输出背景分支 artifacts：`background/foreground_mask.png`、`background/bg_only.png`、`background/bg_only_cloud.ply`、`background/background_manifest.json`。
- 可交互场景必须通过 manifest 加载物体 mesh 和显式物理参数；不要只依赖 USD PhysicsMaterial，因为 Genesis 当前不会自动读取接触摩擦。

## 默认验证命令

在本项目虚拟环境中运行：

```bash
source .venv/bin/activate
python -m pytest tests -q
```

服务器 smoke 测试默认数据：

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate
rsf smoke \
  --left /mnt/workspace/wenqian/scene_edit_v0/test_data/041_L.png \
  --right /mnt/workspace/wenqian/scene_edit_v0/test_data/041_R.png \
  --calib /mnt/workspace/wenqian/scene_edit_v0/test_data/041_calib.json \
  --out runs/smoke_041
```

## 复用旧项目的边界

- `scene_edit_v0`：可复用其 S2M2 测试数据和思路，但新项目要自己实现 client / manifest / QA，不把代码直接写回去。
- `knowin-world`：可参考 manifest、Genesis/USD 加载和物理参数说明；渲染可 subprocess 调用其 `.venv`。不要修改它来适配本项目。
- `ego_hand_pipeline`：只作为 SAM3/MoGe/HaWoR 服务接口参考；它的定位是手部 pseudo-EEF，不承载完整场景重建。
- `ego_eef`：不是本项目依赖；不要引入 K1、IK、HORA replay 或 pseudo-action 逻辑。

## 已知风险

- SAM3D `T_model_to_camera` 对齐已知可能偏移；V1 必须把 pose/scale alignment 当核心问题，而不是只保存服务输出。
- Genesis 不会自动读取 USD PhysicsMaterial 的接触摩擦；物体质量和摩擦必须在 manifest 或加载时显式设置。
- S2M2 只从服务器可达；本地开发时需要 mock 或跳过真实服务 smoke。
- OpenCV H.264 编码在服务器上可能不可用；生成视频时要允许 `mp4v` fallback 或后处理转码。
