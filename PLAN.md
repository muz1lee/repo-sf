# SimFoundry-Style Real-to-Sim Scene Reproduction Plan

## Summary
- 工作线：**项目落地先行、论文后置**。目标不是只做 object mesh，而是复现 SimFoundry 图示主链路：真实双目/视频输入 -> extraction -> foreground removal -> object digital twins -> background reconstruction -> automatic alignment -> 可交互 sim scene。
- 论文依据：SimFoundry 将流程拆成 Extraction、Generation、Augmentation 三段，从单个真实视频重建 sim-ready digital twin，并可扩展 object/scene/task cousins。参考：[arXiv 2606.28276](https://arxiv.org/abs/2606.28276)、[NVIDIA SimFoundry page](https://research.nvidia.com/labs/gear/simfoundry/)。
- 新项目位置：服务器 `wenqian_h200`，代码统一放在 `/mnt/workspace/wenqian/real2sim_scene_foundry`。不改 `ego_hand_pipeline`、`ego_eef`、`knowin-world` 现有仓库。
- 当前成功标准：输入真实双目图片，输出 object mesh、metric pose/scale、物理参数、background artifacts、USD/manifest、Genesis/Isaac 可交互入口，以及 render-vs-input overlay / interaction report。输入视频或多帧时，再启用 BG-only video -> 3DGS 背景训练。

## Key Changes
- 建独立 Python package + CLI：`rsf run --input ... --out ...`，子命令包含 `extract`、`reconstruct`、`align`、`export`、`render`、`smoke`。
- RGB 视频入口：`rsf video-prep --video phone.mp4 --out runs/<id> --frame-stride 10`，先输出 sampled frames、reference frame 和 `video_manifest.json`；相机位姿和 3DGS 训练是后续步骤。
- 服务适配器：
  - SAM3：`http://101.132.143.105:5081/segment`
  - SAM3D：`http://101.132.143.105:5077/api/process`
  - MoGe focal：`http://101.132.143.105:5014/api/focal`
  - S2M2：`http://10.10.4.244:5060-5067/api/process`
  - Gemini：通过环境变量读取 API key；无 key 时允许 YAML 手写 object list。
- 坐标约定固定：
  - 相机系：OpenCV `x right, y down, z forward`，单位米。
  - 仿真世界系：ground-plane aligned `z up`，单位米。
  - 每个 object 存 `T_object_to_camera`、`T_object_to_world`、`scale_m`、`source_backend`、`confidence`。
- 输出结构：
  - `runs/<id>/scene_manifest.json`
  - `runs/<id>/background/{foreground_mask.png,bg_only.png,bg_only_cloud.ply,background_manifest.json}`
  - `runs/<id>/objects/<object_id>/{mask.png,crop.png,mesh.glb,pose.json}`
  - `runs/<id>/exports/scene.usda`
  - `runs/<id>/exports/run_interactive_scene.py`
  - `runs/<id>/qa/{overlay.png,render.png,qa_report.json,interaction_report.json}`

## Implementation Plan
- Phase 1：项目骨架和环境
  - 在 `/mnt/workspace/wenqian/real2sim_scene_foundry` 初始化独立 git repo 和轻量 venv，依赖 `numpy/opencv/Pillow/requests/scipy/trimesh/pyyaml/pytest`。
  - 渲染阶段调用 `/mnt/workspace/wenqian/knowin-world/.venv/bin/python`，因为该 venv 已确认有 `genesis/pxr/trimesh/cv2`；服务 HTTP client 不放进这个 venv。
- Phase 2：Extraction
  - 视频默认取第 0 帧作为 representative frame，和论文一致；也支持用户指定 frame index。
  - 单 RGB 路径用 MoGe 估 K/focal；双目路径用 calibration + S2M2 生成 metric XYZ map。
  - Qwen 先产 bbox/point；SAM3 只吃 bbox/point 产每物体 mask，避免大 bbox 或纯文本分割把相邻物体混进来。
  - union object masks 做 foreground removal，输出 `bg_only.png` 和 `bg_only_cloud.ply`；有 inpaint 服务时走 HTTP，否则 OpenCV fallback。
- Phase 3：Mesh + Pose
  - 每个 object crop/mask 调 SAM3D 生成 mesh 和初始 pose。
  - 用 object mask 内的 metric point cloud 做 similarity alignment，修正 SAM3D 的尺度和相机系姿态。
  - 对齐质量输出：mask projection IoU、2D center error、depth residual；低质量 object 写入 `needs_manual_refine=true`。
  - V1 不做 GUI，提供 `pose_overrides.yaml` 手动调 `tx/ty/tz/rx/ry/rz/scale` 后重渲 overlay。
- Phase 4：Physics + Export
  - 生成 USD/GLB 资产和 scene manifest；桌面/地面来自 ground plane 或默认 plane。
  - 质量、摩擦由 Gemini 类别建议 + conservative defaults 写入 manifest；不要依赖 USD PhysicsMaterial，因为 Genesis 当前不会自动读取接触摩擦。
  - 动态物体先跑短 physics settle，检查穿透、飞出、NaN。
- Phase 5：Render + QA
  - 默认用 Genesis 渲染 smoke preview；USD/manifest 保持 Isaac-compatible，后续接 IsaacLab loader。
  - 生成 overlay：原图、SAM3 mask、投影 mesh 轮廓、渲染图并排。
  - `qa_report.json` 作为 go/no-go：服务版本、输入 hash、object 数、alignment metrics、physics settle 状态。
- Phase 6：Background 3DGS
  - 单个双目 pair 不足以训练论文里的背景 3DGS；当前输出 proxy background，并在 manifest 中标记 `requires_video_or_multiview`。
  - 当输入是视频/多帧目录时，使用 foreground masks 生成 BG-only video，再接 3DGS runner，输出 splat 路径和 camera trajectory。
  - 手机 RGB 视频路径先用 `video-prep` 抽帧；下一步需要 COLMAP/SLAM 估每帧相机位姿，再对每帧做 Qwen/SAM foreground removal。
- Phase 7：Interactive Sim Scene
  - 生成 `exports/run_interactive_scene.py`，用 `/mnt/workspace/wenqian/knowin-world/.venv/bin/python` 加载 manifest。
  - Genesis loader 必须按 manifest 显式应用 mass/friction，并运行 settle steps；本项目 venv 不直接安装 Genesis。

## Current Gap to Full Stereo-to-Interactive Scene

当前项目已经有独立 package、Qwen-before-SAM extraction、双目 XYZ、点云落盘、SAM3D object asset、metric alignment、manifest、USD stub、背景 inpaint artifacts 和 Genesis interactive launcher。它仍不是论文最终完整版；距离“图示完整复现 + 可交互场景”还缺真正视频 3DGS 背景训练、真实 Genesis/Isaac settle 运行结果、以及更稳的 support plane / collision proxy。

1. **Extraction artifacts**
   - 保存 representative RGB、calibration、S2M2 metric `XYZ`、整场景点云 `scene_cloud.ply`、每个物体 mask/crop/object cloud。
   - 支持 YAML/Gemini object list，允许多物体而不是单 label。
   - 对 mask 噪声做最小清理，并在 QA 中记录 mask 面积、valid depth ratio、object point count。

2. **Object asset generation**
   - 主流程从 metric bbox mesh 升级到 SAM3D mesh，并保留 raw SAM3D 输出。
   - 每个物体生成 visual mesh、collision proxy、texture/vertex color fallback。
   - SAM3D pose/scale 只作为初始化，不能直接当 metric truth。

3. **Metric pose alignment**
   - 用 mask 内 object point cloud 对 SAM3D mesh 做 similarity alignment。
   - 输出 `T_object_to_camera`、`T_object_to_world`、`scale_m`、depth residual、2D center error、projection IoU。
   - 低质量对象标记 `needs_manual_refine=true`，并支持 `pose_overrides.yaml` 二次渲染。

4. **Scene composition**
   - 从场景点云估计 table/ground/support plane，建立 z-up world frame。
   - 多物体按 world pose 放入同一个 scene graph，生成 Isaac-compatible USD 和 manifest。
   - 单双目 pair 使用 inpainted RGB + background point cloud proxy；视频/多帧输入使用 BG-only 3DGS。

5. **Physics and interaction**
   - 为每个物体写 mass/friction/restitution 和 collision mesh。
   - Genesis/Isaac loader 显式应用物理参数，运行短 physics settle 检查 NaN、飞出、穿桌。
   - 提供可交互预览：`rsf interactive --run-dir ...` 生成 launcher；真实 viewer 用 knowin-world venv 运行。

6. **QA and acceptance**
   - MeshLab 可检查：`scene_cloud.ply` 看全局深度/桌面，`objects/<id>/object_cloud.ply` 看分割物体点云。
   - `qa_report.json` 给 go/no-go：服务版本、输入 hash、object 数、alignment 指标、physics settle 状态。
   - overlay 必须同时显示输入图、mask、投影轮廓、渲染结果。

## Immediate Next Milestones

1. **M1 点云落盘**：在现有双目流程中输出 `scene_cloud.ply` 和 `objects/<id>/object_cloud.ply`，先让 MeshLab 能直接检查 metric depth 和 mask 质量。
2. **M2 Extraction 命令成形**：把 `rsf extract` 做成稳定 artifact producer，输出 RGB、calib、XYZ summary、object list、mask/crop/cloud，不要求 mesh。
3. **M3 SAM3D asset path**：接入 SAM3D 主流程，保存 raw mesh、normalized mesh、初始 pose，并保留 metric bbox fallback。
4. **M4 Alignment**：实现 SAM3D mesh 到 object cloud 的尺度和位姿修正，生成 projection/depth QA。已完成 Qwen/SAM3/SAM3D 主线。
5. **M5 Background branch**：输出 foreground mask、BG-only RGB、BG-only cloud、background manifest。已完成单帧 proxy 分支。
6. **M6 Interactive preview**：用 `knowin-world` venv 启 Genesis/Isaac loader，加载 manifest/USD，运行 settle 并打开 viewer 或保存交互脚本入口。已完成 launcher、proxy report 和 Genesis no-viewer settle smoke；viewer 窗口模式留给人工打开。
7. **M7 Video 3DGS background**：支持视频/多帧输入，生成 BG-only video，接 3DGS runner，输出 splat 资产和 camera trajectory。
8. **M8 Support plane + physics QA**：从背景点云估计支撑平面，修正 z-up world frame，真实 Genesis/Isaac settle 100 steps 并写回 QA。

## Test Plan
- Unit tests：
  - camera project/backproject round-trip。
  - S2M2/SAM3/MoGe/SAM3D client 使用 mock response。
  - SAM3D 坐标转换和 similarity scale synthetic case。
  - scene manifest validation：缺 mesh、非法 pose、负质量、未知坐标系均报错。
- Server smoke：
  - 用 `/mnt/workspace/wenqian/scene_edit_v0/test_data/041_L.png`、`041_R.png`、`041_calib.json` 跑双目路径。
  - 验证 S2M2 返回 `HxWx3` XYZ、SAM3 mask 非空、至少一个 object 完成 mesh+pose+render。
- Acceptance：
  - `scene.usda` 可加载渲染，`render.png` 非空。
  - object projection 与 mask 大体重合：center error < 30 px 或 IoU > 0.4。
  - physics settle 100 steps 后无 NaN、无明显穿桌、无爆炸位移。
  - 全流程产物保存在 `/mnt/workspace/wenqian/real2sim_scene_foundry/runs/<id>`。

## Assumptions
- 用户已明确要求完整复现图示主流程；当前不再把 automatic background 分支排除在外，但单帧双目输入只能产 proxy background，真正 3DGS 需要视频/多帧。
- SAM3D `:5077` 已知几何可用但 pose/scale 投影有偏差，V1 把“对齐修正”作为核心任务。
- 服务器 `1023` 理解为 SSH 入口 `wenqian_h200`；所有持久代码都在 `/mnt/workspace/wenqian` 下。
- 现有仓库 dirty state 不碰；新项目通过服务 API 和渲染脚本复用能力，不直接迁移旧仓库代码。
