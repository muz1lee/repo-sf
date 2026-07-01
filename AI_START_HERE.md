# AI_START_HERE - real2sim_scene_foundry

This file is the mandatory startup card for any new AI/Codex conversation on this project. New chats do not retain prior context. Read this file first, then read `AGENTS.md`, then read the current execution plan.

## Mandatory Startup Sequence

Run these commands first:

```bash
ssh wenqian_h200
cd /mnt/workspace/wenqian/real2sim_scene_foundry
git status --short
sed -n '1,260p' AI_START_HERE.md
sed -n '1,260p' AGENTS.md
sed -n '1,220p' docs/superpowers/plans/2026-06-30-simfoundry-simulator-export-reproduction.md
```

Then activate the project environment for Python work:

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate
```

Before claiming anything is complete, run:

```bash
python -m pytest tests -q
```

For simulator/export work, tests are not enough. You must also produce and inspect run artifacts under `runs/<run_id>/`.

## Absolute Project Identity

- Work line: **项目落地先行、论文后置**.
- Project goal: reproduce the upper half of SimFoundry Figure 2 through simulator export.
- Authoritative server: `ssh wenqian_h200`.
- Authoritative repository: `/mnt/workspace/wenqian/real2sim_scene_foundry`.
- Local mirror: `/Users/knowin-wenqian/Claude/Projects/07_simulation/real2sim_scene_foundry`, reference only unless explicitly told otherwise.
- Do not put persistent project code in `/tmp`.
- Do not modify these repositories unless the user explicitly asks:
  - `/mnt/workspace/wenqian/knowin-world`
  - `/mnt/workspace/wenqian/ego_hand_pipeline`
  - `/mnt/workspace/wenqian/ego_eef`
  - `/mnt/workspace/wenqian/scene_edit_v0`

## Current Target

Complete SimFoundry-style reconstruction to **Simulator export**:

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

Out of scope for the current execution line:

- object cousins
- scene cousins
- task cousins
- policy training
- policy evaluation
- full articulated object pipeline, unless the user explicitly reopens that scope

## Do Not Drift

These do **not** count as completion:

- 8-vertex / 12-face bbox mesh as visual asset
- `scene_cloud.ply` as final background visual
- `table_collision.glb` as visual tabletop
- identity rotation hidden by box geometry as full 6D pose
- `Genesis convexify=True` as final collision decomposition without provenance
- browser viewer successfully loading proxy assets
- service smoke test with no exported scene artifacts
- unit tests passing while export artifacts are missing

Fallbacks are allowed only for diagnostics. Any fallback-backed artifact must be labeled as `diagnostic_only`, `debug_proxy`, `collision_proxy`, `runtime_unavailable`, or another explicit non-final status. If a real implementation is blocked, write the blocked reason and evidence instead of silently simplifying.

## Current Required Plan

Read and follow this file:

```text
docs/superpowers/plans/2026-06-30-simfoundry-simulator-export-reproduction.md
```

That plan is the current authority for:

- known reproduction gaps
- mandatory execution queue
- asset provenance rules
- simulator export target contract
- blocked statuses
- final acceptance definition

Future agents must build a checklist from that plan, update the checklist after every completed item, and report exact artifacts and verification commands. Do not ask the user for obvious route choices; follow the plan and continue until blocked by external runtime/service limits.

## Canonical Evidence Run

Current reference run:

```text
runs/my_table_m7_20260630_185332
```

Known facts from that run:

- Objects: `bottle`, `cup`, `tissue_pack`.
- `objects/*/mesh_aligned.glb` are box proxies, each with 8 vertices and 12 faces.
- SAM3D returned `ply_camera_base64`, not `mesh_glb_base64` / `glb_base64` / `mesh`.
- `objects/*/sam3d/point_cloud.ply` contains the raw SAM3D point cloud evidence.
- `scene_cloud.ply` is a full-scene cloud and contains foreground tabletop objects.
- `background/bg_only_cloud.ply` exists and is the minimum background proxy.
- `video/3dgs_status.json` reports completed 3DGS training.
- Latest 3DGS config: `video/3dgs/unnamed/splatfacto/2026-06-30_205848/config.yml`.
- Current object visual path is no longer the active blocker: canonical export paths use `objects/*/visual.glb` with `objects/*/visual_asset_report.json`; bbox meshes are debug/collision/proxy artifacts only.
- `background/3dgs_native/splat_rgb.ply` is a candidate/native-side background asset, not proof that Genesis/USD/Isaac rendered native 3DGS.
- `qa/background_3dgs_render.png` is the current final visual background for the accepted B route; `qa/background_3dgs_render_report.json` records `status=rendered`, `backend=external_3dgs_renderer`, `simulator_native=false`.
- Isaac worker verification currently uses a preserved loaded worker report from the local operator bridge; direct `wenqian_h200 -> worker` execution is still not reliable.
- Genesis no-viewer settle passed for foreground/collision physics. Genesis currently loads foreground objects and support plane; it does not natively render 3DGS.

Use this diagnostic script when checking the current run:

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import json
import trimesh

run = Path("runs/my_table_m7_20260630_185332")
for p in sorted(run.glob("objects/*/mesh_aligned.glb")):
    m = trimesh.load(p, force="mesh")
    print(p, "vertices", len(m.vertices), "faces", len(m.faces), "bytes", p.stat().st_size)

for p in sorted(run.glob("objects/*/sam3d/sam3d_metadata.json")):
    d = json.load(open(p))
    keys = [k for k in ["mesh_glb_base64", "glb_base64", "mesh", "ply_camera_base64"] if d.get(k)]
    print(p, keys)

print("scene_cloud", (run / "scene_cloud.ply").exists())
print("bg_only_cloud", (run / "background/bg_only_cloud.ply").exists())
print("3dgs_status", json.load(open(run / "video/3dgs_status.json")).get("status"))
print("3dgs_config", json.load(open(run / "video/3dgs_status.json"))["outputs"].get("latest_config"))
PY
```

## Service And Runtime Registry

Service status can change. If a service fails, health check it before calling the code wrong.

| Capability | Address / Path | Purpose | Notes |
| --- | --- | --- | --- |
| Main server | `ssh wenqian_h200` | Development and execution | Use this server for authoritative work |
| Project repo | `/mnt/workspace/wenqian/real2sim_scene_foundry` | Current project | Persistent code and docs go here |
| SAM3 segmentation | `http://101.132.143.105:5081/segment` | Object masks from text/box prompts | Root 404 does not imply `/segment` is down |
| SAM3D object | not registered | Object geometry / initial pose/scale | No confirmed SAM3D HTTP endpoint is registered for this project; do not assume one without current registry evidence. |
| MoGe focal | `http://101.132.143.105:5014/api/focal` | Single-image focal / intrinsics | Local health path can be `http://localhost:5014/healthz` |
| HaWoR | `http://101.132.143.105:5012/api/recon` | Hand reconstruction reference | Not V1 mainline |
| S2M2 stereo depth | `http://10.10.4.244:5060-5067/api/process` | Stereo metric XYZ map | Server-side reachable; local Mac cannot reach it |
| scene-edit inpaint | `http://101.132.143.105:5091/inpaint` / `:5092/inpaint` | Foreground removal / BG-only RGB | Prefer available HTTP service, but final export cannot hide fallback use |
| SAM3D weights | `/mnt/workspace/SAM3D/object/checkpoints` | Model assets | Do not move |
| Genesis/USD runtime | `/mnt/workspace/wenqian/knowin-world/.venv/bin/python` | Render/export/settle runtime | Use as subprocess; do not modify `knowin-world` |
| Isaac worker | `ssh -p 1024 root@101.132.143.105` + `/isaac-sim/python.sh` | Isaac/Omniverse USD load verification | Project run bundles must be transferred to the worker; authoritative code remains on `wenqian_h200` |
| 3DGS runtime | `.venv_3dgs` under this project | Nerfstudio/splatfacto | Keep separate from `.venv` |
| Smoke data | `/mnt/workspace/wenqian/scene_edit_v0/test_data` | Read-only stereo smoke inputs | Do not write new project code there |

2026-07-01 note: direct `wenqian_h200 -> Isaac worker` SSH currently fails with `Permission denied (publickey,password)` even after non-interactive host-key setup. The passing canonical Isaac validation was run through the local operator bridge: stream the minimal run bundle from `wenqian_h200` to the worker, execute `/isaac-sim/python.sh exports/isaac_scene.py --run-dir <worker_run_dir>`, then copy `qa/isaac_load_report.json` back to the authoritative run.

## Main Commands

Baseline tests:

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate
python -m pytest tests -q
```

Stereo smoke:

```bash
rsf smoke \
  --left /mnt/workspace/wenqian/scene_edit_v0/test_data/041_L.png \
  --right /mnt/workspace/wenqian/scene_edit_v0/test_data/041_R.png \
  --calib /mnt/workspace/wenqian/scene_edit_v0/test_data/041_calib.json \
  --out runs/smoke_041
```

Video front end:

```bash
scripts/rsf_video_m7.sh \
  --video scripts/test_data/my_table.mp4 \
  --out runs/my_table_m7_<timestamp> \
  --frame-stride 10 \
  --reference-frame-index 0
```

Video object scene:

```bash
scripts/rsf_video_scene.sh \
  --run-dir runs/my_table_m7_<timestamp> \
  --target-label bottle \
  --target-label cup \
  --target-label "tissue pack" \
  --max-objects 3 \
  --settle-steps 100
```

Target export commands to implement or run:

```bash
rsf export-manifest --run-dir runs/<run_id>
rsf export-sim --run-dir runs/<run_id> --backend genesis --backend usd --backend isaac
rsf qa-sim --run-dir runs/<run_id>
```

If these commands do not exist yet, implementing them is part of the current work. Do not skip them.

## Required Checklist Format For Future Agents

Every future execution response should include and maintain this checklist:

```markdown
## Execution Checklist

- [ ] 0. Read project rules and confirm dirty worktree
- [ ] 1. Classify current assets and block proxy-as-visual success
- [ ] 2. Build real visual object asset path
- [ ] 3. Separate visual / collision / debug proxy assets
- [ ] 4. Stop using full `scene_cloud.ply` as final background visual
- [ ] 5. Register BG-only / 3DGS background to simulator world
- [ ] 6. Upgrade object placement to explicit 6D pose contract
- [ ] 7. Build strict `sim_export_manifest.json`
- [ ] 8. Build real USD / Genesis / Isaac export bundle
- [ ] 9. Build export-level QA
- [ ] 10. Run canonical end-to-end export and report pass/block status
```

After every completed item, update it like this:

```markdown
- [x] 2. Build real visual object asset path
  - Files changed: ...
  - Artifacts: ...
  - Verification: ...
  - Status: passed / blocked
  - Notes: ...
```

## Current Execution Checklist Status - 2026-07-01

Current canonical status: **B-route simulator export passes**. Object visual path is completed: canonical object visuals now point to `objects/*/visual.glb` with `visual_asset_report.json`, while bbox meshes are debug/collision/proxy artifacts. The final visual background is now the external Nerfstudio 3DGS render `qa/background_3dgs_render.png`; `background/bg_only_cloud.ply` remains recorded as a proxy/diagnostic geometry artifact, not the final visual background.

Important distinction: 3DGS config files, candidate assets under `background/3dgs_native/`, and backend asset-report sections can mean **asset ready**. They do not mean **rendered background**. The accepted B route uses `rsf render-3dgs-background` to call Nerfstudio externally and records `source_kind=external_3dgs_renderer`, `simulator_native=false`. Isaac native 3DGS/splat runtime remains unsupported; Isaac currently validates USD foreground/collision/physics loading using the worker report, not native splat rendering.

- [x] 0. Read project rules and confirm dirty worktree
  - Files changed: `AI_START_HERE.md`, `AGENTS.md`, `PLAN.md`
  - Artifacts: documented Isaac worker `ssh -p 1024 root@101.132.143.105`, `/isaac-sim/python.sh`
  - Verification: `git status --short`; project rules read from `AI_START_HERE.md`, `AGENTS.md`, `PLAN.md`, and the simulator export plan
  - Status: completed
  - Notes: worktree is dirty with active implementation files and untracked new modules/tests.
- [x] 1. Classify current assets and block proxy-as-visual success
  - Files changed: `src/real2sim_scene_foundry/sim_export_manifest.py`, `src/real2sim_scene_foundry/visual_assets.py`, related tests
  - Artifacts: `sim_export_manifest.json`; object asset provenance records; explicit debug proxy records
  - Verification: canonical `objects/*/mesh_aligned.glb` are 8 vertices / 12 faces bbox proxies; canonical `objects/*/visual.glb` are high-topology SAM3D mesh-job outputs; manifest now blocks bbox `mesh_aligned.glb` if it is referenced as final visual
  - Status: completed
  - Notes: legacy bbox files may still exist on disk, but current production references them only for legacy/proxy detection and blocking.
- [x] 2. Build real visual object asset path
  - Files changed: `src/real2sim_scene_foundry/clients.py`, `src/real2sim_scene_foundry/pipeline.py`, `src/real2sim_scene_foundry/visual_assets.py`, `tests/test_reconstruct_align_pipeline.py`, `tests/test_visual_assets.py`, related tests
  - Artifacts: `objects/{bottle,cup,tissue_pack}/visual.glb`; `objects/*/debug_bbox.glb`; `objects/*/visual_point_cloud.ply` for PLY-only SAM3D path
  - Verification: `python -m pytest tests/test_clients.py tests/test_reconstruct_align_pipeline.py tests/test_visual_assets.py tests/test_sim_export_manifest.py -q` -> passed in subagent A; integrated full suite `97 passed`
  - Status: completed
  - Notes: new pipeline no longer exposes bbox proxy as `mesh_path` for PLY-only SAM3D; bbox stays in `alignment_mesh_path` / `debug_proxy_path`.
- [x] 3. Separate visual / collision / debug proxy assets
  - Files changed: `src/real2sim_scene_foundry/collision_assets.py`, `src/real2sim_scene_foundry/interactive.py`, `src/real2sim_scene_foundry/support_plane.py`, `src/real2sim_scene_foundry/manifest.py`, tests
  - Artifacts: per-object `visual.glb`, `collision.glb`, `debug_bbox.glb`, `physics.json`
  - Verification: canonical manifest object visual paths are `objects/*/visual.glb`, collision paths are `objects/*/collision.glb`, debug paths are `objects/*/debug_bbox.glb`; QA collision section passes
  - Status: completed
  - Notes: collision remains a visual-derived collision asset with recorded provenance, not a claim of perfect high-fidelity physics geometry.
- [x] 4. Stop using full `scene_cloud.ply` as final background visual
  - Files changed: `src/real2sim_scene_foundry/background.py`, `src/real2sim_scene_foundry/composite_viewer.py`, `src/real2sim_scene_foundry/sim_export_manifest.py`, tests
  - Artifacts: `scene_manifest.json`, `background/registration.json`
  - Verification: final background path is `background/bg_only_cloud.ply`; `scene_cloud.ply` remains debug-only and is not final background visual
  - Status: completed
  - Notes: this only removes full-scene foreground leakage; BG-only cloud is still a background proxy.
- [x] 5. Register BG-only / 3DGS background to simulator world
  - Files changed: `src/real2sim_scene_foundry/background_registration.py`, `src/real2sim_scene_foundry/background_3dgs_render.py`, `src/real2sim_scene_foundry/sim_export_manifest.py`, `src/real2sim_scene_foundry/export_qa.py`, `src/real2sim_scene_foundry/cli.py`, related tests
  - Artifacts: `background/registration.json`; `background/3dgs_native/splat_rgb.ply` candidate asset; `qa/background_3dgs_render.png`; `qa/background_3dgs_render_report.json`; `background/bg_only_cloud.ply` as recorded proxy/diagnostic geometry
  - Verification: `rsf render-3dgs-background --run-dir runs/my_table_m7_20260630_185332 --split test --camera-idx 0` -> `status=rendered`, image size `1280x720`; targeted tests `37 passed`; canonical manifest background `source_kind=external_3dgs_renderer`, `external_render_verified=true`
  - Status: completed for accepted B route
  - Notes: Isaac/Genesis do not render native 3DGS. The visual background is an external Nerfstudio render sidecar with `simulator_native=false`; this is distinct from native splat runtime support.
- [x] 6. Upgrade object placement to explicit 6D pose contract
  - Files changed: `src/real2sim_scene_foundry/sim_export_manifest.py`, tests
  - Artifacts: `objects/*/pose_refinement_report.json`, `pose_overrides.yaml`
  - Verification: manifest reports object poses `status=ready`, `pose_source=manual_refined_from_auto`
  - Status: completed
  - Notes: identity rotations are explicitly manually accepted where present, not default success.
- [x] 7. Build strict `sim_export_manifest.json`
  - Files changed: `src/real2sim_scene_foundry/sim_export_manifest.py`, `src/real2sim_scene_foundry/cli.py`, `tests/test_sim_export_manifest.py`
  - Artifacts: `runs/my_table_m7_20260630_185332/sim_export_manifest.json`
  - Verification: `rsf export-manifest --run-dir runs/my_table_m7_20260630_185332`; canonical manifest reports `status=passed`, `blocking_reasons=[]`
  - Status: completed
  - Notes: strict manifest still blocks bbox visuals, full-scene cloud, BG-only-only backgrounds, and failed/missing external 3DGS renders.
- [x] 8. Build real USD / Genesis / Isaac export bundle
  - Files changed: `src/real2sim_scene_foundry/usd_export.py`, `src/real2sim_scene_foundry/genesis_export.py`, `src/real2sim_scene_foundry/isaac_export.py`, `tests/test_simulator_exports.py`
  - Artifacts: `exports/scene.usda`, `exports/genesis_scene.py`, `exports/isaac_scene.py`, backend QA reports
  - Verification: `rsf export-sim --run-dir runs/my_table_m7_20260630_185332 --backend genesis --backend usd --backend isaac`; `python -m pytest tests/test_simulator_exports.py -q` -> passed; export reports include `object_visual_refs`, `object_collision_refs`, `debug_proxy_refs`, `background_asset`, `background_runtime`
  - Status: completed for accepted B route
  - Notes: Isaac report is preserved from the worker load validation (`report_source=preserved_existing_worker_report`, `status=loaded`). Native 3DGS remains unsupported; external 3DGS render is the visual background route.
- [x] 9. Build export-level QA
  - Files changed: `src/real2sim_scene_foundry/export_qa.py`, `src/real2sim_scene_foundry/cli.py`, `tests/test_export_qa.py`
  - Artifacts: `qa/sim_export_report.json`, `qa/isaac_load_report.json`, `qa/genesis_settle_report.json`
  - Verification: `rsf qa-sim --run-dir runs/my_table_m7_20260630_185332`; canonical QA reports `overall_status=passed`, `blocking_reasons=[]`
  - Status: completed
  - Notes: QA still fails closed when external 3DGS render is missing/blocked or when Isaac runtime/report is unavailable.
- [x] 10. Run canonical end-to-end export and report pass/block status
  - Files changed: run artifacts under `runs/my_table_m7_20260630_185332`
  - Artifacts: current `sim_export_manifest.json`, `qa/sim_export_report.json`, backend export reports
  - Verification: `python -m pytest tests -q` -> `105 passed, 2 warnings`; canonical `rsf export-manifest`, `rsf export-sim --backend genesis --backend usd --backend isaac`, `rsf qa-sim` all ran successfully
  - Status: passed for accepted B route
  - Notes: `sim_export_manifest.json` reports `status=passed`; `qa/sim_export_report.json` reports `overall_status=passed`. Caveat: `native_runtime_verified=false` for 3DGS; the accepted background route is `external_3dgs_renderer`.

## Recommended Parallelization

Use subagents only for independent slices with disjoint write scopes.

Parallelizable:

- Visual assets / SAM3D response handling:
  - `src/real2sim_scene_foundry/clients.py`
  - `src/real2sim_scene_foundry/pipeline.py`
  - `src/real2sim_scene_foundry/visual_assets.py`
- Background / 3DGS / viewer source:
  - `src/real2sim_scene_foundry/background.py`
  - `src/real2sim_scene_foundry/video_scene.py`
  - `src/real2sim_scene_foundry/background_registration.py`
  - `src/real2sim_scene_foundry/composite_viewer.py`
- Collision / physics separation:
  - `src/real2sim_scene_foundry/collision_assets.py`
  - `src/real2sim_scene_foundry/interactive.py`
  - `src/real2sim_scene_foundry/support_plane.py`
  - `src/real2sim_scene_foundry/manifest.py`
- Export manifest / USD / QA:
  - `src/real2sim_scene_foundry/sim_export_manifest.py`
  - `src/real2sim_scene_foundry/usd_export.py`
  - `src/real2sim_scene_foundry/genesis_export.py`
  - `src/real2sim_scene_foundry/isaac_export.py`
  - `src/real2sim_scene_foundry/export_qa.py`

Keep main-thread integration for:

- `src/real2sim_scene_foundry/cli.py`
- final manifest schema merge
- final end-to-end run
- final QA status

Do not run the final canonical export until visual assets, collision assets, background registration, export bundle, and QA report are all integrated.

## Final Response Standard

A final response on implementation work must include:

```markdown
## Checklist Status
- [x] ...
- [ ] ...

## Built Artifacts
...

## Verification
...

## Blocked / Not Fully Implemented
...

## Next Exact Step
...
```

If any proxy/fallback remains in the final path, say so plainly and do not call the reproduction complete.
