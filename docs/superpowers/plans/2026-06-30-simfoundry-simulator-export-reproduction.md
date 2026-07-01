# SimFoundry Simulator Export Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `real2sim_scene_foundry` from a runnable Figure-2-style prototype to a complete SimFoundry reconstruction pipeline through simulator export: real capture -> extracted foreground objects and background -> metric 6D object poses -> registered background -> stable physics assets -> Genesis and Isaac-compatible scene export with QA evidence.

**Architecture:** Keep the existing artifact-first pipeline, but promote it from fallback-oriented artifacts to a strict export contract. The next phase should make pose, background registration, collision, physics, and simulator export first-class stages with explicit manifests and acceptance reports. Unit tests remain useful for pure math and serialization, but success is gated by end-to-end exported scenes and visual/physical QA, not by mock tests or fallback runs.

**Tech Stack:** Python package `real2sim_scene_foundry`; external services SAM3, SAM3D, Qwen/Gemini-compatible VLM, S2M2, MoGe; Nerfstudio/splatfacto for 3DGS; Genesis and USD/OpenUSD for simulator export; optional IsaacLab loader executed outside this project if the runtime is available.

## Global Constraints

- Work line: **项目落地先行、论文后置**.
- Server authoritative repository: `/mnt/workspace/wenqian/real2sim_scene_foundry`.
- Do not modify `/mnt/workspace/wenqian/knowin-world`, `/mnt/workspace/wenqian/ego_hand_pipeline`, `/mnt/workspace/wenqian/ego_eef`, or `/mnt/workspace/wenqian/scene_edit_v0` unless explicitly requested.
- Current user requirement: do not treat fallback implementations, smoke-only runs, or minimal unit tests as the goal.
- Complete reproduction scope in this plan means SimFoundry Figure 2 upper pipeline through **Simulator export**. Object/scene/task cousins, policy training, and policy evaluation are excluded from this plan unless a later request expands scope.
- A run is accepted only if it produces inspectable artifacts: object meshes and collision assets, 6D poses, background assets, scene manifest, USD/GLB export, Genesis/Isaac loading entrypoint, render-vs-input QA, and physics stability report.
- Fallback paths may remain for diagnostics, but any fallback-backed output must be marked `diagnostic_only` and cannot satisfy the final acceptance gate.
- All production dependency additions must be declared in `pyproject.toml` or isolated in purpose-specific runtime environments such as `.venv_3dgs`; do not install into system Python.
- Future implementation agents must not ask the user to choose between obvious engineering routes. Use the recommended route in this document, write blocked reports when an external service/runtime is unavailable, and continue with the next independent task.
- The implementation order in **Execution Queue For Future Agents** is mandatory. Do not skip ahead to viewer polish, fallback meshes, or extra tests while core visual assets, background registration, and simulator export remain incomplete.

---

## Execution Queue For Future Agents

This section is deliberately prescriptive. Follow it in order.

1. **Classify current assets and block proxy-as-visual success.**
   - Add explicit provenance fields for `visual_asset`, `collision_asset`, `debug_proxy`, and `background_visual`.
   - Mark current 8-vertex box GLBs as `debug_proxy` or `collision_proxy`, never as final visual assets.
   - Acceptance: `sim_export_manifest.json` refuses `overall_status=passed` if any object visual asset is a bbox fallback.

2. **Fix visual object assets before improving viewer aesthetics.**
   - First try to make SAM3D return a real `mesh_glb_base64`/`glb_base64`.
   - If SAM3D only returns point clouds, generate a separate visual candidate from `objects/<id>/sam3d/point_cloud.ply` and label its source honestly.
   - Use point-cloud reconstruction only as an explicit candidate, not as proof of full SimFoundry-quality mesh.
   - Acceptance: every object has `visual.glb` or `visual_point_cloud.ply` with source and quality report; `mesh_aligned.glb` box is no longer used as the visual asset.

3. **Separate visual geometry from physical geometry.**
   - Keep bbox/convex proxies for Genesis/Isaac collision if needed.
   - Export `collision.glb` and `physics.json` separately from visual assets.
   - Acceptance: viewer can toggle visual asset and collision proxy independently.

4. **Stop using full-scene point cloud as the background visual.**
   - `scene_cloud.ply` is a debug full-scene cloud and contains foreground objects.
   - Use `background/bg_only_cloud.ply` as the minimum background proxy.
   - Use trained 3DGS as the target background visual when export/runtime support is ready.
   - Acceptance: composite viewer defaults to BG-only/registered background, never full `scene_cloud.ply`.

5. **Register background to simulator world.**
   - Write `background/registration.json`.
   - Store `T_3dgs_world_to_sim_world` and any scale source.
   - Acceptance: foreground visual assets and BG-only/3DGS background align in a reference-camera overlay.

6. **Upgrade object placement from center/scale to 6D pose.**
   - Use service-backed pose if available.
   - Otherwise use manual refinement only when recorded as `manual_refined_from_auto`.
   - Acceptance: `T_object_to_camera` and `T_object_to_world` are real 6D transforms with source reports, not implicit identity rotations hidden by box geometry.

7. **Build the simulator export bundle.**
   - Export USD with visual references, collision references, physical properties, support surface, reference camera, and backend loader scripts.
   - Genesis export must load physical scene and run settle.
   - Isaac export must load when runtime exists or write `runtime_unavailable` when it does not.

8. **Run export-level QA and only then call the reproduction complete.**
   - Required output: `qa/sim_export_report.json`.
   - Required status for success: `overall_status=passed`.
   - Any fallback, missing visual asset, unregistered background, or proxy-only object must produce a blocked status with exact reason.

Do not add object/scene/task cousins, policy training, or policy evaluation before this queue reaches an export-grade scene.

## Why This Plan Exists

The current project has already passed the early "can we produce artifacts" threshold. It has a package, CLI, stereo extraction, Qwen-before-SAM object proposals, S2M2/MoGe metric geometry, SAM3D-derived object visual assets, background inpainting artifacts, Nerfstudio 3DGS training, a scene manifest, a USD/export bundle, support-plane normalization, and a Genesis launcher.

That is not yet a faithful SimFoundry simulator-export reproduction. The remaining gap is no longer "add more fallbacks" or "add one more mock test." The next engineering goal is to make the output load as a simulator scene whose geometry, background, object transforms, collision, and physical properties are coherent enough to inspect and interact with.

The canonical evidence run at the time of writing is:

```text
/mnt/workspace/wenqian/real2sim_scene_foundry/runs/my_table_m7_20260630_185332
```

Key facts from that run:

- Objects reconstructed: `bottle`, `cup`, `tissue_pack`.
- Background 3DGS status: `completed`, latest checkpoint `video/3dgs/unnamed/splatfacto/2026-06-30_205848/nerfstudio_models/step-000002999.ckpt`.
- BG-only video: 131 frames; 36 frames reused previous masks after proposal failure.
- Object alignment QA:
  - `bottle`: bbox IoU `0.5296815286624204`, center error `18.200274723201296 px`, depth residual `0.02502436039168421 m`.
  - `cup`: bbox IoU `0.7993753903810119`, center error `1.5811388300841898 px`, depth residual `0.020245671272277832 m`.
  - `tissue_pack`: bbox IoU `0.8328236493374108`, center error `3.0413812651491097 px`, depth residual `0.020976364612579346 m`.
- Genesis no-viewer settle: `passed`, 100 steps, no NaN, no fall-out, max penetration `3.6522746086120605e-05 m`.
- Current Genesis export loads foreground meshes and support-plane/table collision. It does not load/render the 3DGS background inside Genesis.

Additional proxy diagnosis from the same run:

- `objects/bottle/mesh_aligned.glb`: 8 vertices, 12 faces, 1140 bytes.
- `objects/cup/mesh_aligned.glb`: 8 vertices, 12 faces, 1144 bytes.
- `objects/tissue_pack/mesh_aligned.glb`: 8 vertices, 12 faces, 1144 bytes.
- `objects/*/sam3d/sam3d_metadata.json` contains `ply_camera_base64`, but not `mesh_glb_base64`, `glb_base64`, or `mesh`.
- `SAM3DClient.process()` currently converts `ply_camera_base64` into `objects/<id>/sam3d/point_cloud.ply`, then `_export_proxy_mesh_from_ply()` exports a 5%-95% bounding box as `mesh.glb`.
- `run_reconstruct_align()` later turns that box into `objects/<id>/mesh_aligned.glb`.
- Therefore the composite viewer is not rendering real assets as boxes. It is correctly loading assets that are already box proxies.
- `scene_cloud.ply` exists and is a full-scene cloud. It contains foreground objects.
- `background/bg_only_cloud.ply` also exists and should be the minimum background proxy.
- `video/3dgs/unnamed/splatfacto/2026-06-30_205848/config.yml` exists as the trained 3DGS config, but current viewer/export does not render that splat as the background.

The exact current simplification is:

```text
visual object asset     -> box proxy from SAM3D point-cloud bbox
collision object asset  -> effectively the same box/proxy mesh
background visual       -> external Nerfstudio 3DGS render sidecar for accepted B route
background proxy        -> bg_only_cloud.ply remains diagnostic/geometry proxy, not final visual
table visual            -> table_collision.glb box, really a physics support proxy
pose                    -> center/scale alignment with mostly identity rotations
viewer                  -> Three.js inspection viewer, not Genesis/Isaac physics
```

This diagnosis must be preserved as provenance. Future agents should not describe the current run as having native Isaac/Genesis 3DGS rendering. The current accepted background path is external 3DGS rendering with explicit `simulator_native=false`.

### 2026-07-01 Status Update

- Object visual path is completed in the current mainline: object visuals are exported through `objects/*/visual.glb` with `objects/*/visual_asset_report.json`, while bbox meshes are kept as debug/collision/proxy artifacts.
- B route is completed in the canonical run: `rsf render-3dgs-background` produced `qa/background_3dgs_render.png` and `qa/background_3dgs_render_report.json` with `status=rendered`.
- 3DGS config files, candidate assets, or backend asset-report sections can show that an asset is ready. They do not show that Genesis/USD/Isaac rendered native 3DGS. asset ready != native simulator rendered.
- Isaac worker bridge is still a repeatability caveat: a local operator bridge produced canonical validation evidence, and `export-sim --backend isaac` preserves that loaded report; direct/repeatable `wenqian_h200 -> Isaac worker` execution is still unreliable.
- Canonical `sim_export_manifest.json` is unblocked and `qa/sim_export_report.json` reports `overall_status=passed` for the accepted B route.

## Paper Context To Preserve

SimFoundry's reconstruction path is:

1. **Extraction**
   - Input raw RGB video.
   - Select representative frame.
   - Estimate depth and intrinsics.
   - Lift RGB-D to point cloud.
   - Use a scene VLM and SAM3 to identify and segment foreground objects.
   - Iteratively remove extracted foreground objects via RGB-D inpainting.
   - Output per-object RGB-D crops/masks and background-only assets.

2. **Generation**
   - Upsample object images.
   - Generate per-object meshes with a 2D-to-3D model.
   - Estimate/refine 6D pose and scale by aligning generated meshes against RGB-D, masks, and scene geometry.
   - For articulated objects, segment parts and generate joint parameters.
   - Generate collision meshes, assign physical properties, perform physics depenetration, and export to robotics simulators.

3. **Background Reconstruction**
   - Automatic path: same capture video -> foreground mask propagation -> two-pass video inpainting -> metric depth and camera poses -> depth-supervised 3DGS -> rigid bridge into simulator world.
   - Manual path: second foreground-free video -> 3DGS -> interactive SE(3)+scale alignment.

4. **Augmentation**
   - Object, scene, and task cousins.
   - This plan does not implement cousins because the current request stops at simulator export.

The project should keep using local services that exist on our server, but every substitution must be explicit. A practical reproduction can use Qwen instead of Gemini, SAM3D instead of Hunyuan/TRELLIS, MoGe+COLMAP instead of DepthAnything3, and Genesis before IsaacLab, but the artifact contract must still match the paper's intent.

## Current Reproduction Gap Ledger

### Gap 1: Foreground Video Segmentation Is Not Paper-Equivalent

**Paper target:** Query object categories on a keyframe, refine masks with image segmentation, propagate the merged foreground mask through video with a video segmentation model, then run temporally consistent two-pass video inpainting.

**Current state:** `video_bg_only.py` runs Qwen/SAM on sampled frames and reuses the previous mask when proposals fail. This produced a completed BG-only frame set, but the mask stream is not temporally grounded the same way as SAM2 propagation, and 36/131 frames in the latest run reused stale masks.

**Required export-grade behavior:** Build a video foreground mask track with explicit per-frame provenance:

- `source=propagated_from_keyframe` when a mask comes from video propagation.
- `source=per_frame_refinement` when a frame is locally corrected.
- `source=reused_previous_mask` is allowed only as `diagnostic_only`.
- The final 3DGS training dataset must not silently include stale masks.

**Do not drift:** Improving the browser viewer before fixing the mask/background asset source is not useful. The viewer must not hide stale masks by rendering the full-scene cloud.

### Gap 2: Metric Camera/Background Bridge Is Incomplete

**Paper target:** Recover metric depth and camera poses for the same video stream, train a background splat, and compute a rigid transform that bridges the splat world into the ground-plane simulator world.

**Current state:** The project trains a Nerfstudio `splatfacto` model from BG-only frames and COLMAP poses. MoGe gives dense geometry for the reference frame. The manifest records the splat checkpoint, but the transform between `3dgs_world`, reference camera, and `z_up_ground_plane_meters` is not a first-class exported asset.

**Required export-grade behavior:** Store and validate these transforms:

- `T_reference_camera_to_world`
- `T_3dgs_world_to_reference_camera`
- `T_3dgs_world_to_sim_world`
- scale convention for COLMAP/Nerfstudio if metric scale is recovered externally

The export stage must be able to place the background asset and foreground meshes in the same visual coordinate frame, even if Genesis cannot render the splat natively.

**Do not drift:** `scene_cloud.ply` is allowed only for debugging depth quality. It is not a valid background visual for final export because it includes foreground tabletop objects.

### Gap 3: Object Pose Is Not Full 6D

**Paper target:** Estimate per-object 6D pose plus scale, refine with RGB-D and pose models such as FoundationPose, and optionally manually tune a small number of important objects.

**Current state:** `mesh_aligned.glb` is scaled/centered against object point clouds. In the latest run, `T_object_to_camera` rotations are effectively identity. The object meshes can appear in the scene, but orientation is not yet a solved pose-estimation output.

**Required export-grade behavior:** Introduce a real pose refinement stage that writes:

- `T_canonical_mesh_to_camera`
- `T_canonical_mesh_to_world`
- `scale_m`
- `rotation_source`
- `translation_source`
- `scale_source`
- `pose_refinement_report.json`

The stage must support service-backed pose refinement if available and a manual refinement editor for export-critical objects. A manually edited pose is acceptable for final reproduction only if the manifest records it as `source=manual_refined_from_auto`, never as an automatic result.

**Do not drift:** A box mesh can make bad rotations invisible. Pose acceptance must use visual/canonical object assets and projection overlays, not bbox-only geometry.

### Gap 3.5: SAM3D Visual Mesh Is Currently Missing

**Paper target:** The object generation stage produces visual meshes for the foreground objects, then aligns those meshes to the metric scene.

**Current state:** The latest run did not receive `mesh_glb_base64`, `glb_base64`, or `mesh` from SAM3D. It received only `ply_camera_base64`, and the client converted that PLY into a bbox mesh. The resulting `mesh_aligned.glb` files are 8-vertex boxes.

**Required export-grade behavior:** Treat SAM3D point clouds and bbox meshes as separate assets:

- `objects/<id>/sam3d/point_cloud.ply`: raw geometry evidence.
- `objects/<id>/visual.glb`: final visual mesh, not a bbox fallback.
- `objects/<id>/visual_point_cloud.ply`: optional visual point cloud if no mesh backend is available.
- `objects/<id>/collision.glb`: simplified physics proxy.
- `objects/<id>/debug_bbox.glb`: diagnostic bbox, never final visual.

**Recommended implementation order:**

1. Inspect SAM3D service response and request options to recover real mesh output.
2. If real SAM3D mesh output is unavailable, add a visual reconstruction backend from point cloud with explicit source and quality report.
3. Add an export gate that blocks final status when `visual_asset.source` is `bbox_from_point_cloud`.

**Do not drift:** Do not spend time making the box look better in the viewer. The box is a valid collision/debug proxy, not a visual asset.

### Gap 4: Visual Mesh, Collision Mesh, And Physical Asset Are Not Separated

**Paper target:** Generate visual mesh, collision geometry using a decomposition method such as CoACD, physical parameters, then depenetrate and cache stable poses.

**Current state:** Genesis uses `gs.morphs.Mesh(... convexify=True, decimate=True)` directly from visual meshes. USD stub records mesh path, mass, friction, and translation. There is no explicit collision asset per object in the project manifest.

**Required export-grade behavior:** Each object must have distinct asset roles:

```json
{
  "visual_mesh_path": "objects/cup/visual.glb",
  "collision_mesh_path": "objects/cup/collision.glb",
  "sim_mesh_path": "objects/cup/sim.glb",
  "mass_kg": 0.2,
  "friction": 0.4,
  "restitution": 0.0,
  "collision_source": "coacd_or_vhacd_or_documented_genesis_convex"
}
```

If the runtime cannot run CoACD/VHACD, Genesis convexification can be kept as a diagnostic path, but final export must mark collision provenance clearly.

**Do not drift:** Do not delete the bbox proxy. It is useful for collision and metric debugging. The fix is separation and provenance, not removal.

### Gap 5: 3DGS Background Is Trained But Not Exported Into The Simulator Scene

**Paper target:** Compose foreground object meshes with a photorealistic background reconstruction.

**Current state:** 3DGS is inspectable in Nerfstudio and `background/3dgs_native/splat_rgb.ply` exists as a candidate/native-side asset; Genesis loads objects and a table/plane. The final background path is still the BG-only proxy unless a backend report proves native 3DGS loading/rendering. asset ready != simulator rendered.

**Required export-grade behavior:** Produce a simulator export bundle with two synchronized views:

- **Physics view:** collision objects, support surfaces, dynamic foreground objects.
- **Visual view:** foreground visual meshes plus registered background, either as native splat rendering where available or as a documented render bridge.

At export time, the manifest must say whether each backend loads the background natively:

```json
{
  "simulator_exports": {
    "genesis": {
      "physics_background": "background/table_collision.glb",
      "visual_background": "not_native_3dgs",
      "visual_background_viewer": "exports/composite_viewer/index.html"
    },
    "isaac": {
      "physics_background": "background/table_collision.glb",
      "visual_background": "usd_gaussian_splat_or_point_proxy"
    }
  }
}
```

**Current viewer-specific simplification:** The composite viewer reads a point-cloud background proxy. In the latest run, that proxy is `scene_cloud.ply`, which is full-scene geometry. This explains why tabletop foreground objects still appear in the background.

**Required viewer behavior before final export:**

- Default background source must be `background/bg_only_cloud.ply` when native 3DGS rendering is not available.
- Full `scene_cloud.ply` may be exposed behind a `Debug full scene cloud` toggle.
- The viewer config must say whether it is rendering `bg_only_cloud`, `scene_cloud_debug`, `registered_3dgs`, or `3dgs_unavailable`.
- If `video/3dgs_status.json` has `status=completed`, the viewer/export must surface the 3DGS config and registration status instead of silently falling back to full-scene cloud.
- A 3DGS asset report or candidate PLY is not enough for final acceptance. Accepted visual-background routes are either native simulator/viewer splat rendering, or the B-route sidecar `external_3dgs_renderer` report with a real rendered image. The canonical run currently uses the B route: `qa/background_3dgs_render_report.json` has `status=rendered`, `render_path=qa/background_3dgs_render.png`, and `simulator_native=false`.

**Do not drift:** A viewer that displays full `scene_cloud.ply` plus box object proxies is a debug viewer, not a SimFoundry reconstruction viewer.

### Gap 6: USD Export Is Still A Stub

**Paper target:** Export a sim-ready scene to downstream robotics simulators such as IsaacLab.

**Current state:** `exports/scene.usda` stores object xforms and custom metadata. It is useful as a manifest-adjacent artifact, but it is not a complete robotics simulator scene.

**Required export-grade behavior:** Export at least:

- world root with meters-per-unit and z-up.
- support surface collision body.
- rigid object prims with visual mesh references.
- collision mesh references or collision approximation metadata.
- mass/friction/restitution in simulator-consumable form.
- camera prims matching the reference camera and QA camera.
- optional background visual prim or documented sidecar for 3DGS/point cloud.
- a loader script that proves the USD can be opened by the target runtime.

### Gap 7: QA Does Not Yet Prove Rendered Export Fidelity

**Paper target:** Reconstruction fidelity is evaluated against geometry and visual alignment; simulator evaluation depends on faithful initial state.

**Current state:** QA includes mask overlays, bbox IoU, center error, depth residual, support plane report, and Genesis settle. This is good but not sufficient to prove simulator export fidelity.

**Required export-grade behavior:** Every accepted run must include:

- `qa/reference_input.png`
- `qa/render_export.png`
- `qa/render_vs_input_overlay.png`
- `qa/object_projection_overlay.png`
- `qa/background_registration_overlay.png`
- `qa/sim_export_report.json`
- `qa/genesis_settle_report.json`
- `qa/isaac_load_report.json` if Isaac runtime is available

The report must include go/no-go status for pose, background, collision, physics, and export loading.

## Target Artifact Contract

The final accepted run directory should have this structure:

```text
runs/<run_id>/
  camera.json
  extraction_manifest.json
  scene_manifest.json
  sim_export_manifest.json
  left.png or video/reference.png
  xyz.npy
  scene_cloud.ply
  background/
    foreground_mask.png
    bg_only.png
    bg_only_cloud.ply
    table_collision.glb
    background_manifest.json
    registration.json
  video/
    video_manifest.json
    bg_only_status.json
    3dgs_status.json
    nerfstudio_bg/
    3dgs/
  objects/<object_id>/
    mask.png
    crop.png
    object_cloud.ply
    sam3d/
      point_cloud.ply
      sam3d_metadata.json
    visual.glb
    visual_point_cloud.ply
    collision.glb
    sim.glb
    debug_bbox.glb
    pose.json
    pose_refinement_report.json
    physics.json
  exports/
    scene.usda
    scene.usdc
    scene.glb
    genesis_scene.py
    isaac_scene.py
    composite_viewer/
      index.html
      viewer_config.json
  qa/
    qa_report.json
    sim_export_report.json
    genesis_settle_report.json
    reference_input.png
    render_export.png
    render_vs_input_overlay.png
    object_projection_overlay.png
    background_registration_overlay.png
```

## Implementation Phases

### Phase A: Freeze The Export Contract And Gap Report

**Purpose:** Stop changing the target implicitly. Make the paper-to-project gap and export acceptance contract part of the repo.

**Files:**

- Create: `docs/superpowers/plans/2026-06-30-simfoundry-simulator-export-reproduction.md`
- Modify: `PLAN.md` only if we want to point readers to this document as the new export-stage plan.

**Steps:**

- [ ] Add this document to the server repository.
- [ ] Add a short note to `PLAN.md` stating that M10+ export work is tracked here.
- [ ] Verify the server worktree is clean before implementation starts:

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
git status --short
```

Expected before future code work: only this documentation file, or a committed clean tree.

**Acceptance:**

- The document explains paper context, current evidence, gap ledger, target artifact contract, phases, and final acceptance.
- It does not claim fallback or mock tests are sufficient.

### Phase B: Add A Strict Simulator Export Manifest

**Purpose:** Make simulator export a first-class stage rather than metadata spread across `scene_manifest.json`, USD stub, and interaction reports.

**Files:**

- Create: `src/real2sim_scene_foundry/sim_export_manifest.py`
- Create: `tests/test_sim_export_manifest.py`
- Modify: `src/real2sim_scene_foundry/cli.py`

**Interface:**

```python
@dataclass(frozen=True)
class SimExportManifest:
    run_dir: str
    source_scene_manifest: str
    coordinate_frames: dict[str, str]
    objects: list[SimExportObject]
    background: SimExportBackground
    exports: dict[str, SimExportBackend]
    qa: dict[str, str]
```

**Acceptance, not just tests:**

- Running `rsf export-manifest --run-dir runs/my_table_m7_20260630_185332` writes `sim_export_manifest.json`.
- The file lists each object visual mesh, collision mesh, physics values, 6D transform source, backend export paths, and QA artifact paths.
- Missing collision meshes or missing background registration cause `status=blocked_missing_export_asset`, not silent fallback success.
- A bbox fallback object produces `status=blocked_proxy_visual_asset` unless a real `visual.glb` or accepted `visual_point_cloud.ply` is present.

### Phase C: Upgrade Object Pose From Centered Similarity To Export-Grade 6D

**Purpose:** Fix the biggest reconstruction gap: current object rotations are not meaningful enough for simulator export.

**Files:**

- Create: `src/real2sim_scene_foundry/pose_refinement.py`
- Create: `src/real2sim_scene_foundry/pose_overrides.py`
- Modify: `src/real2sim_scene_foundry/pipeline.py`
- Modify: `src/real2sim_scene_foundry/manifest.py`
- Modify: `src/real2sim_scene_foundry/composite_viewer.py`

**Required behavior:**

1. Preserve SAM3D raw geometry output and canonical mesh frame.
2. Estimate initial 6D pose from one of:
   - SAM3D returned pose if present and validated.
   - service-backed FoundationPose-style pose if available.
   - manual-refined pose loaded from `pose_overrides.yaml`.
3. Refine scale/translation against object point cloud.
4. Validate projected silhouette/bbox/depth residual.
5. Record pose source in `pose_refinement_report.json`.

**Manual refinement policy:**

Manual refinement is allowed for export-critical objects, because the paper itself supports human intervention. It must be explicit:

```yaml
objects:
  bottle:
    source: manual_refined_from_auto
    translation_xyz_m: [0.1302, -0.0684, 1.0537]
    rotation_quat_wxyz: [0.7071, -0.7071, 0.0, 0.0]
    scale: 1.0
```

**Acceptance, not just tests:**

- `objects/<id>/pose_refinement_report.json` exists for every object.
- `T_object_to_camera` contains nontrivial rotation when the object orientation is not axis-aligned.
- `qa/object_projection_overlay.png` shows mesh projection against mask for every object.
- Export is blocked if any object has `needs_manual_refine=true` and no manual override is provided.

### Phase D: Create Explicit Visual, Point-Cloud, Collision, And Sim Assets

**Purpose:** Separate pretty geometry from physical geometry. This is required for stable simulator export.

**Files:**

- Create: `src/real2sim_scene_foundry/visual_assets.py`
- Create: `src/real2sim_scene_foundry/collision_assets.py`
- Modify: `src/real2sim_scene_foundry/pipeline.py`
- Modify: `src/real2sim_scene_foundry/interactive.py`
- Modify: `src/real2sim_scene_foundry/manifest.py`
- Modify: `src/real2sim_scene_foundry/composite_viewer.py`

**Required behavior:**

- Write `visual.glb` from SAM3D or processed visual mesh.
- If the service only returns `ply_camera_base64`, write `visual_point_cloud.ply` and a visual reconstruction quality report; do not promote bbox fallback to visual mesh.
- Write `collision.glb` from CoACD/VHACD if available.
- If Genesis convexification is used, record it as `collision_source=genesis_runtime_convexification` and mark the run as not final unless the final acceptance explicitly allows that source.
- Write `physics.json` with mass, friction, restitution, density basis, and source.
- Update Genesis launcher to load collision settings from manifest rather than assuming visual mesh equals collision mesh.
- Update composite viewer to show visual asset by default and collision/debug proxy only when toggled.

**Acceptance, not just tests:**

- Every exported object has `visual.glb` or `visual_point_cloud.ply`, `collision.glb`, `debug_bbox.glb`, and `physics.json`.
- `sim_export_report.json` records collision source per object.
- Genesis settle report uses the new collision assets or explicitly records why the runtime substituted a collision approximation.
- A run with only 8-vertex bbox meshes cannot pass visual asset acceptance.

### Phase E: Register Background 3DGS To Simulator World

**Purpose:** Turn trained 3DGS from a side artifact into a registered scene asset.

**Files:**

- Create: `src/real2sim_scene_foundry/background_registration.py`
- Modify: `src/real2sim_scene_foundry/video_scene.py`
- Modify: `src/real2sim_scene_foundry/composite_viewer.py`
- Modify: `src/real2sim_scene_foundry/manifest.py`

**Required behavior:**

- Compute and write `background/registration.json`.
- Store `T_3dgs_world_to_sim_world`.
- Store reference camera metadata and source of scale.
- Render or export a background registration preview.
- The composite viewer must use the registered transform and default to `background/bg_only_cloud.ply` or registered 3DGS, not full `scene_cloud.ply`.

**Acceptance, not just tests:**

- `background/registration.json` exists and contains numeric 4x4 transforms.
- `qa/background_registration_overlay.png` shows foreground object meshes and background proxy/splat aligned from the reference viewpoint.
- `scene_manifest.json` and `sim_export_manifest.json` both refer to the same background registration asset.
- `exports/composite_viewer/viewer_config.json` records `background.source_kind` as `bg_only_cloud` or `registered_3dgs`, not `full_scene_cloud`, for final export runs.

### Phase F: Replace USD Stub With Real Simulator Export Bundle

**Purpose:** Produce simulator-loadable output rather than a metadata-only USD.

**Files:**

- Create: `src/real2sim_scene_foundry/usd_export.py`
- Create: `src/real2sim_scene_foundry/genesis_export.py`
- Create: `src/real2sim_scene_foundry/isaac_export.py`
- Modify: `src/real2sim_scene_foundry/cli.py`
- Keep: `src/real2sim_scene_foundry/interactive.py` as a launcher implementation detail or migrate it into `genesis_export.py`.

**CLI:**

```bash
rsf export-sim \
  --run-dir runs/my_table_m7_20260630_185332 \
  --backend genesis \
  --backend usd \
  --backend isaac
```

**Required outputs:**

- `exports/scene.usda`
- `exports/scene.usdc` when USD tooling is available
- `exports/genesis_scene.py`
- `exports/isaac_scene.py`
- `exports/composite_viewer/index.html`
- `sim_export_manifest.json`

**Genesis acceptance:**

```bash
/mnt/workspace/wenqian/knowin-world/.venv/bin/python \
  runs/my_table_m7_20260630_185332/exports/genesis_scene.py \
  --run-dir runs/my_table_m7_20260630_185332 \
  --settle-steps 100 \
  --backend cpu \
  --no-viewer
```

Expected:

- `qa/genesis_settle_report.json` has `status=completed`.
- `stability_status=passed`.
- `nan_detected=false`.
- `fall_out_detected=false`.
- `max_penetration_depth_m <= 0.005`.

**Isaac acceptance if runtime is available:**

```bash
rsf verify-isaac-export --run-dir runs/my_table_m7_20260630_185332
```

Expected:

- `qa/isaac_load_report.json` has `status=loaded`.
- All object prim paths exist.
- Rigid body and collision metadata are present.
- If Isaac is unavailable, the report must say `status=runtime_unavailable`, not `passed`.

### Phase G: Add Export-Level QA

**Purpose:** Make the final report answer whether the simulator export is usable.

**Files:**

- Create: `src/real2sim_scene_foundry/export_qa.py`
- Modify: `src/real2sim_scene_foundry/cli.py`
- Modify: `src/real2sim_scene_foundry/composite_viewer.py`

**CLI:**

```bash
rsf qa-sim --run-dir runs/my_table_m7_20260630_185332
```

**Required report sections:**

- `pose_alignment`
- `background_registration`
- `collision_assets`
- `physics_settle`
- `usd_export`
- `genesis_export`
- `isaac_export`
- `overall_status`

**Visual outputs:**

- `qa/render_export.png`
- `qa/render_vs_input_overlay.png`
- `qa/object_projection_overlay.png`
- `qa/background_registration_overlay.png`

**Acceptance:**

- The run is accepted only if `qa/sim_export_report.json` has `overall_status=passed`.
- The report must contain exact failing reasons if blocked, such as `missing_collision_mesh`, `pose_needs_manual_refine`, `background_unregistered`, or `genesis_settle_failed`.

### Phase H: Run A Canonical End-To-End Export Reproduction

**Purpose:** Produce one final run that can be used as the project-level proof point.

**Input recommendation:**

Use the latest video run as the first export-grade target:

```text
runs/my_table_m7_20260630_185332
```

If new capture is needed, keep the run name explicit:

```text
runs/simfoundry_export_<scene_name>_<YYYYMMDD_HHMMSS>
```

**Execution sequence:**

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate

scripts/rsf_video_m7.sh \
  --video scripts/test_data/my_table.mp4 \
  --out runs/simfoundry_export_my_table_<timestamp> \
  --frame-stride 10 \
  --reference-frame-index 0

scripts/rsf_video_scene.sh \
  --run-dir runs/simfoundry_export_my_table_<timestamp> \
  --target-label bottle \
  --target-label cup \
  --target-label "tissue pack" \
  --max-objects 3 \
  --settle-steps 100

rsf export-sim \
  --run-dir runs/simfoundry_export_my_table_<timestamp> \
  --backend genesis \
  --backend usd \
  --backend isaac

rsf qa-sim --run-dir runs/simfoundry_export_my_table_<timestamp>
```

**Final acceptance:**

- `sim_export_manifest.json` exists.
- `qa/sim_export_report.json` has `overall_status=passed`.
- `exports/scene.usda` contains real mesh references and rigid/collision metadata.
- `exports/genesis_scene.py` loads and settles successfully.
- `exports/composite_viewer/index.html` visually overlays object meshes, support plane, and registered background.
- If Isaac runtime is not installed and no prior worker report is available, `qa/isaac_load_report.json` explicitly records `runtime_unavailable`; this blocks full requested simulator-export completion. If a worker `status=loaded` report exists, `export-sim --backend isaac` preserves it with `report_source=preserved_existing_worker_report`.

## Engineering Order Of Operations

1. Implement Phase B first because it defines the export contract that every later phase writes into.
2. Implement Phase C before background work because pose errors dominate the perceived scene mismatch.
3. Implement Phase D before final physics verification because collision provenance determines whether settle results mean anything.
4. Implement Phase E before final visual QA because background registration must be judged in the same frame as object meshes.
5. Implement Phase F only after object/background assets have stable paths and transforms.
6. Implement Phase G after export exists so QA can inspect real exported assets rather than pipeline internals.
7. Run Phase H as a release candidate, then inspect visual artifacts manually before calling the reproduction complete.

## Risk Register

| Risk | Impact | Engineering response |
| --- | --- | --- |
| SAM3D mesh frame or scale is inconsistent across objects | Object placement looks plausible in point cloud but wrong in simulator | Preserve raw SAM3D metadata, write explicit canonical-to-camera transform, and block export when pose source is ambiguous |
| Qwen/SAM per-frame masks are temporally unstable | 3DGS background contains foreground ghosting or holes | Add propagated mask track and treat stale-mask reuse as diagnostic-only |
| COLMAP scale is not metric | 3DGS background does not align with metric foreground objects | Store scale source and estimate bridge against MoGe/S2M2/reference geometry |
| Genesis cannot render 3DGS natively | Visual scene in Genesis differs from paper figure | Separate physics and visual views; use `external_3dgs_renderer` sidecar background for the accepted B route until native splat rendering exists |
| Isaac runtime is unavailable | Cannot prove Isaac export load | Generate USD and loader script; mark Isaac verification as blocked by runtime, not passed |
| Collision decomposition is unavailable | Physics settle uses runtime convexification and may hide geometry errors | Keep diagnostic path but require explicit collision provenance for final acceptance |

## Definition Of Done

The reproduction is complete for the requested scope when a fresh run satisfies all of the following:

1. `scene_manifest.json` contains all objects, 6D transforms, physics values, background registration, and support plane.
2. `sim_export_manifest.json` contains backend-specific export records for USD and Genesis, and Isaac when available.
3. Every object has visual mesh, collision mesh, pose report, and physics report.
4. The background visual is either native-rendered 3DGS in the same world frame as object meshes, or an accepted B-route external 3DGS render sidecar with `simulator_native=false` and explicit registration/provenance. BG-only point cloud alone is not final.
5. Genesis no-viewer settle passes with final report written to QA.
6. USD export loads with mesh and collision metadata visible to the loader.
7. Visual QA overlays show the export rendered from the reference camera against the input.
8. `sim_export_manifest.json` is unblocked and `qa/sim_export_report.json` records `overall_status=passed`.
9. The final answer to the user can point to the run directory and list exact artifacts, not just test output.

## Verification Commands For This Documentation Step

After adding this document:

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate
python -m pytest tests -q
git status --short
```

Expected:

- Tests continue passing.
- `git status --short` shows this document as a new tracked candidate unless it has been committed.
