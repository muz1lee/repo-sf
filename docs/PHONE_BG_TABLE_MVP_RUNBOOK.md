# Phone BG-Table MVP Runbook

This runbook is the formal path for validating a new iPhone / ARKit RGB-D capture up to the current milestone:
registered 3DGS visual background, fitted tabletop collision, Genesis cube-drop physics validation, and strict honest QA.

The formal artifact names use `mvp`, not debug or legacy smoke names. Legacy smoke commands and files may exist for backward compatibility, but they must not be used as final evidence.

## Inputs

Use a phone RGB-D capture bundle or a converted Record3D export that provides:

- RGB frames
- metric depth in meters
- explicit intrinsics
- explicit poses
- confidence/tracking evidence when available
- enough camera baseline around the tabletop

Do not use a plain MP4 as a replacement for RGB-D capture.

## Formal Output Contract

A successful run should produce these formal artifacts:

```text
scene_manifest.phone.json
sim_export_manifest.json
exports/bg_table_mvp_config.json
exports/bg_table_mvp.usda
exports/bg_table_mvp_genesis.py
exports/scene.usda
exports/genesis_scene.py
qa/table_physics_mvp_report.json
qa/table_physics_mvp_runtime_report.json
qa/genesis_settle_report.json
qa/table_collision_report.json
qa/table_collision_report_v2.json
qa/phone_bg_table_mvp_report.json
qa/sim_export_report.json
qa/honest_milestone.json
```

Expected strict claim status for this milestone:

```text
engineering_interactive_scene_status = passed
simfoundry_upper_reproduction_status = partial
simfoundry_full_pipeline_status = out_of_scope
```

`simfoundry_upper_reproduction_status` remains `partial` because 3DGS is a browser/native sidecar visual, not Genesis/Isaac simulator-native rendering.

## Existing Imported Run

For a run that already has phone import, 3DGS registration, tabletop mask/plane, and table collision:

```bash
cd /mnt/workspace/wenqian/real2sim_scene_foundry
source .venv/bin/activate

rsf build-table-collision \
  --run-dir runs/<phone_run> \
  --source background/table_polygon_world.json \
  --write

rsf qa-bg-table \
  --run-dir runs/<phone_run> \
  --frames 0 \
  --write-overlays

rsf export-bg-table-mvp \
  --run-dir runs/<phone_run>

/mnt/workspace/wenqian/knowin-world/.venv/bin/python \
  runs/<phone_run>/exports/bg_table_mvp_genesis.py \
  --run-dir runs/<phone_run> \
  --settle-steps 160 \
  --backend cpu \
  --no-viewer

rsf export-phone-bg-table-mvp \
  --run-dir runs/<phone_run>

rsf qa-sim \
  --run-dir runs/<phone_run> \
  --strict-claims
```

## Capture-to-MVP Checklist

For a fresh capture, run or verify these stages in order:

1. Import/validate phone capture or Record3D export.
2. Export Nerfstudio transforms with a declared pose world.
3. Train or load 3DGS and write `background/3dgs_native/splat_rgb.ply`.
4. Register 3DGS to sim world with explicit evidence.
5. Segment the tabletop in one or more RGB frames.
6. Fit tabletop plane/polygon from semantic mask plus ARKit depth.
7. Build table collision slab from `background/table_polygon_world.json`.
8. Run BG-table QA and inspect overlays.
9. Export formal BG-table MVP artifacts.
10. Run Genesis cube-drop validation through `exports/bg_table_mvp_genesis.py`.
11. Run `export-phone-bg-table-mvp` and strict `qa-sim`.
12. Open a viewer only after the formal reports identify the artifacts and claim status.

## Pass/Block Interpretation

Passed at this milestone means:

- 3DGS and table collision are registered to the same sim world.
- Table collision is generated from semantic tabletop mask plus metric phone depth.
- Genesis can settle a cube on the table collision without fall-through or large drift.
- The formal manifest and strict QA state the remaining gaps explicitly.

Blocked or partial means:

- Missing explicit camera/depth evidence blocks strict QA.
- Weak table projection or occlusion blocks export-grade tabletop collision.
- Missing fresh Isaac validation keeps Isaac blocked.
- Missing simulator-native 3DGS keeps SimFoundry upper reproduction partial.

## Legacy Boundary

Do not use legacy debug/smoke artifacts as final evidence. They are allowed only for compatibility with older runs or existing demos. Formal phone BG-table MVP claims must come from the `mvp` artifacts listed above.
