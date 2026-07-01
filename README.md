# real2sim-scene-foundry

Engineering reproduction of the upper real-to-sim pipeline from SimFoundry-style scene reconstruction. The project turns stereo or video observations into object assets, metric scene manifests, collision geometry, simulator exports, QA reports, and an interactive inspection viewer.

This repository is an engineering workbench, not a claim of full SimFoundry paper reproduction. The current canonical run is fail-closed: object meshes, table collision, Genesis settle, USD/Genesis/Isaac export scaffolds, and viewer provenance exist, while native/live 3DGS background registration remains blocked.

## Current Status

- Object visuals: `objects/*/visual.glb` are the final visual paths; legacy `mesh_aligned.glb` files are debug/proxy only.
- Background: trained 3DGS sidecar assets can exist, but the viewer currently uses an external reference-view PNG sidecar unless a native splat runtime is integrated.
- Physics: object collision meshes and support surfaces are exported separately from visual meshes.
- Export: `sim_export_manifest.json` and `qa/sim_export_report.json` are intended to fail closed when background registration, native runtime, weak table QA, or stale Isaac evidence is detected.
- Viewer: `rsf composite-viewer` is an inspection UI that explicitly reports whether the background is live 3DGS, external sidecar, or diagnostic point cloud.

## Repository Layout

```text
src/real2sim_scene_foundry/   Python package and CLI implementation
tests/                        Unit and regression tests
scripts/                      Server-side helper scripts for video/3DGS workflows
docs/                         Gap ledgers, runtime probes, and publishing notes
AGENTS.md                     Project rules for future agents
AI_START_HERE.md              Handoff facts for new AI/Codex sessions
PLAN.md                       Development plan and current milestones
```

Generated artifacts are intentionally excluded from Git: `runs/`, `outputs/`, virtual environments, videos, point clouds, meshes, checkpoints, USD exports, and rendered images.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m pytest tests -q
```

The CLI entrypoint is:

```bash
rsf --help
```

## Common Commands

Canonical stereo smoke input, when available on the project server:

```bash
rsf smoke \
  --left /mnt/workspace/wenqian/scene_edit_v0/test_data/041_L.png \
  --right /mnt/workspace/wenqian/scene_edit_v0/test_data/041_R.png \
  --calib /mnt/workspace/wenqian/scene_edit_v0/test_data/041_calib.json \
  --out runs/smoke_041
```

Refresh key canonical artifacts:

```bash
rsf support-plane --run-dir runs/my_table_m7_20260630_185332 --force
rsf refine-pose --run-dir runs/my_table_m7_20260630_185332
rsf render-3dgs-background --run-dir runs/my_table_m7_20260630_185332
rsf export-manifest --run-dir runs/my_table_m7_20260630_185332
rsf export-sim --run-dir runs/my_table_m7_20260630_185332 --backend genesis --backend usd --backend isaac
rsf qa-sim --run-dir runs/my_table_m7_20260630_185332
rsf composite-viewer --run-dir runs/my_table_m7_20260630_185332 --export-only
```

Start the viewer server:

```bash
rsf composite-viewer --run-dir runs/my_table_m7_20260630_185332 --host 127.0.0.1 --port 7010
```

If the viewer is running on a remote server, forward the port before opening the browser:

```bash
ssh -fN -o ExitOnForwardFailure=yes -L 7010:127.0.0.1:7010 wenqian_h200
```

## Environment

See `.env.example` for optional service variables. SAM3D intentionally has no hard-coded default endpoint; set `SAM3D_PROCESS_URL` only after verifying the current service registry.

The project can call external services for segmentation, depth, inpainting, 3D asset generation, 3DGS training, Genesis, and Isaac validation. Tests are written to run without those services.

## GitHub Publishing Notes

Before pushing, read `docs/GITHUB_PUBLISHING.md`. In particular:

- Do not commit `runs/`, `.venv*/`, model weights, videos, GLB/PLY/USD outputs, checkpoints, or screenshots.
- Decide whether the repository is private or public before publishing files that mention internal server paths or service addresses.
- Choose a license before making a public repository.

## License

No license has been selected yet. Until a license is added, treat the repository as private/internal or all-rights-reserved.
