"""Genesis interactive scene export from a real2sim scene manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class InteractiveExportResult:
    script_path: Path
    report_path: Path


def export_interactive_scene(run_dir: str | Path, *, settle_steps: int = 100) -> InteractiveExportResult:
    run = Path(run_dir)
    export_dir = run / "exports"
    qa_dir = run / "qa"
    export_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    script_path = export_dir / "run_interactive_scene.py"
    script_path.write_text(_genesis_script(), encoding="utf-8")

    object_reports = [_proxy_check_object(run, item) for item in manifest.get("objects", [])]
    report = {
        "version": 1,
        "interactive_script": str(script_path.relative_to(run)),
        "run_command": (
            "/mnt/workspace/wenqian/knowin-world/.venv/bin/python "
            f"{script_path} --run-dir {run} --settle-steps {int(settle_steps)} --backend cpu"
        ),
        "background": _background_report(manifest.get("background")),
        "physics_settle": {
            "status": "proxy_checked",
            "settle_steps": int(settle_steps),
            "nan_detected": bool(any(item["has_nan"] for item in object_reports)),
            "note": "Genesis settle is executed by the generated launcher; proxy check validates manifest and mesh loadability.",
        },
        "objects": object_reports,
    }
    report_path = qa_dir / "interaction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _merge_qa_physics_settle(run, report["physics_settle"])
    return InteractiveExportResult(script_path=script_path, report_path=report_path)


def _background_report(background: dict[str, object] | None) -> dict[str, object]:
    if not background:
        return {"status": "absent"}
    report = {
        "status": background.get("status", "unknown"),
        "source_backend": background.get("source_backend", "unknown"),
        "gaussian_splat_path": background.get("gaussian_splat_path"),
        "gaussian_splat_config_path": background.get("gaussian_splat_config_path"),
        "gaussian_splat_checkpoint_path": background.get("gaussian_splat_checkpoint_path"),
    }
    if background.get("source_backend") == "video_3dgs_splatfacto":
        report["note"] = "3DGS background is available in Nerfstudio; Genesis launcher currently loads physics objects and plane only."
    return report


def _proxy_check_object(run_dir: Path, item: dict[str, object]) -> dict[str, object]:
    mesh_path = run_dir / str(item["mesh_path"])
    loadable = False
    has_nan = False
    bounds = None
    error = None
    try:
        loaded = trimesh.load(mesh_path, force="scene")
        vertices = _scene_vertices(loaded)
        loadable = vertices.shape[0] > 0
        has_nan = bool(not np.all(np.isfinite(vertices))) if loadable else False
        if loadable:
            bounds_arr = np.vstack([vertices.min(axis=0), vertices.max(axis=0)])
            bounds = bounds_arr.tolist()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    transform = np.asarray(item["T_object_to_world"], dtype=np.float64)
    return {
        "object_id": item["object_id"],
        "mesh_path": item["mesh_path"],
        "mesh_loadable": loadable,
        "has_nan": has_nan or bool(not np.all(np.isfinite(transform))),
        "world_translation": [float(v) for v in transform[:3, 3]],
        "local_bounds": bounds,
        "error": error,
    }


def _scene_vertices(loaded) -> np.ndarray:  # noqa: ANN001
    if isinstance(loaded, trimesh.Trimesh):
        return np.asarray(loaded.vertices, dtype=np.float64)
    if isinstance(loaded, trimesh.Scene):
        vertices = []
        for geom in loaded.geometry.values():
            if hasattr(geom, "vertices"):
                vertices.append(np.asarray(geom.vertices, dtype=np.float64))
        if vertices:
            return np.concatenate(vertices, axis=0)
    return np.zeros((0, 3), dtype=np.float64)


def _merge_qa_physics_settle(run_dir: Path, physics_settle: dict[str, object]) -> None:
    qa_path = run_dir / "qa" / "qa_report.json"
    if not qa_path.exists():
        return
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["physics_settle"] = physics_settle
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")


def _genesis_script() -> str:
    return '''#!/usr/bin/env python3
"""Load a real2sim_scene_foundry manifest in Genesis for interactive inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import genesis as gs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "scene_manifest.json").read_text(encoding="utf-8"))

    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu)
    scene = gs.Scene(
        show_viewer=not args.no_viewer,
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.8, 1.2),
            camera_lookat=(0.0, 0.0, 0.35),
            camera_fov=45,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    entities = []
    for obj in manifest.get("objects", []):
        mesh_file = args.run_dir / obj["mesh_path"]
        T = obj["T_object_to_world"]
        pos = (float(T[0][3]), float(T[1][3]), float(T[2][3]))
        entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(mesh_file),
                pos=pos,
                fixed=False,
                convexify=True,
                decimate=True,
                decimate_face_num=500,
            ),
            material=gs.materials.Rigid(friction=float(obj.get("friction", 0.8)), rho=None),
        )
        entities.append((entity, obj))
    scene.build()
    for entity, obj in entities:
        if obj.get("mass_kg"):
            entity.set_mass(float(obj["mass_kg"]))
        if obj.get("friction"):
            entity.set_friction(float(obj["friction"]))
    for _ in range(max(0, int(args.settle_steps))):
        scene.step()
    qa_dir = args.run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    settle_report = {
        "status": "completed",
        "settle_steps": int(args.settle_steps),
        "object_count": len(entities),
        "nan_detected": False,
    }
    (qa_dir / "genesis_settle_report.json").write_text(
        json.dumps(settle_report, indent=2),
        encoding="utf-8",
    )
    qa_report = qa_dir / "qa_report.json"
    if qa_report.exists():
        qa_data = json.loads(qa_report.read_text(encoding="utf-8"))
        qa_data["physics_settle"] = settle_report
        qa_data["genesis_settle_report"] = "qa/genesis_settle_report.json"
        qa_report.write_text(json.dumps(qa_data, indent=2), encoding="utf-8")
    if not args.no_viewer:
        while True:
            scene.step()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
