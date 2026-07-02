"""Minimal simulator smoke export for BG-table collision validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class BgTablePhysicsSmokeResult:
    config_path: Path
    genesis_script_path: Path
    usd_path: Path
    cube_visual_path: Path
    report_path: Path
    report: dict[str, Any]


def export_bg_table_physics_smoke(
    run_dir: str | Path,
    *,
    cube_size_m: float = 0.08,
    drop_height_m: float = 0.12,
    settle_steps: int = 120,
) -> BgTablePhysicsSmokeResult:
    run = Path(run_dir)
    exports = run / "exports"
    qa = run / "qa"
    assets = exports / "bg_table_physics_smoke_assets"
    exports.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    polygon = _load_json(run / "background" / "table_polygon_world.json")
    table_report = _load_json(run / "table" / "collision_polygon_slab_report.json")
    bg_report = _load_json(run / "qa" / "bg_table_report.json")
    registration = _load_json(run / "background" / "registration.json")

    table_mesh_rel = "table/collision_polygon_slab.glb"
    table_mesh_path = run / table_mesh_rel
    if not table_mesh_path.is_file():
        raise FileNotFoundError(f"missing table collision mesh: {table_mesh_path}")

    polygon_xy = _polygon_xy(polygon)
    center_xy = _polygon_center_xy(polygon_xy)
    top_z = float(polygon.get("top_z_m", polygon.get("table_top_z_m", table_report.get("top_z_m", 0.0))) or 0.0)
    cube_size = float(cube_size_m)
    drop_height = float(drop_height_m)
    initial_center = [center_xy[0], center_xy[1], top_z + cube_size * 0.5 + drop_height]
    expected_resting_center_z = top_z + cube_size * 0.5

    cube_visual_path = assets / "test_cube.glb"
    trimesh.creation.box(extents=(cube_size, cube_size, cube_size)).export(cube_visual_path)

    config = {
        "version": 1,
        "scope": "bg_table_physics_smoke",
        "support_surface": {
            "mesh_path": table_mesh_rel,
            "usd_path": "table/collision_polygon_slab.usd" if (run / "table" / "collision_polygon_slab.usd").is_file() else None,
            "source_backend": polygon.get("source_backend") or table_report.get("source_backend"),
            "geometry_type": table_report.get("geometry_type", "polygon_slab"),
            "top_z_m": top_z,
            "polygon_world_xy": polygon_xy,
            "table_collision_final": True,
        },
        "test_object": {
            "object_id": "table_smoke_cube",
            "shape": "cube",
            "size_m": cube_size,
            "mass_kg": 0.1,
            "friction": 0.8,
            "initial_center_world_m": initial_center,
            "expected_resting_center_z_m": expected_resting_center_z,
            "visual_asset_path": _rel(run, cube_visual_path),
        },
        "physics_contract": {
            "settle_steps": int(settle_steps),
            "expected_contact_gap_abs_m_max": 0.03,
            "max_horizontal_drift_m": 0.05,
            "fall_below_table_margin_m": 0.10,
        },
        "background_visual": _background_visual_contract(run, bg_report, registration),
    }

    config_path = exports / "bg_table_physics_smoke_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    genesis_script_path = exports / "bg_table_physics_smoke_genesis.py"
    genesis_script_path.write_text(_genesis_smoke_script(), encoding="utf-8")
    usd_path = exports / "bg_table_physics_smoke.usda"
    usd_path.write_text(_smoke_usda(run, usd_path.parent, config), encoding="utf-8")

    report = {
        "version": 1,
        "status": "script_written",
        "scope": "bg_table_physics_smoke",
        "config_path": "exports/bg_table_physics_smoke_config.json",
        "genesis_script_path": "exports/bg_table_physics_smoke_genesis.py",
        "usd_path": "exports/bg_table_physics_smoke.usda",
        "cube_visual_path": _rel(run, cube_visual_path),
        "run_command": (
            "/mnt/workspace/wenqian/knowin-world/.venv/bin/python "
            f"{genesis_script_path} --run-dir {run} --settle-steps {int(settle_steps)} --backend cpu --no-viewer"
        ),
        "support_surface": config["support_surface"],
        "test_object": config["test_object"],
        "physics_contract": config["physics_contract"],
        "background_visual": config["background_visual"],
        "runtime_report_path": "qa/table_physics_smoke_runtime_report.json",
        "claim": "table collision exported for physics smoke; 3DGS is visual-only provenance, not simulator-native physics",
    }
    report_path = qa / "table_physics_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return BgTablePhysicsSmokeResult(
        config_path=config_path,
        genesis_script_path=genesis_script_path,
        usd_path=usd_path,
        cube_visual_path=cube_visual_path,
        report_path=report_path,
        report=report,
    )


def _background_visual_contract(run: Path, bg_report: dict[str, Any], registration: dict[str, Any]) -> dict[str, Any]:
    splat_path = run / "background" / "3dgs_native" / "splat_rgb.ply"
    return {
        "mode": "registered_3dgs_visual_only" if splat_path.is_file() else "none",
        "splat_path": "background/3dgs_native/splat_rgb.ply" if splat_path.is_file() else None,
        "registration_status": registration.get("status"),
        "qa_status": bg_report.get("status"),
        "simulator_native_3dgs": False,
        "physics_role": "none",
        "provenance": "3DGS is not loaded as Genesis physics or native simulator background in this smoke scene.",
    }


def _polygon_xy(polygon: dict[str, Any]) -> list[list[float]]:
    points = polygon.get("polygon_world_xy")
    if not points:
        polygons = polygon.get("polygons_world_xy")
        if isinstance(polygons, list) and polygons:
            points = polygons[0]
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("table polygon must contain at least three world XY points")
    return [[float(point[0]), float(point[1])] for point in points]


def _polygon_center_xy(points: list[list[float]]) -> list[float]:
    arr = np.asarray(points, dtype=np.float64)
    return [float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _ref(export_dir: Path, target: Path) -> str:
    return os.path.relpath(target, export_dir).replace(os.sep, "/")


def _matrix4d_translate(xyz: list[float]) -> str:
    x, y, z = (float(value) for value in xyz[:3])
    return (
        "( (1, 0, 0, 0), "
        "(0, 1, 0, 0), "
        "(0, 0, 1, 0), "
        f"({x:.8g}, {y:.8g}, {z:.8g}, 1) )"
    )


def _smoke_usda(run: Path, export_dir: Path, config: dict[str, Any]) -> str:
    support = config["support_surface"]
    obj = config["test_object"]
    return "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "World"',
            '    upAxis = "Z"',
            "    metersPerUnit = 1",
            ")",
            "",
            'def Xform "World"',
            "{",
            '    custom string rsf:scope = "bg_table_physics_smoke"',
            '    custom string rsf:background_visual_role = "visual_only_not_physics"',
            '    def Xform "SupportSurface" (',
            '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
            f"        prepend references = @{_ref(export_dir, run / support['mesh_path'])}@",
            "    )",
            "    {",
            '        custom string rsf:role = "phone_depth_table_collision"',
            f'        custom string rsf:source_backend = "{support.get("source_backend", "unknown")}"',
            "        bool physics:collisionEnabled = true",
            "    }",
            '    def Cube "TestCube" (',
            '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]',
            "    )",
            "    {",
            f"        double size = {float(obj['size_m']):.8g}",
            "        bool physics:collisionEnabled = true",
            f"        float physics:mass = {float(obj['mass_kg']):.8g}",
            f"        matrix4d xformOp:transform = {_matrix4d_translate(obj['initial_center_world_m'])}",
            '        uniform token[] xformOpOrder = ["xformOp:transform"]',
            "    }",
            "}",
            "",
        ]
    )


def _genesis_smoke_script() -> str:
    return '''#!/usr/bin/env python3
"""Run a minimal Genesis physics smoke for the BG-table collision slab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import genesis as gs
import numpy as np


def tensor_to_array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def entity_pos(entity):
    if hasattr(entity, "get_pos"):
        return [float(v) for v in tensor_to_array(entity.get_pos()).reshape(-1)[:3]]
    return [0.0, 0.0, 0.0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    run = args.run_dir
    config = json.loads((run / "exports" / "bg_table_physics_smoke_config.json").read_text(encoding="utf-8"))
    support = config["support_surface"]
    obj = config["test_object"]
    contract = config["physics_contract"]
    settle_steps = int(args.settle_steps if args.settle_steps is not None else contract["settle_steps"])

    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu)
    scene = gs.Scene(
        show_viewer=not args.no_viewer,
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.0, -1.2, support["top_z_m"] + 0.7), camera_lookat=(0.0, 0.0, support["top_z_m"]), camera_fov=45),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(run / support["mesh_path"]),
            fixed=True,
            collision=True,
            convexify=False,
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(friction=1.0),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=tuple(float(v) for v in obj["initial_center_world_m"]),
            size=(float(obj["size_m"]), float(obj["size_m"]), float(obj["size_m"])),
            fixed=False,
            collision=True,
        ),
        material=gs.materials.Rigid(friction=float(obj["friction"])),
    )
    scene.build()
    if hasattr(cube, "set_mass"):
        cube.set_mass(float(obj["mass_kg"]))
    for _ in range(settle_steps):
        scene.step()

    final_center = entity_pos(cube)
    initial_center = [float(v) for v in obj["initial_center_world_m"]]
    size = float(obj["size_m"])
    top_z = float(support["top_z_m"])
    expected_center_z = float(obj["expected_resting_center_z_m"])
    bottom_z = float(final_center[2]) - size * 0.5
    contact_gap = bottom_z - top_z
    horizontal_drift = float(np.linalg.norm(np.asarray(final_center[:2]) - np.asarray(initial_center[:2])))
    nan_detected = bool(not np.all(np.isfinite(np.asarray(final_center, dtype=np.float64))))
    fall_below_table = bool(final_center[2] < top_z - float(contract["fall_below_table_margin_m"]))
    contact_ok = abs(contact_gap) <= float(contract["expected_contact_gap_abs_m_max"])
    drift_ok = horizontal_drift <= float(contract["max_horizontal_drift_m"])
    stability_status = "passed" if (not nan_detected and not fall_below_table and contact_ok and drift_ok) else "failed"
    report = {
        "version": 1,
        "status": "completed",
        "stability_status": stability_status,
        "settle_steps": settle_steps,
        "support_surface": support,
        "test_object": {
            "object_id": obj["object_id"],
            "shape": obj["shape"],
            "size_m": size,
            "initial_center_world_m": initial_center,
            "final_center_world_m": final_center,
            "expected_resting_center_z_m": expected_center_z,
            "bottom_z_m": bottom_z,
            "contact_gap_m": contact_gap,
            "horizontal_drift_m": horizontal_drift,
        },
        "checks": {
            "contact_ok": bool(contact_ok),
            "horizontal_drift_ok": bool(drift_ok),
            "nan_detected": nan_detected,
            "fall_below_table": fall_below_table,
            "expected_contact_gap_abs_m_max": float(contract["expected_contact_gap_abs_m_max"]),
            "max_horizontal_drift_m": float(contract["max_horizontal_drift_m"]),
        },
        "background_visual": config.get("background_visual", {}),
    }
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "table_physics_smoke_runtime_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if stability_status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''
