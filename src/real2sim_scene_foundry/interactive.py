"""Genesis interactive scene export from a real2sim scene manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .collision_assets import ensure_collision_assets


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
    ensure_collision_assets(run)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    script_path = export_dir / "run_interactive_scene.py"
    script_path.write_text(_genesis_script(), encoding="utf-8")

    object_reports = [_proxy_check_object(run, item) for item in manifest.get("objects", [])]
    collision_sources_by_object = {item["object_id"]: item["collision_source"] for item in object_reports}
    genesis_collision_sources_by_object = {item["object_id"]: item["genesis_collision_source"] for item in object_reports}
    report = {
        "version": 1,
        "interactive_script": str(script_path.relative_to(run)),
        "run_command": (
            "/mnt/workspace/wenqian/knowin-world/.venv/bin/python "
            f"{script_path} --run-dir {run} --settle-steps {int(settle_steps)} --backend cpu"
        ),
        "background": _background_report(manifest.get("background")),
        "support_plane": _support_plane_report(manifest.get("support_plane")),
        "physics_settle": {
            "status": "proxy_checked",
            "settle_steps": int(settle_steps),
            "nan_detected": bool(any(item["has_nan"] for item in object_reports)),
            "settled_pose_delta_report": "qa/settled_pose_delta_report.json",
            "visual_physics_coherence": _visual_physics_coherence(),
            "collision_sources_by_object": collision_sources_by_object,
            "genesis_collision_sources_by_object": genesis_collision_sources_by_object,
            "genesis_collision_status": "runtime_convexification_enabled",
            "note": "Genesis settle is executed by the generated launcher; proxy check validates manifest, collision mesh loadability, and provenance.",
        },
        "objects": object_reports,
    }
    report_path = qa_dir / "interaction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _merge_qa_physics_settle(run, report["physics_settle"])
    return InteractiveExportResult(script_path=script_path, report_path=report_path)


def _support_plane_report(support_plane: dict[str, object] | None) -> dict[str, object]:
    if not support_plane:
        return {"status": "absent"}
    return {
        "status": support_plane.get("status", "unknown"),
        "source_backend": support_plane.get("source_backend", "unknown"),
        "height_world_m": support_plane.get("height_world_m"),
        "original_height_world_m": support_plane.get("original_height_world_m"),
        "applied_to_world_frame": support_plane.get("applied_to_world_frame", False),
        "normal_world": support_plane.get("normal_world"),
        "table_collision_mesh_path": support_plane.get("table_collision_mesh_path"),
        "table_collision_source_backend": support_plane.get("table_collision_source_backend"),
        "table_collision_geometry_type": support_plane.get("table_collision_geometry_type"),
        "table_collision_final": support_plane.get("table_collision_final", False),
        "table_bounds_world_xy": support_plane.get("table_bounds_world_xy"),
        "table_collision_pos_world": support_plane.get("table_collision_pos_world"),
        "table_collision_size_xyz": support_plane.get("table_collision_size_xyz"),
    }


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


def _visual_physics_coherence() -> dict[str, object]:
    return {
        "status": "partial",
        "visual_assets_referenced_by_runtime": False,
        "collision_assets_referenced_by_runtime": True,
        "background_asset_referenced_by_runtime": False,
        "reason": (
            "Genesis interactive settle runtime uses collision meshes and support surface only; "
            "foreground visual meshes and native background visuals are not referenced by this runtime."
        ),
    }


def _proxy_check_object(run_dir: Path, item: dict[str, object]) -> dict[str, object]:
    collision_asset = item.get("collision_asset") if isinstance(item.get("collision_asset"), dict) else {}
    visual_asset = item.get("visual_asset") if isinstance(item.get("visual_asset"), dict) else {}
    debug_proxy = item.get("debug_proxy") if isinstance(item.get("debug_proxy"), dict) else {}
    physics = _load_physics_record(run_dir, item)
    collision_path_value = collision_asset.get("path") or item.get("mesh_path")
    mesh_path = run_dir / str(collision_path_value)
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
    quat = _matrix_to_quat_wxyz(transform[:3, :3])
    collision_source = str(collision_asset.get("source") or physics.get("collision_source") or "legacy_mesh_path_runtime_collision")
    genesis_collision_source = f"{collision_source}+genesis_runtime_convexification"
    return {
        "object_id": item["object_id"],
        "mesh_path": str(collision_path_value),
        "legacy_mesh_path": item.get("mesh_path"),
        "visual_asset": visual_asset,
        "collision_asset": collision_asset,
        "debug_proxy": debug_proxy,
        "physics": physics,
        "collision_source": collision_source,
        "genesis_collision_source": genesis_collision_source,
        "genesis_collision_status": "runtime_convexification_enabled",
        "mesh_loadable": loadable,
        "collision_mesh_loadable": loadable,
        "has_nan": has_nan or bool(not np.all(np.isfinite(transform))),
        "world_translation": [float(v) for v in transform[:3, 3]],
        "world_quat_wxyz": [float(v) for v in quat],
        "local_bounds": bounds,
        "error": error,
    }


def _load_physics_record(run_dir: Path, item: dict[str, object]) -> dict[str, object]:
    physics = item.get("physics") if isinstance(item.get("physics"), dict) else {}
    record = dict(physics)
    physics_path = record.get("path")
    if physics_path:
        path = run_dir / str(physics_path)
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            loaded["path"] = str(physics_path)
            return loaded
    record.setdefault("path", None)
    record.setdefault("mass_kg", item.get("mass_kg"))
    record.setdefault("friction", item.get("friction"))
    record.setdefault("restitution", item.get("restitution", 0.0))
    collision_asset = item.get("collision_asset") if isinstance(item.get("collision_asset"), dict) else {}
    record.setdefault("collision_source", collision_asset.get("source", "legacy_mesh_path_runtime_collision"))
    return record

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


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> list[float]:
    rot = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(rot)))
        if axis == 0:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / scale,
                    0.25 * scale,
                    (rot[0, 1] + rot[1, 0]) / scale,
                    (rot[0, 2] + rot[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[0, 2] - rot[2, 0]) / scale,
                    (rot[0, 1] + rot[1, 0]) / scale,
                    0.25 * scale,
                    (rot[1, 2] + rot[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [
                    (rot[1, 0] - rot[0, 1]) / scale,
                    (rot[0, 2] + rot[2, 0]) / scale,
                    (rot[1, 2] + rot[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return (quat / norm).tolist()


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
import math
from pathlib import Path

import genesis as gs
import numpy as np


def matrix_to_quat_wxyz(matrix):
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (rot[2, 1] - rot[1, 2]) / scale,
            (rot[0, 2] - rot[2, 0]) / scale,
            (rot[1, 0] - rot[0, 1]) / scale,
        ], dtype=np.float64)
    else:
        axis = int(np.argmax(np.diag(rot)))
        if axis == 0:
            scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array([
                (rot[2, 1] - rot[1, 2]) / scale,
                0.25 * scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
            ], dtype=np.float64)
        elif axis == 1:
            scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array([
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                0.25 * scale,
                (rot[1, 2] + rot[2, 1]) / scale,
            ], dtype=np.float64)
        else:
            scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array([
                (rot[1, 0] - rot[0, 1]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
                (rot[1, 2] + rot[2, 1]) / scale,
                0.25 * scale,
            ], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(v) for v in quat / norm)


def tensor_to_array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def tensor_to_vec(value):
    arr = tensor_to_array(value).reshape(-1)
    return [float(v) for v in arr]


def entity_aabb(entity):
    arr = tensor_to_array(entity.get_AABB(allow_fast_approx=False))
    points = arr.reshape(-1, 3)
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    return [[float(v) for v in lo], [float(v) for v in hi]]


def load_object_physics(run_dir, obj):
    physics = dict(obj.get("physics", {}) or {})
    physics_path = physics.get("path")
    if physics_path:
        path = run_dir / physics_path
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            loaded.setdefault("path", physics_path)
            return loaded
    physics.setdefault("mass_kg", obj.get("mass_kg", 0.2))
    physics.setdefault("friction", obj.get("friction", 0.8))
    physics.setdefault("restitution", obj.get("restitution", 0.0))
    collision_asset = obj.get("collision_asset", {}) or {}
    physics.setdefault("collision_source", collision_asset.get("source", "legacy_mesh_path_runtime_collision"))
    return physics


def displacement(a, b):
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(av - bv))


def fall_out(pos, bounds, margin=0.15):
    if not bounds:
        return False
    (x0, y0), (x1, y1) = bounds
    x, y = float(pos[0]), float(pos[1])
    return bool(x < x0 - margin or x > x1 + margin or y < y0 - margin or y > y1 + margin)


def aabb_fall_out(aabb, bounds, margin=0.15):
    if not bounds:
        return False
    center = [
        (float(aabb[0][0]) + float(aabb[1][0])) * 0.5,
        (float(aabb[0][1]) + float(aabb[1][1])) * 0.5,
    ]
    return fall_out(center, bounds, margin=margin)


def visual_physics_coherence():
    return {
        "status": "partial",
        "visual_assets_referenced_by_runtime": False,
        "collision_assets_referenced_by_runtime": True,
        "background_asset_referenced_by_runtime": False,
        "reason": (
            "Genesis interactive settle runtime uses collision meshes and support surface only; "
            "foreground visual meshes and native background visuals are not referenced by this runtime."
        ),
    }


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
    support_plane = manifest.get("support_plane", {})
    table_collision_mesh_path = support_plane.get("table_collision_mesh_path")
    table_collision_pos = tuple(float(v) for v in support_plane.get("table_collision_pos_world", (0.0, 0.0, -0.02)))
    table_collision_size = support_plane.get("table_collision_size_xyz")
    support_geometry_type = str(support_plane.get("table_collision_geometry_type") or "")
    use_support_mesh = support_geometry_type in {"polygon_slab", "convex_hull_slab"} and table_collision_mesh_path
    support_surface_runtime = {
        "mesh_path": table_collision_mesh_path,
        "geometry_type": support_geometry_type or "unknown",
        "used_polygon_slab_mesh": bool(use_support_mesh),
        "used_box_proxy": bool((not use_support_mesh) and table_collision_size),
        "support_mesh_file_meshes_are_zup": bool(use_support_mesh),
        "table_collision_final": bool(support_plane.get("table_collision_final", False)),
        "table_collision_source_backend": support_plane.get("table_collision_source_backend"),
    }
    if use_support_mesh:
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(args.run_dir / table_collision_mesh_path),
                pos=table_collision_pos,
                quat=tuple(float(v) for v in support_plane.get("table_collision_quat_wxyz", (1.0, 0.0, 0.0, 0.0))),
                fixed=True,
                collision=True,
                convexify=False,
                file_meshes_are_zup=True,
            ),
            material=gs.materials.Rigid(friction=0.9, rho=None),
        )
    elif table_collision_size:
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=table_collision_pos,
                size=tuple(float(v) for v in table_collision_size),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(friction=0.9, rho=None),
        )
    elif table_collision_mesh_path:
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(args.run_dir / table_collision_mesh_path),
                pos=table_collision_pos,
                quat=tuple(float(v) for v in support_plane.get("table_collision_quat_wxyz", (1.0, 0.0, 0.0, 0.0))),
                fixed=True,
                convexify=True,
                decimate=True,
                decimate_face_num=500,
            ),
            material=gs.materials.Rigid(friction=0.9, rho=None),
        )
    else:
        scene.add_entity(gs.morphs.Plane())
    entities = []
    collision_sources_by_object = {}
    genesis_collision_sources_by_object = {}
    genesis_collision_status_by_object = {}
    for obj in manifest.get("objects", []):
        collision_asset = obj.get("collision_asset", {})
        if not collision_asset.get("path"):
            legacy_mesh_path = obj.get("mesh_path")
            collision_asset = {
                "path": legacy_mesh_path,
                "source": "legacy_mesh_path_runtime_collision",
                "status": "runtime_substitution",
            }
        collision_mesh_file = args.run_dir / collision_asset["path"]
        physics = load_object_physics(args.run_dir, obj)
        T = obj["T_object_to_world"]
        pos = (float(T[0][3]), float(T[1][3]), float(T[2][3]))
        quat = matrix_to_quat_wxyz(T)
        collision_source = str(collision_asset.get("source") or physics.get("collision_source") or "unknown_collision_source")
        genesis_collision_status = "runtime_convexification_enabled"
        genesis_collision_source = f"{collision_source}+genesis_runtime_convexification"
        object_id = str(obj["object_id"])
        collision_sources_by_object[object_id] = collision_source
        genesis_collision_sources_by_object[object_id] = genesis_collision_source
        genesis_collision_status_by_object[object_id] = genesis_collision_status
        entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(collision_mesh_file),
                pos=pos,
                quat=quat,
                fixed=False,
                convexify=True,
                decimate=True,
                align=False,
                decimate_face_num=500,
            ),
            material=gs.materials.Rigid(friction=float(physics.get("friction", obj.get("friction", 0.8))), rho=None),
        )
        entities.append((entity, obj, list(pos), list(quat), collision_source, genesis_collision_source, genesis_collision_status, str(collision_asset["path"]), physics))
    scene.build()
    for entity, obj, _initial_pos, _initial_quat, _collision_source, _genesis_collision_source, _genesis_collision_status, _collision_asset_path, physics in entities:
        if physics.get("mass_kg"):
            entity.set_mass(float(physics["mass_kg"]))
        if physics.get("friction"):
            entity.set_friction(float(physics["friction"]))
    for _ in range(max(0, int(args.settle_steps))):
        scene.step()
    qa_dir = args.run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    final_pose_by_object = {}
    table_bounds = support_plane.get("table_bounds_world_xy")
    max_displacement = 0.0
    max_penetration = 0.0
    fall_out_detected = False
    nan_detected = False
    pose_delta_threshold_m = 0.02
    pose_written_back = False
    requires_viewer_export_delta_display = False
    settled_pose_delta_objects = []
    for entity, obj, initial_pos, initial_quat, collision_source, genesis_collision_source, genesis_collision_status, collision_asset_path, physics in entities:
        object_id = str(obj["object_id"])
        final_pos = tensor_to_vec(entity.get_pos(relative=True))[:3]
        final_quat = tensor_to_vec(entity.get_quat(relative=True))[:4]
        final_aabb = entity_aabb(entity)
        translation_delta = [
            float(v)
            for v in (np.asarray(final_pos, dtype=np.float64) - np.asarray(initial_pos, dtype=np.float64)).reshape(-1)[:3]
        ]
        disp = float(np.linalg.norm(np.asarray(translation_delta, dtype=np.float64)))
        vertical_delta = float(final_pos[2] - initial_pos[2])
        penetration = max(0.0, -float(final_aabb[0][2]))
        object_fall_out = aabb_fall_out(final_aabb, table_bounds)
        object_has_nan = not np.all(np.isfinite(np.asarray(final_pos + final_quat + final_aabb[0] + final_aabb[1], dtype=np.float64)))
        pose_delta_exceeds_threshold = disp > pose_delta_threshold_m
        pose_delta_status = "moved_not_written_back" if pose_delta_exceeds_threshold and not pose_written_back else "within_threshold"
        viewer_export_action = (
            "display_initial_and_settled_pose_or_write_back"
            if pose_delta_exceeds_threshold and not pose_written_back
            else "no_delta_display_required"
        )
        max_displacement = max(max_displacement, disp)
        max_penetration = max(max_penetration, penetration)
        fall_out_detected = fall_out_detected or object_fall_out
        nan_detected = nan_detected or object_has_nan
        requires_viewer_export_delta_display = requires_viewer_export_delta_display or (pose_delta_exceeds_threshold and not pose_written_back)
        settled_pose_delta_objects.append({
            "object_id": object_id,
            "initial_pos": initial_pos,
            "final_pos": final_pos,
            "translation_delta_m": translation_delta,
            "displacement_m": disp,
            "threshold_m": float(pose_delta_threshold_m),
            "status": pose_delta_status,
            "pose_written_back": bool(pose_written_back),
            "viewer_export_action": viewer_export_action,
        })
        final_pose_by_object[object_id] = {
            "initial_position": initial_pos,
            "initial_quat_wxyz": initial_quat,
            "final_position": final_pos,
            "final_quat_wxyz": final_quat,
            "final_aabb_world": final_aabb,
            "translation_delta_m": translation_delta,
            "displacement_m": disp,
            "vertical_displacement_m": vertical_delta,
            "threshold_m": float(pose_delta_threshold_m),
            "pose_delta_status": pose_delta_status,
            "pose_written_back": bool(pose_written_back),
            "viewer_export_action": viewer_export_action,
            "penetration_depth_m": penetration,
            "fall_out": object_fall_out,
            "nan_detected": object_has_nan,
            "collision_asset_path": collision_asset_path,
            "collision_source": collision_source,
            "genesis_collision_source": genesis_collision_source,
            "genesis_collision_status": genesis_collision_status,
            "mass_kg": float(physics.get("mass_kg", obj.get("mass_kg", 0.0))),
            "friction": float(physics.get("friction", obj.get("friction", 0.0))),
            "restitution": float(physics.get("restitution", 0.0)),
        }
    stability_thresholds = {
        "max_penetration_depth_m": 0.005,
        "max_displacement_m": 0.5,
    }
    stability_status = (
        "passed"
        if (
            not nan_detected
            and not fall_out_detected
            and max_penetration <= stability_thresholds["max_penetration_depth_m"]
            and max_displacement <= stability_thresholds["max_displacement_m"]
        )
        else "failed"
    )
    coherence = visual_physics_coherence()
    settle_report = {
        "status": "completed",
        "stability_status": stability_status,
        "stability_thresholds": stability_thresholds,
        "settle_steps": int(args.settle_steps),
        "object_count": len(entities),
        "nan_detected": nan_detected,
        "support_plane": manifest.get("support_plane", {"status": "absent"}),
        "support_surface_runtime": support_surface_runtime,
        "table_collision_mesh_path": table_collision_mesh_path,
        "collision_sources_by_object": collision_sources_by_object,
        "genesis_collision_sources_by_object": genesis_collision_sources_by_object,
        "genesis_collision_status_by_object": genesis_collision_status_by_object,
        "final_pose_by_object": final_pose_by_object,
        "pose_delta_threshold_m": float(pose_delta_threshold_m),
        "pose_written_back": bool(pose_written_back),
        "requires_viewer_export_delta_display": bool(requires_viewer_export_delta_display),
        "settled_pose_delta_report": "qa/settled_pose_delta_report.json",
        "visual_physics_coherence": coherence,
        "max_displacement_m": max_displacement,
        "max_penetration_depth_m": max_penetration,
        "fall_out_detected": fall_out_detected,
    }
    settled_pose_delta_report = {
        "status": "partial" if requires_viewer_export_delta_display or coherence["status"] == "partial" else "passed",
        "settle_report_path": "qa/genesis_settle_report.json",
        "threshold_m": float(pose_delta_threshold_m),
        "max_displacement_m": float(max_displacement),
        "pose_written_back": bool(pose_written_back),
        "writeback_policy": "not_written_back_current_contract",
        "requires_viewer_export_delta_display": bool(requires_viewer_export_delta_display),
        "objects": settled_pose_delta_objects,
        "visual_physics_coherence": coherence,
    }
    (qa_dir / "genesis_settle_report.json").write_text(
        json.dumps(settle_report, indent=2),
        encoding="utf-8",
    )
    (qa_dir / "settled_pose_delta_report.json").write_text(
        json.dumps(settled_pose_delta_report, indent=2),
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
