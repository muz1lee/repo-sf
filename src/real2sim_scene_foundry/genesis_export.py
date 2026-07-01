"""Genesis export helpers for simulator export bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sim_export_manifest import build_sim_export_manifest, write_sim_export_manifest
from .usd_export import build_runtime_asset_report


@dataclass(frozen=True)
class GenesisExportResult:
    script_path: Path
    report_path: Path
    report: dict[str, Any]


def export_genesis_scene(run_dir: str | Path, *, settle_steps: int = 100) -> GenesisExportResult:
    run = Path(run_dir)
    export_dir = run / "exports"
    qa_dir = run / "qa"
    export_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    script_path = export_dir / "genesis_scene.py"
    script_path.write_text(_genesis_script(), encoding="utf-8")
    manifest = build_sim_export_manifest(run).to_dict()
    report = {
        "status": "script_written",
        "script_path": "exports/genesis_scene.py",
        "settle_steps": int(settle_steps),
        "run_command": (
            "/mnt/workspace/wenqian/knowin-world/.venv/bin/python "
            f"{script_path} --run-dir {run} --settle-steps {int(settle_steps)} --backend cpu --no-viewer"
        ),
        "loads_manifest": "sim_export_manifest.json",
        "loads_physical_scene": True,
        "writes_settle_report": "qa/genesis_settle_report.json",
        "writes_settled_pose_delta_report": "qa/settled_pose_delta_report.json",
    }
    report.update(
        build_runtime_asset_report(
            manifest,
            backend="genesis",
            visual_assets_referenced=False,
            collision_assets_referenced=True,
            background_asset_referenced=False,
            debug_proxy_visible=False,
            evidence=(
                "Genesis loader uses explicit collision meshes and support surface for physics settle; "
                "foreground visual meshes and native 3DGS background are not loaded by this runtime script."
            ),
        )
    )
    report["visual_physics_coherence"] = _genesis_visual_physics_coherence()
    report_path = qa_dir / "genesis_export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sim_export_manifest(run)
    return GenesisExportResult(script_path=script_path, report_path=report_path, report=report)


def apply_genesis_settle_writeback(run_dir: str | Path, *, writeback: bool = False) -> dict[str, Any]:
    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "scene_manifest.json"
    settle_report = _load_json(qa_dir / "genesis_settle_report.json")
    delta_report = _load_json(qa_dir / "settled_pose_delta_report.json")
    reasons: list[str] = []
    if settle_report.get("status") != "completed":
        reasons.append("genesis_settle_not_completed")
    if settle_report.get("stability_status") not in {None, "passed"}:
        reasons.append("genesis_settle_failed")
    if not delta_report:
        reasons.append("settled_pose_delta_report_missing")
    manifest = _load_json(manifest_path)
    if not manifest:
        reasons.append("scene_manifest_missing")

    object_transforms = _settled_object_transforms(delta_report)
    if not object_transforms:
        reasons.append("settled_object_transforms_missing")

    status = "completed" if not reasons else "blocked"
    pose_written_back = False
    if manifest and object_transforms:
        settled_manifest = json.loads(json.dumps(manifest))
        for obj in settled_manifest.get("objects", []):
            object_id = str(obj.get("object_id"))
            transform = object_transforms.get(object_id)
            if transform is not None:
                obj.setdefault("initial_T_object_to_world", obj.get("T_object_to_world"))
                obj["T_object_to_world"] = transform
                obj["settled_pose_source"] = "genesis_settle"
        (run / "scene_manifest.settled.json").write_text(json.dumps(settled_manifest, indent=2), encoding="utf-8")
        if writeback and status == "completed":
            manifest_path.write_text(json.dumps(settled_manifest, indent=2), encoding="utf-8")
            for obj in settled_manifest.get("objects", []):
                object_id = str(obj.get("object_id"))
                transform = object_transforms.get(object_id)
                if transform is None:
                    continue
                pose_path = run / "objects" / object_id / "pose.json"
                pose = _load_json(pose_path)
                pose["object_id"] = object_id
                pose.setdefault("initial_T_object_to_world", obj.get("initial_T_object_to_world"))
                pose["T_object_to_world"] = transform
                pose["settled_pose_source"] = "genesis_settle"
                pose_path.parent.mkdir(parents=True, exist_ok=True)
                pose_path.write_text(json.dumps(pose, indent=2), encoding="utf-8")
            pose_written_back = True

    report = {
        "version": 1,
        "status": status,
        "blocking_reasons": _dedupe(reasons),
        "writeback_requested": bool(writeback),
        "pose_written_back": bool(pose_written_back),
        "settled_manifest_path": "scene_manifest.settled.json" if (run / "scene_manifest.settled.json").is_file() else None,
        "settle_report_path": "qa/genesis_settle_report.json" if (qa_dir / "genesis_settle_report.json").is_file() else None,
        "settled_pose_delta_report_path": "qa/settled_pose_delta_report.json" if (qa_dir / "settled_pose_delta_report.json").is_file() else None,
        "object_count": len(object_transforms),
        "report_path": "qa/genesis_settle_writeback_report.json",
    }
    (qa_dir / "genesis_settle_writeback_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _settled_object_transforms(delta_report: dict[str, Any]) -> dict[str, list[list[float]]]:
    raw = delta_report.get("objects", [])
    if isinstance(raw, dict):
        iterable = [dict(value, object_id=key) if isinstance(value, dict) else {"object_id": key} for key, value in raw.items()]
    elif isinstance(raw, list):
        iterable = [item for item in raw if isinstance(item, dict)]
    else:
        iterable = []
    transforms: dict[str, list[list[float]]] = {}
    for item in iterable:
        object_id = str(item.get("object_id") or "")
        transform = item.get("settled_T_object_to_world") or item.get("T_object_to_world_settled") or item.get("T_settled_object_to_world")
        if object_id and _is_transform(transform):
            transforms[object_id] = transform
    return transforms


def _is_transform(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(isinstance(row, list) and len(row) == 4 for row in value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _genesis_visual_physics_coherence() -> dict[str, Any]:
    return {
        "status": "partial",
        "visual_assets_referenced_by_runtime": False,
        "collision_assets_referenced_by_runtime": True,
        "background_asset_referenced_by_runtime": False,
        "reason": (
            "Genesis settle runtime uses collision meshes and support surface only; "
            "foreground visual meshes and native background visuals are not referenced by this runtime."
        ),
    }


def _genesis_script() -> str:
    return '''#!/usr/bin/env python3
"""Load a real2sim simulator export manifest in Genesis and run settle QA."""

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
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
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
    return [float(v) for v in tensor_to_array(value).reshape(-1)]


def entity_pos(entity):
    if hasattr(entity, "get_pos"):
        return tensor_to_vec(entity.get_pos())[:3]
    return [0.0, 0.0, 0.0]


def entity_has_nan(entity):
    values = entity_pos(entity)
    return bool(not np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def fall_out(pos, bounds, margin=0.15):
    if not bounds:
        return False
    (x0, y0), (x1, y1) = bounds
    x, y = float(pos[0]), float(pos[1])
    return bool(x < x0 - margin or x > x1 + margin or y < y0 - margin or y > y1 + margin)


def fall_below_support(pos, support_top_z, margin=0.25):
    return bool(float(pos[2]) < float(support_top_z) - float(margin))


def asset_ref(obj, key, referenced_by_runtime):
    asset = obj.get(key, {}) or {}
    return {
        "object_id": obj.get("object_id"),
        "path": asset.get("path"),
        "status": asset.get("status", "unknown"),
        "source": asset.get("source", "unknown"),
        "diagnostic_only": bool(asset.get("diagnostic_only", False)),
        "referenced_by_runtime": bool(referenced_by_runtime and asset.get("path")),
    }


def native_3dgs_status(background):
    if background.get("gaussian_splat_config_path") or background.get("gaussian_splat_checkpoint_path"):
        return "unsupported_unverified"
    return "not_requested"


def dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def background_completion_report(background, native_status):
    visual_background = background.get("visual_asset", {})
    registration = background.get("registration", {})
    external_render = background.get("external_render", {})
    reasons = []
    caveats = []
    visual_status = str(visual_background.get("status") or "missing_background_visual")
    if visual_status != "ready":
        reasons.append(visual_status)
    if registration.get("status") != "registered":
        reasons.append("background_unregistered")
    reasons.extend(str(item) for item in registration.get("blocking_reasons", []))
    external_sidecar = (
        visual_background.get("source_kind") == "external_3dgs_renderer"
        or external_render.get("backend") == "external_3dgs_renderer"
    )
    simulator_native = bool(visual_background.get("simulator_native") and visual_background.get("native_runtime_verified"))
    if external_sidecar:
        caveats.append("external_3dgs_render_sidecar_not_native_simulator_background")
    if visual_background.get("has_3dgs_sidecar") and not simulator_native:
        caveats.append("native_3dgs_runtime_not_verified")
    if native_status not in {"not_requested", "native_3dgs_loaded", "registered_3dgs_loaded", "verified_native_3dgs"}:
        caveats.append("native_3dgs_status:" + str(native_status))
    blocking_reasons = dedupe(reasons)
    caveats = dedupe(caveats)
    status = "blocked" if blocking_reasons else "partial" if caveats else "passed"
    return {
        "status": status,
        "complete": status == "passed",
        "blocking_reasons": blocking_reasons,
        "caveats": caveats,
        "background_asset_referenced": False,
        "external_sidecar": bool(external_sidecar),
        "external_render_verified": bool(visual_background.get("external_render_verified")),
        "simulator_native": simulator_native,
        "registration_status": registration.get("status", "unknown"),
        "native_3dgs_status": native_status,
    }


def visual_physics_coherence():
    return {
        "status": "partial",
        "visual_assets_referenced_by_runtime": False,
        "collision_assets_referenced_by_runtime": True,
        "background_asset_referenced_by_runtime": False,
        "reason": (
            "Genesis settle runtime uses collision meshes and support surface only; "
            "foreground visual meshes and native background visuals are not referenced by this runtime."
        ),
    }


def runtime_asset_report(manifest):
    background = manifest.get("background", {})
    visual_background = background.get("visual_asset", {})
    registration = background.get("registration", {})
    native_status = native_3dgs_status(background)
    return {
        "object_visual_refs": [asset_ref(obj, "visual_asset", False) for obj in manifest.get("objects", [])],
        "object_collision_refs": [asset_ref(obj, "collision_asset", True) for obj in manifest.get("objects", [])],
        "debug_proxy_refs": [asset_ref(obj, "debug_proxy", False) for obj in manifest.get("objects", [])],
        "debug_proxy_visible": False,
        "background_asset": {
            "role": visual_background.get("role", "background_visual"),
            "path": visual_background.get("path"),
            "status": visual_background.get("status", "unknown"),
            "source_kind": visual_background.get("source_kind", "unknown"),
            "source_backend": background.get("source_backend", "unknown"),
            "diagnostic_only": bool(visual_background.get("diagnostic_only", False)),
            "registration_path": registration.get("path"),
            "registration_status": registration.get("status", "unknown"),
            "gaussian_splat_config_path": background.get("gaussian_splat_config_path"),
            "gaussian_splat_checkpoint_path": background.get("gaussian_splat_checkpoint_path"),
            "referenced_by_runtime": False,
        },
        "background_runtime": {
            "backend": "genesis",
            "native_3dgs_supported": False,
            "native_3dgs_status": native_status,
            "background_asset_referenced": False,
            "evidence": "Genesis settle script loads collision meshes and support surface only; native 3DGS rendering is not integrated or verified.",
        },
        "background_completion": background_completion_report(background, native_status),
        "visual_physics_coherence": visual_physics_coherence(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir
    manifest = json.loads((run_dir / "sim_export_manifest.json").read_text(encoding="utf-8"))

    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu)
    scene = gs.Scene(
        show_viewer=not args.no_viewer,
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.0, -1.8, 1.2), camera_lookat=(0.0, 0.0, 0.2), camera_fov=45),
    )
    support = manifest.get("support_surface", {})
    support_top_z = float(support.get("height_world_m") or 0.0)
    support_size = support.get("table_collision_size_xyz")
    support_pos = support.get("table_collision_pos_world")
    support_mesh_path = support.get("mesh_path")
    support_geometry_type = str(support.get("table_collision_geometry_type") or "")
    use_support_mesh = support_geometry_type in {"polygon_slab", "convex_hull_slab"} and support_mesh_path
    support_surface_runtime = {
        "mesh_path": support_mesh_path,
        "geometry_type": support_geometry_type or "unknown",
        "used_polygon_slab_mesh": bool(use_support_mesh),
        "used_box_proxy": bool((not use_support_mesh) and support_size and support_pos),
        "support_mesh_file_meshes_are_zup": bool(use_support_mesh),
        "table_collision_final": bool(support.get("table_collision_final", False)),
        "table_collision_source_backend": support.get("table_collision_source_backend"),
    }
    if use_support_mesh:
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(run_dir / support_mesh_path),
                pos=tuple(float(v) for v in support_pos or (0.0, 0.0, 0.0)),
                fixed=True,
                collision=True,
                convexify=False,
                file_meshes_are_zup=True,
            ),
            material=gs.materials.Rigid(friction=1.0),
        )
    elif support_size and support_pos:
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=tuple(float(v) for v in support_pos),
                size=tuple(float(v) for v in support_size),
                fixed=True,
                collision=True,
            ),
            material=gs.materials.Rigid(friction=1.0),
        )
    elif support_mesh_path:
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(run_dir / support_mesh_path),
                pos=tuple(float(v) for v in support_pos or (0.0, 0.0, 0.0)),
                fixed=True,
                collision=True,
                convexify=False,
            ),
            material=gs.materials.Rigid(friction=1.0),
        )

    entities = {}
    initial_pose_by_object = {}
    bounds = support.get("table_bounds_world_xy")
    for obj in manifest.get("objects", []):
        collision_asset = obj["collision_asset"]
        physics = obj["physics"]
        if collision_asset.get("status") != "ready" or not collision_asset.get("path"):
            continue
        transform = np.asarray(obj["pose"]["T_object_to_world"], dtype=np.float64)
        asset_scale = float(obj.get("pose", {}).get("asset_scale", obj.get("asset_scale", 1.0)) or 1.0)
        pos = tuple(float(v) for v in transform[:3, 3])
        quat = matrix_to_quat_wxyz(transform)
        entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(run_dir / collision_asset["path"]),
                pos=pos,
                quat=quat,
                scale=asset_scale,
                fixed=False,
                collision=True,
                convexify=True,
            ),
            material=gs.materials.Rigid(friction=float(physics.get("friction", 0.8))),
        )
        entities[obj["object_id"]] = entity
        initial_pose_by_object[obj["object_id"]] = {
            "pos": list(pos),
            "quat_wxyz": list(quat),
            "mass_kg": float(physics.get("mass_kg", 0.25)),
            "asset_scale": float(asset_scale),
        }

    scene.build()
    for object_id, entity in entities.items():
        if hasattr(entity, "set_mass"):
            entity.set_mass(float(initial_pose_by_object[object_id]["mass_kg"]))
    for _ in range(int(args.settle_steps)):
        scene.step()

    object_reports = []
    nan_detected = False
    fall_out_detected = False
    fall_below_support_detected = False
    excessive_displacement_detected = False
    max_allowed_displacement_m = 0.5
    pose_delta_threshold_m = 0.02
    pose_written_back = False
    requires_viewer_export_delta_display = False
    max_displacement_m = 0.0
    final_pose_by_object = {}
    settled_pose_delta_objects = []
    for object_id, entity in entities.items():
        final_pos = entity_pos(entity)
        initial_pos = initial_pose_by_object[object_id]["pos"]
        translation_delta = [
            float(v)
            for v in (np.asarray(final_pos, dtype=np.float64) - np.asarray(initial_pos, dtype=np.float64)).reshape(-1)[:3]
        ]
        displacement = float(np.linalg.norm(np.asarray(translation_delta, dtype=np.float64)))
        has_nan = entity_has_nan(entity)
        fell_out = fall_out(final_pos, bounds)
        fell_below = fall_below_support(final_pos, support_top_z)
        excessive_displacement = displacement > max_allowed_displacement_m
        pose_delta_exceeds_threshold = displacement > pose_delta_threshold_m
        pose_delta_status = "moved_not_written_back" if pose_delta_exceeds_threshold and not pose_written_back else "within_threshold"
        viewer_export_action = (
            "display_initial_and_settled_pose_or_write_back"
            if pose_delta_exceeds_threshold and not pose_written_back
            else "no_delta_display_required"
        )
        nan_detected = nan_detected or has_nan
        fall_out_detected = fall_out_detected or fell_out
        fall_below_support_detected = fall_below_support_detected or fell_below
        excessive_displacement_detected = excessive_displacement_detected or excessive_displacement
        requires_viewer_export_delta_display = requires_viewer_export_delta_display or (pose_delta_exceeds_threshold and not pose_written_back)
        max_displacement_m = max(max_displacement_m, displacement)
        final_pose_by_object[object_id] = {"pos": final_pos}
        settled_pose_delta_objects.append({
            "object_id": object_id,
            "initial_pos": initial_pos,
            "final_pos": final_pos,
            "translation_delta_m": translation_delta,
            "displacement_m": displacement,
            "threshold_m": float(pose_delta_threshold_m),
            "status": pose_delta_status,
            "pose_written_back": bool(pose_written_back),
            "viewer_export_action": viewer_export_action,
        })
        object_reports.append({
            "object_id": object_id,
            "initial_pos": initial_pos,
            "final_pos": final_pos,
            "translation_delta_m": translation_delta,
            "displacement_m": displacement,
            "threshold_m": float(pose_delta_threshold_m),
            "pose_delta_status": pose_delta_status,
            "pose_written_back": bool(pose_written_back),
            "viewer_export_action": viewer_export_action,
            "nan_detected": has_nan,
            "fall_out_detected": fell_out,
            "fall_below_support_detected": fell_below,
            "excessive_displacement_detected": excessive_displacement,
        })

    stability_status = "passed" if not nan_detected and not fall_out_detected and not fall_below_support_detected and not excessive_displacement_detected else "failed"
    report = {
        "status": "completed",
        "stability_status": stability_status,
        "settle_steps": int(args.settle_steps),
        "nan_detected": bool(nan_detected),
        "fall_out_detected": bool(fall_out_detected),
        "fall_below_support_detected": bool(fall_below_support_detected),
        "excessive_displacement_detected": bool(excessive_displacement_detected),
        "max_allowed_displacement_m": float(max_allowed_displacement_m),
        "pose_delta_threshold_m": float(pose_delta_threshold_m),
        "pose_written_back": bool(pose_written_back),
        "requires_viewer_export_delta_display": bool(requires_viewer_export_delta_display),
        "max_displacement_m": float(max_displacement_m),
        "max_penetration_depth_m": 0.0,
        "support_top_z_m": float(support_top_z),
        "support_surface_runtime": support_surface_runtime,
        "initial_pose_by_object": initial_pose_by_object,
        "final_pose_by_object": final_pose_by_object,
        "objects": object_reports,
        "settled_pose_delta_report": "qa/settled_pose_delta_report.json",
    }
    runtime_report = runtime_asset_report(manifest)
    report.update(runtime_report)
    visual_coherence = runtime_report["visual_physics_coherence"]
    settled_pose_delta_report = {
        "status": "partial" if requires_viewer_export_delta_display or visual_coherence["status"] == "partial" else "passed",
        "settle_report_path": "qa/genesis_settle_report.json",
        "threshold_m": float(pose_delta_threshold_m),
        "max_displacement_m": float(max_displacement_m),
        "pose_written_back": bool(pose_written_back),
        "writeback_policy": "not_written_back_current_contract",
        "requires_viewer_export_delta_display": bool(requires_viewer_export_delta_display),
        "objects": settled_pose_delta_objects,
        "visual_physics_coherence": visual_coherence,
    }
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "genesis_settle_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (qa_dir / "settled_pose_delta_report.json").write_text(json.dumps(settled_pose_delta_report, indent=2), encoding="utf-8")
    return 0 if stability_status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''
