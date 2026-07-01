"""Strict simulator export manifest builders and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .pose_refinement import measure_object_support_alignment


MIN_TABLETOP_SELECTED_CANDIDATE_RATIO = 0.08
MIN_TABLE_COLLISION_EXTENT_M = 0.12
SCHEMA_NAME = "real2sim_scene_foundry.sim_export_manifest"
SCHEMA_VERSION = 1
READY_STATUS = "ready"
BLOCKED_STATUS = "blocked"

SIM_EXPORT_MANIFEST_SCHEMA: dict[str, Any] = {
    "schema_name": SCHEMA_NAME,
    "version": SCHEMA_VERSION,
    "required_top_level_keys": [
        "schema_name",
        "version",
        "run_dir",
        "source_scene_manifest",
        "coordinate_frames",
        "objects",
        "background",
        "support_surface",
        "exports",
        "qa",
        "status",
        "blocking_reasons",
    ],
    "object_required_keys": [
        "object_id",
        "label",
        "mask_path",
        "crop_path",
        "visual_asset",
        "collision_asset",
        "debug_proxy",
        "pose",
        "physics",
    ],
    "blocking_statuses": [
        "blocked_proxy_visual_asset",
        "blocked_legacy_bbox_visual_reference",
        "missing_visual_asset",
        "missing_collision_mesh",
        "missing_background_visual",
        "blocked_background_proxy_visual",
        "background_3dgs_runtime_not_verified",
        "background_external_3dgs_render_failed",
        "background_unregistered",
        "missing_support_surface_collision",
        "support_surface_is_proxy",
        "table_collision_not_tabletop_mask_derived",
        "table_collision_projection_failed",
        "table_collision_missing_visual_qa",
        "table_collision_box_proxy",
        "table_collision_support_coverage_too_small",
        "table_collision_extent_too_small",
        "pose_needs_manual_refine",
        "pose_support_penetration",
        "pose_support_geometry_unreadable",
        "missing_backend_export",
    ],
}


@dataclass(frozen=True)
class SimExportManifest:
    run_dir: str
    source_scene_manifest: str
    coordinate_frames: dict[str, str]
    objects: list[dict[str, Any]]
    background: dict[str, Any]
    support_surface: dict[str, Any]
    exports: dict[str, dict[str, Any]]
    qa: dict[str, str]
    status: str
    blocking_reasons: list[str]
    version: int = SCHEMA_VERSION
    schema_name: str = SCHEMA_NAME

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_name": self.schema_name,
            "version": self.version,
            "run_dir": self.run_dir,
            "source_scene_manifest": self.source_scene_manifest,
            "coordinate_frames": self.coordinate_frames,
            "objects": self.objects,
            "background": self.background,
            "support_surface": self.support_surface,
            "exports": self.exports,
            "qa": self.qa,
            "status": self.status,
            "blocking_reasons": self.blocking_reasons,
        }
        validate_manifest_dict(data)
        return data


def build_sim_export_manifest(run_dir: str | Path) -> SimExportManifest:
    run = Path(run_dir)
    scene_manifest_path = run / "scene_manifest.json"
    scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    blocking_reasons: list[str] = []

    support_plane_data = scene_manifest.get("support_plane", {})
    objects = [_object_record(run, item, support_plane_data, blocking_reasons) for item in scene_manifest.get("objects", [])]
    background = _background_record(run, scene_manifest.get("background", {}), blocking_reasons)
    support_surface = _support_surface_record(run, scene_manifest.get("support_plane", {}), blocking_reasons)
    exports = _exports_record(run, blocking_reasons)
    coordinate_frames = dict(scene_manifest.get("coordinate_frames", {}))
    coordinate_frames.setdefault("camera", "opencv_x_right_y_down_z_forward_meters")
    coordinate_frames.setdefault("world", "z_up_ground_plane_meters")

    reasons = _dedupe(blocking_reasons)
    return SimExportManifest(
        run_dir=str(run),
        source_scene_manifest="scene_manifest.json",
        coordinate_frames=coordinate_frames,
        objects=objects,
        background=background,
        support_surface=support_surface,
        exports=exports,
        qa={
            "qa_report": "qa/qa_report.json",
            "sim_export_report": "qa/sim_export_report.json",
            "genesis_settle_report": "qa/genesis_settle_report.json",
            "isaac_load_report": "qa/isaac_load_report.json",
            "render_export": "qa/render_export.png",
            "render_vs_input_overlay": "qa/render_vs_input_overlay.png",
            "object_projection_overlay": "qa/object_projection_overlay.png",
            "background_registration_overlay": "qa/background_registration_overlay.png",
            "table_collision_overlay": "qa/table_collision_overlay.png",
            "table_collision_report": "qa/table_collision_report.json",
        },
        status="passed" if not reasons else BLOCKED_STATUS,
        blocking_reasons=reasons,
    )


def write_sim_export_manifest(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    manifest = build_sim_export_manifest(run)
    path = run / "sim_export_manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def load_sim_export_manifest(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    path = run / "sim_export_manifest.json"
    if not path.is_file():
        path = write_sim_export_manifest(run)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_dict(data)
    return data


def validate_manifest_dict(data: dict[str, Any]) -> None:
    for key in SIM_EXPORT_MANIFEST_SCHEMA["required_top_level_keys"]:
        if key not in data:
            raise ValueError(f"sim export manifest missing required key: {key}")
    if data["schema_name"] != SCHEMA_NAME:
        raise ValueError(f"unexpected schema_name: {data['schema_name']}")
    if int(data["version"]) != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema version: {data['version']}")
    if data["status"] == "passed" and data["blocking_reasons"]:
        raise ValueError("passed sim export manifest cannot contain blocking_reasons")
    for obj in data["objects"]:
        for key in SIM_EXPORT_MANIFEST_SCHEMA["object_required_keys"]:
            if key not in obj:
                raise ValueError(f"sim export object missing required key: {key}")


def _object_record(
    run: Path,
    item: dict[str, Any],
    support_plane: dict[str, Any],
    blocking_reasons: list[str],
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    object_dir = run / "objects" / object_id
    visual_asset, debug_proxy = _visual_asset_record(run, object_dir, item, blocking_reasons)
    collision_asset = _collision_asset_record(run, object_dir, object_id, blocking_reasons)
    physics = _physics_record(run, object_dir, item, object_id, blocking_reasons)
    pose = _pose_record(run, object_dir, item, object_id, support_plane, blocking_reasons)
    return {
        "object_id": object_id,
        "label": item.get("label", object_id),
        "mask_path": item.get("mask_path"),
        "crop_path": item.get("crop_path"),
        "visual_asset": visual_asset,
        "collision_asset": collision_asset,
        "debug_proxy": debug_proxy,
        "pose": pose,
        "physics": physics,
        "confidence": item.get("confidence"),
        "source_backend": item.get("source_backend", "unknown"),
    }


def _visual_asset_record(
    run: Path,
    object_dir: Path,
    item: dict[str, Any],
    blocking_reasons: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    object_id = str(item["object_id"])
    visual_glb = object_dir / "visual.glb"
    visual_ply = object_dir / "visual_point_cloud.ply"
    debug_bbox = object_dir / "debug_bbox.glb"
    legacy_mesh_rel = item.get("mesh_path")
    legacy_mesh = run / str(legacy_mesh_rel) if legacy_mesh_rel else None
    legacy_mesh_exists = legacy_mesh is not None and legacy_mesh.is_file()

    debug_proxy = _asset_record(
        "debug_proxy",
        _rel(run, debug_bbox if debug_bbox.is_file() else legacy_mesh if legacy_mesh_exists else None),
        "bbox_or_legacy_mesh",
        READY_STATUS if debug_bbox.is_file() or legacy_mesh_exists else "missing",
        diagnostic_only=True,
    )
    legacy_bbox_visual_reference = (
        legacy_mesh_exists
        and legacy_mesh.name == "mesh_aligned.glb"
        and legacy_mesh != visual_glb
        and _looks_like_bbox_proxy(legacy_mesh)
    )

    if visual_glb.is_file():
        if legacy_bbox_visual_reference:
            blocking_reasons.append(f"blocked_legacy_bbox_visual_reference:{object_id}")
        if _looks_like_bbox_proxy(visual_glb):
            blocking_reasons.append(f"blocked_proxy_visual_asset:{object_id}")
            return (
                _asset_record(
                    "visual_asset",
                    _rel(run, visual_glb),
                    "sam3d_or_visual_reconstruction",
                    "blocked_proxy_visual_asset",
                    diagnostic_only=True,
                ),
                debug_proxy,
            )
        return (
            _asset_record("visual_asset", _rel(run, visual_glb), "sam3d_or_visual_reconstruction", READY_STATUS),
            debug_proxy,
        )
    if visual_ply.is_file():
        quality_report = _load_json(object_dir / "visual_quality_report.json")
        if quality_report.get("status") == "accepted":
            return (
                _asset_record("visual_point_cloud", _rel(run, visual_ply), "point_cloud_visual_candidate", READY_STATUS),
                debug_proxy,
            )
        blocking_reasons.append(f"visual_point_cloud_not_accepted:{object_id}")
        return (
            _asset_record(
                "visual_point_cloud",
                _rel(run, visual_ply),
                "point_cloud_visual_candidate",
                "blocked_visual_point_cloud_candidate",
                diagnostic_only=True,
            ),
            debug_proxy,
        )
    if legacy_mesh_exists and _looks_like_bbox_proxy(legacy_mesh):
        blocking_reasons.append(f"blocked_proxy_visual_asset:{object_id}")
        return (
            _asset_record(
                "debug_proxy",
                _rel(run, legacy_mesh),
                "bbox_from_point_cloud_or_metric_bbox",
                "blocked_proxy_visual_asset",
                diagnostic_only=True,
            ),
            debug_proxy,
        )
    if legacy_mesh_exists:
        blocking_reasons.append(f"missing_explicit_visual_asset:{object_id}")
        return (
            _asset_record(
                "legacy_mesh_candidate",
                _rel(run, legacy_mesh),
                "legacy_scene_manifest_mesh_path",
                "blocked_missing_explicit_visual_asset",
                diagnostic_only=True,
            ),
            debug_proxy,
        )
    blocking_reasons.append(f"missing_visual_asset:{object_id}")
    return (_asset_record("visual_asset", None, "missing", "missing", diagnostic_only=True), debug_proxy)


def _collision_asset_record(run: Path, object_dir: Path, object_id: str, blocking_reasons: list[str]) -> dict[str, Any]:
    collision = object_dir / "collision.glb"
    if collision.is_file():
        return _asset_record("collision_asset", _rel(run, collision), "explicit_collision_mesh", READY_STATUS)
    blocking_reasons.append(f"missing_collision_mesh:{object_id}")
    return _asset_record("collision_asset", None, "missing", "missing", diagnostic_only=True)


def _physics_record(
    run: Path,
    object_dir: Path,
    item: dict[str, Any],
    object_id: str,
    blocking_reasons: list[str],
) -> dict[str, Any]:
    physics_path = object_dir / "physics.json"
    physics = _load_json(physics_path)
    source = str(physics.get("source", "physics_json" if physics else "scene_manifest"))
    mass = float(physics.get("mass_kg", item.get("mass_kg", 0.0)))
    friction = float(physics.get("friction", item.get("friction", 0.0)))
    restitution = float(physics.get("restitution", item.get("restitution", 0.0)))
    status = READY_STATUS
    if mass <= 0.0 or friction <= 0.0 or restitution < 0.0:
        status = "blocked_invalid_physics"
        blocking_reasons.append(f"invalid_physics:{object_id}")
    return {
        "status": status,
        "path": _rel(run, physics_path) if physics_path.is_file() else None,
        "mass_kg": mass,
        "friction": friction,
        "restitution": restitution,
        "source": source,
    }


def _pose_record(
    run: Path,
    object_dir: Path,
    item: dict[str, Any],
    object_id: str,
    support_plane: dict[str, Any],
    blocking_reasons: list[str],
) -> dict[str, Any]:
    report_path = object_dir / "pose_refinement_report.json"
    report = _load_json(report_path)
    transform_camera = item.get("T_object_to_camera")
    transform_world = item.get("T_object_to_world")
    valid = _valid_transform(transform_camera) and _valid_transform(transform_world)
    needs_manual_refine = bool(item.get("needs_manual_refine", False))
    accepted_refinement = _accepted_pose_refinement(report)
    effective_needs_manual_refine = needs_manual_refine and not accepted_refinement
    status = READY_STATUS
    if not valid:
        status = "blocked_invalid_pose_transform"
        blocking_reasons.append(f"invalid_pose_transform:{object_id}")
    if effective_needs_manual_refine:
        status = "blocked_pose_needs_manual_refine"
        blocking_reasons.append(f"pose_needs_manual_refine:{object_id}")
    support_alignment = measure_object_support_alignment(run, item, support_plane)
    if support_alignment.get("status") == "penetrating_support":
        status = "blocked_pose_support_penetration"
        blocking_reasons.append(f"pose_support_penetration:{object_id}")
    elif support_alignment.get("status") == "unreadable_visual_geometry":
        status = "blocked_pose_support_geometry_unreadable"
        blocking_reasons.append(f"pose_support_geometry_unreadable:{object_id}")
    return {
        "status": status,
        "path": "objects/{}/pose.json".format(object_id) if (object_dir / "pose.json").is_file() else None,
        "pose_refinement_report": _rel(run, report_path) if report_path.is_file() else None,
        "T_object_to_camera": transform_camera,
        "T_object_to_world": transform_world,
        "asset_scale": _asset_scale(item),
        "scale_m": float(item.get("scale_m", 0.0)),
        "needs_manual_refine": effective_needs_manual_refine,
        "refinement_status": report.get("status"),
        "pose_source": report.get("source", item.get("source_backend", "scene_manifest")),
        "rotation_source": report.get("rotation_source", item.get("source_backend", "scene_manifest")),
        "translation_source": report.get("translation_source", item.get("source_backend", "scene_manifest")),
        "scale_source": report.get("scale_source", item.get("source_backend", "scene_manifest")),
        "support_alignment": support_alignment,
    }



def _accepted_pose_refinement(report: dict[str, Any]) -> bool:
    if str(report.get("status", "")).lower() != "accepted":
        return False
    allowed_sources = {
        "manual_refined_from_auto",
        "service_pose_refined",
        "foundationpose_service",
        "sam3d_service_pose_refined",
    }
    sources = {
        str(report.get("source", "")).lower(),
        str(report.get("rotation_source", "")).lower(),
        str(report.get("translation_source", "")).lower(),
    }
    return bool(sources & allowed_sources)


def _asset_scale(item: dict[str, Any]) -> float:
    value = float(item.get("asset_scale", 1.0) or 1.0)
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _background_record(run: Path, background: dict[str, Any], blocking_reasons: list[str]) -> dict[str, Any]:
    point_cloud_rel = str(background.get("point_cloud_path") or "background/bg_only_cloud.ply")
    point_cloud = run / point_cloud_rel
    registration_path = run / "background" / "registration.json"
    registration = _load_json(registration_path)
    external_render = _background_external_render_record(run)
    external_render_verified = _background_external_render_verified(run, external_render)
    source_kind = _background_source_kind(background, point_cloud_rel)
    has_3dgs_sidecar = _has_3dgs_sidecar(run, background)
    native_runtime_verified = _background_native_runtime_verified(run)
    visual_status = READY_STATUS if point_cloud.is_file() else "missing"
    visual_path = point_cloud_rel if point_cloud.is_file() else None
    if external_render_verified:
        visual_status = READY_STATUS
        visual_path = external_render["render_path"]
        source_kind = "external_3dgs_renderer"
    elif not point_cloud.is_file():
        blocking_reasons.append("missing_background_visual")
    if not external_render_verified and external_render.get("status") == "blocked":
        reason = str(external_render.get("blocked_reason") or "unknown")
        blocking_reasons.append(f"background_external_3dgs_render_failed:{reason}")
    if not external_render_verified and point_cloud_rel == "scene_cloud.ply":
        visual_status = "blocked_full_scene_cloud"
        blocking_reasons.append("blocked_full_scene_background")
    elif not external_render_verified and point_cloud.is_file() and point_cloud_rel == "background/bg_only_cloud.ply":
        visual_status = "blocked_bg_only_proxy_visual"
        blocking_reasons.append("blocked_background_proxy_visual:bg_only_cloud")
        if has_3dgs_sidecar and not native_runtime_verified:
            blocking_reasons.append("background_3dgs_runtime_not_verified")

    registration_status = str(registration.get("status", "missing"))
    registration_blocking_reasons: list[str] = []
    if registration_status != "registered":
        blocking_reasons.append("background_unregistered")
    registration_sources = registration.get("transform_sources")
    if not isinstance(registration_sources, dict):
        registration_sources = {}
    transform_source = str(registration_sources.get("T_3dgs_world_to_sim_world") or "")
    registration_blocker = _background_3dgs_registration_blocker(
        transform_source=transform_source,
        has_3dgs_sidecar=has_3dgs_sidecar,
        external_render_verified=external_render_verified,
    )
    if registration_blocker:
        blocking_reasons.append(registration_blocker)
        registration_blocking_reasons.append(registration_blocker)
        registration_status = "blocked_3dgs_registration_placeholder"
    return {
        "visual_asset": {
            "role": "background_visual",
            "path": visual_path,
            "source_kind": source_kind,
            "status": visual_status,
            "diagnostic_only": visual_status != READY_STATUS,
            "has_3dgs_sidecar": has_3dgs_sidecar,
            "native_runtime_verified": native_runtime_verified,
            "external_render_verified": external_render_verified,
            "simulator_native": bool(native_runtime_verified),
            "bg_only_proxy_path": point_cloud_rel if point_cloud.is_file() else None,
            "external_render_report": "qa/background_3dgs_render_report.json"
            if (run / "qa" / "background_3dgs_render_report.json").is_file()
            else None,
        },
        "bg_only_image_path": background.get("bg_only_image_path"),
        "foreground_mask_path": background.get("foreground_mask_path"),
        "gaussian_splat_config_path": background.get("gaussian_splat_config_path"),
        "gaussian_splat_checkpoint_path": background.get("gaussian_splat_checkpoint_path"),
        "registration": {
            "path": "background/registration.json" if registration_path.is_file() else None,
            "status": registration_status,
            "T_3dgs_world_to_sim_world": registration.get("T_3dgs_world_to_sim_world"),
            "T_3dgs_world_to_sim_world_source": transform_source or None,
            "scale_source": registration.get("scale_source"),
            "blocking_reasons": registration_blocking_reasons,
        },
        "external_render": {
            "path": "qa/background_3dgs_render_report.json"
            if (run / "qa" / "background_3dgs_render_report.json").is_file()
            else None,
            "status": external_render.get("status"),
            "backend": external_render.get("backend"),
            "render_path": external_render.get("render_path"),
            "simulator_native": external_render.get("simulator_native"),
            "registered_3dgs_rendered": external_render.get("registered_3dgs_rendered"),
            "blocked_reason": external_render.get("blocked_reason"),
        },
        "source_backend": background.get("source_backend", "unknown"),
    }


def _background_3dgs_registration_blocker(
    *,
    transform_source: str,
    has_3dgs_sidecar: bool,
    external_render_verified: bool,
) -> str | None:
    if not has_3dgs_sidecar and not external_render_verified:
        return None
    normalized = transform_source.strip().lower()
    if not normalized:
        return "background_3dgs_registration_placeholder"
    placeholder_markers = ("placeholder", "not_registered", "not_available", "unknown")
    if any(marker in normalized for marker in placeholder_markers):
        return "background_3dgs_registration_placeholder"
    return None


def _support_surface_record(run: Path, support_plane: dict[str, Any], blocking_reasons: list[str]) -> dict[str, Any]:
    mesh_rel = support_plane.get("table_collision_mesh_path")
    mesh_path = run / str(mesh_rel) if mesh_rel else None
    mesh_exists = mesh_path is not None and mesh_path.is_file()
    table_report_path = run / "qa" / "table_collision_report.json"
    table_report = _load_json(table_report_path)
    source_backend = str(support_plane.get("source_backend") or "unknown")
    table_source_backend = str(support_plane.get("table_collision_source_backend") or source_backend)
    geometry_type = str(support_plane.get("table_collision_geometry_type") or table_report.get("geometry_type") or "unknown")
    visual_qa_path = str(
        support_plane.get("table_collision_visual_qa_path") or table_report.get("visual_qa_path") or ""
    )
    blockers: list[str] = []

    if not mesh_exists:
        blockers.append("missing_support_surface_collision")
    proxy_sources = {"background_support_points_rect", "object_bounds_rect", "estimated_support_box_proxy"}
    if source_backend in proxy_sources or table_source_backend in proxy_sources:
        blockers.extend(["support_surface_is_proxy", "table_collision_not_tabletop_mask_derived"])
    derived_sources = {"tabletop_mask", "tabletop_mask_polygon_slab", "tabletop_mask_convex_hull"}
    if table_source_backend not in derived_sources or table_report.get("derived_from_tabletop_mask") is not True:
        blockers.append("table_collision_not_tabletop_mask_derived")
    if geometry_type not in {"polygon_slab", "convex_hull_slab"}:
        blockers.append("table_collision_not_tabletop_mask_derived")
    if not table_report:
        blockers.append("table_collision_missing_visual_qa")
    elif table_report.get("status") != "passed":
        blockers.append("table_collision_projection_failed")
    blockers.extend(_table_collision_metric_blockers(support_plane, table_report))
    if not visual_qa_path or not (run / visual_qa_path).is_file():
        blockers.append("table_collision_missing_visual_qa")
    if mesh_exists and _looks_like_bbox_proxy(mesh_path):
        blockers.extend(["support_surface_is_proxy", "table_collision_box_proxy"])

    blockers = _dedupe(blockers)
    blocking_reasons.extend(blockers)
    support_status = "passed" if mesh_exists and not blockers else "blocked"
    table_collision_final = bool(support_plane.get("table_collision_final") is True and support_status == "passed")
    return {
        "status": READY_STATUS if support_status == "passed" else "blocked" if mesh_exists else "missing",
        "support_surface_status": support_status,
        "source_backend": source_backend,
        "mesh_path": str(mesh_rel) if mesh_exists else None,
        "height_world_m": support_plane.get("height_world_m"),
        "normal_world": support_plane.get("normal_world", [0.0, 0.0, 1.0]),
        "table_collision_pos_world": support_plane.get("table_collision_pos_world"),
        "table_collision_size_xyz": support_plane.get("table_collision_size_xyz"),
        "table_bounds_world_xy": support_plane.get("table_bounds_world_xy"),
        "tabletop_mask_path": support_plane.get("tabletop_mask_path"),
        "table_collision_source_backend": table_source_backend,
        "table_collision_geometry_type": geometry_type,
        "table_collision_final": table_collision_final,
        "table_collision_visual_qa_path": visual_qa_path or None,
        "table_collision_report_path": "qa/table_collision_report.json" if table_report_path.is_file() else None,
        "blocking_reasons": blockers,
    }


def _table_collision_metric_blockers(support_plane: dict[str, Any], table_report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    selected_ratio = _first_float(table_report.get("selected_candidate_ratio"))
    if selected_ratio is not None and selected_ratio < MIN_TABLETOP_SELECTED_CANDIDATE_RATIO:
        blockers.append("table_collision_support_coverage_too_small")
    extent_x = _first_float(table_report.get("polygon_extent_x_m"))
    extent_y = _first_float(table_report.get("polygon_extent_y_m"))
    size = support_plane.get("table_collision_size_xyz")
    if (extent_x is None or extent_y is None) and isinstance(size, (list, tuple)) and len(size) >= 2:
        extent_x = _first_float(size[0])
        extent_y = _first_float(size[1])
    if extent_x is not None and extent_y is not None and min(extent_x, extent_y) < MIN_TABLE_COLLISION_EXTENT_M:
        blockers.append("table_collision_extent_too_small")
    return blockers


def _exports_record(run: Path, blocking_reasons: list[str]) -> dict[str, dict[str, Any]]:
    exports = {
        "usd": _backend_record(run, "usd", "exports/scene.usda", "qa/usd_export_report.json"),
        "genesis": _backend_record(run, "genesis", "exports/genesis_scene.py", "qa/genesis_export_report.json"),
        "isaac": _backend_record(run, "isaac", "exports/isaac_scene.py", "qa/isaac_load_report.json"),
    }
    for backend, record in exports.items():
        if record["status"] == "missing":
            blocking_reasons.append(f"missing_backend_export:{backend}")
        elif record["status"] == "runtime_unavailable":
            blocking_reasons.append(f"{backend}_runtime_unavailable")
    return exports


def _backend_record(run: Path, backend: str, artifact: str, report: str) -> dict[str, Any]:
    artifact_path = run / artifact
    report_path = run / report
    status = READY_STATUS if artifact_path.is_file() else "missing"
    report_data = _load_json(report_path)
    if report_data.get("status") == "runtime_unavailable":
        status = "runtime_unavailable"
    return {
        "backend": backend,
        "status": status,
        "path": artifact if artifact_path.is_file() else None,
        "report_path": report if report_path.is_file() else None,
        "runtime_status": report_data.get("status"),
    }


def _background_external_render_record(run: Path) -> dict[str, Any]:
    return _load_json(run / "qa" / "background_3dgs_render_report.json")


def _background_external_render_verified(run: Path, report: dict[str, Any]) -> bool:
    render_rel = report.get("render_path")
    if (
        report.get("status") != "rendered"
        or report.get("backend") != "external_3dgs_renderer"
        or report.get("registered_3dgs_rendered") is not True
        or not render_rel
    ):
        return False
    return (run / str(render_rel)).is_file()


def _asset_record(role: str, path: str | None, source: str, status: str, *, diagnostic_only: bool = False) -> dict[str, Any]:
    return {
        "role": role,
        "path": path,
        "source": source,
        "status": status,
        "diagnostic_only": diagnostic_only,
    }


def _background_source_kind(background: dict[str, Any], point_cloud_rel: str) -> str:
    if background.get("gaussian_splat_config_path"):
        return "registered_3dgs_sidecar" if point_cloud_rel != "scene_cloud.ply" else "full_scene_cloud_debug"
    if point_cloud_rel == "background/bg_only_cloud.ply":
        return "bg_only_cloud"
    return "unknown_background_proxy"


def _has_3dgs_sidecar(run: Path, background: dict[str, Any]) -> bool:
    if background.get("gaussian_splat_config_path") or background.get("gaussian_splat_checkpoint_path"):
        return True
    status = _load_json(run / "video" / "3dgs_status.json")
    outputs = status.get("outputs") if isinstance(status.get("outputs"), dict) else {}
    return bool(status.get("status") == "completed" or outputs.get("latest_config") or outputs.get("latest_checkpoint"))


def _background_native_runtime_verified(run: Path) -> bool:
    for rel in [
        "qa/background_runtime_report.json",
        "qa/usd_export_report.json",
        "qa/genesis_export_report.json",
        "qa/isaac_load_report.json",
    ]:
        report = _load_json(run / rel)
        if report.get("background_runtime_verified") is True:
            return True
        status = str(
            report.get("background_visual_status")
            or report.get("visual_background_status")
            or report.get("background_status")
            or ""
        ).lower()
        if status in {"native_3dgs_loaded", "registered_3dgs_loaded", "verified_native_3dgs"}:
            return True
    return False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(metric):
        return None
    return metric


def _valid_transform(value: Any) -> bool:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return transform.shape == (4, 4) and bool(np.all(np.isfinite(transform)))


def _looks_like_bbox_proxy(path: Path) -> bool:
    if path.name in {"debug_bbox.glb", "bbox.glb"}:
        return True
    try:
        loaded = trimesh.load(path, force="scene")
        vertices = []
        faces = 0
        if isinstance(loaded, trimesh.Trimesh):
            vertices.append(np.asarray(loaded.vertices))
            faces += len(loaded.faces)
        elif isinstance(loaded, trimesh.Scene):
            for geom in loaded.geometry.values():
                if hasattr(geom, "vertices"):
                    vertices.append(np.asarray(geom.vertices))
                if hasattr(geom, "faces"):
                    faces += len(geom.faces)
        vertex_count = int(sum(len(v) for v in vertices))
        return vertex_count <= 8 and faces <= 12
    except Exception:  # noqa: BLE001 - invalid mesh is handled as a blocked legacy candidate elsewhere.
        return path.name in {"mesh_aligned.glb", "mesh.glb"}


def _rel(run: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return path.relative_to(run).as_posix()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped



def export_sim(run_dir: str | Path, *, backends: list[str]) -> dict[str, Path]:
    requested = list(dict.fromkeys(backends))
    artifacts: dict[str, Path] = {"sim_export_manifest": write_sim_export_manifest(run_dir)}
    for backend in requested:
        if backend == "usd":
            from .usd_export import export_usd_scene

            artifacts["usd"] = export_usd_scene(run_dir).scene_path
        elif backend == "genesis":
            from .genesis_export import export_genesis_scene

            artifacts["genesis"] = export_genesis_scene(run_dir).script_path
        elif backend == "isaac":
            from .isaac_export import export_isaac_scene

            artifacts["isaac"] = export_isaac_scene(run_dir).script_path
        else:
            raise ValueError(f"unknown simulator export backend: {backend}")
    artifacts["sim_export_manifest"] = write_sim_export_manifest(run_dir)
    return artifacts
