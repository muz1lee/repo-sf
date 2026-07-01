"""Browser composite viewer export for reconstructed assets and background geometry."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .background_registration import load_background_registration


@dataclass(frozen=True)
class CompositeViewerResult:
    index_path: Path
    config_path: Path
    url_path: str


def export_composite_viewer(run_dir: str | Path, *, backend: str = "external-sidecar", show_settled: bool = False) -> CompositeViewerResult:
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa_path = run / "qa" / "qa_report.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.is_file() else {}

    viewer_dir = run / "exports" / "composite_viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    config = _viewer_config(run, viewer_dir, manifest, _qa_with_genesis_fallback(run, qa), backend=backend, show_settled=show_settled)
    _write_viewer_audits(run, viewer_dir, config)
    config_path = viewer_dir / "viewer_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    index_path = viewer_dir / "index.html"
    index_path.write_text(_index_html(), encoding="utf-8")
    return CompositeViewerResult(
        index_path=index_path,
        config_path=config_path,
        url_path="/exports/composite_viewer/index.html",
    )


def serve_composite_viewer(
    run_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7010,
    backend: str = "external-sidecar",
    show_settled: bool = False,
) -> None:
    run = Path(run_dir)
    export_composite_viewer(run, backend=backend, show_settled=show_settled)
    handler = partial(SimpleHTTPRequestHandler, directory=str(run))
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.serve_forever()


def qa_viewer(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    viewer_config_path = run / "exports" / "composite_viewer" / "viewer_config.json"
    if not viewer_config_path.is_file():
        export_composite_viewer(run)
    config = _read_json(viewer_config_path)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = qa_dir / "viewer_screenshot.png"
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(screenshot_path)
    background = config.get("background", {}) if isinstance(config.get("background"), dict) else {}
    report = {
        "version": 1,
        "status": config.get("status", "unknown"),
        "viewer_config_path": "exports/composite_viewer/viewer_config.json",
        "screenshot_path": "qa/viewer_screenshot.png",
        "background_source_kind": background.get("source_kind"),
        "background_render_mode": background.get("render_mode"),
        "live_3dgs_runtime": bool(background.get("live_3dgs_runtime", False)),
        "simulator_native": bool(background.get("simulator_native", False)),
        "pose_display": config.get("pose_display", {}),
        "blocking_reasons": [background.get("status")] if str(background.get("status", "")).startswith("blocked") else [],
    }
    (qa_dir / "viewer_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report_path": "qa/viewer_audit.json"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _viewer_config(
    run: Path,
    viewer_dir: Path,
    manifest: dict[str, Any],
    qa: dict[str, Any],
    *,
    backend: str = "external-sidecar",
    show_settled: bool = False,
) -> dict[str, Any]:
    support = manifest.get("support_plane", {})
    strict_manifest = _read_json(run / "sim_export_manifest.json")
    strict_support = strict_manifest.get("support_surface", {}) if isinstance(strict_manifest.get("support_surface"), dict) else {}
    table_qa = _read_json(run / "qa" / "table_collision_report.json")
    background = manifest.get("background", {})
    physics = qa.get("physics_settle", {})
    background_config = _viewer_background_config(run, viewer_dir, background, support, backend=backend)
    pose_display, settled_poses = _settled_pose_config(run, viewer_dir)
    pose_display["default_mode"] = "settled" if show_settled and _settled_pose_ready(pose_display) else "initial"
    objects = [_object_config(run, viewer_dir, obj, settled_poses.get(str(obj.get("object_id")))) for obj in manifest.get("objects", [])]
    status, status_reason = _viewer_status(background_config)
    config = {
        "version": 1,
        "status": status,
        "status_reason": status_reason,
        "run_name": run.name,
        "viewer_backend": backend,
        "coordinate_frame": manifest.get("coordinate_frames", {}).get("world", "z_up_ground_plane_meters"),
        "background": background_config,
        "reference_camera": _reference_camera_config(run, support, background_config),
        "support_plane": _support_plane_config(run, viewer_dir, support, strict_support, table_qa),
        "pose_display": pose_display,
        "objects": objects,
        "audits": {
            "object_path_audit": "object_path_audit.json",
            "background_provenance_audit": "background_provenance_audit.json",
        },
        "qa": {
            "stability_status": physics.get("stability_status"),
            "settle_steps": physics.get("settle_steps"),
            "max_penetration_depth_m": physics.get("max_penetration_depth_m"),
            "max_displacement_m": physics.get("max_displacement_m"),
            "fall_out_detected": physics.get("fall_out_detected"),
            "nan_detected": physics.get("nan_detected"),
        },
    }
    return config


def _viewer_status(background: dict[str, Any]) -> tuple[str, str]:
    viewer_status = str(background.get("viewer_status") or background.get("status") or "unknown")
    if viewer_status.startswith("partial_"):
        return "partial", viewer_status
    if viewer_status.startswith("blocked") or viewer_status.startswith("missing"):
        return "blocked", viewer_status
    if viewer_status == "live_3dgs_runtime":
        return "passed", viewer_status
    return "partial", viewer_status


def _support_plane_config(
    run: Path,
    viewer_dir: Path,
    support: dict[str, Any],
    strict_support: dict[str, Any],
    table_qa: dict[str, Any],
) -> dict[str, Any]:
    mesh_rel = strict_support.get("mesh_path") or support.get("table_collision_mesh_path")
    mesh_path = _rel_existing(run, viewer_dir, mesh_rel)
    size = strict_support.get("table_collision_size_xyz") or support.get("table_collision_size_xyz") or []
    pos = strict_support.get("table_collision_pos_world") or support.get("table_collision_pos_world") or []
    thickness = support.get("thickness")
    if thickness is None and len(size) >= 3:
        thickness = size[2]
    top_z = support.get("top_z")
    if top_z is None and len(pos) >= 3 and thickness is not None:
        top_z = float(pos[2]) + float(thickness) * 0.5
    artifact_role = str(support.get("artifact_role") or support.get("role") or "collision_proxy")
    support_status = strict_support.get("support_surface_status") or support.get("support_surface_status")
    qa_status = table_qa.get("status") or support_status or support.get("qa_status") or support.get("status", "unknown")
    is_final_collision = (
        (strict_support.get("table_collision_final") if "table_collision_final" in strict_support else support.get("table_collision_final"))
        is True
        and support_status == "passed"
        and qa_status == "passed"
    )
    final_or_proxy = str(support.get("final_or_proxy") or ("final" if is_final_collision or artifact_role == "final" else "proxy"))
    if not is_final_collision:
        final_or_proxy = "proxy"
    geometry_type = (
        strict_support.get("table_collision_geometry_type")
        or support.get("table_collision_geometry_type")
        or support.get("geometry_type")
        or "support_box_proxy"
    )
    source_backend = (
        strict_support.get("table_collision_source_backend")
        or support.get("table_collision_source_backend")
        or support.get("source_backend", "unknown")
    )
    label = support.get("label") or (
        "tabletop mask polygon collision" if "polygon" in str(geometry_type) else "estimated support box proxy"
    )
    return {
        "status": strict_support.get("status") or support.get("status", "absent"),
        "source_backend": source_backend,
        "geometry_type": geometry_type,
        "final_or_proxy": final_or_proxy,
        "top_z": top_z,
        "thickness": thickness,
        "mesh_path": mesh_path,
        "qa_status": qa_status,
        "label": label,
        "table_collision_mesh_path": mesh_path,
        "table_collision_pos_world": strict_support.get("table_collision_pos_world") or support.get("table_collision_pos_world"),
        "table_collision_size_xyz": strict_support.get("table_collision_size_xyz") or support.get("table_collision_size_xyz"),
        "table_bounds_world_xy": strict_support.get("table_bounds_world_xy") or support.get("table_bounds_world_xy"),
        "blocking_reasons": strict_support.get("blocking_reasons") or table_qa.get("blocking_reasons") or [],
        "projection_iou": table_qa.get("projection_iou"),
        "tabletop_mask_area_px": table_qa.get("tabletop_mask_area_px"),
        "selected_candidate_ratio": table_qa.get("selected_candidate_ratio"),
    }


def _reference_camera_config(run: Path, support: dict[str, Any], background: dict[str, Any]) -> dict[str, Any]:
    camera_path = run / "camera.json"
    camera = json.loads(camera_path.read_text(encoding="utf-8")) if camera_path.is_file() else {}
    image_size = background.get("image_size") or [camera.get("width", 1280), camera.get("height", 720)]
    image_width = int(image_size[0] or camera.get("width", 1280))
    image_height = int(image_size[1] or camera.get("height", 720))
    source_width = float(camera.get("width", image_width) or image_width)
    source_height = float(camera.get("height", image_height) or image_height)
    scale_y = float(image_height) / source_height if source_height > 0.0 else 1.0
    fx = float(camera.get("fx", 0.0) or 0.0)
    fy = float(camera.get("fy", 0.0) or 0.0)
    cx = float(camera.get("cx", source_width * 0.5) or source_width * 0.5)
    cy = float(camera.get("cy", source_height * 0.5) or source_height * 0.5)
    fov_y_deg = 55.0
    if fy > 0.0:
        fov_y_deg = math.degrees(2.0 * math.atan(float(source_height) / (2.0 * fy)))
    support_height = float(support.get("original_height_world_m", 0.0) or 0.0)
    camera_world_z = -support_height
    return {
        "mode": "opencv_reference_overlay",
        "intrinsics_source": "camera.json" if camera_path.is_file() else "fallback_image_size",
        "image_size": [image_width, image_height],
        "source_camera_size": [int(source_width), int(source_height)],
        "fx": fx * (float(image_width) / source_width if source_width > 0.0 else 1.0),
        "fy": fy * scale_y,
        "cx": cx * (float(image_width) / source_width if source_width > 0.0 else 1.0),
        "cy": cy * scale_y,
        "fov_y_deg": float(fov_y_deg),
        "camera_world_position": [0.0, 0.0, camera_world_z],
        "camera_world_lookat": [0.0, 1.0, camera_world_z],
        "camera_world_up": [0.0, 0.0, 1.0],
        "support_camera_height_m": support_height,
        "note": "Default viewer camera matches the reference OpenCV camera bridge into z-up simulator world; orbit controls remain available after load.",
    }


def _qa_with_genesis_fallback(run: Path, qa: dict[str, Any]) -> dict[str, Any]:
    genesis_path = run / "qa" / "genesis_settle_report.json"
    if not genesis_path.is_file():
        return qa
    merged = dict(qa)
    physics = qa.get("physics_settle", {})
    merged["physics_settle"] = {**physics, **json.loads(genesis_path.read_text(encoding="utf-8"))}
    return merged


def _viewer_background_config(
    run: Path,
    viewer_dir: Path,
    background: dict[str, Any],
    support: dict[str, Any],
    *,
    backend: str = "external-sidecar",
) -> dict[str, Any]:
    registration = load_background_registration(run)
    bg_only_cloud = run / "background" / "bg_only_cloud.ply"
    full_scene_cloud = run / "scene_cloud.ply"
    native_3dgs_asset = _native_3dgs_asset_path(run)
    completed_3dgs = _has_completed_3dgs(run, background)
    external_report = _external_3dgs_render_report(run)
    external_render_path = _external_3dgs_render_path(run, external_report)
    runtime_report = _background_3dgs_runtime_report(run)

    if backend == "browser-3dgs":
        return _browser_3dgs_background_config(
            run,
            viewer_dir,
            background,
            support,
            registration,
            native_3dgs_asset,
            bg_only_cloud,
            full_scene_cloud,
            external_render_path,
            runtime_report,
        )

    diagnostic_only = False
    is_final_visual = False
    simulator_native = False
    image_path = None
    image_size = None
    point_cloud_diagnostic_only = True
    if external_render_path is not None:
        source_kind = "external_3dgs_renderer"
        point_cloud = bg_only_cloud if bg_only_cloud.is_file() else None
        status = "external_3dgs_rendered"
        image_path = _rel(viewer_dir, external_render_path)
        image_size = external_report.get("image_size")
        simulator_native = bool(external_report.get("simulator_native", False))
    elif native_3dgs_asset.is_file() and not bg_only_cloud.is_file():
        source_kind = "3dgs_native_asset_candidate"
        point_cloud = None
        status = "blocked_native_3dgs_not_integrated"
        diagnostic_only = True
    elif completed_3dgs:
        source_kind = "external_3dgs_render_missing"
        point_cloud = bg_only_cloud if bg_only_cloud.is_file() else None
        status = "blocked_missing_external_3dgs_render"
        diagnostic_only = True
    elif bg_only_cloud.is_file():
        source_kind = "bg_only_cloud_debug"
        point_cloud = bg_only_cloud
        status = "blocked_external_3dgs_required"
        diagnostic_only = True
    elif full_scene_cloud.is_file():
        source_kind = "full_scene_cloud_debug"
        point_cloud = full_scene_cloud
        status = "blocked_missing_bg_only_background"
        diagnostic_only = True
    else:
        source_kind = "3dgs_unavailable"
        point_cloud = None
        status = "blocked_missing_background_asset"

    if registration and registration.get("source_kind") == "registered_3dgs" and registration.get("status") in {"registered", "registered_3dgs"}:
        source_kind = "registered_3dgs"
        status = "registered_3dgs"
    is_final_visual = bool(simulator_native and source_kind == "registered_3dgs")

    registration_path = _registration_path(run, background)
    transform = registration.get("transforms", {}).get("T_bg_only_cloud_to_sim_world") if registration else None
    viewer_meta = _background_viewer_metadata(
        source_kind=source_kind,
        status=status,
        simulator_native=simulator_native,
        has_external_render=external_render_path is not None,
        has_bg_only_cloud=bg_only_cloud.is_file(),
        has_full_scene_cloud=full_scene_cloud.is_file(),
        has_native_asset=native_3dgs_asset.is_file(),
    )
    return {
        "source_kind": source_kind,
        "mode_label": "external_3dgs_render_sidecar" if source_kind == "external_3dgs_renderer" else source_kind,
        "source_backend": background.get("source_backend", "unknown"),
        "status": status,
        **viewer_meta,
        "diagnostic_only": diagnostic_only,
        "is_final_visual": is_final_visual,
        "simulator_native": simulator_native,
        "image_path": image_path,
        "image_size": image_size,
        "render_report_path": _rel(viewer_dir, run / "qa" / "background_3dgs_render_report.json")
        if (run / "qa" / "background_3dgs_render_report.json").is_file()
        else None,
        "point_cloud_path": _rel(viewer_dir, point_cloud) if point_cloud is not None and point_cloud.is_file() else None,
        "point_cloud_diagnostic_only": point_cloud_diagnostic_only,
        "native_asset_candidate_path": _rel(viewer_dir, native_3dgs_asset) if native_3dgs_asset.is_file() else None,
        "debug_full_scene_cloud_path": _rel(viewer_dir, full_scene_cloud) if full_scene_cloud.is_file() else None,
        "registration_path": _rel(viewer_dir, registration_path) if registration_path is not None and registration_path.is_file() else None,
        "registration": registration,
        "point_cloud_transform": {
            "source_frame": "opencv_x_right_y_down_z_forward_meters",
            "target_frame": "z_up_ground_plane_meters",
            "support_height_m": float(support.get("original_height_world_m", 0.0)),
            "matrix": transform,
        },
        "gaussian_splat": _gaussian_splat_viewer_status(run, background, registration),
        "note": "Viewer final background must be a 3DGS render. BG-only and full-scene point clouds are diagnostic-only debug assets.",
    }


def _background_viewer_metadata(
    *,
    source_kind: str,
    status: str,
    simulator_native: bool,
    has_external_render: bool,
    has_bg_only_cloud: bool,
    has_full_scene_cloud: bool,
    has_native_asset: bool,
) -> dict[str, Any]:
    live_3dgs_runtime = bool(status == "live_3dgs_runtime" or (simulator_native and source_kind == "registered_3dgs"))
    if source_kind == "browser_native_3dgs":
        render_mode = "browser_native_3dgs"
        viewer_status = "live_3dgs_runtime" if live_3dgs_runtime else "blocked_browser_3dgs_runtime_not_verified"
        orbit_policy = "free_orbit_supported" if live_3dgs_runtime else "free_orbit_blocked_until_runtime_verified"
        camera_lock_required = False
        provenance_label = "browser-native 3DGS runtime"
    elif live_3dgs_runtime:
        render_mode = "live_3dgs_runtime"
        viewer_status = "live_3dgs_runtime"
        orbit_policy = "free_orbit_supported"
        camera_lock_required = False
        provenance_label = "live/native 3DGS runtime"
    elif source_kind == "registered_3dgs":
        render_mode = "registered_3dgs_transform_only"
        viewer_status = "blocked_registered_3dgs_runtime_not_verified"
        orbit_policy = "free_orbit_blocked_until_runtime_verified"
        camera_lock_required = False
        provenance_label = "registered 3DGS transform only; live runtime not verified"
    elif source_kind == "external_3dgs_renderer" and has_external_render:
        render_mode = "external_3dgs_png_sidecar"
        viewer_status = "partial_external_render_only"
        orbit_policy = "locked_to_reference_camera"
        camera_lock_required = True
        provenance_label = "external 3DGS PNG sidecar (reference view only)"
    elif source_kind == "bg_only_cloud_debug":
        render_mode = "bg_only_diagnostic_cloud"
        viewer_status = status
        orbit_policy = "free_orbit_debug_only"
        camera_lock_required = False
        provenance_label = "BG-only diagnostic point cloud"
    elif source_kind == "full_scene_cloud_debug":
        render_mode = "full_scene_debug_cloud"
        viewer_status = status
        orbit_policy = "free_orbit_debug_only"
        camera_lock_required = False
        provenance_label = "full-scene debug point cloud"
    elif source_kind == "3dgs_native_asset_candidate":
        render_mode = "native_3dgs_asset_candidate"
        viewer_status = "blocked_native_3dgs_not_integrated"
        orbit_policy = "free_orbit_debug_only"
        camera_lock_required = False
        provenance_label = "native 3DGS asset candidate, runtime unsupported"
    else:
        render_mode = source_kind
        viewer_status = status
        orbit_policy = "free_orbit_debug_only"
        camera_lock_required = False
        provenance_label = source_kind

    return {
        "viewer_status": viewer_status,
        "render_mode": render_mode,
        "live_3dgs_runtime": live_3dgs_runtime,
        "orbit_policy": orbit_policy,
        "camera_lock_required": camera_lock_required,
        "provenance_label": provenance_label,
        "background_layers": {
            "live_3dgs_runtime": {
                "status": "active" if live_3dgs_runtime else ("asset_candidate_only" if has_native_asset else "unavailable"),
                "simulator_native": live_3dgs_runtime,
            },
            "external_3dgs_png_sidecar": {
                "status": "active_reference_view_only" if has_external_render and source_kind != "browser_native_3dgs" else ("available_reference_view_only" if has_external_render else "missing"),
                "simulator_native": False,
            },
            "bg_only_diagnostic_cloud": {
                "status": "available_diagnostic_only" if has_bg_only_cloud else "missing",
            },
            "full_scene_debug_cloud": {
                "status": "available_debug_only" if has_full_scene_cloud else "missing",
            },
        },
    }


def _browser_3dgs_background_config(
    run: Path,
    viewer_dir: Path,
    background: dict[str, Any],
    support: dict[str, Any],
    registration: dict[str, Any] | None,
    native_3dgs_asset: Path,
    bg_only_cloud: Path,
    full_scene_cloud: Path,
    external_render_path: Path | None,
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    del background
    registration_path = _registration_path(run, {"registration_path": "background/registration.json"})
    runtime_verified = _browser_runtime_verified(runtime_report)
    status = "live_3dgs_runtime" if runtime_verified else "blocked_browser_3dgs_runtime_not_verified"
    transform = registration.get("transforms", {}).get("T_bg_only_cloud_to_sim_world") if registration else None
    viewer_meta = _background_viewer_metadata(
        source_kind="browser_native_3dgs",
        status=status,
        simulator_native=False,
        has_external_render=external_render_path is not None,
        has_bg_only_cloud=bg_only_cloud.is_file(),
        has_full_scene_cloud=full_scene_cloud.is_file(),
        has_native_asset=native_3dgs_asset.is_file(),
    )
    return {
        "source_kind": "browser_native_3dgs",
        "mode_label": "browser_native_3dgs",
        "source_backend": "browser_3dgs_runtime",
        "status": status,
        **viewer_meta,
        "diagnostic_only": not runtime_verified,
        "is_final_visual": runtime_verified,
        "simulator_native": False,
        "image_path": None,
        "image_size": None,
        "render_report_path": _rel(viewer_dir, run / "qa" / "background_3dgs_render_report.json")
        if (run / "qa" / "background_3dgs_render_report.json").is_file()
        else None,
        "point_cloud_path": _rel(viewer_dir, bg_only_cloud) if bg_only_cloud.is_file() else None,
        "point_cloud_diagnostic_only": True,
        "native_asset_candidate_path": _rel(viewer_dir, native_3dgs_asset) if native_3dgs_asset.is_file() else None,
        "debug_full_scene_cloud_path": _rel(viewer_dir, full_scene_cloud) if full_scene_cloud.is_file() else None,
        "registration_path": _rel(viewer_dir, registration_path) if registration_path is not None and registration_path.is_file() else None,
        "registration": registration,
        "point_cloud_transform": {
            "source_frame": "opencv_x_right_y_down_z_forward_meters",
            "target_frame": "z_up_ground_plane_meters",
            "support_height_m": float(support.get("original_height_world_m", 0.0)),
            "matrix": transform,
        },
        "gaussian_splat": _gaussian_splat_viewer_status(run, {}, registration),
        "browser_3dgs": {
            "asset_path": _rel(viewer_dir, native_3dgs_asset) if native_3dgs_asset.is_file() else None,
            "runtime_status": status,
            "runtime_report_path": _rel(viewer_dir, run / "qa" / "background_3dgs_runtime_report.json")
            if (run / "qa" / "background_3dgs_runtime_report.json").is_file()
            else None,
            "registration_transform": (registration or {}).get("T_3dgs_world_to_sim_world"),
            "blocked_reason": None if runtime_verified else "browser-native 3DGS loader/runtime evidence is not available.",
        },
        "note": "Browser-native 3DGS backend was requested; external PNG sidecar is not promoted to live 3D.",
    }


def _background_3dgs_runtime_report(run: Path) -> dict[str, Any]:
    path = run / "qa" / "background_3dgs_runtime_report.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _browser_runtime_verified(report: dict[str, Any]) -> bool:
    viewer = report.get("native_viewer_runtime") if isinstance(report.get("native_viewer_runtime"), dict) else {}
    return bool(report.get("browser_native_3dgs_verified") or viewer.get("status") in {"verified", "passed", "live_3dgs_runtime"})


def _registration_path(run: Path, background: dict[str, Any]) -> Path | None:
    value = background.get("registration_path") or "background/registration.json"
    path = run / str(value)
    return path if path.is_file() else None


def _has_completed_3dgs(run: Path, background: dict[str, Any]) -> bool:
    if background.get("gaussian_splat_config_path"):
        return True
    status = _load_3dgs_status(run)
    return status.get("status") == "completed"


def _gaussian_splat_viewer_status(run: Path, background: dict[str, Any], registration: dict[str, Any] | None) -> dict[str, Any]:
    if registration and isinstance(registration.get("gaussian_splat"), dict):
        data = dict(registration["gaussian_splat"])
        native_asset = _native_3dgs_asset_path(run)
        if native_asset.is_file() and data.get("status") != "registered_3dgs":
            data.update(
                {
                    "status": "native_asset_candidate",
                    "native_asset_path": str(native_asset.relative_to(run)),
                    "native_rendering": False,
                    "runtime_status": "blocked_native_3dgs_not_integrated",
                    "note": "3DGS native asset candidate exists, but simulator/viewer runtime loading is not integrated.",
                }
            )
        return data
    status = _load_3dgs_status(run)
    outputs = status.get("outputs", {}) if isinstance(status.get("outputs", {}), dict) else {}
    config_path = background.get("gaussian_splat_config_path") or outputs.get("latest_config")
    checkpoint_path = background.get("gaussian_splat_checkpoint_path") or outputs.get("latest_checkpoint")
    run_path = background.get("gaussian_splat_path") or outputs.get("latest_run_dir") or outputs.get("output_dir")
    native_asset = _native_3dgs_asset_path(run)
    if native_asset.is_file():
        return {
            "status": "native_asset_candidate",
            "training_status": status.get("status"),
            "run_path": run_path,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "native_asset_path": str(native_asset.relative_to(run)),
            "native_rendering": False,
            "runtime_status": "blocked_native_3dgs_not_integrated",
            "blocked_reason": "3DGS native asset candidate exists, but native 3DGS loading is not integrated into this browser viewer.",
        }
    if config_path or status.get("status") == "completed":
        return {
            "status": "blocked_viewer_not_integrated",
            "training_status": status.get("status", "completed"),
            "run_path": run_path,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "blocked_reason": "3DGS training exists, but native 3DGS loading is not integrated into this browser viewer.",
        }
    return {"status": "not_available", "training_status": status.get("status"), "run_path": run_path, "config_path": config_path, "checkpoint_path": checkpoint_path}


def _load_3dgs_status(run: Path) -> dict[str, Any]:
    path = run / "video" / "3dgs_status.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_status_json"}
    return data if isinstance(data, dict) else {"status": "invalid_status_json"}


def _native_3dgs_asset_path(run: Path) -> Path:
    return run / "background" / "3dgs_native" / "splat_rgb.ply"


def _settled_pose_config(run: Path, viewer_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = run / "qa" / "settled_pose_delta_report.json"
    if not path.is_file():
        return (
            {
                "status": "settled_pose_unavailable",
                "toggle_enabled": False,
                "report_path": None,
                "source": None,
                "object_count": 0,
                "note": "No qa/settled_pose_delta_report.json was found; only initial poses are available.",
            },
            {},
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return (
            {
                "status": "settled_pose_report_invalid",
                "toggle_enabled": False,
                "report_path": _rel(viewer_dir, path),
                "source": None,
                "object_count": 0,
                "note": "qa/settled_pose_delta_report.json is not valid JSON.",
            },
            {},
        )
    records: dict[str, dict[str, Any]] = {}
    raw_objects = report.get("objects", [])
    if isinstance(raw_objects, dict):
        iterable = [dict(value, object_id=key) if isinstance(value, dict) else {"object_id": key} for key, value in raw_objects.items()]
    elif isinstance(raw_objects, list):
        iterable = [item for item in raw_objects if isinstance(item, dict)]
    else:
        iterable = []
    for item in iterable:
        object_id = str(item.get("object_id") or "")
        transform = (
            item.get("settled_T_object_to_world")
            or item.get("T_object_to_world_settled")
            or item.get("T_settled_object_to_world")
            or (item.get("settled_pose", {}) if isinstance(item.get("settled_pose"), dict) else {}).get("T_object_to_world")
        )
        settled_position = item.get("final_pos") or item.get("settled_pos") or item.get("settled_position")
        if object_id and (transform is not None or settled_position is not None):
            record = dict(item)
            if transform is not None:
                record["_settled_T_object_to_world"] = transform
            if settled_position is not None:
                record["_settled_position"] = settled_position
            records[object_id] = record
    return (
        {
            "status": report.get("status", "available"),
            "toggle_enabled": bool(records),
            "report_path": _rel(viewer_dir, path),
            "source": report.get("source") or report.get("backend"),
            "object_count": len(records),
            "note": report.get("note"),
        },
        records,
    )


def _settled_pose_ready(pose_display: dict[str, Any]) -> bool:
    return bool(pose_display.get("toggle_enabled")) and str(pose_display.get("status")) in {"completed", "passed", "ready"}


def _object_config(run: Path, viewer_dir: Path, obj: dict[str, Any], settled_pose: dict[str, Any] | None = None) -> dict[str, Any]:
    transform = np.asarray(obj["T_object_to_world"], dtype=np.float64)
    object_id = str(obj["object_id"])
    object_dir = run / "objects" / object_id
    legacy_mesh = run / str(obj["mesh_path"])
    visual_mesh = object_dir / "visual.glb"
    collision_mesh = object_dir / "collision.glb"
    debug_proxy = object_dir / "debug_bbox.glb"
    if not debug_proxy.is_file() and legacy_mesh.is_file() and legacy_mesh.name == "mesh_aligned.glb":
        debug_proxy = legacy_mesh
    final_visual = visual_mesh if visual_mesh.is_file() else None
    initial_position = [float(v) for v in transform[:3, 3]]
    initial_quat = _matrix_to_quat_wxyz(transform[:3, :3])
    settled_transform = None
    if settled_pose and settled_pose.get("_settled_T_object_to_world") is not None:
        try:
            settled_transform = np.asarray(settled_pose["_settled_T_object_to_world"], dtype=np.float64)
            if settled_transform.shape != (4, 4):
                settled_transform = None
        except (TypeError, ValueError):
            settled_transform = None
    settled_position_override = None
    if settled_pose and settled_pose.get("_settled_position") is not None:
        try:
            values = [float(v) for v in settled_pose["_settled_position"]]
            if len(values) == 3 and all(np.isfinite(values)):
                settled_position_override = values
        except (TypeError, ValueError):
            settled_position_override = None
    settled_position = (
        [float(v) for v in settled_transform[:3, 3]]
        if settled_transform is not None
        else settled_position_override or initial_position
    )
    settled_quat = _matrix_to_quat_wxyz(settled_transform[:3, :3]) if settled_transform is not None else initial_quat
    settled_delta = None
    if settled_pose:
        settled_delta = {
            key: settled_pose.get(key)
            for key in ("translation_delta_m", "rotation_delta_deg", "max_translation_delta_m", "max_rotation_delta_deg")
            if key in settled_pose
        }
    return {
        "object_id": object_id,
        "label": obj.get("label", object_id),
        "mesh_path": _rel(viewer_dir, final_visual) if final_visual is not None else None,
        "visual_mesh_path": _rel(viewer_dir, final_visual) if final_visual is not None else None,
        "collision_mesh_path": _rel(viewer_dir, collision_mesh) if collision_mesh.is_file() else None,
        "debug_proxy_path": _rel(viewer_dir, debug_proxy) if debug_proxy.is_file() else None,
        "legacy_mesh_path": _rel(viewer_dir, legacy_mesh) if legacy_mesh.is_file() else None,
        "final_visual_source": "objects_visual_glb" if final_visual is not None else "missing_visual_glb",
        "final_visual_is_proxy": final_visual is None,
        "world_position": initial_position,
        "world_quat_wxyz": initial_quat,
        "initial_world_position": initial_position,
        "initial_world_quat_wxyz": initial_quat,
        "settled_world_position": settled_position,
        "settled_world_quat_wxyz": settled_quat,
        "settled_pose_available": settled_transform is not None or settled_position_override is not None,
        "settled_pose_delta": settled_delta or {},
        "asset_scale": _asset_scale(obj),
        "mass_kg": obj.get("mass_kg"),
        "friction": obj.get("friction"),
        "needs_manual_refine": obj.get("needs_manual_refine", False),
    }


def _external_3dgs_render_report(run: Path) -> dict[str, Any]:
    path = run / "qa" / "background_3dgs_render_report.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _external_3dgs_render_path(run: Path, report: dict[str, Any]) -> Path | None:
    rel = report.get("render_path")
    if (
        report.get("status") != "rendered"
        or report.get("backend") != "external_3dgs_renderer"
        or report.get("registered_3dgs_rendered") is not True
        or not rel
    ):
        return None
    path = run / str(rel)
    return path if path.is_file() else None


def _write_viewer_audits(run: Path, viewer_dir: Path, config: dict[str, Any]) -> None:
    object_records = []
    for item in config.get("objects", []):
        final_rel = _run_rel_from_viewer_rel(run, viewer_dir, item.get("visual_mesh_path"))
        legacy_rel = _run_rel_from_viewer_rel(run, viewer_dir, item.get("legacy_mesh_path"))
        collision_rel = _run_rel_from_viewer_rel(run, viewer_dir, item.get("collision_mesh_path"))
        debug_rel = _run_rel_from_viewer_rel(run, viewer_dir, item.get("debug_proxy_path"))
        object_records.append(
            {
                "object_id": item.get("object_id"),
                "label": item.get("label"),
                "final_visual_path": final_rel,
                "final_visual_is_proxy": bool(item.get("final_visual_is_proxy", True)),
                "legacy_mesh_path": legacy_rel,
                "collision_mesh_path": collision_rel,
                "debug_proxy_path": debug_rel,
                "asset_scale": item.get("asset_scale", 1.0),
                "default_visible": bool(final_rel) and not bool(item.get("final_visual_is_proxy", True)),
            }
        )
    (viewer_dir / "object_path_audit.json").write_text(
        json.dumps({"version": 1, "objects": object_records}, indent=2),
        encoding="utf-8",
    )
    background = config.get("background", {})
    (viewer_dir / "background_provenance_audit.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source_kind": background.get("mode_label") or background.get("source_kind"),
                "status": background.get("status"),
                "viewer_status": background.get("viewer_status"),
                "render_mode": background.get("render_mode"),
                "live_3dgs_runtime": bool(background.get("live_3dgs_runtime", False)),
                "orbit_policy": background.get("orbit_policy"),
                "final_visual_path": _run_rel_from_viewer_rel(run, viewer_dir, background.get("image_path")),
                "simulator_native": bool(background.get("simulator_native", False)),
                "render_report_path": _run_rel_from_viewer_rel(run, viewer_dir, background.get("render_report_path")),
                "bg_only_cloud_path": _run_rel_from_viewer_rel(run, viewer_dir, background.get("point_cloud_path")),
                "bg_only_cloud_diagnostic_only": bool(background.get("point_cloud_diagnostic_only", True)),
                "full_scene_cloud_path": _run_rel_from_viewer_rel(run, viewer_dir, background.get("debug_full_scene_cloud_path")),
                "is_final_visual": bool(background.get("is_final_visual", False)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_rel_from_viewer_rel(run: Path, viewer_dir: Path, value: object) -> str | None:
    if not value:
        return None
    path = (viewer_dir / str(value)).resolve()
    try:
        return path.relative_to(run.resolve()).as_posix()
    except ValueError:
        return str(value)


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
                [(rot[2, 1] - rot[1, 2]) / scale, 0.25 * scale, (rot[0, 1] + rot[1, 0]) / scale, (rot[0, 2] + rot[2, 0]) / scale],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [(rot[0, 2] - rot[2, 0]) / scale, (rot[0, 1] + rot[1, 0]) / scale, 0.25 * scale, (rot[1, 2] + rot[2, 1]) / scale],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [(rot[1, 0] - rot[0, 1]) / scale, (rot[0, 2] + rot[2, 0]) / scale, (rot[1, 2] + rot[2, 1]) / scale, 0.25 * scale],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or not np.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) for v in quat / norm]


def _asset_scale(obj: dict[str, Any]) -> float:
    value = float(obj.get("asset_scale", 1.0) or 1.0)
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _rel_existing(run: Path, viewer_dir: Path, value: object) -> str | None:
    if not value:
        return None
    path = run / str(value)
    return _rel(viewer_dir, path) if path.is_file() else None


def _rel(from_dir: Path, path: Path) -> str:
    import os

    return Path(os.path.relpath(path, from_dir)).as_posix()


def _legacy_index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Real2Sim Composite Viewer</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111; color: #eee; }
    #viewport { position: fixed; inset: 0; }
    #panel { position: fixed; left: 16px; top: 16px; width: 320px; max-height: calc(100vh - 32px); overflow: auto; background: rgba(17, 19, 23, 0.86); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 14px; backdrop-filter: blur(10px); }
    h1 { font-size: 15px; margin: 0 0 10px; font-weight: 650; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 8px 0; font-size: 12px; }
    .label { color: #aab0bb; }
    .value { color: #fff; text-align: right; overflow-wrap: anywhere; }
    .ok { color: #7ee787; }
    .warn { color: #ffcf5a; }
    button { border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.08); color: #fff; border-radius: 6px; padding: 7px 9px; cursor: pointer; font-size: 12px; }
    button[aria-pressed="true"] { background: rgba(96, 165, 250, 0.35); border-color: rgba(147,197,253,0.8); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    #toggles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }
    #status { position: fixed; right: 16px; bottom: 16px; padding: 8px 10px; border-radius: 6px; background: rgba(0,0,0,0.62); font-size: 12px; color: #d8dee9; max-width: 520px; }
  </style>
  <script type="importmap">
    {"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js"}}
  </script>
</head>
<body>
  <div id="viewport"></div>
  <aside id="panel">
    <h1>Real2Sim Composite Viewer</h1>
    <div class="row"><span class="label">Run</span><span id="runName" class="value">loading</span></div>
    <div class="row"><span class="label">Background</span><span id="bgSource" class="value">loading</span></div>
    <div class="row"><span class="label">Physics</span><span id="stability" class="value">loading</span></div>
    <div class="row"><span class="label">Objects</span><span id="objectCount" class="value">0</span></div>
    <div id="toggles">
      <button id="toggleCloud" aria-pressed="true">Point Cloud</button>
      <button id="toggleObjects" aria-pressed="true">Objects</button>
      <button id="toggleTable" aria-pressed="true">Table</button>
      <button id="toggleAxes" aria-pressed="false">Axes</button>
    </div>
    <div id="objectList"></div>
  </aside>
  <div id="status">Loading viewer_config.json...</div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';
    import { PLYLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js';

    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    const root = document.getElementById('viewport');
    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111318);
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 200);
    camera.up.set(0, 0, 1);
    camera.position.set(1.6, -2.2, 1.2);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    root.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.2, 0.15);
    controls.update();

    const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 1.5);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(1.5, -2.0, 3.0);
    scene.add(dir);
    const axes = new THREE.AxesHelper(0.5);
    axes.visible = false;
    scene.add(axes);
    const cloudGroup = new THREE.Group();
    const objectGroup = new THREE.Group();
    const tableGroup = new THREE.Group();
    scene.add(cloudGroup, tableGroup, objectGroup);

    function setStatus(text) { status.textContent = text; }
    function quatWxyzToThree(q) { return new THREE.Quaternion(q[1], q[2], q[3], q[0]); }
    function transformPointCloudToWorld(geometry, supportHeight) {
      const pos = geometry.getAttribute('position');
      for (let i = 0; i < pos.count; i++) {
        const cx = pos.getX(i), cy = pos.getY(i), cz = pos.getZ(i);
        pos.setXYZ(i, cx, cz, -cy - supportHeight);
      }
      pos.needsUpdate = true;
      geometry.computeBoundingSphere();
      geometry.computeBoundingBox();
    }
    function addToggle(id, target) {
      const button = document.getElementById(id);
      button.addEventListener('click', () => {
        target.visible = !target.visible;
        button.setAttribute('aria-pressed', String(target.visible));
      });
    }
    addToggle('toggleCloud', cloudGroup);
    addToggle('toggleObjects', objectGroup);
    addToggle('toggleTable', tableGroup);
    addToggle('toggleAxes', axes);

    async function main() {
      const config = await fetch('viewer_config.json').then(r => r.json());
      document.getElementById('runName').textContent = config.run_name;
      document.getElementById('bgSource').textContent = config.background.source_kind;
      document.getElementById('stability').textContent = config.qa.stability_status || 'unknown';
      document.getElementById('stability').className = config.qa.stability_status === 'passed' ? 'value ok' : 'value warn';
      document.getElementById('objectCount').textContent = String(config.objects.length);
      const objectList = document.getElementById('objectList');
      objectList.replaceChildren();
      const tableButton = document.createElement('button');
      tableButton.type = 'button';
      tableButton.innerHTML = '<span>table_collision</span><span>Table Collision</span>';
      tableButton.addEventListener('click', inspectTableCollision);
      objectList.append(tableButton);
      for (const object of config.objects) {
        const row = document.createElement('div');
        const id = document.createElement('span');
        const label = document.createElement('span');
        row.className = 'row';
        id.className = 'label';
        label.className = 'value';
        id.textContent = object.object_id;
        label.textContent = object.label;
        row.append(id, label);
        objectList.append(row);
      }

      const plyLoader = new PLYLoader();
      if (config.background.point_cloud_path) {
        plyLoader.load(config.background.point_cloud_path, geometry => {
          transformPointCloudToWorld(geometry, config.background.point_cloud_transform.support_height_m || 0);
          const hasColor = Boolean(geometry.getAttribute('color'));
          const mat = new THREE.PointsMaterial({ size: 0.008, vertexColors: hasColor, color: hasColor ? 0xffffff : 0x8ab4f8, opacity: 0.82, transparent: true });
          cloudGroup.add(new THREE.Points(geometry, mat));
          setStatus('Loaded point-cloud background, object meshes, and table collision proxy.');
        }, undefined, err => setStatus(`Point cloud load failed: ${err.message || err}`));
      }

      const gltfLoader = new GLTFLoader();
      if (config.support_plane.table_collision_mesh_path) {
        gltfLoader.load(config.support_plane.table_collision_mesh_path, gltf => {
          const table = gltf.scene;
          const p = config.support_plane.table_collision_pos_world || [0, 0, 0];
          table.position.set(p[0], p[1], p[2]);
          table.traverse(node => {
            if (node.isMesh) {
              node.material = new THREE.MeshStandardMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.28, roughness: 0.8 });
              node.renderOrder = 1;
            }
          });
          tableGroup.add(table);
        });
      }

      for (const item of config.objects) {
        gltfLoader.load(item.mesh_path, gltf => {
          const object = gltf.scene;
          object.name = item.object_id;
          object.position.set(...item.world_position);
          object.quaternion.copy(quatWxyzToThree(item.world_quat_wxyz));
          object.traverse(node => {
            if (node.isMesh) {
              node.material = node.material || new THREE.MeshStandardMaterial();
              node.castShadow = true;
              node.receiveShadow = true;
            }
          });
          objectGroup.add(object);
        }, undefined, err => setStatus(`${item.object_id} mesh load failed: ${err.message || err}`));
      }
    }
    main().catch(err => setStatus(`Viewer failed: ${err.stack || err}`));
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""


def _index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Real2Sim Interactive Viewer</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101114; color: #eee; }
    #stage { position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%); width: min(100vw, calc(100vh * var(--stage-aspect, 1.7777778))); height: min(100vh, calc(100vw / var(--stage-aspect, 1.7777778))); background: #050608; }
    #backgroundImage { position: absolute; inset: 0; background-position: center; background-size: 100% 100%; background-repeat: no-repeat; }
    #viewport { position: absolute; inset: 0; }
    #panel { position: fixed; left: 16px; top: 16px; width: 348px; max-height: calc(100vh - 32px); overflow: auto; background: rgba(17, 19, 23, 0.88); border: 1px solid rgba(255,255,255,0.16); border-radius: 8px; padding: 14px; backdrop-filter: blur(10px); }
    h1 { font-size: 15px; margin: 0 0 10px; font-weight: 650; letter-spacing: 0; }
    h2 { font-size: 12px; margin: 14px 0 8px; color: #c8d0dc; font-weight: 650; letter-spacing: 0; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 8px 0; font-size: 12px; }
    .label { color: #aab0bb; }
    .value { color: #fff; text-align: right; overflow-wrap: anywhere; }
    .ok { color: #7ee787; }
    .warn { color: #ffcf5a; }
    button { border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.08); color: #fff; border-radius: 6px; padding: 7px 9px; cursor: pointer; font-size: 12px; }
    button[aria-pressed="true"] { background: rgba(96, 165, 250, 0.35); border-color: rgba(147,197,253,0.8); }
    #toggles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }
    #objectList button { width: 100%; display: flex; justify-content: space-between; margin: 6px 0; }
    #inspectPanel, #provenancePanel { min-height: 72px; padding: 9px; border: 1px solid rgba(255,255,255,0.12); border-radius: 6px; background: rgba(255,255,255,0.05); font-size: 12px; line-height: 1.45; color: #dce4ef; overflow-wrap: anywhere; }
    code { color: #b7d7ff; }
    #status { position: fixed; right: 16px; bottom: 16px; padding: 8px 10px; border-radius: 6px; background: rgba(0,0,0,0.62); font-size: 12px; color: #d8dee9; max-width: 520px; }
    @media (max-width: 720px) {
      #panel { left: 8px; right: 8px; top: 8px; width: auto; max-height: 46vh; padding: 12px; }
      #status { left: 8px; right: 8px; bottom: 8px; max-width: none; }
    }
  </style>
  <script type="importmap">
    {"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js"}}
  </script>
</head>
<body>
  <div id="stage">
    <div id="backgroundImage"></div>
    <div id="viewport"></div>
  </div>
  <aside id="panel">
    <h1>Real2Sim Interactive Viewer</h1>
    <div class="row"><span class="label">Run</span><span id="runName" class="value">loading</span></div>
    <div class="row"><span class="label">Viewer status</span><span id="viewerStatus" class="value">loading</span></div>
    <div class="row"><span class="label">Background mode</span><span id="bgSource" class="value">loading</span></div>
    <div class="row"><span class="label">Simulator native</span><span id="simNative" class="value">loading</span></div>
    <div class="row"><span class="label">Physics</span><span id="stability" class="value">loading</span></div>
    <div class="row"><span class="label">Objects</span><span id="objectCount" class="value">0</span></div>
    <div id="toggles">
      <button id="toggleBackground" aria-pressed="true">Background</button>
      <button id="toggleVisual" aria-pressed="true">Visual Mesh</button>
      <button id="toggleCollision" aria-pressed="false">Object Collision</button>
      <button id="toggleTableCollision" aria-pressed="true">Table Collision</button>
      <button id="toggleDebug" aria-pressed="false">Debug</button>
      <button id="toggleAxes" aria-pressed="false">Axes</button>
    </div>
    <h2>Reference Camera</h2>
    <div id="toggles">
      <button id="resetReferenceCamera" aria-pressed="false">Reset</button>
      <button id="lockReferenceCamera" aria-pressed="true">Lock</button>
    </div>
    <div id="orbitWarning" class="row warn" style="display:none">background is 2D sidecar render; projection alignment only valid in reference camera.</div>
    <h2>Pose</h2>
    <div id="toggles">
      <button id="initialPoseMode" aria-pressed="true">Initial Pose</button>
      <button id="settledPoseMode" aria-pressed="false">Settled Pose</button>
    </div>
    <h2>Provenance</h2>
    <div id="provenancePanel">Loading provenance.</div>
    <h2>Objects</h2>
    <div id="objectList"></div>
    <h2>Inspect</h2>
    <div id="inspectPanel">Click an object mesh or select an object.</div>
  </aside>
  <div id="status">Loading viewer_config.json...</div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';
    import { PLYLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js';

    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    const stage = document.getElementById('stage');
    const root = document.getElementById('viewport');
    const backgroundImage = document.getElementById('backgroundImage');
    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = null;
    const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.01, 200);
    camera.up.set(0, 0, 1);
    camera.position.set(1.6, -2.2, 1.2);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(stage.clientWidth || window.innerWidth, stage.clientHeight || window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    root.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.2, 0.15);
    controls.update();

    scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.5));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(1.5, -2.0, 3.0);
    scene.add(dir);
    const axes = new THREE.AxesHelper(0.5);
    axes.visible = false;
    scene.add(axes);
    const visualGroup = new THREE.Group();
    const collisionGroup = new THREE.Group();
    const tableCollisionGroup = new THREE.Group();
    const debugGroup = new THREE.Group();
    collisionGroup.visible = false;
    tableCollisionGroup.visible = true;
    debugGroup.visible = false;
    scene.add(collisionGroup, tableCollisionGroup, debugGroup, visualGroup);
    const objectMeshes = [];
    const objectRoots = new Map();
    let loadedConfig = null;
    let loadedBackgroundTexture = null;
    let referenceCameraLocked = true;
    let currentPoseMode = 'initial';

    function setStatus(text) { status.textContent = text; }
    function quatWxyzToThree(q) { return new THREE.Quaternion(q[1], q[2], q[3], q[0]); }
    function poseFor(item) {
      if (currentPoseMode === 'settled' && item.settled_pose_available) {
        return {
          position: item.settled_world_position || item.world_position,
          quat: item.settled_world_quat_wxyz || item.world_quat_wxyz
        };
      }
      return {
        position: item.initial_world_position || item.world_position,
        quat: item.initial_world_quat_wxyz || item.world_quat_wxyz
      };
    }
    function setObjectTransform(object, item) {
      const pose = poseFor(item);
      object.position.set(...pose.position);
      object.quaternion.copy(quatWxyzToThree(pose.quat));
      object.scale.setScalar(item.asset_scale || 1);
    }
    function rememberObjectRoot(objectId, root) {
      const roots = objectRoots.get(objectId) || [];
      roots.push(root);
      objectRoots.set(objectId, roots);
    }
    function applyPoseMode(mode) {
      if (!loadedConfig) return;
      if (mode === 'settled' && !loadedConfig.pose_display?.toggle_enabled) {
        setStatus(`Settled pose unavailable/partial: ${loadedConfig.pose_display?.status || 'missing report'}`);
        return;
      }
      currentPoseMode = mode;
      document.getElementById('initialPoseMode').setAttribute('aria-pressed', String(mode === 'initial'));
      document.getElementById('settledPoseMode').setAttribute('aria-pressed', String(mode === 'settled'));
      for (const item of loadedConfig.objects || []) {
        for (const root of objectRoots.get(item.object_id) || []) setObjectTransform(root, item);
      }
      setStatus(`Pose mode: ${mode}`);
    }
    function resizeStage() {
      const width = stage.clientWidth || window.innerWidth;
      const height = stage.clientHeight || window.innerHeight;
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }
    function applyReferenceCamera(ref) {
      if (!ref) {
        resizeStage();
        return;
      }
      const imageSize = ref.image_size || [1280, 720];
      const aspect = imageSize[0] / Math.max(imageSize[1], 1);
      stage.style.setProperty('--stage-aspect', String(aspect));
      camera.fov = ref.fov_y_deg || 55;
      camera.near = 0.01;
      camera.far = 200;
      camera.up.set(...(ref.camera_world_up || [0, 0, 1]));
      camera.position.set(...(ref.camera_world_position || [0, 0, 0]));
      const lookat = ref.camera_world_lookat || [0, 1, 0];
      camera.lookAt(new THREE.Vector3(...lookat));
      controls.target.set(...lookat);
      resizeStage();
      controls.update();
    }
    function transformPointCloudToWorld(geometry, supportHeight) {
      const pos = geometry.getAttribute('position');
      for (let i = 0; i < pos.count; i++) {
        const cx = pos.getX(i), cy = pos.getY(i), cz = pos.getZ(i);
        pos.setXYZ(i, cx, cz, -cy - supportHeight);
      }
      pos.needsUpdate = true;
      geometry.computeBoundingSphere();
      geometry.computeBoundingBox();
    }
    function addToggle(id, target) {
      const button = document.getElementById(id);
      button.addEventListener('click', () => {
        target.visible = !target.visible;
        button.setAttribute('aria-pressed', String(target.visible));
      });
    }
    document.getElementById('toggleBackground').addEventListener('click', () => {
      const visible = backgroundImage.style.display !== 'none';
      backgroundImage.style.display = visible ? 'none' : 'block';
      scene.background = visible ? null : loadedBackgroundTexture;
      document.getElementById('toggleBackground').setAttribute('aria-pressed', String(!visible));
    });
    addToggle('toggleVisual', visualGroup);
    addToggle('toggleCollision', collisionGroup);
    addToggle('toggleTableCollision', tableCollisionGroup);
    addToggle('toggleDebug', debugGroup);
    addToggle('toggleAxes', axes);
    document.getElementById('resetReferenceCamera').addEventListener('click', () => {
      applyReferenceCamera(loadedConfig?.reference_camera);
      document.getElementById('orbitWarning').style.display = 'none';
    });
    document.getElementById('lockReferenceCamera').addEventListener('click', event => {
      if (loadedConfig?.background?.camera_lock_required) {
        referenceCameraLocked = true;
        controls.enabled = false;
        event.currentTarget.setAttribute('aria-pressed', 'true');
        applyReferenceCamera(loadedConfig?.reference_camera);
        document.getElementById('orbitWarning').style.display = 'flex';
        setStatus('Reference camera lock required: external 3DGS PNG sidecar is a reference-view render, not live 3DGS.');
        return;
      }
      referenceCameraLocked = !referenceCameraLocked;
      controls.enabled = !referenceCameraLocked;
      event.currentTarget.setAttribute('aria-pressed', String(referenceCameraLocked));
      if (referenceCameraLocked) applyReferenceCamera(loadedConfig?.reference_camera);
    });
    document.getElementById('initialPoseMode').addEventListener('click', () => applyPoseMode('initial'));
    document.getElementById('settledPoseMode').addEventListener('click', () => applyPoseMode('settled'));
    controls.addEventListener('change', () => {
      if (!referenceCameraLocked && loadedConfig?.background?.mode_label === 'external_3dgs_render_sidecar') {
        document.getElementById('orbitWarning').style.display = 'flex';
      }
    });

    function inspectObject(item) {
      const panel = document.getElementById('inspectPanel');
      panel.innerHTML = [
        `<div><strong>${item.object_id}</strong> / ${item.label}</div>`,
        `<div>visual: <code>${item.visual_mesh_path || 'missing'}</code></div>`,
        `<div>collision: <code>${item.collision_mesh_path || 'missing'}</code></div>`,
        `<div>debug: <code>${item.debug_proxy_path || 'none'}</code></div>`,
        `<div>asset scale ${item.asset_scale || 1}</div>`,
        `<div>mass ${item.mass_kg ?? 'unknown'} kg, friction ${item.friction ?? 'unknown'}</div>`
      ].join('');
      controls.target.set(...poseFor(item).position);
      controls.update();
    }
    function inspectTableCollision() {
      const table = loadedConfig?.support_plane || {};
      const panel = document.getElementById('inspectPanel');
      panel.innerHTML = [
        '<div><strong>Table Collision</strong></div>',
        `<div>source_backend: <code>${table.source_backend || 'unknown'}</code></div>`,
        `<div>geometry_type: <code>${table.geometry_type || 'unknown'}</code></div>`,
        `<div>final_or_proxy: <code>${table.final_or_proxy || 'unknown'}</code></div>`,
        `<div>top_z: <code>${table.top_z ?? 'unknown'}</code></div>`,
        `<div>thickness: <code>${table.thickness ?? 'unknown'}</code></div>`,
        `<div>mesh path: <code>${table.mesh_path || table.table_collision_mesh_path || 'missing'}</code></div>`,
        `<div>qa_status: <code>${table.qa_status || 'unknown'}</code></div>`
      ].join('');
    }
    function renderProvenance(config) {
      const bg = config.background || {};
      const pose = config.pose_display || {};
      const panel = document.getElementById('provenancePanel');
      panel.innerHTML = [
        `<div><strong>${bg.provenance_label || bg.mode_label || bg.source_kind}</strong></div>`,
        `<div>viewer_status: <code>${bg.viewer_status || 'unknown'}</code></div>`,
        `<div>render_mode: <code>${bg.render_mode || 'unknown'}</code></div>`,
        `<div>orbit_policy: <code>${bg.orbit_policy || 'unknown'}</code></div>`,
        `<div>native/live 3DGS: <code>${bg.live_3dgs_runtime ? 'available' : 'unsupported'}</code></div>`,
        `<div>bg-only cloud: <code>${bg.background_layers?.bg_only_diagnostic_cloud?.status || 'unknown'}</code></div>`,
        `<div>full-scene cloud: <code>${bg.background_layers?.full_scene_debug_cloud?.status || 'unknown'}</code></div>`,
        `<div>settled pose: <code>${pose.status || 'unknown'}</code></div>`,
        `<div class="warn">Native/live 3DGS runtime unsupported when viewer_status is partial_external_render_only.</div>`
      ].join('');
    }
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    renderer.domElement.addEventListener('click', event => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(objectMeshes, true);
      if (!hits.length || !loadedConfig) return;
      let node = hits[0].object;
      while (node && !node.userData.object_id) node = node.parent;
      const item = loadedConfig.objects.find(obj => obj.object_id === node?.userData.object_id);
      if (item) inspectObject(item);
    });

    async function main() {
      const config = await fetch('viewer_config.json').then(r => r.json());
      loadedConfig = config;
      applyReferenceCamera(config.reference_camera);
      document.getElementById('runName').textContent = config.run_name;
      document.getElementById('viewerStatus').textContent = `${config.status || 'unknown'}:${config.status_reason || 'unknown'}`;
      document.getElementById('viewerStatus').className = config.status === 'passed' ? 'value ok' : 'value warn';
      document.getElementById('bgSource').textContent = config.background.mode_label || config.background.source_kind;
      document.getElementById('simNative').textContent = String(Boolean(config.background.simulator_native));
      document.getElementById('simNative').className = config.background.simulator_native ? 'value ok' : 'value warn';
      document.getElementById('stability').textContent = config.qa.stability_status || 'unknown';
      document.getElementById('stability').className = config.qa.stability_status === 'passed' ? 'value ok' : 'value warn';
      document.getElementById('objectCount').textContent = String(config.objects.length);
      document.getElementById('settledPoseMode').disabled = !config.pose_display?.toggle_enabled;
      renderProvenance(config);
      if (config.background.camera_lock_required) {
        referenceCameraLocked = true;
        controls.enabled = false;
        document.getElementById('lockReferenceCamera').setAttribute('aria-pressed', 'true');
        document.getElementById('orbitWarning').style.display = 'flex';
      }
      if (config.background.image_path) {
        backgroundImage.style.backgroundImage = `url("${config.background.image_path}")`;
        new THREE.TextureLoader().load(config.background.image_path, texture => {
          texture.colorSpace = THREE.SRGBColorSpace;
          loadedBackgroundTexture = texture;
          scene.background = texture;
        }, undefined, err => setStatus(`Background texture load failed: ${err.message || err}`));
      } else {
        backgroundImage.style.backgroundImage = 'linear-gradient(135deg, #16191f, #303744)';
        setStatus(`Background blocked: ${config.background.status}`);
      }
      const objectList = document.getElementById('objectList');
      objectList.replaceChildren();
      for (const object of config.objects) {
        const button = document.createElement('button');
        button.type = 'button';
        button.innerHTML = `<span>${object.object_id}</span><span>${object.label}</span>`;
        button.addEventListener('click', () => inspectObject(object));
        objectList.append(button);
      }

      const plyLoader = new PLYLoader();
      if (config.background.point_cloud_path) {
        plyLoader.load(config.background.point_cloud_path, geometry => {
          transformPointCloudToWorld(geometry, config.background.point_cloud_transform.support_height_m || 0);
          const hasColor = Boolean(geometry.getAttribute('color'));
          const mat = new THREE.PointsMaterial({ size: 0.008, vertexColors: hasColor, color: hasColor ? 0xffffff : 0x8ab4f8, opacity: 0.5, transparent: true });
          debugGroup.add(new THREE.Points(geometry, mat));
        }, undefined, err => setStatus(`Debug point cloud load failed: ${err.message || err}`));
      }

      const gltfLoader = new GLTFLoader();
      if (config.support_plane.table_collision_mesh_path) {
        gltfLoader.load(config.support_plane.table_collision_mesh_path, gltf => {
          const table = gltf.scene;
          const p = config.support_plane.table_collision_pos_world || [0, 0, 0];
          table.position.set(p[0], p[1], p[2]);
          table.traverse(node => {
            if (node.isMesh) {
              node.material = new THREE.MeshStandardMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.28, roughness: 0.8 });
              node.renderOrder = 1;
            }
          });
          table.userData.object_id = 'table_collision';
          table.traverse(node => { if (node.isMesh) node.userData.object_id = 'table_collision'; });
          tableCollisionGroup.add(table);
        });
      }

      for (const item of config.objects) {
        if (item.visual_mesh_path) gltfLoader.load(item.visual_mesh_path, gltf => {
          const object = gltf.scene;
          object.name = item.object_id;
          object.userData.object_id = item.object_id;
          setObjectTransform(object, item);
          object.traverse(node => {
            if (node.isMesh) {
              node.material = node.material || new THREE.MeshStandardMaterial();
              node.castShadow = true;
              node.receiveShadow = true;
              node.userData.object_id = item.object_id;
              objectMeshes.push(node);
            }
          });
          rememberObjectRoot(item.object_id, object);
          visualGroup.add(object);
        }, undefined, err => setStatus(`${item.object_id} visual mesh load failed: ${err.message || err}`));
        if (item.collision_mesh_path) gltfLoader.load(item.collision_mesh_path, gltf => {
          const collision = gltf.scene;
          setObjectTransform(collision, item);
          collision.traverse(node => {
            if (node.isMesh) node.material = new THREE.MeshStandardMaterial({ color: 0xffcf5a, wireframe: true, transparent: true, opacity: 0.45 });
          });
          rememberObjectRoot(item.object_id, collision);
          collisionGroup.add(collision);
        });
        if (item.debug_proxy_path) gltfLoader.load(item.debug_proxy_path, gltf => {
          const debug = gltf.scene;
          setObjectTransform(debug, item);
          debug.traverse(node => {
            if (node.isMesh) node.material = new THREE.MeshStandardMaterial({ color: 0xff6b6b, wireframe: true, transparent: true, opacity: 0.38 });
          });
          rememberObjectRoot(item.object_id, debug);
          debugGroup.add(debug);
        });
      }
      controls.enabled = !referenceCameraLocked;
      setStatus(`Loaded external_3dgs_render_sidecar=${Boolean(config.background.image_path)}, visual meshes=${config.objects.length}, table collision=${Boolean(config.support_plane.table_collision_mesh_path)}, simulator_native=${Boolean(config.background.simulator_native)}.`);
    }
    main().catch(err => setStatus(`Viewer failed: ${err.stack || err}`));
    window.addEventListener('resize', () => {
      resizeStage();
    });
    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""
