"""BG-table MVP: registered 3DGS background plus tabletop collision only."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

from .table_collision_qa import _camera_matrix, _mask_iou, _project_vertices, _transform, _world_to_camera


SCOPE_NAME = "background_3dgs_and_table_collision_only"
TABLE_COLLISION_SOURCE = "arkit_depth_ransac_plane"
SEMANTIC_TABLE_COLLISION_SOURCE = "semantic_masked_arkit_depth_ransac_plane"
ALLOWED_TABLE_COLLISION_SOURCES = {TABLE_COLLISION_SOURCE, SEMANTIC_TABLE_COLLISION_SOURCE}
TABLETOP_IOU_THRESHOLD = 0.65
VISIBLE_IOU_THRESHOLD = 0.70
MAX_OVERREACH_RATIO = 0.15
MAX_UNDERCOVERAGE_RATIO = 0.20
MAX_DEPTH_PLANE_MEDIAN_ABS_RESIDUAL_M = 0.02
MAX_DEPTH_PLANE_P90_ABS_RESIDUAL_M = 0.05
DEFAULT_SLAB_THICKNESS_M = 0.03
ARKIT_TO_SIM_BRIDGE_SOURCE = "phone_sim_alignment_arkit_to_sim_world"
NERFSTUDIO_GAUSSIAN_ASSET_AXIS_BRIDGE = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class TableCollisionBuildResult:
    mesh_path: Path
    usd_path: Path
    report_path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class TabletopSemanticFitResult:
    polygon_path: Path
    semantic_mask_path: Path
    refined_mask_path: Path
    report_path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class TabletopSemanticMaskResult:
    mask_path: Path
    report_path: Path
    report: dict[str, Any]


def write_bg_table_scope(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / "bg_table_scope.json"
    scope = {
        "version": 1,
        "scope": SCOPE_NAME,
        "included": {
            "registered_3dgs_background": True,
            "table_polygon_slab_collision": True,
            "phone_camera_frustums": True,
            "qa_overlays_and_provenance": True,
        },
        "ignore": {
            "object_assets": True,
            "object_poses": True,
            "object_physics": True,
        },
        "object_scope": {
            "ignored": True,
            "reason": "BG-table MVP validates only background 3DGS registration and tabletop collision alignment.",
        },
    }
    path.write_text(json.dumps(scope, indent=2), encoding="utf-8")
    return path


def segment_tabletop_semantic_mask(
    run_dir: str | Path,
    *,
    sam3_client: Any,
    prompt: str = "tabletop / coffee table top",
    bbox_xyxy: tuple[int, int, int, int] | None = None,
    frame_index: int = 0,
) -> TabletopSemanticMaskResult:
    run = Path(run_dir)
    bg_dir = run / "background"
    qa_dir = run / "qa"
    bg_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    mask_path = bg_dir / "tabletop_semantic_mask.png"
    report_path = qa_dir / "tabletop_semantic_mask_report.json"
    trajectory = _load_json(run / "trajectory.json")
    frames = _select_trajectory_frames(trajectory, [int(frame_index)])
    frame = frames[0] if frames else {}
    image_rel = frame.get("rgb_path") or f"frames/frame_{int(frame_index):06d}.jpg"
    image_path = run / str(image_rel)
    if not image_path.is_file():
        report = {
            "version": 1,
            "status": "blocked",
            "source_backend": "sam3_box_prompt" if bbox_xyxy else "sam3_text_prompt",
            "frame_index": int(frame_index),
            "image_path": str(image_rel),
            "semantic_mask_path": None,
            "blocking_reasons": ["rgb_frame_missing"],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TabletopSemanticMaskResult(mask_path, report_path, report)

    if bbox_xyxy is not None:
        mask = np.asarray(sam3_client.segment_box(image_path, bbox_xyxy), dtype=bool)
        source_backend = "sam3_box_prompt"
    else:
        mask = np.asarray(sam3_client.segment_text(image_path, prompt), dtype=bool)
        source_backend = "sam3_text_prompt"
    if mask.ndim != 2 or int(np.count_nonzero(mask)) == 0:
        report = {
            "version": 1,
            "status": "blocked",
            "source_backend": source_backend,
            "frame_index": int(frame_index),
            "image_path": str(image_rel),
            "semantic_mask_path": None,
            "mask_area_px": int(np.count_nonzero(mask)) if mask.ndim == 2 else None,
            "blocking_reasons": ["sam3_tabletop_mask_empty_or_invalid"],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TabletopSemanticMaskResult(mask_path, report_path, report)

    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    report = {
        "version": 1,
        "status": "passed",
        "source_backend": source_backend,
        "frame_index": int(frame_index),
        "image_path": str(image_rel),
        "prompt": None if bbox_xyxy is not None else prompt,
        "bbox_xyxy": list(bbox_xyxy) if bbox_xyxy is not None else None,
        "semantic_mask_path": "background/tabletop_semantic_mask.png",
        "mask_area_px": int(np.count_nonzero(mask)),
        "claim_boundary": {
            "semantic_tabletop_localization": "sam3_mask_candidate",
            "metric_geometry": "not_claimed_until_fit_tabletop_from_mask_uses_phone_depth",
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TabletopSemanticMaskResult(mask_path, report_path, report)


def fit_tabletop_from_semantic_mask(
    run_dir: str | Path,
    *,
    mask: str | Path = "background/tabletop_semantic_mask.png",
    frame_index: int = 0,
    hull: str = "convex-hull",
    plane_distance_threshold_m: float = DEFAULT_PLANE_DISTANCE_THRESHOLD_M if "DEFAULT_PLANE_DISTANCE_THRESHOLD_M" in globals() else 0.02,
    write: bool = False,
) -> TabletopSemanticFitResult:
    run = Path(run_dir)
    bg_dir = run / "background"
    qa_dir = run / "qa"
    bg_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    report_path = qa_dir / "tabletop_semantic_fit_report.json"
    polygon_path = bg_dir / "table_polygon_world.json"
    semantic_mask_path = bg_dir / "tabletop_semantic_mask.png"
    refined_mask_path = bg_dir / "tabletop_mask.png"

    camera = _load_json(run / "camera.json")
    trajectory = _load_json(run / "trajectory.json")
    frames = _select_trajectory_frames(trajectory, [int(frame_index)])
    mask_src = _resolve_run_path(run, mask)
    blocking_reasons: list[str] = []
    if not mask_src.is_file():
        blocking_reasons.append("tabletop_semantic_mask_missing")
    if not frames:
        blocking_reasons.append("trajectory_frame_missing")
    k = _camera_matrix(camera)
    if k is None:
        blocking_reasons.append("camera_intrinsics_missing")
    if hull not in {"convex-hull", "clipped-convex-hull", "rotated-rectangle"}:
        blocking_reasons.append("unsupported_hull_method")

    frame = frames[0] if frames else {}
    t_camera_to_world = _transform(frame.get("T_camera_to_world")) if frame else None
    if t_camera_to_world is None:
        blocking_reasons.append("camera_extrinsics_missing")
    depth_path = run / str(frame.get("depth_path", "")) if frame else run / "<missing-depth>"
    confidence_path = run / str(frame.get("confidence_path", "")) if frame else run / "<missing-confidence>"
    if not depth_path.is_file():
        blocking_reasons.append("depth_frame_missing")
    if not confidence_path.is_file():
        blocking_reasons.append("confidence_frame_missing")

    if blocking_reasons:
        report = _semantic_fit_blocked_report(run, blocking_reasons, mask_src, frame_index, hull)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TabletopSemanticFitResult(polygon_path, semantic_mask_path, refined_mask_path, report_path, report)

    semantic_mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
    depth = np.load(depth_path)
    confidence = np.asarray(Image.open(confidence_path).convert("L"))
    if semantic_mask is None or depth.ndim != 2 or semantic_mask.shape != depth.shape or confidence.shape != depth.shape:
        reasons = []
        if semantic_mask is None:
            reasons.append("tabletop_semantic_mask_unreadable")
        if depth.ndim != 2:
            reasons.append("depth_frame_not_2d")
        if semantic_mask is not None and semantic_mask.shape != depth.shape:
            reasons.append("semantic_mask_depth_shape_mismatch")
        if confidence.shape != depth.shape:
            reasons.append("confidence_depth_shape_mismatch")
        report = _semantic_fit_blocked_report(run, reasons, mask_src, frame_index, hull)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TabletopSemanticFitResult(polygon_path, semantic_mask_path, refined_mask_path, report_path, report)

    semantic_binary = semantic_mask > 0
    valid = semantic_binary & np.isfinite(depth) & (depth > 0.0) & (confidence > 0)
    points_world, rows, cols = _backproject_selected_depth(depth, valid, k, t_camera_to_world)
    if points_world.shape[0] < 3:
        report = _semantic_fit_blocked_report(run, ["insufficient_semantic_depth_points"], mask_src, frame_index, hull)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TabletopSemanticFitResult(polygon_path, semantic_mask_path, refined_mask_path, report_path, report)

    plane = _fit_plane_ransac_points(points_world, distance_threshold_m=float(plane_distance_threshold_m))
    if plane is None:
        report = _semantic_fit_blocked_report(run, ["semantic_depth_plane_ransac_failed"], mask_src, frame_index, hull)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TabletopSemanticFitResult(polygon_path, semantic_mask_path, refined_mask_path, report_path, report)
    normal, offset, inliers = plane
    if int(np.count_nonzero(inliers)) < 3:
        report = _semantic_fit_blocked_report(run, ["insufficient_near_plane_semantic_points"], mask_src, frame_index, hull)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TabletopSemanticFitResult(polygon_path, semantic_mask_path, refined_mask_path, report_path, report)

    near_points = points_world[inliers]
    near_rows = rows[inliers]
    near_cols = cols[inliers]
    polygon_xy = _tabletop_hull_xy(near_points[:, :2], method=hull)
    if polygon_xy is None:
        report = _semantic_fit_blocked_report(run, ["semantic_depth_polygon_hull_failed"], mask_src, frame_index, hull)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return TabletopSemanticFitResult(polygon_path, semantic_mask_path, refined_mask_path, report_path, report)

    refined_mask = np.zeros(depth.shape, dtype=np.uint8)
    refined_mask[near_rows, near_cols] = 255
    top_z = float(np.median(near_points[:, 2]))
    residuals = np.abs(points_world @ normal + offset)
    inlier_residuals = residuals[inliers]
    extent = np.max(polygon_xy, axis=0) - np.min(polygon_xy, axis=0)
    polygon = {
        "version": 1,
        "status": "passed",
        "source_backend": SEMANTIC_TABLE_COLLISION_SOURCE,
        "semantic_source_backend": "tabletop_semantic_mask",
        "geometry_type": f"{hull}_from_semantic_masked_depth_plane",
        "coordinate_world": "sim_world",
        "coordinate_frame": "sim_world",
        "unit": "meter",
        "polygon_world_xy": [[float(x), float(y)] for x, y in polygon_xy.tolist()],
        "polygons_world_xy": [[[float(x), float(y)] for x, y in polygon_xy.tolist()]],
        "polygon_count": 1,
        "top_z_m": top_z,
        "support_height_m": top_z,
        "semantic_mask_path": "background/tabletop_semantic_mask.png",
        "refined_plane_mask_path": "background/tabletop_mask.png",
        "plane_world": {
            "normal": [float(v) for v in normal.tolist()],
            "offset_m": float(offset),
            "equation": "normal dot X + offset = 0",
        },
        "polygon_extent_x_m": float(extent[0]),
        "polygon_extent_y_m": float(extent[1]),
        "input_semantic_depth_point_count": int(points_world.shape[0]),
        "near_plane_point_count": int(np.count_nonzero(inliers)),
    }
    report = {
        "version": 1,
        "status": "passed",
        "source_backend": SEMANTIC_TABLE_COLLISION_SOURCE,
        "frame_index": int(frame_index),
        "frame_id": frame.get("frame_id"),
        "hull_method": hull,
        "semantic_mask_path": "background/tabletop_semantic_mask.png",
        "refined_plane_mask_path": "background/tabletop_mask.png",
        "polygon_path": "background/table_polygon_world.json",
        "depth_path": _rel(run, depth_path),
        "confidence_path": _rel(run, confidence_path),
        "metrics": {
            "semantic_mask_area_px": int(np.count_nonzero(semantic_binary)),
            "masked_depth_point_count": int(points_world.shape[0]),
            "near_plane_mask_area_px": int(np.count_nonzero(refined_mask)),
            "plane_inlier_count": int(np.count_nonzero(inliers)),
            "plane_inlier_ratio": float(np.count_nonzero(inliers) / max(1, points_world.shape[0])),
            "plane_distance_threshold_m": float(plane_distance_threshold_m),
            "plane_residual_median_m": float(np.median(inlier_residuals)),
            "plane_residual_p90_m": float(np.percentile(inlier_residuals, 90.0)),
            "top_z_m": top_z,
        },
        "claim_boundary": {
            "semantic_tabletop_localization": "provided_by_mask_not_inferred_from_3dgs",
            "metric_geometry": "phone_depth_confidence_and_explicit_camera_pose",
            "not_collision_source": ["3dgs_splat_centers", "bbox_proxy", "full_scene_largest_plane"],
        },
        "blocking_reasons": [],
    }
    if write:
        if mask_src.resolve() != semantic_mask_path.resolve():
            shutil.copyfile(mask_src, semantic_mask_path)
        else:
            Image.fromarray(semantic_mask.astype(np.uint8)).save(semantic_mask_path)
        Image.fromarray(refined_mask).save(refined_mask_path)
        polygon_path.write_text(json.dumps(polygon, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TabletopSemanticFitResult(polygon_path, semantic_mask_path, refined_mask_path, report_path, report)


def build_table_collision(
    run_dir: str | Path,
    *,
    source: str | Path = "background/table_polygon_world.json",
    write: bool = False,
    thickness_m: float = DEFAULT_SLAB_THICKNESS_M,
) -> TableCollisionBuildResult:
    run = Path(run_dir)
    source_path = _resolve_run_path(run, source)
    polygon = _load_json(source_path)
    _validate_world_units(polygon)
    polygons_xy = _polygon_xy_sets(polygon)
    if not polygons_xy:
        raise ValueError("background/table_polygon_world.json does not contain a valid polygon_world_xy")
    top_z = float(polygon.get("top_z_m", polygon.get("table_top_z_m", 0.0)) or 0.0)
    thickness = float(thickness_m)
    if thickness <= 0.0:
        raise ValueError("table slab thickness must be positive")

    meshes = [_world_polygon_slab_mesh(poly, top_z=top_z, thickness=thickness) for poly in polygons_xy]
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    table_dir = run / "table"
    table_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = table_dir / "collision_polygon_slab.glb"
    usd_path = table_dir / "collision_polygon_slab.usd"
    report_path = table_dir / "collision_polygon_slab_report.json"
    if write:
        mesh.export(mesh_path)
        usd_path.write_text(_slab_usd_ascii(mesh, mesh_path=mesh_path.name), encoding="utf-8")
    scope_path = write_bg_table_scope(run)
    all_xy = np.concatenate(polygons_xy, axis=0)
    report = {
        "version": 1,
        "status": "passed",
        "source": "background_table_polygon_world",
        "source_backend": polygon.get("source_backend") or TABLE_COLLISION_SOURCE,
        "source_path": _rel(run, source_path),
        "mesh_path": _rel(run, mesh_path),
        "usd_path": _rel(run, usd_path),
        "scope_path": _rel(run, scope_path),
        "coordinate_world": "sim_world",
        "coordinate_frame": "sim_world",
        "unit": "meter",
        "top_z_m": top_z,
        "thickness_m": thickness,
        "thickness_direction": "down_negative_z",
        "slab_bottom_z_m": float(top_z - thickness),
        "polygon_count": len(polygons_xy),
        "polygon_extent_x_m": float(np.max(all_xy[:, 0]) - np.min(all_xy[:, 0])),
        "polygon_extent_y_m": float(np.max(all_xy[:, 1]) - np.min(all_xy[:, 1])),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "collision_role": "physics_collider",
        "visual_role": "debug_visible_collision_only",
        "not_collision_source": {
            "3dgs_splat_centers": True,
            "table_bbox_proxy": True,
            "object_assets": True,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TableCollisionBuildResult(mesh_path=mesh_path, usd_path=usd_path, report_path=report_path, report=report)


def qa_bg_table(
    run_dir: str | Path,
    *,
    frames: str = "0,20,40,60",
    write_overlays: bool = False,
) -> dict[str, Any]:
    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    scope_path = write_bg_table_scope(run)
    registration = _load_json(run / "background" / "registration.json")
    phone_alignment = _load_json(run / "background" / "phone_sim_alignment.json")
    polygon = _load_json(run / "background" / "table_polygon_world.json")
    camera = _load_json(run / "camera.json")
    trajectory = _load_json(run / "trajectory.json")
    frame_indices = _parse_frame_indices(frames)
    hard_reasons: list[str] = []
    partial_reasons: list[str] = []

    transform_3dgs = registration.get("T_3dgs_world_to_sim_world") or (
        registration.get("transforms", {}) if isinstance(registration.get("transforms"), dict) else {}
    ).get("T_3dgs_world_to_sim_world")
    if registration.get("status") != "registered":
        hard_reasons.append("background_registration_not_registered")
    t_3dgs_to_sim = _transform(transform_3dgs)
    if t_3dgs_to_sim is None:
        hard_reasons.append("missing_T_3dgs_world_to_sim_world")
    registration_bridge = _background_registration_coordinate_bridge_contract(registration)
    hard_reasons.extend(registration_bridge["blocking_reasons"])

    table_contract = _table_polygon_contract(polygon)
    hard_reasons.extend(table_contract.pop("blocking_reasons"))
    projection = _table_projection_metrics(run, polygon, camera)
    hard_reasons.extend(projection.pop("blocking_reasons"))
    depth_metrics = _depth_plane_metrics(run, polygon, camera, trajectory, frame_indices)
    hard_reasons.extend(depth_metrics.pop("blocking_reasons"))
    overlay_evidence = _load_or_render_3dgs_overlay_evidence(
        run,
        frame_indices=frame_indices,
        allow_render=write_overlays and t_3dgs_to_sim is not None,
        transform_3dgs=transform_3dgs,
    )
    if overlay_evidence.get("status") != "rendered":
        partial_reasons.append("registered_3dgs_overlay_not_rendered")
    overlay_paths = (
        _write_bg_table_overlays(run, polygon, camera, trajectory, overlay_evidence, frame_indices)
        if write_overlays and overlay_evidence.get("status") == "rendered"
        else []
    )
    splat_diag = _splat_center_diagnostic(run, polygon, t_3dgs_to_sim)

    if projection.get("tabletop_iou") is None or projection["tabletop_iou"] < TABLETOP_IOU_THRESHOLD:
        hard_reasons.append("tabletop_iou_below_threshold")
    if projection.get("visible_iou") is None or projection["visible_iou"] < VISIBLE_IOU_THRESHOLD:
        hard_reasons.append("visible_iou_below_threshold")
    if projection.get("overreach_ratio") is None or projection["overreach_ratio"] > MAX_OVERREACH_RATIO:
        hard_reasons.append("overreach_ratio_above_threshold")
    if projection.get("undercoverage_ratio") is None or projection["undercoverage_ratio"] > MAX_UNDERCOVERAGE_RATIO:
        hard_reasons.append("undercoverage_ratio_above_threshold")
    if (
        depth_metrics.get("median_abs_residual_m") is None
        or depth_metrics["median_abs_residual_m"] > MAX_DEPTH_PLANE_MEDIAN_ABS_RESIDUAL_M
    ):
        hard_reasons.append("depth_plane_median_abs_residual_above_threshold")

    hard_blocking_reasons = _dedupe(hard_reasons)
    partial_blocking_reasons = _dedupe(partial_reasons)
    blocking_reasons = _dedupe([*hard_blocking_reasons, *partial_blocking_reasons])
    status = "blocked" if hard_blocking_reasons else ("partial" if partial_blocking_reasons else "passed")
    report = {
        "version": 1,
        "status": status,
        "scope": SCOPE_NAME,
        "blocking_reasons": blocking_reasons,
        "hard_blocking_reasons": hard_blocking_reasons,
        "partial_blocking_reasons": partial_blocking_reasons,
        "scope_path": _rel(run, scope_path),
        "background": {
            "registration_status": registration.get("status"),
            "registration_path": "background/registration.json" if (run / "background" / "registration.json").is_file() else None,
            "T_3dgs_world_to_sim_world": transform_3dgs,
            "transform_source": (registration.get("transform_sources") or {}).get("T_3dgs_world_to_sim_world"),
            "coordinate_bridge_status": registration_bridge["status"],
            "coordinate_bridge": registration_bridge,
            "splat_rgb_path": "background/3dgs_native/splat_rgb.ply"
            if (run / "background" / "3dgs_native" / "splat_rgb.ply").is_file()
            else None,
        },
        "phone_sim_alignment": {
            "status": phone_alignment.get("status"),
            "source_backend": phone_alignment.get("source_backend"),
            "metrics": phone_alignment.get("metrics", {}),
        },
        "table_polygon": table_contract,
        "table_collision": {
            "source": "background_table_polygon_world",
            "source_backend": table_contract.get("source_backend"),
            "polygon_path": "background/table_polygon_world.json",
            "mesh_path": "table/collision_polygon_slab.glb" if (run / "table" / "collision_polygon_slab.glb").is_file() else None,
            "usd_path": "table/collision_polygon_slab.usd" if (run / "table" / "collision_polygon_slab.usd").is_file() else None,
            "projection_iou": projection.get("tabletop_iou"),
            "tabletop_iou": projection.get("tabletop_iou"),
            "projection_mask_source": projection.get("projection_mask_source"),
            "projection_mask_path": projection.get("projection_mask_path"),
            "visible_iou": projection.get("visible_iou"),
            "overreach_ratio": projection.get("overreach_ratio"),
            "undercoverage_ratio": projection.get("undercoverage_ratio"),
            "raw_projection_iou": projection.get("raw_tabletop_iou"),
            "raw_overreach_ratio": projection.get("raw_overreach_ratio"),
            "depth_plane_median_abs_residual_m": depth_metrics.get("median_abs_residual_m"),
            "depth_plane_p90_abs_residual_m": depth_metrics.get("p90_abs_residual_m"),
        },
        "thresholds": {
            "table_collision.projection_iou": TABLETOP_IOU_THRESHOLD,
            "table_collision.visible_iou": VISIBLE_IOU_THRESHOLD,
            "table_collision.overreach_ratio": MAX_OVERREACH_RATIO,
            "table_collision.undercoverage_ratio": MAX_UNDERCOVERAGE_RATIO,
            "table_collision.depth_plane_median_abs_residual_m": MAX_DEPTH_PLANE_MEDIAN_ABS_RESIDUAL_M,
        },
        "overlays": [_rel(run, path) for path in overlay_paths],
        "3dgs_overlay": overlay_evidence,
        "splat_center_diagnostic": splat_diag,
        "object_scope": {
            "ignored": True,
            "object_assets": "ignored",
            "object_poses": "ignored",
            "object_physics": "ignored",
        },
        "claim": "registered 3DGS background and tabletop collision are aligned in the same sim world" if status == "passed" else None,
        "simulator_native_3dgs": False,
        "simulator_native_3dgs_evidence": "not_claimed_without_native_runtime_evidence",
    }
    report_path = qa_dir / "bg_table_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _table_polygon_contract(polygon: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    source_backend = polygon.get("source_backend")
    coordinate_world = polygon.get("coordinate_world")
    coordinate_frame = polygon.get("coordinate_frame")
    unit = polygon.get("unit")
    if source_backend not in ALLOWED_TABLE_COLLISION_SOURCES:
        reasons.append("table_polygon_source_backend_not_supported")
    if coordinate_world != "sim_world":
        reasons.append("table_polygon_coordinate_world_not_sim_world")
    if coordinate_frame != "sim_world":
        reasons.append("table_polygon_coordinate_frame_not_sim_world")
    if unit != "meter":
        reasons.append("table_polygon_unit_not_meter")
    return {
        "status": polygon.get("status"),
        "source_backend": source_backend,
        "coordinate_world": coordinate_world,
        "coordinate_frame": coordinate_frame,
        "unit": unit,
        "top_z_m": polygon.get("top_z_m", polygon.get("table_top_z_m")),
        "blocking_reasons": reasons,
    }


def _background_registration_coordinate_bridge_contract(registration: dict[str, Any]) -> dict[str, Any]:
    bridge = registration.get("coordinate_bridge") if isinstance(registration.get("coordinate_bridge"), dict) else {}
    transform_source = str((registration.get("transform_sources") or {}).get("T_3dgs_world_to_sim_world") or "")
    convention = registration.get("coordinate_convention") if isinstance(registration.get("coordinate_convention"), dict) else {}
    gs_convention = str(convention.get("3dgs") or "").lower()
    bridge_source_world = str(bridge.get("source_world") or "").lower()
    requires_arkit_bridge = "arkit" in gs_convention or bridge_source_world == "arkit"
    reasons: list[str] = []
    if requires_arkit_bridge and (
        bridge.get("status") != "closed" or transform_source != ARKIT_TO_SIM_BRIDGE_SOURCE
    ):
        reasons.append("background_registration_coordinate_bridge_not_closed")
    return {
        "status": str(bridge.get("status") or ("blocked" if reasons else "not_required")),
        "source": bridge.get("source"),
        "source_world": bridge.get("source_world"),
        "target_world": bridge.get("target_world"),
        "requires_arkit_bridge": requires_arkit_bridge,
        "transform_source": transform_source or None,
        "phone_sim_alignment_path": bridge.get("phone_sim_alignment_path"),
        "blocking_reasons": reasons,
    }


def _load_or_render_3dgs_overlay_evidence(
    run: Path,
    *,
    frame_indices: list[int],
    allow_render: bool,
    transform_3dgs: Any,
) -> dict[str, Any]:
    manifest = _registered_3dgs_table_render_manifest(run, frame_indices)
    if manifest.get("status") == "rendered":
        return manifest
    sidecar = _single_registered_3dgs_render_evidence(run, frame_indices)
    if sidecar.get("status") == "rendered":
        return sidecar
    if allow_render:
        rendered = _render_registered_3dgs_frames(run, frame_indices=frame_indices, transform_3dgs=transform_3dgs)
        if rendered.get("status") == "rendered":
            return rendered
        return rendered
    return {
        "status": "blocked",
        "runtime": None,
        "uses_T_3dgs_world_to_sim_world": False,
        "blocking_reason": manifest.get("blocking_reason") or sidecar.get("blocking_reason") or "registered_3dgs_overlay_not_rendered",
        "checked_paths": [
            "qa/registered_3dgs_table_renders.json",
            "qa/background_3dgs_render_report.json",
        ],
        "live_3dgs_runtime": False,
        "simulator_native": False,
    }


def _registered_3dgs_table_render_manifest(run: Path, frame_indices: list[int]) -> dict[str, Any]:
    path = run / "qa" / "registered_3dgs_table_renders.json"
    data = _load_json(path)
    if data.get("status") != "rendered":
        return {"status": "blocked", "blocking_reason": "registered_3dgs_table_render_manifest_missing_or_not_rendered"}
    if data.get("uses_T_3dgs_world_to_sim_world") is not True:
        return {"status": "blocked", "blocking_reason": "registered_3dgs_render_did_not_use_transform"}
    frames = _normalize_render_frames(run, data.get("frames"), frame_indices)
    if frames is None:
        return {"status": "blocked", "blocking_reason": "registered_3dgs_render_frames_missing"}
    return {
        "status": "rendered",
        "backend": data.get("backend"),
        "runtime": data.get("runtime") or data.get("backend"),
        "manifest_path": "qa/registered_3dgs_table_renders.json",
        "uses_T_3dgs_world_to_sim_world": True,
        "live_3dgs_runtime": bool(data.get("live_3dgs_runtime", False)),
        "simulator_native": bool(data.get("simulator_native", False)),
        "frames": frames,
        "render_paths_by_frame": {str(item["frame_index"]): item["render_path"] for item in frames},
    }


def _single_registered_3dgs_render_evidence(run: Path, frame_indices: list[int]) -> dict[str, Any]:
    requested = frame_indices or [0]
    if len(requested) != 1:
        return {"status": "blocked", "blocking_reason": "single_3dgs_render_cannot_cover_selected_frames"}
    report_path = run / "qa" / "background_3dgs_render_report.json"
    report = _load_json(report_path)
    render_rel = report.get("render_path")
    render_path = run / str(render_rel) if render_rel else None
    if report.get("status") != "rendered" or report.get("registered_3dgs_rendered") is not True or render_path is None or not render_path.is_file():
        return {"status": "blocked", "blocking_reason": "background_3dgs_render_report_missing_or_not_rendered"}
    frame_index = int(requested[0])
    frames = [{"frame_index": frame_index, "render_path": str(render_rel)}]
    return {
        "status": "rendered",
        "backend": report.get("backend"),
        "runtime": "external-3dgs-renderer",
        "manifest_path": "qa/background_3dgs_render_report.json",
        "uses_T_3dgs_world_to_sim_world": True,
        "live_3dgs_runtime": False,
        "simulator_native": False,
        "frames": frames,
        "render_paths_by_frame": {str(frame_index): str(render_rel)},
    }


def _render_registered_3dgs_frames(run: Path, *, frame_indices: list[int], transform_3dgs: Any) -> dict[str, Any]:
    from .background_3dgs_render import render_external_3dgs_background

    requested = frame_indices or [0]
    rendered_frames: list[dict[str, Any]] = []
    for frame_index in requested:
        result = render_external_3dgs_background(run, camera_idx=int(frame_index))
        if result.report.get("status") != "rendered":
            return {
                "status": "blocked",
                "runtime": "external-3dgs-renderer",
                "uses_T_3dgs_world_to_sim_world": True,
                "blocking_reason": result.report.get("blocked_reason") or "registered_3dgs_render_failed",
                "render_report": result.report,
                "live_3dgs_runtime": False,
                "simulator_native": False,
            }
        render_rel = result.report.get("render_path")
        if not render_rel or not (run / str(render_rel)).is_file():
            return {
                "status": "blocked",
                "runtime": "external-3dgs-renderer",
                "uses_T_3dgs_world_to_sim_world": True,
                "blocking_reason": "registered_3dgs_render_output_missing",
                "render_report": result.report,
                "live_3dgs_runtime": False,
                "simulator_native": False,
            }
        per_frame_path = run / "qa" / f"bg3dgs_render_{int(frame_index):06d}.png"
        shutil.copyfile(run / str(render_rel), per_frame_path)
        rendered_frames.append({"frame_index": int(frame_index), "render_path": _rel(run, per_frame_path)})
    manifest = {
        "version": 1,
        "status": "rendered",
        "backend": "external_3dgs_renderer",
        "runtime": "external-3dgs-renderer",
        "uses_T_3dgs_world_to_sim_world": True,
        "T_3dgs_world_to_sim_world": transform_3dgs,
        "live_3dgs_runtime": False,
        "simulator_native": False,
        "frames": rendered_frames,
    }
    manifest_path = run / "qa" / "registered_3dgs_table_renders.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return _registered_3dgs_table_render_manifest(run, frame_indices)


def _normalize_render_frames(run: Path, value: Any, frame_indices: list[int]) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    requested = set(frame_indices) if frame_indices else None
    frames: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        frame_index = item.get("frame_index")
        if frame_index is None:
            frame_index = _frame_number(str(item.get("frame_id") or ""))
        try:
            frame_index = int(frame_index)
        except (TypeError, ValueError):
            continue
        render_rel = item.get("render_path")
        if not render_rel or not (run / str(render_rel)).is_file():
            continue
        if requested is None or frame_index in requested:
            frames.append({"frame_index": frame_index, "frame_id": item.get("frame_id"), "render_path": str(render_rel)})
    if requested is not None and {item["frame_index"] for item in frames} != requested:
        return None
    return frames


def _table_projection_metrics(run: Path, polygon: dict[str, Any], camera: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    semantic_mask_path = run / "background" / "tabletop_semantic_mask.png"
    fallback_mask_path = run / "background" / "tabletop_mask.png"
    if semantic_mask_path.is_file():
        mask_path = semantic_mask_path
        mask_source = "semantic_tabletop_mask"
    else:
        mask_path = fallback_mask_path
        mask_source = "refined_or_legacy_tabletop_mask"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
    if mask is None:
        reasons.append("tabletop_mask_missing_or_unreadable")
        return {"blocking_reasons": reasons, "projection_mask_source": mask_source, "projection_mask_path": _rel(run, mask_path)}
    k = _camera_matrix(camera)
    if k is None:
        reasons.append("camera_intrinsics_missing")
    t_world_to_camera = _world_to_camera(camera, support_height_m=float(polygon.get("support_height_m", 0.0) or 0.0))
    if t_world_to_camera is None:
        reasons.append("camera_extrinsics_missing")
    vertex_sets = _polygon_world_vertex_sets(polygon)
    if not vertex_sets:
        reasons.append("table_polygon_vertices_missing")
    if reasons:
        return {"blocking_reasons": reasons}

    projected_mask = np.zeros(mask.shape, dtype=np.uint8)
    for vertices in vertex_sets:
        points_px = _project_vertices(vertices, k, t_world_to_camera)
        if points_px is None:
            points_px = _project_visible_polygon_sample(vertices, k, t_world_to_camera, mask.shape)
        if points_px is None:
            reasons.append("table_polygon_projection_invalid")
            continue
        hull = cv2.convexHull(points_px.astype(np.float32)).reshape(-1, 2)
        cv2.fillPoly(projected_mask, [np.rint(hull).astype(np.int32)], 255)
    if reasons:
        return {"blocking_reasons": reasons}

    projected = projected_mask > 0
    target = mask > 0
    raw_projected_area = int(np.count_nonzero(projected))
    target_area = int(np.count_nonzero(target))
    raw_intersection = int(np.count_nonzero(projected & target))
    raw_tabletop_iou = _mask_iou(projected_mask, mask)
    raw_overreach = _safe_ratio(raw_projected_area - raw_intersection, raw_projected_area)
    visible_projected = projected & target
    visible_projected_mask = visible_projected.astype(np.uint8) * 255
    visible_iou = _mask_iou(visible_projected_mask, mask)
    visible_projected_area = int(np.count_nonzero(visible_projected))
    intersection = int(np.count_nonzero(visible_projected & target))
    undercoverage = _safe_ratio(target_area - intersection, target_area)
    overlay_path = run / "qa" / "bg_table_projection_overlay.png"
    _write_projection_overlay(overlay_path, mask, projected_mask)
    return {
        "blocking_reasons": [],
        "tabletop_iou": raw_tabletop_iou,
        "visible_iou": visible_iou,
        "overreach_ratio": raw_overreach,
        "undercoverage_ratio": undercoverage,
        "projected_polygon_area_px": raw_projected_area,
        "visible_projected_polygon_area_px": visible_projected_area,
        "raw_projected_polygon_area_px": raw_projected_area,
        "tabletop_mask_area_px": target_area,
        "intersection_area_px": intersection,
        "raw_intersection_area_px": raw_intersection,
        "raw_tabletop_iou": raw_tabletop_iou,
        "raw_overreach_ratio": raw_overreach,
        "overlay_path": _rel(run, overlay_path),
        "projection_mask_source": mask_source,
        "projection_mask_path": _rel(run, mask_path),
    }


def _depth_plane_metrics(
    run: Path,
    polygon: dict[str, Any],
    camera: dict[str, Any],
    trajectory: dict[str, Any],
    frame_indices: list[int],
) -> dict[str, Any]:
    reasons: list[str] = []
    k = _camera_matrix(camera)
    if k is None:
        return {"blocking_reasons": ["camera_intrinsics_missing"]}
    frames = _select_trajectory_frames(trajectory, frame_indices)
    if not frames:
        return {"blocking_reasons": ["trajectory_frames_missing"]}
    polygons_xy = _polygon_xy_sets(polygon)
    if not polygons_xy:
        return {"blocking_reasons": ["table_polygon_vertices_missing"]}
    top_z = float(polygon.get("top_z_m", polygon.get("table_top_z_m", 0.0)) or 0.0)
    mask_metrics = _depth_plane_metrics_from_tabletop_mask(run, camera, frames, k, top_z)
    if mask_metrics is not None:
        return mask_metrics
    residuals: list[np.ndarray] = []
    for frame in frames:
        depth_rel = frame.get("depth_path")
        transform = _transform(frame.get("T_camera_to_world"))
        if not depth_rel or transform is None:
            continue
        depth_path = run / str(depth_rel)
        if not depth_path.is_file():
            continue
        depth = np.load(depth_path)
        if depth.ndim != 2:
            continue
        points = _backproject_depth(depth, k, transform, stride=max(1, int(round(max(depth.shape) / 256))))
        if points.size == 0:
            continue
        inside = _points_inside_any_polygon(points[:, :2], polygons_xy)
        selected = points[inside]
        if selected.size:
            residuals.append(np.abs(selected[:, 2] - top_z))
    if not residuals:
        metrics = ((_load_json(run / "background" / "phone_sim_alignment.json").get("metrics") or {}))
        median = metrics.get("plane_residual_median_m")
        p90 = metrics.get("plane_residual_p90_m")
        if median is not None and p90 is not None:
            return {
                "blocking_reasons": [],
                "median_abs_residual_m": float(median),
                "p90_abs_residual_m": float(p90),
                "source": "phone_sim_alignment_metrics_fallback",
            }
        reasons.append("no_phone_depth_points_inside_table_polygon")
        return {"blocking_reasons": reasons, "median_abs_residual_m": None, "p90_abs_residual_m": None}
    values = np.concatenate(residuals)
    return {
        "blocking_reasons": [],
        "median_abs_residual_m": float(np.median(values)),
        "p90_abs_residual_m": float(np.percentile(values, 90.0)),
        "sample_count": int(values.size),
        "source": "phone_depth_point_cloud_inside_table_polygon",
    }


def _project_visible_polygon_sample(
    vertices_world: np.ndarray,
    k: np.ndarray,
    t_world_to_camera: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    xy = vertices_world[:, :2]
    z = float(np.mean(vertices_world[:, 2]))
    x0, y0 = np.min(xy, axis=0)
    x1, y1 = np.max(xy, axis=0)
    grid_n = 180
    xs = np.linspace(float(x0), float(x1), grid_n)
    ys = np.linspace(float(y0), float(y1), grid_n)
    gx, gy = np.meshgrid(xs, ys)
    candidates_xy = np.column_stack([gx.ravel(), gy.ravel()])
    inside = _points_inside_any_polygon(candidates_xy, [xy])
    edge_samples = _edge_samples(xy, samples_per_edge=64)
    points_xy = np.vstack([candidates_xy[inside], edge_samples, xy])
    if points_xy.shape[0] < 3:
        return None
    vertices = np.column_stack([points_xy, np.full(points_xy.shape[0], z, dtype=np.float64)])
    homogeneous = np.column_stack([vertices, np.ones(len(vertices), dtype=np.float64)])
    camera_points = (t_world_to_camera @ homogeneous.T).T[:, :3]
    valid = camera_points[:, 2] > 1e-5
    if int(np.count_nonzero(valid)) < 3:
        return None
    camera_points = camera_points[valid]
    pixels_h = (k @ camera_points.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    height, width = image_shape
    finite = np.all(np.isfinite(pixels), axis=1)
    near_image = (
        (pixels[:, 0] >= -width)
        & (pixels[:, 0] <= 2 * width)
        & (pixels[:, 1] >= -height)
        & (pixels[:, 1] <= 2 * height)
    )
    pixels = pixels[finite & near_image]
    if pixels.shape[0] < 3:
        return None
    return pixels


def _edge_samples(polygon_xy: np.ndarray, *, samples_per_edge: int) -> np.ndarray:
    samples = []
    for idx in range(len(polygon_xy)):
        start = polygon_xy[idx]
        end = polygon_xy[(idx + 1) % len(polygon_xy)]
        weights = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        samples.append(start[None, :] * (1.0 - weights[:, None]) + end[None, :] * weights[:, None])
    return np.vstack(samples) if samples else np.empty((0, 2), dtype=np.float64)


def _depth_plane_metrics_from_tabletop_mask(
    run: Path,
    camera: dict[str, Any],
    frames: list[dict[str, Any]],
    k: np.ndarray,
    top_z: float,
) -> dict[str, Any] | None:
    mask_path = run / "background" / "tabletop_mask.png"
    if not mask_path.is_file():
        return None
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    for frame in frames:
        depth_rel = frame.get("depth_path")
        t_camera_to_world = _transform(frame.get("T_camera_to_world"))
        if t_camera_to_world is None:
            t_camera_to_world = _transform(camera.get("T_camera_to_world"))
        if not depth_rel or t_camera_to_world is None:
            continue
        depth_path = run / str(depth_rel)
        if not depth_path.is_file():
            continue
        depth = np.load(depth_path)
        if depth.shape != mask.shape:
            continue
        valid = mask & np.isfinite(depth) & (depth > 0.0)
        if int(np.count_nonzero(valid)) < 16:
            continue
        rows, cols = np.where(valid)
        z = depth[rows, cols].astype(np.float64)
        x = (cols.astype(np.float64) - float(k[0, 2])) * z / float(k[0, 0])
        y = (rows.astype(np.float64) - float(k[1, 2])) * z / float(k[1, 1])
        points_world = (t_camera_to_world @ np.column_stack([x, y, z, np.ones_like(z)]).T).T[:, :3]
        residual = np.abs(points_world[:, 2] - top_z)
        residual = residual[np.isfinite(residual)]
        if residual.size:
            return {
                "blocking_reasons": [],
                "median_abs_residual_m": float(np.median(residual)),
                "p90_abs_residual_m": float(np.percentile(residual, 90.0)),
                "sample_count": int(residual.size),
                "source": "phone_depth_points_selected_by_tabletop_mask",
            }
    return None


def _write_bg_table_overlays(
    run: Path,
    polygon: dict[str, Any],
    camera: dict[str, Any],
    trajectory: dict[str, Any],
    overlay_evidence: dict[str, Any],
    frame_indices: list[int],
) -> list[Path]:
    k = _camera_matrix(camera)
    if k is None:
        return []
    polygons = _polygon_world_vertex_sets(polygon)
    render_paths_by_frame = overlay_evidence.get("render_paths_by_frame") if isinstance(overlay_evidence.get("render_paths_by_frame"), dict) else {}
    out: list[Path] = []
    for frame in _select_trajectory_frames(trajectory, frame_indices):
        frame_id = str(frame.get("frame_id") or "")
        frame_number = _frame_number(frame_id)
        render_rel = render_paths_by_frame.get(str(frame_number))
        if render_rel is None:
            continue
        image_path = run / str(render_rel)
        if not image_path.is_file():
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"))
        overlay = image.copy()
        t_camera_to_world = _transform(frame.get("T_camera_to_world"))
        if t_camera_to_world is None:
            t_camera_to_world = _transform(camera.get("T_camera_to_world"))
        if t_camera_to_world is None:
            continue
        try:
            t_world_to_camera = np.linalg.inv(t_camera_to_world)
        except np.linalg.LinAlgError:
            continue
        k_for_image = _scale_intrinsics_to_image(k, camera, image.shape[:2])
        for vertices in polygons:
            points = _project_vertices(vertices, k_for_image, t_world_to_camera)
            if points is not None:
                pts = np.rint(points).astype(np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.fillPoly(overlay, [pts], color=(0, 80, 80))
                overlay = cv2.addWeighted(overlay, 0.72, image, 0.28, 0)
        out_path = run / "qa" / f"bg3dgs_table_overlay_{frame_number:06d}.png"
        Image.fromarray(overlay).save(out_path)
        out.append(out_path)
    return out


def _scale_intrinsics_to_image(k: np.ndarray, camera: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    source_width = float(camera.get("width", width) or width)
    source_height = float(camera.get("height", height) or height)
    if source_width <= 0.0 or source_height <= 0.0:
        return k
    scaled = np.array(k, dtype=np.float64, copy=True)
    scaled[0, :] *= float(width) / source_width
    scaled[1, :] *= float(height) / source_height
    scaled[2, :] = k[2, :]
    return scaled


def _splat_center_diagnostic(run: Path, polygon: dict[str, Any], t_3dgs_world_to_sim: np.ndarray | None) -> dict[str, Any]:
    centers, _colors, metadata = _load_splat_centers(run / "background" / "3dgs_native" / "splat_rgb.ply", max_points=100000)
    asset_bridge = _splat_asset_axis_bridge(metadata)
    bridge_report = _splat_axis_bridge_report(asset_bridge)
    if centers.size == 0 or t_3dgs_world_to_sim is None:
        return {
            "status": "unavailable",
            "near_plane_inside_polygon_ratio": None,
            "asset_axis_bridge": bridge_report,
            "collision_geometry_source": "not_used",
        }
    diagnostic_transform = t_3dgs_world_to_sim @ asset_bridge
    centers_sim = _transform_points(centers, diagnostic_transform)
    top_z = float(polygon.get("top_z_m", polygon.get("table_top_z_m", 0.0)) or 0.0)
    near = np.abs(centers_sim[:, 2] - top_z) <= 0.05
    near_count = int(np.count_nonzero(near))
    if near_count == 0:
        return {
            "status": "no_near_plane_splats",
            "near_plane_inside_polygon_ratio": None,
            "near_plane_count": 0,
            "asset_axis_bridge": bridge_report,
            "T_splat_asset_to_sim_world": diagnostic_transform.tolist(),
            "transform_source": _splat_diagnostic_transform_source(asset_bridge),
            "collision_geometry_source": "not_used",
        }
    inside = _points_inside_any_polygon(centers_sim[near, :2], _polygon_xy_sets(polygon))
    inside_count = int(np.count_nonzero(inside))
    return {
        "status": "computed",
        "near_plane_count": near_count,
        "near_plane_inside_polygon_count": inside_count,
        "near_plane_inside_polygon_ratio": float(inside_count / near_count),
        "near_plane_threshold_m": 0.05,
        "asset_axis_bridge": bridge_report,
        "T_splat_asset_to_sim_world": diagnostic_transform.tolist(),
        "transform_source": _splat_diagnostic_transform_source(asset_bridge),
        "source": "splat_centers_diagnostic_only_not_collision_source",
        "collision_geometry_source": "not_used",
    }


def _splat_asset_axis_bridge(metadata: dict[str, Any]) -> np.ndarray:
    if metadata.get("schema") == "nerfstudio_gaussian_ply":
        return NERFSTUDIO_GAUSSIAN_ASSET_AXIS_BRIDGE.copy()
    return np.eye(4, dtype=np.float64)


def _splat_axis_bridge_report(transform: np.ndarray) -> dict[str, Any]:
    applied = not np.allclose(transform, np.eye(4, dtype=np.float64))
    return {
        "status": "applied_for_nerfstudio_gaussian_ply_diagnostic" if applied else "not_required_for_plain_ply_diagnostic",
        "matrix": transform.tolist(),
        "diagnostic_only": True,
        "not_written_to_registration_json": True,
    }


def _splat_diagnostic_transform_source(transform: np.ndarray) -> str:
    if np.allclose(transform, np.eye(4, dtype=np.float64)):
        return "registration_T_3dgs_world_to_sim_world"
    return "diagnostic_splat_asset_to_sim_world"


def _semantic_fit_blocked_report(run: Path, reasons: list[str], mask_path: Path, frame_index: int, hull: str) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "blocked",
        "source_backend": SEMANTIC_TABLE_COLLISION_SOURCE,
        "frame_index": int(frame_index),
        "hull_method": hull,
        "semantic_mask_path": _rel(run, mask_path),
        "refined_plane_mask_path": None,
        "polygon_path": None,
        "blocking_reasons": _dedupe(reasons),
        "claim_boundary": {
            "semantic_tabletop_localization": "blocked_without_masked_depth_plane",
            "metric_geometry": "blocked_without_phone_depth_confidence_and_pose",
        },
    }


def _backproject_selected_depth(
    depth: np.ndarray,
    selected: np.ndarray,
    k: np.ndarray,
    t_camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(selected)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64), rows, cols
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - float(k[0, 2])) * z / float(k[0, 0])
    y = (rows.astype(np.float64) - float(k[1, 2])) * z / float(k[1, 1])
    points_camera = np.column_stack([x, y, z, np.ones_like(z)])
    points_world = (t_camera_to_world @ points_camera.T).T[:, :3]
    finite = np.all(np.isfinite(points_world), axis=1)
    return points_world[finite], rows[finite], cols[finite]


def _fit_plane_ransac_points(points: np.ndarray, *, distance_threshold_m: float) -> tuple[np.ndarray, float, np.ndarray] | None:
    if points.shape[0] < 3:
        return None
    rng = np.random.default_rng(23)
    iterations = min(512, max(128, points.shape[0] // 16))
    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(iterations):
        sample = points[rng.choice(points.shape[0], size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        distances = np.abs(points @ normal + offset)
        inliers = distances <= float(distance_threshold_m)
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or int(np.count_nonzero(best_inliers)) < 3:
        return None
    inlier_points = points[best_inliers]
    centroid = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[2] < 0.0:
        normal = -normal
    offset = -float(normal @ centroid)
    distances = np.abs(points @ normal + offset)
    inliers = distances <= float(distance_threshold_m)
    return normal, offset, inliers


def _tabletop_hull_xy(points_xy: np.ndarray, *, method: str) -> np.ndarray | None:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or points_xy.shape[0] < 3:
        return None
    finite = np.all(np.isfinite(points_xy), axis=1)
    points = points_xy[finite].astype(np.float32)
    if points.shape[0] < 3:
        return None
    if method == "rotated-rectangle":
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect).astype(np.float64)
        return box if box.shape[0] >= 3 else None
    hull = cv2.convexHull(points).reshape(-1, 2).astype(np.float64)
    return hull if hull.shape[0] >= 3 else None


def _validate_world_units(polygon: dict[str, Any]) -> None:
    coordinate_world = polygon.get("coordinate_world")
    coordinate_frame = polygon.get("coordinate_frame")
    unit = polygon.get("unit")
    if coordinate_world != "sim_world":
        raise ValueError("table polygon must declare coordinate_world=sim_world")
    if coordinate_frame != "sim_world":
        raise ValueError("table polygon must declare coordinate_frame=sim_world")
    if unit != "meter":
        raise ValueError("table polygon must declare unit=meter")


def _polygon_xy_sets(polygon: dict[str, Any]) -> list[np.ndarray]:
    raw = polygon.get("polygons_world_xy")
    if isinstance(raw, list):
        out = []
        for item in raw:
            poly = _xy_array(item)
            if poly is not None:
                out.append(poly)
        if out:
            return out
    single = _xy_array(polygon.get("polygon_world_xy"))
    return [single] if single is not None else []


def _polygon_world_vertex_sets(polygon: dict[str, Any]) -> list[np.ndarray]:
    top_z = float(polygon.get("top_z_m", polygon.get("table_top_z_m", 0.0)) or 0.0)
    return [np.column_stack([poly, np.full(len(poly), top_z, dtype=np.float64)]) for poly in _polygon_xy_sets(polygon)]


def _xy_array(value: object) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _world_polygon_slab_mesh(polygon_xy: np.ndarray, *, top_z: float, thickness: float) -> trimesh.Trimesh:
    n = int(len(polygon_xy))
    top = np.column_stack([polygon_xy, np.full(n, top_z, dtype=np.float64)])
    bottom = np.column_stack([polygon_xy, np.full(n, top_z - thickness, dtype=np.float64)])
    vertices = np.vstack([top, bottom])
    faces: list[list[int]] = []
    for idx in range(1, n - 1):
        faces.append([0, idx, idx + 1])
        faces.append([n, n + idx + 1, n + idx])
    for idx in range(n):
        nxt = (idx + 1) % n
        faces.append([idx, nxt, n + nxt])
        faces.append([idx, n + nxt, n + idx])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def _slab_usd_ascii(mesh: trimesh.Trimesh, *, mesh_path: str) -> str:
    points = ",\n            ".join(f"({float(x):.9g}, {float(y):.9g}, {float(z):.9g})" for x, y, z in mesh.vertices)
    face_counts = ", ".join("3" for _ in mesh.faces)
    face_indices = ", ".join(str(int(v)) for face in mesh.faces for v in face)
    return f'''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
)

def Xform "World"
{{
    def Mesh "TableCollision"
    {{
        custom string rsf:source = "{TABLE_COLLISION_SOURCE}"
        custom string rsf:mesh_sidecar = "{mesh_path}"
        uniform token subdivisionScheme = "none"
        point3f[] points = [
            {points}
        ]
        int[] faceVertexCounts = [{face_counts}]
        int[] faceVertexIndices = [{face_indices}]
    }}
}}
'''


def _backproject_depth(depth: np.ndarray, k: np.ndarray, t_camera_to_world: np.ndarray, *, stride: int) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if stride > 1:
        keep = (rows % stride == 0) & (cols % stride == 0)
        rows = rows[keep]
        cols = cols[keep]
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - float(k[0, 2])) * z / float(k[0, 0])
    y = (rows.astype(np.float64) - float(k[1, 2])) * z / float(k[1, 1])
    points_camera = np.column_stack([x, y, z, np.ones_like(z)])
    points_world = (t_camera_to_world @ points_camera.T).T[:, :3]
    return points_world[np.all(np.isfinite(points_world), axis=1)]


def _load_splat_centers(path: Path, *, max_points: int) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if not path.is_file():
        return np.empty((0, 3), dtype=np.float64), None, {}
    data = path.read_bytes()
    header_end = data.find(b"end_header")
    if header_end < 0:
        return np.empty((0, 3), dtype=np.float64), None, {}
    header_stop = data.find(b"\n", header_end)
    header_stop = len(data) if header_stop < 0 else header_stop + 1
    header = data[:header_stop].decode("utf-8", errors="replace").splitlines()
    fmt = "ascii"
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "format":
            fmt = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and len(parts) >= 3 and parts[0] == "property":
            properties.append((parts[1], parts[2]))
    metadata = _splat_ply_metadata(fmt, vertex_count, properties)
    if vertex_count <= 0:
        return np.empty((0, 3), dtype=np.float64), None, metadata
    count = min(vertex_count, int(max_points))
    if fmt == "ascii":
        rows = data[header_stop:].decode("utf-8", errors="ignore").splitlines()[:count]
        values = [row.split() for row in rows if row.strip()]
        name_to_idx = {name: idx for idx, (_typ, name) in enumerate(properties)}
        centers = np.asarray(
            [[float(row[name_to_idx["x"]]), float(row[name_to_idx["y"]]), float(row[name_to_idx["z"]])] for row in values],
            dtype=np.float64,
        )
        colors = _ascii_colors(values, name_to_idx)
        return centers, colors, metadata
    if fmt != "binary_little_endian":
        return np.empty((0, 3), dtype=np.float64), None, metadata
    dtype = np.dtype([(name, _ply_dtype(typ)) for typ, name in properties])
    records = np.frombuffer(data, dtype=dtype, count=count, offset=header_stop)
    centers = np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float64)
    colors = None
    if all(name in records.dtype.names for name in ("red", "green", "blue")):
        colors = np.column_stack([records["red"], records["green"], records["blue"]]).astype(np.uint8)
    return centers, colors, metadata


def _splat_ply_metadata(fmt: str, vertex_count: int, properties: list[tuple[str, str]]) -> dict[str, Any]:
    property_names = [name for _typ, name in properties]
    gaussian_fields = {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    return {
        "format": fmt,
        "vertex_count": int(vertex_count),
        "property_names": property_names,
        "schema": "nerfstudio_gaussian_ply" if gaussian_fields.issubset(set(property_names)) else "plain_ply",
    }


def _ascii_colors(values: list[list[str]], name_to_idx: dict[str, int]) -> np.ndarray | None:
    if not all(name in name_to_idx for name in ("red", "green", "blue")):
        return None
    return np.asarray(
        [[int(float(row[name_to_idx["red"]])), int(float(row[name_to_idx["green"]])), int(float(row[name_to_idx["blue"]]))] for row in values],
        dtype=np.uint8,
    )


def _ply_dtype(name: str) -> str:
    mapping = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
    }
    if name not in mapping:
        raise ValueError(f"unsupported PLY property type {name!r}")
    return mapping[name]


def _draw_splat_points(
    image: np.ndarray,
    centers_world: np.ndarray,
    colors: np.ndarray | None,
    k: np.ndarray,
    t_world_to_camera: np.ndarray,
) -> None:
    if centers_world.size == 0:
        return
    homogeneous = np.column_stack([centers_world, np.ones(len(centers_world), dtype=np.float64)])
    camera_points = (t_world_to_camera @ homogeneous.T).T[:, :3]
    valid = camera_points[:, 2] > 1e-6
    if not np.any(valid):
        return
    camera_points = camera_points[valid]
    color_values = colors[valid] if colors is not None and len(colors) == len(centers_world) else None
    pixels_h = (k @ camera_points.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    height, width = image.shape[:2]
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    pixels = np.rint(pixels[inside]).astype(np.int32)
    if color_values is not None:
        color_values = color_values[inside]
    for idx, (x, y) in enumerate(pixels):
        color = tuple(int(v) for v in color_values[idx].tolist()) if color_values is not None else (180, 160, 120)
        cv2.circle(image, (int(x), int(y)), 1, color, -1)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (transform @ homogeneous.T).T[:, :3]


def _points_inside_any_polygon(points_xy: np.ndarray, polygons_xy: list[np.ndarray]) -> np.ndarray:
    if points_xy.size == 0 or not polygons_xy:
        return np.zeros(len(points_xy), dtype=bool)
    inside = np.zeros(len(points_xy), dtype=bool)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    for polygon in polygons_xy:
        px = polygon[:, 0]
        py = polygon[:, 1]
        current = np.zeros(len(points_xy), dtype=bool)
        j = len(polygon) - 1
        for i in range(len(polygon)):
            crosses = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / ((py[j] - py[i]) + 1e-12) + px[i])
            current ^= crosses
            j = i
        inside |= current
    return inside


def _select_trajectory_frames(trajectory: dict[str, Any], indices: list[int]) -> list[dict[str, Any]]:
    frames = [frame for frame in trajectory.get("frames", []) if isinstance(frame, dict)]
    if not indices:
        return frames[:1]
    by_number = {_frame_number(str(frame.get("frame_id") or "")): frame for frame in frames}
    selected = []
    for index in indices:
        if index in by_number:
            selected.append(by_number[index])
        elif 0 <= index < len(frames):
            selected.append(frames[index])
    return selected


def _parse_frame_indices(value: str) -> list[int]:
    out = []
    for part in str(value or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _frame_number(frame_id: str) -> int:
    digits = "".join(ch for ch in frame_id if ch.isdigit())
    return int(digits) if digits else 0


def _write_projection_overlay(path: Path, tabletop_mask: np.ndarray, projected_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = np.zeros((*tabletop_mask.shape, 3), dtype=np.uint8)
    overlay[..., 1] = tabletop_mask
    overlay[..., 2] = projected_mask
    overlay[(tabletop_mask > 0) & (projected_mask > 0)] = [255, 255, 255]
    cv2.imwrite(str(path), overlay)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(max(0.0, float(numerator)) / float(denominator))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _resolve_run_path(run: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run / path


def _rel(run: Path, path: Path) -> str:
    try:
        return path.relative_to(run).as_posix()
    except ValueError:
        return path.as_posix()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
