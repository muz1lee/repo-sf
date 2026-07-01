"""Support-plane estimation and world-frame normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image


TABLETOP_PLANE_RESIDUAL_TOLERANCE_M = 0.02
MIN_TABLETOP_SELECTED_CANDIDATE_RATIO = 0.08
MIN_FINAL_TABLE_EXTENT_M = 0.12


def estimate_and_apply_support_plane(run_dir: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Estimate the support height, shift world poses so that support plane is z=0."""
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = manifest.get("support_plane")
    if not force and _support_plane_is_complete(run, existing):
        refreshed = _refresh_table_collision_projection_qa(run, existing)
        if refreshed != existing:
            manifest["support_plane"] = refreshed
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _merge_qa_support_plane(run, refreshed)
        return refreshed

    objects = manifest.get("objects", [])
    if not objects:
        raise ValueError("scene_manifest.json has no objects for support-plane estimation")
    before_bottoms = _object_bottoms(run, objects)
    phone_plane = _estimate_phone_support_from_depth(run)
    height, source = (
        (float(phone_plane["height_world_m"]), "arkit_depth_point_cloud_plane")
        if phone_plane.get("status") == "passed"
        else _estimate_height_from_point_rings(run, objects)
    )
    if height is None:
        height = float(np.median(list(before_bottoms.values())))
        source = "object_mesh_bottom_median"

    phone_plane_used = phone_plane.get("status") == "passed"
    if phone_plane_used:
        _shift_phone_camera_world(run, support_height_m=float(height))

    for item in objects:
        transform = _transform_array(item["T_object_to_world"])
        transform[2, 3] -= height
        item["T_object_to_world"] = transform.tolist()

    bottoms_after_global_shift = _object_bottoms(run, objects)
    object_corrections: dict[str, float] = {}
    for item in objects:
        object_id = str(item["object_id"])
        correction = -float(bottoms_after_global_shift[object_id])
        transform = _transform_array(item["T_object_to_world"])
        transform[2, 3] += correction
        item["T_object_to_world"] = transform.tolist()
        object_corrections[object_id] = correction
        _update_pose_file(run, item["object_id"], transform)

    after_bottoms = _object_bottoms(run, objects)
    table_report = (
        _write_phone_table_collision_mesh(run, phone_plane, support_height_m=height)
        if phone_plane_used
        else _write_table_collision_mesh(run, objects, support_height_m=height)
    )
    support_plane = {
        "status": "estimated",
        "source_backend": source,
        "normal_world": [0.0, 0.0, 1.0],
        "height_world_m": 0.0,
        "original_height_world_m": float(height),
        "applied_translation_world_m": [0.0, 0.0, float(-height)],
        "applied_to_world_frame": True,
        "object_ids": [str(item["object_id"]) for item in objects],
        "object_bottoms_before_m": {key: float(value) for key, value in before_bottoms.items()},
        "object_bottoms_after_global_shift_m": {key: float(value) for key, value in bottoms_after_global_shift.items()},
        "object_vertical_corrections_m": {key: float(value) for key, value in object_corrections.items()},
        "object_bottoms_after_m": {key: float(value) for key, value in after_bottoms.items()},
        **table_report,
    }
    support_plane = _refresh_table_collision_projection_qa(run, support_plane)
    manifest["support_plane"] = support_plane
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_usda_stub_from_manifest(run / "exports" / "scene.usda", manifest)
    _merge_qa_support_plane(run, support_plane)
    return support_plane


def build_tabletop_support_mask(run_dir: str | Path) -> dict[str, Any]:
    """Build an image-space tabletop mask from metric background points."""
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = manifest.get("objects", [])
    height, _ = _estimate_height_from_point_rings(run, objects)
    if height is None:
        height = float(manifest.get("support_plane", {}).get("original_height_world_m", 0.0))
    return _build_tabletop_support_mask(run, objects, support_height_m=float(height))


def _support_plane_is_complete(run: Path, support_plane: object) -> bool:
    if not isinstance(support_plane, dict):
        return False
    if support_plane.get("applied_to_world_frame") is not True:
        return False
    table_path = support_plane.get("table_collision_mesh_path")
    return bool(
        table_path
        and (run / str(table_path)).is_file()
        and support_plane.get("table_collision_pos_world")
        and support_plane.get("table_collision_size_xyz")
        and "table_collision_final" in support_plane
        and support_plane.get("support_surface_status") in {"passed", "blocked"}
    )


def _estimate_height_from_point_rings(run: Path, objects: list[dict[str, Any]]) -> tuple[float | None, str]:
    xyz_path = run / "xyz.npy"
    if not xyz_path.is_file():
        return None, "unavailable"
    xyz = np.load(xyz_path)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        return None, "unavailable"
    masks: dict[str, np.ndarray] = {}
    union = np.zeros(xyz.shape[:2], dtype=bool)
    for item in objects:
        mask_path = run / str(item.get("mask_path", ""))
        if not mask_path.is_file():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != xyz.shape[:2]:
            continue
        masks[str(item["object_id"])] = mask
        union |= mask
    if not masks:
        return None, "unavailable"

    heights: list[float] = []
    kernel_size = max(7, int(round(min(xyz.shape[:2]) * 0.04)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    valid_xyz = np.all(np.isfinite(xyz), axis=2) & (xyz[..., 2] > 0.0)
    world_z = -np.asarray(xyz[..., 1], dtype=np.float64)
    for mask in masks.values():
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & ~union & valid_xyz
        if np.count_nonzero(ring) < 8:
            continue
        samples = world_z[ring]
        samples = samples[np.isfinite(samples)]
        if samples.size:
            heights.append(float(np.median(samples)))
    if not heights:
        return None, "unavailable"
    return float(np.median(heights)), "background_point_ring_median"


def _estimate_phone_support_from_depth(run: Path) -> dict[str, Any]:
    """Estimate a support plane from imported phone RGB-D frames.

    The phone route is intentionally evidence-gated: it only runs when import
    wrote explicit ARKit intrinsics, poses, and meter-scale depth.
    """
    camera = _load_json_doc(run / "camera.json")
    trajectory = _load_json_doc(run / "trajectory.json")
    if (
        camera.get("intrinsics_source") != "arkit_explicit"
        or camera.get("extrinsics_source") != "arkit_explicit"
        or camera.get("scale_source") != "arkit_sceneDepth_meters"
    ):
        return {"status": "unavailable", "blocked_reason": "phone_explicit_camera_evidence_missing"}
    if str(camera.get("depth_unit", "meter")).lower() not in {"meter", "meters", "m"}:
        return {"status": "blocked", "blocked_reason": "phone_depth_unit_not_meter"}

    frame = _first_phone_depth_frame(run, trajectory)
    if frame is None:
        return {"status": "unavailable", "blocked_reason": "phone_depth_frame_missing"}
    k = _camera_matrix_from_dict(camera)
    transform = _transform_array(frame["T_camera_to_world"])
    depth = np.load(run / frame["depth_path"])
    confidence = np.asarray(Image.open(run / frame["confidence_path"]).convert("L"))
    if depth.ndim != 2 or confidence.shape != depth.shape or k is None:
        return {"status": "blocked", "blocked_reason": "invalid_phone_depth_or_camera_shape"}

    valid = np.isfinite(depth) & (depth > 0.0) & (confidence > 0)
    if int(np.count_nonzero(valid)) < 16:
        return {"status": "blocked", "blocked_reason": "insufficient_confident_phone_depth_points"}

    points_world = _backproject_depth_to_world(depth, valid, k, transform)
    height = float(np.median(points_world[:, 2]))
    residual = np.abs(points_world[:, 2] - height)
    planar_valid_flat = residual <= TABLETOP_PLANE_RESIDUAL_TOLERANCE_M
    planar_valid = np.zeros(depth.shape, dtype=bool)
    planar_valid[np.where(valid)] = planar_valid_flat
    if int(np.count_nonzero(planar_valid)) < 16:
        return {"status": "blocked", "blocked_reason": "insufficient_planar_phone_depth_points"}

    kernel = np.ones((3, 3), dtype=np.uint8)
    selected = cv2.erode(planar_valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    if selected.shape[0] > 2 and selected.shape[1] > 2:
        selected[0, :] = False
        selected[-1, :] = False
        selected[:, 0] = False
        selected[:, -1] = False
    if int(np.count_nonzero(selected)) < 16:
        selected = planar_valid
    selected_points_world = _backproject_depth_to_world(depth, selected, k, transform)
    polygon_xy = _convex_hull_xy(selected_points_world[:, :2])
    if polygon_xy is None:
        return {"status": "blocked", "blocked_reason": "insufficient_phone_table_polygon_points"}

    background_dir = run / "background"
    background_dir.mkdir(parents=True, exist_ok=True)
    tabletop_mask_path = background_dir / "tabletop_mask.png"
    support_report_path = background_dir / "tabletop_support_report.json"
    Image.fromarray(selected.astype(np.uint8) * 255).save(tabletop_mask_path)
    extent_x = float(np.max(polygon_xy[:, 0]) - np.min(polygon_xy[:, 0]))
    extent_y = float(np.max(polygon_xy[:, 1]) - np.min(polygon_xy[:, 1]))
    report = {
        "status": "passed",
        "source": "arkit_depth_point_cloud_plane",
        "mask_path": str(tabletop_mask_path.relative_to(run)),
        "report_path": str(support_report_path.relative_to(run)),
        "support_height_m": height,
        "candidate_count": int(np.count_nonzero(valid)),
        "planar_candidate_count": int(np.count_nonzero(planar_valid)),
        "selected_component_area": int(np.count_nonzero(selected)),
        "selected_candidate_ratio": float(np.count_nonzero(selected) / max(1, np.count_nonzero(valid))),
        "selected_planar_candidate_ratio": float(np.count_nonzero(selected) / max(1, np.count_nonzero(planar_valid))),
        "plane_residual_median_m": float(np.median(residual)),
        "plane_residual_p90_m": float(np.percentile(residual, 90.0)),
        "polygon_extent_x_m": extent_x,
        "polygon_extent_y_m": extent_y,
        "polygon_world_xy": [[float(x), float(y)] for x, y in polygon_xy.tolist()],
        "frame_id": frame["frame_id"],
        "depth_path": frame["depth_path"],
        "confidence_path": frame["confidence_path"],
    }
    support_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"status": "passed", "height_world_m": height, "polygon_xy": polygon_xy, "mask_report": report}


def _first_phone_depth_frame(run: Path, trajectory: dict[str, Any]) -> dict[str, Any] | None:
    frames = trajectory.get("frames", [])
    if not isinstance(frames, list):
        return None
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        depth_path = str(frame.get("depth_path", ""))
        confidence_path = str(frame.get("confidence_path", ""))
        if not depth_path or not confidence_path:
            continue
        if (run / depth_path).is_file() and (run / confidence_path).is_file() and frame.get("T_camera_to_world") is not None:
            return {
                "frame_id": str(frame.get("frame_id") or Path(depth_path).stem),
                "depth_path": depth_path,
                "confidence_path": confidence_path,
                "T_camera_to_world": frame["T_camera_to_world"],
            }
    return None


def _camera_matrix_from_dict(camera: dict[str, Any]) -> np.ndarray | None:
    try:
        if camera.get("K") is not None:
            k = np.asarray(camera["K"], dtype=np.float64)
            return k if k.shape == (3, 3) and np.all(np.isfinite(k)) else None
        return np.asarray(
            [[float(camera["fx"]), 0.0, float(camera["cx"])], [0.0, float(camera["fy"]), float(camera["cy"])], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _backproject_depth_to_world(depth: np.ndarray, mask: np.ndarray, k: np.ndarray, t_camera_to_world: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - float(k[0, 2])) * z / float(k[0, 0])
    y = (rows.astype(np.float64) - float(k[1, 2])) * z / float(k[1, 1])
    points_camera = np.column_stack([x, y, z, np.ones_like(z)])
    points_world = (t_camera_to_world @ points_camera.T).T[:, :3]
    finite = np.all(np.isfinite(points_world), axis=1)
    return points_world[finite]


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray | None:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or points_xy.shape[0] < 3:
        return None
    finite = np.all(np.isfinite(points_xy), axis=1)
    points = points_xy[finite]
    if points.shape[0] < 3:
        return None
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    return hull if hull.shape[0] >= 3 else None


def _shift_phone_camera_world(run: Path, *, support_height_m: float) -> None:
    if abs(float(support_height_m)) <= 1e-9:
        return
    camera_path = run / "camera.json"
    camera = _load_json_doc(camera_path)
    if camera.get("intrinsics_source") == "arkit_explicit" and camera.get("extrinsics_source") == "arkit_explicit":
        shifted = _shift_camera_to_world_z(camera.get("T_camera_to_world"), support_height_m=support_height_m)
        if shifted is not None:
            camera["T_camera_to_world"] = shifted.tolist()
            camera["T_world_to_camera"] = np.linalg.inv(shifted).tolist()
            camera["pose_world"] = "sim"
            camera["support_plane_shift_applied_m"] = float(support_height_m)
            camera_path.write_text(json.dumps(camera, indent=2), encoding="utf-8")

    trajectory_path = run / "trajectory.json"
    trajectory = _load_json_doc(trajectory_path)
    frames = trajectory.get("frames", [])
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            shifted = _shift_camera_to_world_z(frame.get("T_camera_to_world"), support_height_m=support_height_m)
            if shifted is not None:
                frame["T_camera_to_world"] = shifted.tolist()
        trajectory["pose_world"] = "sim"
        trajectory["support_plane_shift_applied_m"] = float(support_height_m)
        trajectory_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")


def _shift_camera_to_world_z(value: object, *, support_height_m: float) -> np.ndarray | None:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    transform = transform.copy()
    transform[2, 3] -= float(support_height_m)
    return transform


def _write_phone_table_collision_mesh(run: Path, phone_plane: dict[str, Any], *, support_height_m: float) -> dict[str, Any]:
    polygon_xy = np.asarray(phone_plane["polygon_xy"], dtype=np.float64)
    mask_report = dict(phone_plane.get("mask_report", {}))
    thickness = 0.04
    x0, y0 = np.min(polygon_xy, axis=0)
    x1, y1 = np.max(polygon_xy, axis=0)
    size = [float(x1 - x0), float(y1 - y0), float(thickness)]
    if min(size[0], size[1]) < MIN_FINAL_TABLE_EXTENT_M:
        collision_report = _write_table_collision_report(
            run,
            {
                "status": "blocked",
                "blocked_reason": "table_collision_extent_too_small",
                "source_backend": "arkit_depth_point_cloud_plane",
                "geometry_type": "polygon_slab",
                "final": False,
                "table_collision_size_xyz": size,
                "min_final_table_extent_m": float(MIN_FINAL_TABLE_EXTENT_M),
            },
        )
        return {
            "table_collision_mesh_path": "",
            "table_collision_source_backend": "arkit_depth_point_cloud_plane",
            "table_collision_geometry_type": "polygon_slab",
            "table_collision_final": False,
            "support_surface_status": "blocked",
            "support_surface_blocked_reason": "table_collision_extent_too_small",
            "table_collision_report_path": collision_report,
        }

    center = [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5), float(-thickness * 0.5)]
    center_xy = np.asarray(center[:2], dtype=np.float64)
    mesh = _polygon_slab_mesh(polygon_xy - center_xy, thickness=thickness)
    table_path = run / "table" / "collision_polygon_slab.glb"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(table_path)

    background_dir = run / "background"
    background_dir.mkdir(parents=True, exist_ok=True)
    polygon_path = background_dir / "table_polygon_world.json"
    polygon_payload = {
        "status": "passed",
        "geometry_type": "polygon_slab",
        "source_backend": "arkit_depth_point_cloud_plane",
        "polygon_world_xy": [[float(x), float(y)] for x, y in polygon_xy.tolist()],
        "polygons_world_xy": [[[float(x), float(y)] for x, y in polygon_xy.tolist()]],
        "polygon_count": 1,
        "visible_tabletop_polygon_count": 1,
        "object_support_footprint_polygon_count": 0,
        "support_height_m": float(support_height_m),
        "top_z_m": 0.0,
        "thickness_m": float(thickness),
        "input_point_count": int(mask_report.get("selected_component_area", 0) or 0),
        "candidate_count": int(mask_report.get("candidate_count", 0) or 0),
        "planar_candidate_count": int(mask_report.get("planar_candidate_count", 0) or 0),
        "selected_component_area": int(mask_report.get("selected_component_area", 0) or 0),
        "selected_candidate_ratio": float(mask_report.get("selected_candidate_ratio", 0.0) or 0.0),
        "selected_planar_candidate_ratio": float(mask_report.get("selected_planar_candidate_ratio", 0.0) or 0.0),
        "polygon_extent_x_m": float(size[0]),
        "polygon_extent_y_m": float(size[1]),
    }
    polygon_path.write_text(json.dumps(polygon_payload, indent=2), encoding="utf-8")
    collision_report_path = _write_table_collision_report(
        run,
        {
            "status": "passed",
            "blocked_reason": None,
            "source_backend": "arkit_depth_point_cloud_plane",
            "geometry_type": "polygon_slab",
            "final": True,
            "mesh_path": str(table_path.relative_to(run)),
            "polygon_path": str(polygon_path.relative_to(run)),
            "table_collision_size_xyz": size,
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "polygon_count": 1,
            "candidate_count": int(mask_report.get("candidate_count", 0) or 0),
            "selected_component_area": int(mask_report.get("selected_component_area", 0) or 0),
            "selected_candidate_ratio": float(mask_report.get("selected_candidate_ratio", 0.0) or 0.0),
            "polygon_extent_x_m": float(size[0]),
            "polygon_extent_y_m": float(size[1]),
        },
    )
    return {
        "table_collision_mesh_path": str(table_path.relative_to(run)),
        "table_collision_source_backend": "arkit_depth_point_cloud_plane",
        "table_collision_geometry_type": "polygon_slab",
        "table_collision_final": True,
        "support_surface_status": "passed",
        "support_surface_blocked_reason": None,
        "table_bounds_world_xy": [[float(x0), float(y0)], [float(x1), float(y1)]],
        "table_collision_pos_world": center,
        "table_collision_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "table_collision_size_xyz": size,
        "table_top_z_m": 0.0,
        "table_thickness_m": float(thickness),
        "tabletop_mask_path": str(mask_report.get("mask_path", "background/tabletop_mask.png")),
        "tabletop_support_report_path": str(mask_report.get("report_path", "background/tabletop_support_report.json")),
        "table_polygon_world_path": str(polygon_path.relative_to(run)),
        "table_collision_report_path": collision_report_path,
    }


def _write_table_collision_mesh(run: Path, objects: list[dict[str, Any]], *, support_height_m: float) -> dict[str, Any]:
    mask_report = _build_tabletop_support_mask(run, objects, support_height_m=support_height_m)
    if mask_report.get("status") == "passed":
        polygon_report = _write_tabletop_polygon_collision(run, objects, mask_report, support_height_m=support_height_m)
        if isinstance(polygon_report, dict) and polygon_report.get("table_collision_final") is True:
            return polygon_report

    bounds, source = _table_bounds_from_background_points(run, objects, support_height_m=support_height_m)
    if bounds is None:
        bounds = _table_bounds_from_objects(run, objects)
        source = "object_bounds_rect"
    return _write_estimated_table_box_collision(run, bounds, source=source, mask_report=mask_report)


def _build_tabletop_support_mask(run: Path, objects: list[dict[str, Any]], *, support_height_m: float) -> dict[str, Any]:
    background_dir = run / "background"
    background_dir.mkdir(parents=True, exist_ok=True)
    mask_path = background_dir / "tabletop_mask.png"
    report_path = background_dir / "tabletop_support_report.json"
    base_report: dict[str, Any] = {
        "candidate_count": 0,
        "component_count": 0,
        "selected_component_area": 0,
        "source": "unavailable",
        "status": "blocked",
        "blocked_reason": "",
        "mask_path": str(mask_path.relative_to(run)),
        "report_path": str(report_path.relative_to(run)),
        "support_height_m": float(support_height_m),
    }

    xyz_path = run / "xyz.npy"
    if not xyz_path.is_file():
        return _write_tabletop_support_report(report_path, {**base_report, "blocked_reason": "missing_xyz"})
    xyz = np.load(xyz_path)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        return _write_tabletop_support_report(report_path, {**base_report, "blocked_reason": "invalid_xyz_shape"})

    shape = xyz.shape[:2]
    object_union = _object_mask_union(run, objects, shape)
    foreground = object_union.copy()
    bg_foreground = _read_mask(run / "background" / "foreground_mask.png", shape)
    if bg_foreground is not None:
        foreground |= bg_foreground

    valid = np.all(np.isfinite(xyz), axis=2) & (xyz[..., 2] > 0.0)
    world_z = -np.asarray(xyz[..., 1], dtype=np.float64) - float(support_height_m)
    candidates = valid & ~foreground & (np.abs(world_z) <= 0.06)
    planar_candidates = candidates & (np.abs(world_z) <= TABLETOP_PLANE_RESIDUAL_TOLERANCE_M)
    candidate_count = int(np.count_nonzero(candidates))
    planar_candidate_count = int(np.count_nonzero(planar_candidates))
    if candidate_count < 16:
        Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(mask_path)
        return _write_tabletop_support_report(
            report_path,
            {**base_report, "candidate_count": candidate_count, "blocked_reason": "insufficient_near_support_candidates"},
        )
    if planar_candidate_count < 16:
        Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(mask_path)
        return _write_tabletop_support_report(
            report_path,
            {
                **base_report,
                "candidate_count": candidate_count,
                "planar_candidate_count": planar_candidate_count,
                "blocked_reason": "insufficient_planar_tabletop_candidates",
                "plane_residual_tolerance_m": float(TABLETOP_PLANE_RESIDUAL_TOLERANCE_M),
            },
        )

    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(planar_candidates.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    cleaned &= ~foreground
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    component_count -= 1
    if component_count <= 0:
        Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(mask_path)
        return _write_tabletop_support_report(
            report_path,
            {
                **base_report,
                "candidate_count": candidate_count,
                "planar_candidate_count": planar_candidate_count,
                "blocked_reason": "no_connected_tabletop_component",
                "plane_residual_tolerance_m": float(TABLETOP_PLANE_RESIDUAL_TOLERANCE_M),
            },
        )

    contact_kernel_size = max(5, int(round(min(shape) * 0.08)))
    if contact_kernel_size % 2 == 0:
        contact_kernel_size += 1
    contact_kernel = np.ones((contact_kernel_size, contact_kernel_size), dtype=np.uint8)
    contact = cv2.dilate(object_union.astype(np.uint8), contact_kernel, iterations=1).astype(bool) & ~foreground
    component_infos: list[dict[str, Any]] = []
    for label in range(1, component_count + 1):
        rows = np.where(labels == label)[0]
        if rows.size == 0:
            continue
        overlap = int(np.count_nonzero((labels == label) & contact))
        component_infos.append(
            {
                "label": int(label),
                "area": int(stats[label, cv2.CC_STAT_AREA]),
                "row_center": float((int(rows.min()) + int(rows.max())) * 0.5),
                "contact_overlap_px": overlap,
            }
        )
    min_component_area = max(16, int(round(planar_candidate_count * 0.005)))
    row_tolerance_px = max(8.0, float(min(shape) * 0.08))
    contact_components = [item for item in component_infos if item["contact_overlap_px"] > 0]
    if contact_components:
        anchor = max(contact_components, key=lambda item: (item["contact_overlap_px"], item["area"]))
        source = "contact_row_components"
    else:
        anchor = max(component_infos, key=lambda item: item["area"])
        source = "largest_row_components"
    selected_labels = [
        item["label"]
        for item in component_infos
        if item["area"] >= min_component_area and abs(float(item["row_center"]) - float(anchor["row_center"])) <= row_tolerance_px
    ]
    if not selected_labels:
        selected_labels = [int(anchor["label"])]
        source = "contact_neighborhood_component" if contact_components else "largest_component"

    selected = np.isin(labels, selected_labels)
    selected &= planar_candidates
    selected_area = int(np.count_nonzero(selected))
    selected_candidate_ratio = float(selected_area / candidate_count) if candidate_count else 0.0
    selected_planar_candidate_ratio = float(selected_area / planar_candidate_count) if planar_candidate_count else 0.0
    extent_x = 0.0
    extent_y = 0.0
    if selected_area:
        selected_x = np.asarray(xyz[..., 0], dtype=np.float64)[selected]
        selected_y = np.asarray(xyz[..., 2], dtype=np.float64)[selected]
        finite = np.isfinite(selected_x) & np.isfinite(selected_y)
        if np.any(finite):
            extent_x = float(np.max(selected_x[finite]) - np.min(selected_x[finite]))
            extent_y = float(np.max(selected_y[finite]) - np.min(selected_y[finite]))
    if selected_area < 16:
        Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(mask_path)
        return _write_tabletop_support_report(
            report_path,
            {
                **base_report,
                "candidate_count": candidate_count,
                "planar_candidate_count": planar_candidate_count,
                "component_count": int(component_count),
                "selected_component_area": selected_area,
                "selected_component_label": int(anchor["label"]),
                "selected_component_labels": [int(label) for label in selected_labels],
                "source": source,
                "blocked_reason": "insufficient_planar_tabletop_component",
                "contact_overlap_px": int(anchor["contact_overlap_px"]),
                "plane_residual_tolerance_m": float(TABLETOP_PLANE_RESIDUAL_TOLERANCE_M),
                "selected_candidate_ratio": selected_candidate_ratio,
                "selected_planar_candidate_ratio": selected_planar_candidate_ratio,
                "selected_extent_x_m": extent_x,
                "selected_extent_y_m": extent_y,
            },
        )
    if selected_candidate_ratio < MIN_TABLETOP_SELECTED_CANDIDATE_RATIO:
        Image.fromarray(selected.astype(np.uint8) * 255).save(mask_path)
        return _write_tabletop_support_report(
            report_path,
            {
                **base_report,
                "candidate_count": candidate_count,
                "planar_candidate_count": planar_candidate_count,
                "component_count": int(component_count),
                "selected_component_area": selected_area,
                "selected_component_label": int(anchor["label"]),
                "selected_component_labels": [int(label) for label in selected_labels],
                "source": source,
                "blocked_reason": "table_collision_support_coverage_too_small",
                "contact_overlap_px": int(anchor["contact_overlap_px"]),
                "plane_residual_tolerance_m": float(TABLETOP_PLANE_RESIDUAL_TOLERANCE_M),
                "selected_candidate_ratio": selected_candidate_ratio,
                "selected_planar_candidate_ratio": selected_planar_candidate_ratio,
                "min_selected_candidate_ratio": float(MIN_TABLETOP_SELECTED_CANDIDATE_RATIO),
                "selected_extent_x_m": extent_x,
                "selected_extent_y_m": extent_y,
            },
        )
    Image.fromarray(selected.astype(np.uint8) * 255).save(mask_path)
    return _write_tabletop_support_report(
        report_path,
        {
            **base_report,
            "candidate_count": candidate_count,
            "planar_candidate_count": planar_candidate_count,
            "component_count": int(component_count),
            "selected_component_area": selected_area,
            "selected_component_label": int(anchor["label"]),
            "selected_component_labels": [int(label) for label in selected_labels],
            "source": source,
            "status": "passed",
            "blocked_reason": None,
            "contact_overlap_px": int(anchor["contact_overlap_px"]),
            "plane_residual_tolerance_m": float(TABLETOP_PLANE_RESIDUAL_TOLERANCE_M),
            "selected_candidate_ratio": selected_candidate_ratio,
            "selected_planar_candidate_ratio": selected_planar_candidate_ratio,
            "min_selected_candidate_ratio": float(MIN_TABLETOP_SELECTED_CANDIDATE_RATIO),
            "selected_extent_x_m": extent_x,
            "selected_extent_y_m": extent_y,
        },
    )


def _write_tabletop_support_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _write_tabletop_polygon_collision(
    run: Path,
    objects: list[dict[str, Any]],
    mask_report: dict[str, Any],
    *,
    support_height_m: float,
) -> dict[str, Any]:
    mask = _read_mask(run / str(mask_report["mask_path"]), None)
    xyz = np.load(run / "xyz.npy")
    if mask is None or mask.shape != xyz.shape[:2]:
        return _write_table_collision_report(
            run,
            {
                "status": "blocked",
                "blocked_reason": "invalid_tabletop_mask",
                "source_backend": "tabletop_mask_polygon_slab",
                "geometry_type": "polygon_slab",
                "final": False,
            },
        )
    visible_polygons_xy = _tabletop_component_polygons_xy(mask, xyz)
    object_footprints_xy = _object_support_footprint_polygons_xy(run, objects)
    polygons_xy = visible_polygons_xy + object_footprints_xy
    if not polygons_xy:
        return _write_table_collision_report(
            run,
            {
                "status": "blocked",
                "blocked_reason": "insufficient_tabletop_polygon_points",
                "source_backend": "tabletop_mask_polygon_slab",
                "geometry_type": "polygon_slab",
                "final": False,
            },
        )

    thickness = 0.04
    all_polygon_points = np.concatenate(polygons_xy, axis=0)
    x0, y0 = np.min(all_polygon_points, axis=0)
    x1, y1 = np.max(all_polygon_points, axis=0)
    size = [float(x1 - x0), float(y1 - y0), float(thickness)]
    if min(size[0], size[1]) < MIN_FINAL_TABLE_EXTENT_M:
        return _write_table_collision_report(
            run,
            {
                "status": "blocked",
                "blocked_reason": "table_collision_extent_too_small",
                "source_backend": "tabletop_mask_polygon_slab",
                "geometry_type": "polygon_slab",
                "final": False,
                "table_collision_size_xyz": size,
                "min_final_table_extent_m": float(MIN_FINAL_TABLE_EXTENT_M),
                "tabletop_mask_path": str(mask_report["mask_path"]),
                "tabletop_support_report_path": str(mask_report["report_path"]),
            },
        )
    center = [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5), float(-thickness * 0.5)]
    center_xy = np.asarray(center[:2], dtype=np.float64)
    meshes = [_polygon_slab_mesh(polygon_xy - center_xy, thickness=thickness) for polygon_xy in polygons_xy]
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    table_path = run / "background" / "table_collision.glb"
    mesh.export(table_path)
    polygon_path = run / "background" / "table_polygon_world.json"
    polygon_payload = {
        "status": "passed",
        "geometry_type": "polygon_slab",
        "source_backend": "tabletop_mask_polygon_slab",
        "polygon_world_xy": [[float(x), float(y)] for x, y in polygons_xy[0].tolist()],
        "polygons_world_xy": [[[float(x), float(y)] for x, y in polygon_xy.tolist()] for polygon_xy in polygons_xy],
        "polygon_count": int(len(polygons_xy)),
        "visible_tabletop_polygon_count": int(len(visible_polygons_xy)),
        "object_support_footprint_polygon_count": int(len(object_footprints_xy)),
        "support_height_m": float(support_height_m),
        "top_z_m": 0.0,
        "thickness_m": float(thickness),
        "input_point_count": int(np.count_nonzero(mask)),
        "candidate_count": int(mask_report.get("candidate_count", 0) or 0),
        "planar_candidate_count": int(mask_report.get("planar_candidate_count", 0) or 0),
        "selected_component_area": int(mask_report.get("selected_component_area", 0) or 0),
        "selected_candidate_ratio": float(mask_report.get("selected_candidate_ratio", 0.0) or 0.0),
        "selected_planar_candidate_ratio": float(mask_report.get("selected_planar_candidate_ratio", 0.0) or 0.0),
        "polygon_extent_x_m": float(size[0]),
        "polygon_extent_y_m": float(size[1]),
    }
    polygon_path.write_text(json.dumps(polygon_payload, indent=2), encoding="utf-8")
    collision_report_path = _write_table_collision_report(
        run,
        {
            "status": "passed",
            "blocked_reason": None,
            "source_backend": "tabletop_mask_polygon_slab",
            "geometry_type": "polygon_slab",
            "final": True,
            "mesh_path": str(table_path.relative_to(run)),
            "polygon_path": str(polygon_path.relative_to(run)),
            "table_collision_size_xyz": size,
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "polygon_count": int(len(polygons_xy)),
            "visible_tabletop_polygon_count": int(len(visible_polygons_xy)),
            "object_support_footprint_polygon_count": int(len(object_footprints_xy)),
            "candidate_count": int(mask_report.get("candidate_count", 0) or 0),
            "selected_component_area": int(mask_report.get("selected_component_area", 0) or 0),
            "selected_candidate_ratio": float(mask_report.get("selected_candidate_ratio", 0.0) or 0.0),
            "polygon_extent_x_m": float(size[0]),
            "polygon_extent_y_m": float(size[1]),
        },
    )
    return {
        "table_collision_mesh_path": str(table_path.relative_to(run)),
        "table_collision_source_backend": "tabletop_mask_polygon_slab",
        "table_collision_geometry_type": "polygon_slab",
        "table_collision_final": True,
        "support_surface_status": "passed",
        "support_surface_blocked_reason": None,
        "table_bounds_world_xy": [[float(x0), float(y0)], [float(x1), float(y1)]],
        "table_collision_pos_world": center,
        "table_collision_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "table_collision_size_xyz": size,
        "table_top_z_m": 0.0,
        "table_thickness_m": float(thickness),
        "tabletop_mask_path": str(mask_report["mask_path"]),
        "tabletop_support_report_path": str(mask_report["report_path"]),
        "table_polygon_world_path": str(polygon_path.relative_to(run)),
        "table_collision_report_path": collision_report_path,
    }


def _tabletop_component_polygons_xy(mask: np.ndarray, xyz: np.ndarray) -> list[np.ndarray]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    total_area = int(np.count_nonzero(mask))
    min_area = max(3, int(round(total_area * 0.001)))
    polygons: list[np.ndarray] = []
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_area:
            continue
        component = labels == label
        points_xy = np.stack([xyz[..., 0][component], xyz[..., 2][component]], axis=1).astype(np.float64)
        points_xy = points_xy[np.all(np.isfinite(points_xy), axis=1)]
        if points_xy.shape[0] < 3:
            continue
        hull = cv2.convexHull(points_xy.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        if hull.shape[0] >= 3:
            polygons.append(hull)
    return polygons


def _object_support_footprint_polygons_xy(run: Path, objects: list[dict[str, Any]]) -> list[np.ndarray]:
    footprints: list[np.ndarray] = []
    padding_m = 0.04
    bottom_tolerance_m = 0.03
    for item in objects:
        vertices = _mesh_vertices(_object_geometry_path(run, item))
        transform = _transform_array(item["T_object_to_world"])
        homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
        world_vertices = (transform @ homogeneous.T).T[:, :3]
        finite = np.all(np.isfinite(world_vertices), axis=1)
        world_vertices = world_vertices[finite]
        if len(world_vertices) < 3:
            continue
        bottom_z = float(np.min(world_vertices[:, 2]))
        support_vertices = world_vertices[world_vertices[:, 2] <= bottom_z + bottom_tolerance_m]
        if len(support_vertices) < 3:
            support_vertices = world_vertices
        x0 = float(np.min(support_vertices[:, 0]) - padding_m)
        x1 = float(np.max(support_vertices[:, 0]) + padding_m)
        y0 = float(np.min(support_vertices[:, 1]) - padding_m)
        y1 = float(np.max(support_vertices[:, 1]) + padding_m)
        if x1 <= x0 or y1 <= y0:
            continue
        footprints.append(
            np.asarray(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float64,
            )
        )
    return footprints


def _polygon_slab_mesh(polygon_xy: np.ndarray, *, thickness: float) -> trimesh.Trimesh:
    n = int(polygon_xy.shape[0])
    top = np.column_stack([polygon_xy, np.full(n, float(thickness) * 0.5, dtype=np.float64)])
    bottom = np.column_stack([polygon_xy, np.full(n, -float(thickness) * 0.5, dtype=np.float64)])
    vertices = np.vstack([top, bottom])
    faces: list[list[int]] = []
    for idx in range(1, n - 1):
        faces.append([0, idx, idx + 1])
        faces.append([n, n + idx + 1, n + idx])
    for idx in range(n):
        nxt = (idx + 1) % n
        faces.append([idx, nxt, n + nxt])
        faces.append([idx, n + nxt, n + idx])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=True)


def _write_estimated_table_box_collision(
    run: Path,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    source: str,
    mask_report: dict[str, Any],
) -> dict[str, Any]:
    (x0, y0), (x1, y1) = bounds
    thickness = 0.04
    extent_x = max(0.2, float(x1 - x0))
    extent_y = max(0.2, float(y1 - y0))
    center = [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5), float(-thickness * 0.5)]
    size = [float(extent_x), float(extent_y), float(thickness)]
    mesh = trimesh.creation.box(extents=size)
    table_path = run / "background" / "table_collision.glb"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(table_path)
    blocked_reason = mask_report.get("blocked_reason") or "tabletop_mask_unavailable"
    collision_report = _write_table_collision_report(
        run,
        {
            "status": "blocked",
            "blocked_reason": blocked_reason,
            "source_backend": source,
            "geometry_type": "estimated_box_proxy",
            "final": False,
            "mesh_path": str(table_path.relative_to(run)),
            "table_bounds_world_xy": [[float(x0), float(y0)], [float(x1), float(y1)]],
            "table_collision_size_xyz": size,
        },
    )
    return {
        "table_collision_mesh_path": str(table_path.relative_to(run)),
        "table_collision_source_backend": source,
        "table_collision_geometry_type": "estimated_box_proxy",
        "table_collision_final": False,
        "support_surface_status": "blocked",
        "support_surface_blocked_reason": blocked_reason,
        "table_bounds_world_xy": [[float(x0), float(y0)], [float(x1), float(y1)]],
        "table_collision_pos_world": center,
        "table_collision_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "table_collision_size_xyz": size,
        "table_top_z_m": 0.0,
        "table_thickness_m": float(thickness),
        "tabletop_mask_path": str(mask_report.get("mask_path", "")),
        "tabletop_support_report_path": str(mask_report.get("report_path", "")),
        "table_collision_report_path": collision_report,
    }


def _write_table_collision_report(run: Path, report: dict[str, Any]) -> str:
    path = run / "background" / "table_collision_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(path.relative_to(run))


def _refresh_table_collision_projection_qa(run: Path, support_plane: dict[str, Any]) -> dict[str, Any]:
    if not (run / "camera.json").is_file() or not (run / "background" / "table_polygon_world.json").is_file():
        return support_plane
    qa_report_path = run / "qa" / "table_collision_report.json"
    try:
        from .table_collision_qa import write_table_collision_projection_qa

        qa_result = write_table_collision_projection_qa(run)
        qa_report = qa_result.report
    except Exception as exc:  # noqa: BLE001 - support-surface QA must fail closed.
        qa_report_path.parent.mkdir(parents=True, exist_ok=True)
        qa_report = {
            "version": 1,
            "status": "blocked",
            "blocking_reasons": ["table_collision_projection_failed"],
            "error": str(exc),
            "visual_qa_path": None,
        }
        qa_report_path.write_text(json.dumps(qa_report, indent=2), encoding="utf-8")

    updated = dict(support_plane)
    updated["table_collision_projection_report_path"] = "qa/table_collision_report.json"
    updated["table_collision_visual_qa_path"] = qa_report.get("visual_qa_path")
    updated["table_collision_projection_iou"] = qa_report.get("projection_iou")
    updated["table_collision_projection_status"] = qa_report.get("status")
    updated["table_collision_projection_blocking_reasons"] = qa_report.get("blocking_reasons", [])
    if qa_report.get("status") != "passed":
        updated["support_surface_status"] = "blocked"
        updated["support_surface_blocked_reason"] = "table_collision_projection_failed"
        updated["table_collision_final"] = False
    return updated


def _object_mask_union(run: Path, objects: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for item in objects:
        mask = _read_mask(run / str(item.get("mask_path", "")), shape)
        if mask is not None:
            union |= mask
    return union


def _read_mask(path: Path, shape: tuple[int, int] | None) -> np.ndarray | None:
    if not path.is_file():
        return None
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if shape is not None and mask.shape != shape:
        return None
    return mask


def _load_json_doc(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _table_bounds_from_background_points(
    run: Path,
    objects: list[dict[str, Any]],
    *,
    support_height_m: float,
) -> tuple[tuple[tuple[float, float], tuple[float, float]] | None, str]:
    xyz_path = run / "xyz.npy"
    if not xyz_path.is_file():
        return None, "unavailable"
    xyz = np.load(xyz_path)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        return None, "unavailable"
    foreground = np.zeros(xyz.shape[:2], dtype=bool)
    for item in objects:
        mask_path = run / str(item.get("mask_path", ""))
        if not mask_path.is_file():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape == xyz.shape[:2]:
            foreground |= mask

    valid = np.all(np.isfinite(xyz), axis=2) & (xyz[..., 2] > 0.0)
    world_z = -np.asarray(xyz[..., 1], dtype=np.float64) - float(support_height_m)
    near_support = valid & ~foreground & (np.abs(world_z) <= 0.06)
    if np.count_nonzero(near_support) < 16:
        return None, "unavailable"
    world_x = np.asarray(xyz[..., 0], dtype=np.float64)[near_support]
    world_y = np.asarray(xyz[..., 2], dtype=np.float64)[near_support]
    x0, x1 = np.percentile(world_x, [2.0, 98.0])
    y0, y1 = np.percentile(world_y, [2.0, 98.0])
    return _pad_bounds(float(x0), float(y0), float(x1), float(y1), padding=0.08), "estimated_support_box_proxy"


def _table_bounds_from_objects(run: Path, objects: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    vertices = []
    for item in objects:
        mesh_vertices = _mesh_vertices(_object_geometry_path(run, item))
        transform = _transform_array(item["T_object_to_world"])
        homogeneous = np.concatenate([mesh_vertices, np.ones((mesh_vertices.shape[0], 1), dtype=np.float64)], axis=1)
        vertices.append((transform @ homogeneous.T).T[:, :3])
    points = np.concatenate(vertices, axis=0)
    return _pad_bounds(
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
        padding=0.25,
    )


def _pad_bounds(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    padding: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0 - padding, y0 - padding), (x1 + padding, y1 + padding)


def _object_bottoms(run: Path, objects: list[dict[str, Any]]) -> dict[str, float]:
    bottoms: dict[str, float] = {}
    for item in objects:
        vertices = _mesh_vertices(_object_geometry_path(run, item))
        transform = _transform_array(item["T_object_to_world"])
        homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
        world_vertices = (transform @ homogeneous.T).T[:, :3]
        bottoms[str(item["object_id"])] = float(np.min(world_vertices[:, 2]))
    return bottoms


def _object_geometry_path(run: Path, item: dict[str, Any]) -> Path:
    collision_asset = item.get("collision_asset")
    if isinstance(collision_asset, dict) and collision_asset.get("path"):
        collision_path = run / str(collision_asset["path"])
        if collision_path.is_file():
            return collision_path
    return run / str(item["mesh_path"])


def _mesh_vertices(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, force="scene")
    vertices = []
    if isinstance(loaded, trimesh.Trimesh):
        vertices.append(np.asarray(loaded.vertices, dtype=np.float64))
    elif isinstance(loaded, trimesh.Scene):
        for geom in loaded.geometry.values():
            if hasattr(geom, "vertices"):
                vertices.append(np.asarray(geom.vertices, dtype=np.float64))
    if not vertices:
        raise ValueError(f"{path} has no mesh vertices")
    return np.concatenate(vertices, axis=0)


def _transform_array(value: object) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("T_object_to_world must be 4x4")
    return transform


def _update_pose_file(run: Path, object_id: object, transform: np.ndarray) -> None:
    pose_path = run / "objects" / str(object_id) / "pose.json"
    if not pose_path.is_file():
        return
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    pose["T_object_to_world"] = transform.tolist()
    pose_path.write_text(json.dumps(pose, indent=2), encoding="utf-8")


def _merge_qa_support_plane(run: Path, support_plane: dict[str, Any]) -> None:
    qa_path = run / "qa" / "qa_report.json"
    if not qa_path.exists():
        return
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["support_plane"] = support_plane
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")


def _write_usda_stub_from_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#usda 1.0",
        "(",
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        ")",
        "",
        'def Xform "World"',
        "{",
    ]
    support = manifest.get("support_plane")
    if isinstance(support, dict):
        lines.append(f'    custom string support_plane_status = "{support.get("status", "unknown")}"')
        lines.append(f"    custom double support_plane_height_world_m = {float(support.get('height_world_m', 0.0)):.8g}")
        if support.get("table_collision_mesh_path"):
            lines.append(f'    custom string table_collision_mesh_path = "{support["table_collision_mesh_path"]}"')
        if support.get("table_collision_pos_world"):
            pos = support["table_collision_pos_world"]
            lines.append(f"    custom double3 table_collision_pos_world = ({float(pos[0]):.8g}, {float(pos[1]):.8g}, {float(pos[2]):.8g})")
        if support.get("table_collision_size_xyz"):
            size = support["table_collision_size_xyz"]
            lines.append(f"    custom double3 table_collision_size_xyz = ({float(size[0]):.8g}, {float(size[1]):.8g}, {float(size[2]):.8g})")
    for obj in manifest.get("objects", []):
        transform = _transform_array(obj["T_object_to_world"])
        tx, ty, tz = (float(transform[i, 3]) for i in range(3))
        lines.extend(
            [
                f'    def Xform "{obj["object_id"]}"',
                "    {",
                f'        custom string mesh_path = "{obj["mesh_path"]}"',
                f'        custom string visual_asset_path = "{obj.get("visual_asset", {}).get("path", obj["mesh_path"])}"',
                f'        custom string collision_asset_path = "{obj.get("collision_asset", {}).get("path", obj["mesh_path"])}"',
                f"        custom double mass_kg = {float(obj['mass_kg']):.8g}",
                f"        custom double friction = {float(obj['friction']):.8g}",
                f"        double3 xformOp:translate = ({tx:.8g}, {ty:.8g}, {tz:.8g})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
            ]
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
