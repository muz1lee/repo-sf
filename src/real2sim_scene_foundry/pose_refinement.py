"""Pose refinement helpers for export-grade object placement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image


DEFAULT_CAMERA_TO_WORLD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PoseSupportRefinementResult:
    report_path: Path
    report: dict[str, Any]


def refine_visual_pose_to_masks(
    run_dir: str | Path,
    *,
    tolerance_m: float = 0.005,
    min_scale_factor: float = 0.25,
    max_scale_factor: float = 4.0,
) -> PoseSupportRefinementResult:
    """Refine per-object visual asset scale against reference-camera masks."""
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support = manifest.get("support_plane", {})
    support_z = _support_z(support)
    camera = _load_reference_camera(run)
    object_reports = []
    for item in manifest.get("objects", []):
        object_reports.append(
            _refine_one_object_to_mask(
                run,
                item,
                camera=camera,
                support_z=support_z,
                support_camera_height=_support_camera_height(support),
                tolerance_m=float(tolerance_m),
                min_scale_factor=float(min_scale_factor),
                max_scale_factor=float(max_scale_factor),
            )
        )
    _expand_support_collision_to_object_footprints(run, manifest, support)
    support["pose_refinement_applied"] = True
    support["visual_bbox_refinement_applied"] = True
    support["object_bottoms_after_pose_refinement_m"] = {
        str(item["object_id"]): float(report["support_alignment"]["after_bottom_z_m"])
        for item, report in zip(manifest.get("objects", []), object_reports, strict=True)
        if report.get("support_alignment", {}).get("after_bottom_z_m") is not None
    }
    manifest["support_plane"] = support
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    blocking = [f"{item['object_id']}:{item.get('blocked_reason', item.get('status'))}" for item in object_reports if item["status"] != "accepted"]
    report = {
        "version": 1,
        "status": "accepted" if not blocking else "blocked",
        "source": "manual_refined_from_auto",
        "operation": "reference_camera_bbox_scale_refinement",
        "support_z_m": float(support_z),
        "support_camera_height_m": float(_support_camera_height(support)),
        "tolerance_m": float(tolerance_m),
        "scale_factor_clamp": [float(min_scale_factor), float(max_scale_factor)],
        "objects": object_reports,
        "blocking_reasons": blocking,
    }
    report_path = qa_dir / "reference_projection_refinement_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return PoseSupportRefinementResult(report_path=report_path, report=report)


def apply_visual_orientation_overrides(
    run_dir: str | Path,
    overrides: dict[str, str],
    *,
    tolerance_m: float = 0.005,
) -> PoseSupportRefinementResult:
    """Apply explicit local-frame visual orientation overrides and keep support contact."""
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support = manifest.get("support_plane", {})
    support_z = _support_z(support)
    object_reports = []
    for item in manifest.get("objects", []):
        object_id = str(item["object_id"])
        if object_id not in overrides:
            continue
        object_reports.append(
            _apply_one_orientation_override(
                run,
                item,
                override=str(overrides[object_id]),
                support_z=support_z,
                tolerance_m=float(tolerance_m),
            )
        )
    _expand_support_collision_to_object_footprints(run, manifest, support)
    support["pose_refinement_applied"] = True
    support["visual_orientation_refinement_applied"] = True
    manifest["support_plane"] = support
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    blocking = [f"{item['object_id']}:{item.get('blocked_reason', item.get('status'))}" for item in object_reports if item["status"] != "accepted"]
    report = {
        "version": 1,
        "status": "accepted" if not blocking else "blocked",
        "source": "manual_refined_from_auto",
        "operation": "manual_visual_orientation_override",
        "support_z_m": float(support_z),
        "tolerance_m": float(tolerance_m),
        "overrides": dict(overrides),
        "objects": object_reports,
        "blocking_reasons": blocking,
    }
    report_path = qa_dir / "visual_orientation_refinement_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return PoseSupportRefinementResult(report_path=report_path, report=report)


def snap_object_poses_to_support(run_dir: str | Path, *, tolerance_m: float = 0.005) -> PoseSupportRefinementResult:
    """Snap each foreground visual mesh to the support plane and write provenance reports."""
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support = manifest.get("support_plane", {})
    support_z = _support_z(support)
    object_reports = []
    for item in manifest.get("objects", []):
        object_report = _snap_one_object(run, item, support_z=support_z, tolerance_m=float(tolerance_m))
        object_reports.append(object_report)
    support["pose_refinement_applied"] = True
    support["object_bottoms_after_pose_refinement_m"] = {
        str(item["object_id"]): float(report["support_alignment"]["after_bottom_z_m"]) for item, report in zip(manifest.get("objects", []), object_reports, strict=True)
    }
    manifest["support_plane"] = support
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    blocking = [f"{item['object_id']}:{item['support_alignment']['status']}" for item in object_reports if item["status"] != "accepted"]
    report = {
        "version": 1,
        "status": "accepted" if not blocking else "blocked",
        "source": "manual_refined_from_auto",
        "operation": "snap_visual_bounds_to_support_plane",
        "support_z_m": float(support_z),
        "tolerance_m": float(tolerance_m),
        "objects": object_reports,
        "blocking_reasons": blocking,
    }
    report_path = qa_dir / "pose_support_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return PoseSupportRefinementResult(report_path=report_path, report=report)


def measure_object_support_alignment(
    run_dir: str | Path,
    item: dict[str, Any],
    support_plane: dict[str, Any] | None = None,
    *,
    tolerance_m: float = 0.005,
) -> dict[str, Any]:
    run = Path(run_dir)
    support_z = _support_z(support_plane or {})
    visual_path = _visual_geometry_path(run, item)
    if visual_path is None:
        return {
            "status": "missing_visual_geometry",
            "mesh_path": None,
            "support_z_m": float(support_z),
            "tolerance_m": float(tolerance_m),
        }
    try:
        bounds = _world_bounds(visual_path, _transform_array(item["T_object_to_world"]), asset_scale=_asset_scale(item))
    except Exception as exc:  # noqa: BLE001 - export manifest records unreadable geometry as blocked provenance.
        return {
            "status": "unreadable_visual_geometry",
            "mesh_path": str(visual_path.relative_to(run)),
            "support_z_m": float(support_z),
            "tolerance_m": float(tolerance_m),
            "error": str(exc),
        }
    bottom = float(bounds["min"][2])
    clearance = bottom - support_z
    if clearance < -float(tolerance_m):
        status = "penetrating_support"
    else:
        status = "support_aligned"
    return {
        "status": status,
        "mesh_path": str(visual_path.relative_to(run)),
        "support_z_m": float(support_z),
        "bottom_z_m": bottom,
        "clearance_m": float(clearance),
        "tolerance_m": float(tolerance_m),
        "world_bounds": bounds,
    }


def refine_pose_rgbd(run_dir: str | Path, *, frames: str = "reference") -> PoseSupportRefinementResult:
    """Write fail-closed RGB-D/mask/mesh pose alignment reports."""
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support = manifest.get("support_plane", {})
    object_reports = []
    for item in manifest.get("objects", []):
        object_reports.append(_write_rgbd_pose_report(run, item, support, frames=frames))
    aggregate = _object_alignment_aggregate(object_reports)
    report = {
        "version": 1,
        "status": aggregate["status"],
        "source": "rgbd_refined_from_auto",
        "operation": "rgbd_mask_mesh_pose_verification",
        "frames": frames,
        "thresholds": _object_alignment_thresholds(),
        "objects": object_reports,
        "blocking_reasons": aggregate["blocking_reasons"],
    }
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    report_path = qa_dir / "object_pose_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return PoseSupportRefinementResult(report_path=report_path, report=report)


def qa_object_alignment(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    manifest = _load_json(run / "scene_manifest.json")
    object_reports = []
    for item in manifest.get("objects", []):
        object_id = str(item.get("object_id"))
        report = _load_json(run / "objects" / object_id / "pose_refinement_report.json")
        status, reasons = _evaluate_object_alignment_report(report)
        object_reports.append(
            {
                "object_id": object_id,
                "status": status,
                "blocking_reasons": reasons,
                "metrics": report.get("metrics", {}),
                "report_path": f"objects/{object_id}/pose_refinement_report.json" if report else None,
            }
        )
    aggregate = _object_alignment_aggregate(object_reports)
    report = {
        "version": 1,
        "status": aggregate["status"],
        "thresholds": _object_alignment_thresholds(),
        "objects": object_reports,
        "blocking_reasons": aggregate["blocking_reasons"],
    }
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "object_pose_alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report_path": "qa/object_pose_alignment_report.json"}


def _write_rgbd_pose_report(run: Path, item: dict[str, Any], support: dict[str, Any], *, frames: str) -> dict[str, Any]:
    object_id = str(item["object_id"])
    object_dir = run / "objects" / object_id
    object_dir.mkdir(parents=True, exist_ok=True)
    support_alignment = measure_object_support_alignment(run, item, support)
    clearance = support_alignment.get("clearance_m")
    support_gap_abs = abs(float(clearance)) if clearance is not None else None
    penetration_depth = max(0.0, -float(clearance)) if clearance is not None else None
    metrics = {
        "silhouette_mask_iou": None,
        "projected_bbox_iou": None,
        "bbox_error_px": None,
        "depth_median_abs_m": None,
        "depth_p90_abs_m": None,
        "support_gap_abs_m": support_gap_abs,
        "penetration_depth_m": penetration_depth,
        "projected_center_error_px": None,
    }
    camera_source = _rgbd_camera_source(run)
    scale_source = str(item.get("scale_source") or item.get("pose_refinement", {}).get("scale_source") or "rgbd_alignment_unverified")
    phone_metrics = _phone_rgbd_alignment_metrics(run, item)
    metrics.update(phone_metrics.get("metrics", {}))
    report = {
        "version": 1,
        "object_id": object_id,
        "label": item.get("label", object_id),
        "status": "blocked",
        "source": "rgbd_refined_from_auto",
        "operation": "rgbd_mask_mesh_pose_verification",
        "frames": frames,
        "optimized_dofs": "yaw_xy_z_uniform_scale",
        "camera_source": camera_source,
        "scale_source": scale_source,
        "metrics": metrics,
        "metric_sources": phone_metrics.get("sources", {}),
        "support_alignment": support_alignment,
        "blocking_reasons": _rgbd_missing_metric_reasons(metrics, camera_source, scale_source),
        "T_object_to_camera": item.get("T_object_to_camera"),
        "T_object_to_world": item.get("T_object_to_world"),
    }
    status, reasons = _evaluate_object_alignment_report(report)
    report["status"] = "accepted" if status == "passed" else "blocked"
    report["blocking_reasons"] = reasons
    overlay = phone_metrics.get("overlay_rgb")
    if isinstance(overlay, np.ndarray):
        Image.fromarray(overlay.astype(np.uint8)).save(object_dir / "pose_overlay_ref_frame.png")
    else:
        Image.new("RGB", (2, 2), color=(0, 0, 0)).save(object_dir / "pose_overlay_ref_frame.png")
    residual_vis = phone_metrics.get("depth_residual_rgb")
    if isinstance(residual_vis, np.ndarray):
        Image.fromarray(residual_vis.astype(np.uint8)).save(object_dir / "depth_residual_ref_frame.png")
    else:
        Image.new("RGB", (2, 2), color=(0, 0, 0)).save(object_dir / "depth_residual_ref_frame.png")
    _write_object_report(object_dir, report)
    return report


def _phone_rgbd_alignment_metrics(run: Path, item: dict[str, Any]) -> dict[str, Any]:
    camera = _load_json(run / "camera.json")
    trajectory = _load_json(run / "trajectory.json")
    if (
        camera.get("intrinsics_source") != "arkit_explicit"
        or camera.get("extrinsics_source") != "arkit_explicit"
        or camera.get("scale_source") != "arkit_sceneDepth_meters"
    ):
        return {"metrics": {}, "sources": {"status": "unavailable", "reason": "phone_rgbd_evidence_missing"}}
    frame = _first_rgbd_frame(run, trajectory)
    k = _camera_matrix(camera)
    t_world_to_camera = _world_to_camera_transform(camera)
    mask = _read_binary_mask(run / str(item.get("mask_path", "")))
    visual_path = _visual_geometry_path(run, item)
    if frame is None or k is None or t_world_to_camera is None or mask is None or visual_path is None:
        return {"metrics": {}, "sources": {"status": "unavailable", "reason": "phone_rgbd_inputs_incomplete"}}

    depth = np.load(run / frame["depth_path"])
    confidence = np.asarray(Image.open(run / frame["confidence_path"]).convert("L"))
    if depth.ndim != 2 or confidence.shape != depth.shape or mask.shape != depth.shape:
        return {"metrics": {}, "sources": {"status": "blocked", "reason": "phone_rgbd_shape_mismatch"}}

    vertices = _mesh_vertices(visual_path) * _asset_scale(item)
    transform = _transform_array(item["T_object_to_world"])
    homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    vertices_world = (transform @ homogeneous.T).T
    vertices_camera = (t_world_to_camera @ vertices_world.T).T[:, :3]
    projected = _project_camera_points(vertices_camera, k)
    if projected is None or projected.shape[0] < 3:
        return {"metrics": {}, "sources": {"status": "blocked", "reason": "mesh_projection_invalid"}}

    silhouette = np.zeros(depth.shape, dtype=np.uint8)
    hull = cv2.convexHull(projected.astype(np.float32)).reshape(-1, 2)
    cv2.fillConvexPoly(silhouette, np.rint(hull).astype(np.int32), 255)
    silhouette_mask = silhouette > 0
    object_mask = mask > 0
    overlap = silhouette_mask & object_mask & (confidence > 0) & np.isfinite(depth) & (depth > 0.0)
    depth_values = None
    if np.any(overlap):
        predicted_depth = float(np.percentile(vertices_camera[vertices_camera[:, 2] > 1e-6, 2], 5.0))
        depth_values = np.abs(depth[overlap].astype(np.float64) - predicted_depth)

    silhouette_iou = _binary_iou(silhouette_mask, object_mask)
    projected_bbox = _bbox_from_binary_mask(silhouette_mask)
    mask_bbox = _bbox_from_binary_mask(object_mask)
    projected_bbox_iou = _bbox_iou_xyxy(projected_bbox, mask_bbox)
    projected_center = _mask_center(silhouette_mask)
    mask_center = _mask_center(object_mask)
    center_error = (
        float(np.linalg.norm(np.asarray(projected_center, dtype=np.float64) - np.asarray(mask_center, dtype=np.float64)))
        if projected_center is not None and mask_center is not None
        else None
    )
    bbox_error = _bbox_corner_error(projected_bbox, mask_bbox)

    overlay = np.zeros((*depth.shape, 3), dtype=np.uint8)
    overlay[..., 1] = object_mask.astype(np.uint8) * 255
    overlay[..., 2] = silhouette_mask.astype(np.uint8) * 255
    overlay[object_mask & silhouette_mask] = [255, 255, 255]
    residual_vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if depth_values is not None:
        residual = np.zeros(depth.shape, dtype=np.float64)
        residual[overlap] = np.clip(depth_values / 0.08, 0.0, 1.0)
        residual_vis[..., 0] = (residual * 255).astype(np.uint8)
        residual_vis[..., 1] = ((1.0 - residual) * overlap * 255).astype(np.uint8)

    return {
        "metrics": {
            "silhouette_mask_iou": silhouette_iou,
            "projected_bbox_iou": projected_bbox_iou,
            "bbox_error_px": bbox_error,
            "depth_median_abs_m": float(np.median(depth_values)) if depth_values is not None and depth_values.size else None,
            "depth_p90_abs_m": float(np.percentile(depth_values, 90.0)) if depth_values is not None and depth_values.size else None,
            "projected_center_error_px": center_error,
        },
        "sources": {
            "status": "computed",
            "frame_id": frame["frame_id"],
            "depth_path": frame["depth_path"],
            "confidence_path": frame["confidence_path"],
            "silhouette_source": "mesh_convex_hull_projection_not_bbox",
            "depth_source": "phone_capture_depth_confidence",
            "intrinsics_source": camera.get("intrinsics_source"),
            "extrinsics_source": camera.get("extrinsics_source"),
            "scale_source": camera.get("scale_source"),
        },
        "overlay_rgb": overlay,
        "depth_residual_rgb": residual_vis,
    }


def _first_rgbd_frame(run: Path, trajectory: dict[str, Any]) -> dict[str, str] | None:
    frames = trajectory.get("frames", [])
    if not isinstance(frames, list):
        return None
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        depth_path = str(frame.get("depth_path", ""))
        confidence_path = str(frame.get("confidence_path", ""))
        if depth_path and confidence_path and (run / depth_path).is_file() and (run / confidence_path).is_file():
            return {"frame_id": str(frame.get("frame_id") or Path(depth_path).stem), "depth_path": depth_path, "confidence_path": confidence_path}
    return None


def _camera_matrix(camera: dict[str, Any]) -> np.ndarray | None:
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


def _world_to_camera_transform(camera: dict[str, Any]) -> np.ndarray | None:
    if camera.get("T_world_to_camera") is not None:
        transform = np.asarray(camera["T_world_to_camera"], dtype=np.float64)
        return transform if transform.shape == (4, 4) and np.all(np.isfinite(transform)) else None
    if camera.get("T_camera_to_world") is None:
        return None
    transform = np.asarray(camera["T_camera_to_world"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    try:
        return np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return None


def _read_binary_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return np.asarray(Image.open(path).convert("L")) > 0
    except Exception:  # noqa: BLE001 - corrupted masks make the metric unavailable.
        return None


def _project_camera_points(points_camera: np.ndarray, k: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-6)
    if np.count_nonzero(valid) < 3:
        return None
    points = points_camera[valid]
    pixels_h = (k @ points.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    finite = np.isfinite(pixels).all(axis=1)
    pixels = pixels[finite]
    return pixels if pixels.shape[0] >= 3 else None


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape:
        return None
    union = np.logical_or(a, b)
    if not np.any(union):
        return None
    return float(np.logical_and(a, b).sum() / union.sum())


def _bbox_from_binary_mask(mask: np.ndarray) -> list[float] | None:
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None
    return [float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())]


def _bbox_iou_xyxy(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    intersection = max(0.0, x1 - x0 + 1.0) * max(0.0, y1 - y0 + 1.0)
    area_a = max(0.0, a[2] - a[0] + 1.0) * max(0.0, a[3] - a[1] + 1.0)
    area_b = max(0.0, b[2] - b[0] + 1.0) * max(0.0, b[3] - b[1] + 1.0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else None


def _bbox_corner_error(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None
    return float(cols.mean()), float(rows.mean())


def _rgbd_camera_source(run: Path) -> str:
    camera = _load_json(run / "camera.json")
    has_intrinsics = any(key in camera for key in ("K", "intrinsics")) or all(key in camera for key in ("fx", "fy", "cx", "cy"))
    has_extrinsics = "T_world_to_camera" in camera or "T_camera_to_world" in camera
    return "explicit" if has_intrinsics and has_extrinsics else "fallback"


def _object_alignment_thresholds() -> dict[str, float]:
    return {
        "silhouette_mask_iou": 0.65,
        "depth_median_abs_m": 0.03,
        "depth_p90_abs_m": 0.08,
        "support_gap_abs_m": 0.005,
        "penetration_depth_m": 0.005,
        "projected_center_error_px": 20.0,
    }


def _evaluate_object_alignment_report(report: dict[str, Any]) -> tuple[str, list[str]]:
    if not report:
        return "blocked", ["pose_refinement_report_missing"]
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    reasons = list(report.get("blocking_reasons", []))
    thresholds = _object_alignment_thresholds()
    checks = [
        ("silhouette_mask_iou", "silhouette_mask_iou_below_threshold", lambda value, threshold: value >= threshold),
        ("depth_median_abs_m", "depth_median_residual_too_high", lambda value, threshold: value <= threshold),
        ("depth_p90_abs_m", "depth_p90_residual_too_high", lambda value, threshold: value <= threshold),
        ("support_gap_abs_m", "support_gap_too_large", lambda value, threshold: value <= threshold),
        ("penetration_depth_m", "penetration_depth_too_large", lambda value, threshold: value <= threshold),
        ("projected_center_error_px", "projected_center_error_too_high", lambda value, threshold: value <= threshold),
    ]
    for key, reason, predicate in checks:
        value = _metric_float(metrics.get(key))
        if value is None:
            reasons.append(f"{key}_unavailable")
        elif not predicate(value, thresholds[key]):
            reasons.append(reason)
    if report.get("camera_source") != "explicit":
        reasons.append("camera_source_fallback")
    if str(report.get("scale_source") or "").lower() in {"bbox_only", "reference_camera_bbox_refinement"}:
        reasons.append("scale_source_bbox_only")
    reasons = _dedupe(reasons)
    return ("passed" if not reasons else "blocked"), reasons


def _rgbd_missing_metric_reasons(metrics: dict[str, Any], camera_source: str, scale_source: str) -> list[str]:
    report = {"metrics": metrics, "camera_source": camera_source, "scale_source": scale_source, "blocking_reasons": []}
    return _evaluate_object_alignment_report(report)[1]


def _object_alignment_aggregate(object_reports: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    for report in object_reports:
        object_id = str(report.get("object_id", "unknown"))
        status = report.get("status")
        if status not in {"accepted", "passed"}:
            reasons.append(f"object_alignment_blocked:{object_id}")
        reasons.extend(f"{object_id}:{reason}" for reason in report.get("blocking_reasons", []))
    reasons = _dedupe(reasons)
    return {"status": "passed" if not reasons else "blocked", "blocking_reasons": reasons}


def _metric_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _snap_one_object(run: Path, item: dict[str, Any], *, support_z: float, tolerance_m: float) -> dict[str, Any]:
    object_id = str(item["object_id"])
    object_dir = run / "objects" / object_id
    visual_path = _visual_geometry_path(run, item)
    if visual_path is None:
        report = {
            "version": 1,
            "object_id": object_id,
            "status": "blocked",
            "source": "manual_refined_from_auto",
            "rotation_source": "manual_refined_from_auto",
            "translation_source": "manual_refined_from_viewer_overlay",
            "scale_source": "rgbd_alignment",
            "support_alignment": {
                "status": "missing_visual_geometry",
                "support_z_m": float(support_z),
                "tolerance_m": float(tolerance_m),
            },
        }
        _write_object_report(object_dir, report)
        return report

    previous_report = _load_json(object_dir / "pose_refinement_report.json")
    T_world_before = _transform_array(item["T_object_to_world"])
    T_camera_before = _transform_array(item.get("T_object_to_camera", DEFAULT_CAMERA_TO_WORLD))
    camera_to_world = _camera_to_world_bridge(T_world_before, T_camera_before)
    asset_scale = _asset_scale(item)
    before_bounds = _world_bounds(visual_path, T_world_before, asset_scale=asset_scale)
    before_bottom = float(before_bounds["min"][2])
    dz = float(support_z - before_bottom)
    if abs(dz) <= tolerance_m:
        dz = 0.0
    T_world_after = T_world_before.copy()
    T_world_after[2, 3] += dz
    T_camera_after = np.linalg.inv(camera_to_world) @ T_world_after
    item["T_object_to_world"] = T_world_after.tolist()
    item["T_object_to_camera"] = T_camera_after.tolist()
    after_bounds = _world_bounds(visual_path, T_world_after, asset_scale=asset_scale)
    after_bottom = float(after_bounds["min"][2])
    alignment_status = "support_aligned" if after_bottom >= support_z - tolerance_m else "penetrating_support"
    support_alignment = _support_alignment_payload(
        visual_path,
        run=run,
        status=alignment_status,
        support_z=support_z,
        before_bottom=before_bottom,
        after_bottom=after_bottom,
        before_bounds=before_bounds,
        after_bounds=after_bounds,
        tolerance_m=tolerance_m,
    )

    pose_path = object_dir / "pose.json"
    if pose_path.is_file():
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        pose["T_object_to_world"] = T_world_after.tolist()
        pose["T_object_to_camera"] = T_camera_after.tolist()
        pose["asset_scale"] = asset_scale
        pose["pose_refinement"] = {
            "source": "manual_refined_from_auto",
            "translation_source": "manual_refined_from_viewer_overlay",
            "operation": "snap_visual_bounds_to_support_plane",
            "translation_delta_world_m": [0.0, 0.0, float(dz)],
            "asset_scale": asset_scale,
            "support_z_m": float(support_z),
            "support_alignment": support_alignment,
        }
        pose_path.write_text(json.dumps(pose, indent=2), encoding="utf-8")

    report = {
        "version": 1,
        "object_id": object_id,
        "label": item.get("label", object_id),
        "status": "accepted" if alignment_status == "support_aligned" else "blocked",
        "source": "manual_refined_from_auto",
        "rotation_source": previous_report.get("rotation_source") or "manual_refined_from_auto",
        "translation_source": "manual_refined_from_viewer_overlay",
        "scale_source": previous_report.get("scale_source") or "rgbd_alignment",
        "asset_scale": asset_scale,
        "manual_review_basis": [
            "visual.glb transformed bounds were inspected against the support plane",
            "translation was adjusted in the shared world pose consumed by viewer/USD/Genesis",
            "visual, collision, and debug proxy asset paths were preserved",
        ],
        "support_alignment": support_alignment,
        "translation_delta_world_m": [0.0, 0.0, float(dz)],
        "T_object_to_camera": T_camera_after.tolist(),
        "T_object_to_world": T_world_after.tolist(),
        "preserved_assets": {
            "mesh_path": item.get("mesh_path"),
            "visual_asset": (item.get("visual_asset") or {}).get("path"),
            "collision_asset": (item.get("collision_asset") or {}).get("path"),
            "debug_proxy": (item.get("debug_proxy") or {}).get("path"),
        },
    }
    _write_object_report(object_dir, report)
    return report


def _support_z(support: dict[str, Any]) -> float:
    return float(support.get("table_top_z_m", support.get("height_world_m", 0.0)) or 0.0)


def _support_alignment_payload(
    visual_path: Path,
    *,
    run: Path,
    status: str,
    support_z: float,
    before_bottom: float,
    after_bottom: float,
    before_bounds: dict[str, list[float]],
    after_bounds: dict[str, list[float]],
    tolerance_m: float,
) -> dict[str, Any]:
    return {
        "status": status,
        "mesh_path": str(visual_path.relative_to(run)),
        "support_z_m": float(support_z),
        "before_bottom_z_m": float(before_bottom),
        "after_bottom_z_m": float(after_bottom),
        "before_world_bounds": before_bounds,
        "after_world_bounds": after_bounds,
        "tolerance_m": float(tolerance_m),
    }


def _support_camera_height(support: dict[str, Any]) -> float:
    return float(support.get("original_height_world_m", support.get("height_world_m", 0.0)) or 0.0)


def _apply_one_orientation_override(
    run: Path,
    item: dict[str, Any],
    *,
    override: str,
    support_z: float,
    tolerance_m: float,
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    object_dir = run / "objects" / object_id
    visual_path = _visual_geometry_path(run, item)
    if visual_path is None:
        report = _blocked_visual_refinement_report(object_id, item, support_z, tolerance_m, "missing_visual_geometry")
        report["rotation_override"] = override
        _write_object_report(object_dir, report)
        return report
    delta = _orientation_override_matrix(override)
    if delta is None:
        report = _blocked_visual_refinement_report(object_id, item, support_z, tolerance_m, "unsupported_rotation_override")
        report["rotation_override"] = override
        _write_object_report(object_dir, report)
        return report

    previous_report = _load_json(object_dir / "pose_refinement_report.json")
    T_world_before = _transform_array(item["T_object_to_world"])
    T_camera_before = _transform_array(item.get("T_object_to_camera", DEFAULT_CAMERA_TO_WORLD))
    camera_to_world = _camera_to_world_bridge(T_world_before, T_camera_before)
    asset_scale = _asset_scale(item)
    before_bounds = _world_bounds(visual_path, T_world_before, asset_scale=asset_scale)
    rotation_before_override = _rotation_before_override(previous_report, override)
    if rotation_before_override is None:
        rotation_before_override = T_world_before[:3, :3].copy()
    T_world_after = T_world_before.copy()
    T_world_after[:3, :3] = rotation_before_override @ delta[:3, :3]
    flipped_bounds = _world_bounds(visual_path, T_world_after, asset_scale=asset_scale)
    dz = float(support_z - float(flipped_bounds["min"][2]))
    if abs(dz) <= tolerance_m:
        dz = 0.0
    T_world_after[2, 3] += dz
    T_camera_after = np.linalg.inv(camera_to_world) @ T_world_after
    after_bounds = _world_bounds(visual_path, T_world_after, asset_scale=asset_scale)
    after_bottom = float(after_bounds["min"][2])
    alignment_status = "support_aligned" if after_bottom >= support_z - tolerance_m else "penetrating_support"
    support_alignment = _support_alignment_payload(
        visual_path,
        run=run,
        status=alignment_status,
        support_z=support_z,
        before_bottom=float(before_bounds["min"][2]),
        after_bottom=after_bottom,
        before_bounds=before_bounds,
        after_bounds=after_bounds,
        tolerance_m=tolerance_m,
    )

    item["T_object_to_world"] = T_world_after.tolist()
    item["T_object_to_camera"] = T_camera_after.tolist()
    item["asset_scale"] = asset_scale
    pose_path = object_dir / "pose.json"
    if pose_path.is_file():
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        pose["T_object_to_world"] = T_world_after.tolist()
        pose["T_object_to_camera"] = T_camera_after.tolist()
        pose["asset_scale"] = asset_scale
        pose["pose_refinement"] = {
            "source": "manual_refined_from_auto",
            "rotation_source": "manual_refined_from_auto",
            "rotation_override": override,
            "rotation_before_override": rotation_before_override.tolist(),
            "rotation_after_override": T_world_after[:3, :3].tolist(),
            "operation": "manual_visual_orientation_override",
            "translation_delta_world_m": [0.0, 0.0, float(dz)],
            "asset_scale": asset_scale,
            "support_z_m": float(support_z),
            "support_alignment": support_alignment,
        }
        pose_path.write_text(json.dumps(pose, indent=2), encoding="utf-8")

    report = {
        "version": 1,
        "object_id": object_id,
        "label": item.get("label", object_id),
        "status": "accepted" if alignment_status == "support_aligned" else "blocked",
        "source": "manual_refined_from_auto",
        "rotation_source": "manual_refined_from_auto",
        "rotation_override": override,
        "rotation_before_override": rotation_before_override.tolist(),
        "rotation_after_override": T_world_after[:3, :3].tolist(),
        "translation_source": previous_report.get("translation_source") or "manual_refined_from_auto",
        "scale_source": previous_report.get("scale_source") or item.get("source_backend", "scene_manifest"),
        "asset_scale": asset_scale,
        "support_alignment": support_alignment,
        "translation_delta_world_m": [0.0, 0.0, float(dz)],
        "T_object_to_camera": T_camera_after.tolist(),
        "T_object_to_world": T_world_after.tolist(),
        "preserved_assets": {
            "mesh_path": item.get("mesh_path"),
            "visual_asset": (item.get("visual_asset") or {}).get("path"),
            "collision_asset": (item.get("collision_asset") or {}).get("path"),
            "debug_proxy": (item.get("debug_proxy") or {}).get("path"),
        },
    }
    _write_object_report(object_dir, report)
    return report


def _rotation_before_override(report: dict[str, Any], override: str) -> np.ndarray | None:
    if report.get("rotation_override") != override:
        return None
    value = report.get("rotation_before_override")
    if value is None:
        return None
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        return None
    return rotation


def _orientation_override_matrix(override: str) -> np.ndarray | None:
    if override == "flip_local_x_180":
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return delta
    if override == "flip_local_y_180":
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        return delta
    if override == "flip_local_z_180":
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        return delta
    return None


def _expand_support_collision_to_object_footprints(
    run: Path,
    manifest: dict[str, Any],
    support: dict[str, Any],
    *,
    margin_m: float = 0.03,
) -> None:
    bounds = support.get("table_bounds_world_xy")
    xs: list[float] = []
    ys: list[float] = []
    if isinstance(bounds, list) and len(bounds) == 2:
        xs.extend([float(bounds[0][0]), float(bounds[1][0])])
        ys.extend([float(bounds[0][1]), float(bounds[1][1])])
    elif support.get("table_collision_pos_world") and support.get("table_collision_size_xyz"):
        pos = [float(v) for v in support["table_collision_pos_world"]]
        size = [float(v) for v in support["table_collision_size_xyz"]]
        xs.extend([pos[0] - size[0] * 0.5, pos[0] + size[0] * 0.5])
        ys.extend([pos[1] - size[1] * 0.5, pos[1] + size[1] * 0.5])
    for item in manifest.get("objects", []):
        path = _collision_geometry_path(run, item) or _visual_geometry_path(run, item)
        if path is None:
            continue
        try:
            footprint = _world_bounds(path, _transform_array(item["T_object_to_world"]), asset_scale=_asset_scale(item))
        except Exception:  # noqa: BLE001 - support expansion is best-effort; manifest validation records geometry failures elsewhere.
            continue
        xs.extend([float(footprint["min"][0]), float(footprint["max"][0])])
        ys.extend([float(footprint["min"][1]), float(footprint["max"][1])])
    if not xs or not ys:
        return
    min_x = min(xs) - float(margin_m)
    max_x = max(xs) + float(margin_m)
    min_y = min(ys) - float(margin_m)
    max_y = max(ys) + float(margin_m)
    thickness = float(support.get("table_thickness_m", 0.04) or 0.04)
    support_z = _support_z(support)
    size = [float(max_x - min_x), float(max_y - min_y), thickness]
    pos = [float((min_x + max_x) * 0.5), float((min_y + max_y) * 0.5), float(support_z - thickness * 0.5)]
    support["table_bounds_world_xy"] = [[float(min_x), float(min_y)], [float(max_x), float(max_y)]]
    support["table_collision_pos_world"] = pos
    support["table_collision_size_xyz"] = size
    support["table_collision_extent_source"] = "visible_support_bounds_plus_object_footprints"
    support["table_collision_footprint_margin_m"] = float(margin_m)
    mesh_rel = support.get("table_collision_mesh_path")
    if mesh_rel:
        mesh_path = run / str(mesh_rel)
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        final_polygon_collision = (
            support.get("table_collision_final") is True
            and support.get("table_collision_geometry_type") in {"polygon_slab", "convex_hull_slab"}
            and support.get("table_collision_source_backend")
            in {"tabletop_mask", "tabletop_mask_polygon_slab", "tabletop_mask_convex_hull"}
        )
        if not final_polygon_collision:
            trimesh.creation.box(extents=size).export(mesh_path)


def _collision_geometry_path(run: Path, item: dict[str, Any]) -> Path | None:
    asset = item.get("collision_asset") if isinstance(item.get("collision_asset"), dict) else {}
    candidate = asset.get("path")
    if not candidate:
        return None
    path = run / str(candidate)
    return path if path.is_file() else None


def _refine_one_object_to_mask(
    run: Path,
    item: dict[str, Any],
    *,
    camera: dict[str, float],
    support_z: float,
    support_camera_height: float,
    tolerance_m: float,
    min_scale_factor: float,
    max_scale_factor: float,
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    object_dir = run / "objects" / object_id
    visual_path = _visual_geometry_path(run, item)
    if visual_path is None:
        report = _blocked_visual_refinement_report(object_id, item, support_z, tolerance_m, "missing_visual_geometry")
        _write_object_report(object_dir, report)
        return report
    mask_bbox = _mask_bbox(run / str(item.get("mask_path", "")))
    if mask_bbox is None:
        report = _blocked_visual_refinement_report(object_id, item, support_z, tolerance_m, "missing_or_empty_mask")
        _write_object_report(object_dir, report)
        return report

    previous_report = _load_json(object_dir / "pose_refinement_report.json")
    T_world_before = _transform_array(item["T_object_to_world"])
    T_camera_before = _transform_array(item.get("T_object_to_camera", DEFAULT_CAMERA_TO_WORLD))
    camera_to_world = _camera_to_world_bridge(T_world_before, T_camera_before)
    previous_scale = _asset_scale(item)
    projected_before = _project_visual_bbox(
        visual_path,
        T_world_before,
        asset_scale=previous_scale,
        camera=camera,
        support_camera_height=support_camera_height,
    )
    scale_factor, scale_status = _bbox_scale_factor(mask_bbox, projected_before, min_scale_factor, max_scale_factor)
    if scale_status != "ready":
        report = _blocked_visual_refinement_report(object_id, item, support_z, tolerance_m, scale_status)
        report["reference_projection"] = {
            "mask_bbox_xyxy": mask_bbox,
            "projected_bbox_before_xyxy": projected_before,
            "scale_status": scale_status,
        }
        _write_object_report(object_dir, report)
        return report

    asset_scale = float(previous_scale * scale_factor)
    item["asset_scale"] = asset_scale
    before_bounds = _world_bounds(visual_path, T_world_before, asset_scale=asset_scale)
    before_bottom = float(before_bounds["min"][2])
    dz = float(support_z - before_bottom)
    if abs(dz) <= tolerance_m:
        dz = 0.0
    T_world_after = T_world_before.copy()
    T_world_after[2, 3] += dz
    T_camera_after = np.linalg.inv(camera_to_world) @ T_world_after
    item["T_object_to_world"] = T_world_after.tolist()
    item["T_object_to_camera"] = T_camera_after.tolist()
    after_bounds = _world_bounds(visual_path, T_world_after, asset_scale=asset_scale)
    after_bottom = float(after_bounds["min"][2])
    projected_after = _project_visual_bbox(
        visual_path,
        T_world_after,
        asset_scale=asset_scale,
        camera=camera,
        support_camera_height=support_camera_height,
    )
    alignment_status = "support_aligned" if after_bottom >= support_z - tolerance_m else "penetrating_support"
    support_alignment = _support_alignment_payload(
        visual_path,
        run=run,
        status=alignment_status,
        support_z=support_z,
        before_bottom=before_bottom,
        after_bottom=after_bottom,
        before_bounds=before_bounds,
        after_bounds=after_bounds,
        tolerance_m=tolerance_m,
    )

    pose_path = object_dir / "pose.json"
    if pose_path.is_file():
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        pose["T_object_to_world"] = T_world_after.tolist()
        pose["T_object_to_camera"] = T_camera_after.tolist()
        pose["asset_scale"] = asset_scale
        pose["pose_refinement"] = {
            "source": "manual_refined_from_auto",
            "translation_source": "manual_refined_from_reference_overlay",
            "scale_source": "reference_camera_bbox_refinement",
            "operation": "reference_camera_bbox_scale_refinement",
            "translation_delta_world_m": [0.0, 0.0, float(dz)],
            "asset_scale": asset_scale,
            "support_z_m": float(support_z),
            "support_alignment": support_alignment,
        }
        pose_path.write_text(json.dumps(pose, indent=2), encoding="utf-8")

    report = {
        "version": 1,
        "object_id": object_id,
        "label": item.get("label", object_id),
        "status": "accepted" if alignment_status == "support_aligned" else "blocked",
        "source": "manual_refined_from_auto",
        "rotation_source": previous_report.get("rotation_source") or "manual_refined_from_auto",
        "translation_source": "manual_refined_from_reference_overlay",
        "scale_source": "reference_camera_bbox_refinement",
        "asset_scale": asset_scale,
        "previous_asset_scale": previous_scale,
        "reference_projection": {
            "camera_source": camera.get("source", "camera.json"),
            "mask_bbox_xyxy": mask_bbox,
            "projected_bbox_before_xyxy": projected_before,
            "projected_bbox_after_xyxy": projected_after,
            "scale_factor": scale_factor,
            "scale_factor_clamp": [float(min_scale_factor), float(max_scale_factor)],
            "support_camera_height_m": float(support_camera_height),
        },
        "support_alignment": support_alignment,
        "translation_delta_world_m": [0.0, 0.0, float(dz)],
        "T_object_to_camera": T_camera_after.tolist(),
        "T_object_to_world": T_world_after.tolist(),
        "preserved_assets": {
            "mesh_path": item.get("mesh_path"),
            "visual_asset": (item.get("visual_asset") or {}).get("path"),
            "collision_asset": (item.get("collision_asset") or {}).get("path"),
            "debug_proxy": (item.get("debug_proxy") or {}).get("path"),
        },
    }
    _write_object_report(object_dir, report)
    return report


def _blocked_visual_refinement_report(
    object_id: str,
    item: dict[str, Any],
    support_z: float,
    tolerance_m: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "version": 1,
        "object_id": object_id,
        "label": item.get("label", object_id),
        "status": "blocked",
        "source": "manual_refined_from_auto",
        "rotation_source": "manual_refined_from_auto",
        "translation_source": "manual_refined_from_reference_overlay",
        "scale_source": "reference_camera_bbox_refinement",
        "blocked_reason": reason,
        "support_alignment": {
            "status": reason,
            "support_z_m": float(support_z),
            "tolerance_m": float(tolerance_m),
        },
    }


def _visual_geometry_path(run: Path, item: dict[str, Any]) -> Path | None:
    asset = item.get("visual_asset") if isinstance(item.get("visual_asset"), dict) else {}
    candidates = [asset.get("path"), item.get("mesh_path")]
    for candidate in candidates:
        if not candidate:
            continue
        path = run / str(candidate)
        if path.is_file():
            return path
    return None


def _world_bounds(mesh_path: Path, transform: np.ndarray, *, asset_scale: float = 1.0) -> dict[str, list[float]]:
    vertices = _mesh_vertices(mesh_path) * float(asset_scale)
    homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    world = (transform @ homogeneous.T).T[:, :3]
    return {
        "min": [float(v) for v in np.min(world, axis=0)],
        "max": [float(v) for v in np.max(world, axis=0)],
    }


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


def _project_visual_bbox(
    mesh_path: Path,
    transform: np.ndarray,
    *,
    asset_scale: float,
    camera: dict[str, float],
    support_camera_height: float,
) -> list[float] | None:
    vertices = _mesh_vertices(mesh_path) * float(asset_scale)
    homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    world = (transform @ homogeneous.T).T[:, :3]
    cam_x = world[:, 0]
    cam_y = float(support_camera_height) - world[:, 2]
    cam_z = world[:, 1]
    valid = cam_z > 1e-6
    if not np.any(valid):
        return None
    u = float(camera["fx"]) * cam_x[valid] / cam_z[valid] + float(camera["cx"])
    v = float(camera["fy"]) * cam_y[valid] / cam_z[valid] + float(camera["cy"])
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        return None
    return [float(np.min(u)), float(np.min(v)), float(np.max(u)), float(np.max(v))]


def _bbox_scale_factor(
    mask_bbox: list[int],
    projected_bbox: list[float] | None,
    min_scale_factor: float,
    max_scale_factor: float,
) -> tuple[float, str]:
    if projected_bbox is None:
        return 1.0, "projection_outside_reference_camera"
    mask_w = float(mask_bbox[2] - mask_bbox[0])
    mask_h = float(mask_bbox[3] - mask_bbox[1])
    proj_w = float(projected_bbox[2] - projected_bbox[0])
    proj_h = float(projected_bbox[3] - projected_bbox[1])
    ratios = []
    if mask_w > 0.0 and proj_w > 1e-6:
        ratios.append(mask_w / proj_w)
    if mask_h > 0.0 and proj_h > 1e-6:
        ratios.append(mask_h / proj_h)
    if not ratios:
        return 1.0, "invalid_reference_bbox"
    raw = float(np.median(np.asarray(ratios, dtype=np.float64)))
    if not np.isfinite(raw) or raw <= 0.0:
        return 1.0, "invalid_reference_bbox"
    return float(np.clip(raw, min_scale_factor, max_scale_factor)), "ready"


def _mask_bbox(path: Path) -> list[int] | None:
    if not path.is_file():
        return None
    mask = np.asarray(Image.open(path).convert("L")) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _load_reference_camera(run: Path) -> dict[str, float]:
    camera_path = run / "camera.json"
    if not camera_path.is_file():
        return {"source": "fallback_image_size", "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    data = json.loads(camera_path.read_text(encoding="utf-8"))
    width = float(data.get("width", 0.0) or 0.0)
    height = float(data.get("height", 0.0) or 0.0)
    return {
        "source": "camera.json",
        "fx": float(data.get("fx", width) or width or 1.0),
        "fy": float(data.get("fy", height) or height or 1.0),
        "cx": float(data.get("cx", width * 0.5) or width * 0.5),
        "cy": float(data.get("cy", height * 0.5) or height * 0.5),
    }


def _asset_scale(item: dict[str, Any]) -> float:
    value = float(item.get("asset_scale", 1.0) or 1.0)
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _camera_to_world_bridge(T_world: np.ndarray, T_camera: np.ndarray) -> np.ndarray:
    try:
        candidate = T_world @ np.linalg.inv(T_camera)
    except np.linalg.LinAlgError:
        return DEFAULT_CAMERA_TO_WORLD.copy()
    if candidate.shape != (4, 4) or not np.all(np.isfinite(candidate)):
        return DEFAULT_CAMERA_TO_WORLD.copy()
    return candidate


def _transform_array(value: object) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("pose transform must be 4x4")
    return transform


def _write_object_report(object_dir: Path, report: dict[str, Any]) -> None:
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "pose_refinement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
