"""Projection QA for tabletop-derived collision surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MIN_TABLETOP_SELECTED_CANDIDATE_RATIO = 0.08
MIN_TABLE_COLLISION_EXTENT_M = 0.12
MIN_TABLE_COLLISION_PROJECTION_IOU = 0.5
MAX_ACCEPTED_FOREGROUND_TO_TABLETOP_AREA_RATIO = 1.0
EXPORT_GRADE_TABLETOP_IOU = 0.65
EXPORT_GRADE_VISIBLE_IOU = 0.70
MAX_EXPORT_GRADE_OVERREACH_RATIO = 0.15
MAX_EXPORT_GRADE_UNDERCOVERAGE_RATIO = 0.20
MAX_EXPORT_GRADE_FOREGROUND_TO_TABLETOP_AREA_RATIO = 0.20


@dataclass(frozen=True)
class TableCollisionQAResult:
    report_path: Path
    overlay_path: Path | None
    report: dict[str, Any]


def write_table_collision_projection_qa(
    run_dir: str | Path, *, iou_threshold: float = MIN_TABLE_COLLISION_PROJECTION_IOU
) -> TableCollisionQAResult:
    run = Path(run_dir)
    qa_dir = run / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    report_path = qa_dir / "table_collision_report.json"
    overlay_path = qa_dir / "table_collision_overlay.png"

    polygon_path = run / "background" / "table_polygon_world.json"
    mask_path = run / "background" / "tabletop_mask.png"
    support_report_path = run / "background" / "tabletop_support_report.json"
    camera_path = run / "camera.json"
    polygon = _load_json(polygon_path)
    support_report = _load_json(support_report_path)
    camera = _load_json(camera_path)
    reasons: list[str] = []

    if not polygon:
        reasons.append("table_polygon_world_missing")
    if not mask_path.is_file():
        reasons.append("tabletop_mask_missing")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
    if mask is None and mask_path.is_file():
        reasons.append("tabletop_mask_unreadable")
    tabletop_mask_area_px = int(np.count_nonzero(mask)) if mask is not None else 0
    foreground_mask = _read_grayscale_mask(run / "background" / "foreground_mask.png", mask.shape if mask is not None else None)
    k = _camera_matrix(camera)
    support_height_m = float(polygon.get("support_height_m", 0.0) or 0.0)
    transform = _world_to_camera(camera, support_height_m=support_height_m)
    if k is None:
        reasons.append("camera_intrinsics_missing")
    if transform is None:
        reasons.append("camera_extrinsics_missing")
    polygon_vertex_sets = _polygon_vertex_sets(polygon)
    world_vertices = np.vstack(polygon_vertex_sets) if polygon_vertex_sets else None
    if not polygon_vertex_sets:
        reasons.append("table_polygon_vertices_missing")

    geometry_type = str(polygon.get("geometry_type") or "unknown")
    source_backend = str(polygon.get("source_backend") or "unknown")
    extent_x_m, extent_y_m = _polygon_extents(world_vertices)
    extent_x_m = _metric_value(polygon, support_report, "polygon_extent_x_m", extent_x_m)
    extent_y_m = _metric_value(polygon, support_report, "polygon_extent_y_m", extent_y_m)
    candidate_count = int(_metric_value(polygon, support_report, "candidate_count", 0.0) or 0)
    selected_component_area = int(_metric_value(polygon, support_report, "selected_component_area", float(tabletop_mask_area_px)) or 0)
    selected_candidate_ratio = _metric_value(
        polygon,
        support_report,
        "selected_candidate_ratio",
        float(selected_component_area / candidate_count) if candidate_count else None,
    )
    projection_vertex_sets = polygon_vertex_sets
    projection_surface = "top_surface"
    thickness_m = float(polygon.get("thickness_m", 0.0) or 0.0)
    if polygon_vertex_sets and geometry_type in {"polygon_slab", "convex_hull_slab"} and thickness_m > 0.0:
        projection_vertex_sets = []
        for vertices_world in polygon_vertex_sets:
            bottom_vertices = np.array(vertices_world, copy=True)
            bottom_vertices[:, 2] -= thickness_m
            projection_vertex_sets.append(np.vstack([vertices_world, bottom_vertices]))
        projection_surface = "slab_silhouette"

    projected_mask = None
    projection_iou = None
    projection_iou_with_tabletop_mask = None
    projected_polygon_area_px = None
    accepted_foreground_area_px = None
    foreground_to_tabletop_area_ratio = None
    exceeds_reference_bounds = None
    if not reasons and mask is not None and k is not None and transform is not None and projection_vertex_sets:
        projected_polygons = []
        bounds_points = []
        projection_invalid = False
        for vertices_world in projection_vertex_sets:
            points_px = _project_vertices(vertices_world, k, transform)
            if points_px is None:
                projection_invalid = True
                break
            bounds_points.append(points_px)
            if projection_surface == "slab_silhouette":
                points_px = cv2.convexHull(points_px.astype(np.float32)).reshape(-1, 2)
            projected_polygons.append(np.rint(points_px).astype(np.int32))
        if projection_invalid:
            reasons.append("table_polygon_projection_invalid")
        else:
            all_points_px = np.vstack(bounds_points)
            exceeds_reference_bounds = bool(
                np.any(all_points_px[:, 0] < 0.0)
                or np.any(all_points_px[:, 0] >= mask.shape[1])
                or np.any(all_points_px[:, 1] < 0.0)
                or np.any(all_points_px[:, 1] >= mask.shape[0])
            )
            projected_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(projected_mask, projected_polygons, 255)
            projected_polygon_area_px = int(np.count_nonzero(projected_mask))
            target_mask = mask
            accepted_foreground_area_px = 0
            if foreground_mask is not None:
                accepted_foreground = (foreground_mask > 0) & (projected_mask > 0)
                accepted_foreground_area_px = int(np.count_nonzero(accepted_foreground))
                target_mask = np.maximum(mask, accepted_foreground.astype(np.uint8) * 255)
            projection_iou = _mask_iou(projected_mask, target_mask)
            projection_iou_with_tabletop_mask = _mask_iou(projected_mask, mask)
            if foreground_mask is not None and accepted_foreground_area_px > 0:
                if tabletop_mask_area_px <= 0:
                    reasons.append("blocked_tabletop_occlusion")
                else:
                    foreground_to_tabletop_area_ratio = float(accepted_foreground_area_px / tabletop_mask_area_px)
                    if (
                        foreground_to_tabletop_area_ratio > MAX_ACCEPTED_FOREGROUND_TO_TABLETOP_AREA_RATIO
                        and projection_iou_with_tabletop_mask < iou_threshold
                    ):
                        reasons.append("blocked_tabletop_occlusion")
            if projection_iou < iou_threshold:
                reasons.append("low_tabletop_projection_iou")

    derived_sources = {"tabletop_mask", "tabletop_mask_polygon_slab", "tabletop_mask_convex_hull"}
    derived = source_backend in derived_sources and geometry_type in {"polygon_slab", "convex_hull_slab"}
    if not derived:
        reasons.append("table_collision_not_tabletop_mask_derived")
    if selected_candidate_ratio is not None and selected_candidate_ratio < MIN_TABLETOP_SELECTED_CANDIDATE_RATIO:
        reasons.append("table_collision_support_coverage_too_small")
    if extent_x_m is not None and extent_y_m is not None and min(extent_x_m, extent_y_m) < MIN_TABLE_COLLISION_EXTENT_M:
        reasons.append("table_collision_extent_too_small")

    if projected_mask is not None and mask is not None:
        _write_overlay(overlay_path, mask, projected_mask)
        overlay_rel: str | None = "qa/table_collision_overlay.png"
    else:
        overlay_rel = None

    blocking_reasons = _dedupe(reasons)
    report = {
        "version": 1,
        "status": "passed" if not blocking_reasons else "blocked",
        "blocking_reasons": blocking_reasons,
        "source_backend": source_backend,
        "geometry_type": geometry_type,
        "derived_from_tabletop_mask": derived,
        "projection_iou": projection_iou,
        "projection_iou_with_tabletop_mask": projection_iou_with_tabletop_mask,
        "projection_surface": projection_surface,
        "polygon_count": len(polygon_vertex_sets),
        "projected_polygon_area_px": projected_polygon_area_px,
        "tabletop_mask_area_px": tabletop_mask_area_px,
        "accepted_foreground_projection_area_px": accepted_foreground_area_px,
        "foreground_to_tabletop_area_ratio": foreground_to_tabletop_area_ratio,
        "max_foreground_to_tabletop_area_ratio": MAX_ACCEPTED_FOREGROUND_TO_TABLETOP_AREA_RATIO,
        "exceeds_reference_bounds": exceeds_reference_bounds,
        "candidate_count": candidate_count,
        "selected_component_area": selected_component_area,
        "selected_candidate_ratio": selected_candidate_ratio,
        "min_selected_candidate_ratio": MIN_TABLETOP_SELECTED_CANDIDATE_RATIO,
        "polygon_extent_x_m": extent_x_m,
        "polygon_extent_y_m": extent_y_m,
        "min_table_collision_extent_m": MIN_TABLE_COLLISION_EXTENT_M,
        "iou_threshold": iou_threshold,
        "polygon_path": "background/table_polygon_world.json" if polygon_path.is_file() else None,
        "tabletop_mask_path": "background/tabletop_mask.png" if mask_path.is_file() else None,
        "tabletop_support_report_path": "background/tabletop_support_report.json" if support_report_path.is_file() else None,
        "camera_path": "camera.json" if camera_path.is_file() else None,
        "visual_qa_path": overlay_rel,
        "overlay_path": overlay_rel,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TableCollisionQAResult(
        report_path=report_path,
        overlay_path=overlay_path if overlay_rel else None,
        report=report,
    )


def write_table_collision_projection_qa_v2(run_dir: str | Path) -> TableCollisionQAResult:
    """Write stricter export-grade tabletop projection QA without changing legacy QA."""
    run = Path(run_dir)
    legacy = write_table_collision_projection_qa(run)
    report = dict(legacy.report)
    camera = _load_json(run / "camera.json")
    camera_intrinsics_source = "explicit" if _camera_matrix(camera) is not None else "missing"
    camera_extrinsics_source = _camera_extrinsics_source(camera)
    projection_iou = _float_or_none(report.get("projection_iou"))
    tabletop_iou = _float_or_none(report.get("projection_iou_with_tabletop_mask"))
    projected_area = _float_or_none(report.get("projected_polygon_area_px"))
    tabletop_area = _float_or_none(report.get("tabletop_mask_area_px"))
    intersection_area = _intersection_from_iou(tabletop_iou, projected_area, tabletop_area)
    overreach_ratio = _safe_ratio(None if intersection_area is None or projected_area is None else projected_area - intersection_area, projected_area)
    undercoverage_ratio = _safe_ratio(None if intersection_area is None or tabletop_area is None else tabletop_area - intersection_area, tabletop_area)

    reasons = list(report.get("blocking_reasons", []))
    if camera_intrinsics_source != "explicit":
        reasons.append("camera_intrinsics_not_explicit")
    if camera_extrinsics_source != "explicit":
        reasons.append("camera_extrinsics_not_explicit")
    if tabletop_iou is None or tabletop_iou < EXPORT_GRADE_TABLETOP_IOU:
        reasons.append("below_export_grade_tabletop_iou")
    if projection_iou is None or projection_iou < EXPORT_GRADE_VISIBLE_IOU:
        reasons.append("below_export_grade_visible_iou")
    if overreach_ratio is not None and overreach_ratio > MAX_EXPORT_GRADE_OVERREACH_RATIO:
        reasons.append("table_collision_overreaches_visible_table")
    if undercoverage_ratio is not None and undercoverage_ratio > MAX_EXPORT_GRADE_UNDERCOVERAGE_RATIO:
        reasons.append("table_collision_under_covers_visible_table")
    foreground_ratio = _float_or_none(report.get("foreground_to_tabletop_area_ratio"))
    if foreground_ratio is not None and foreground_ratio > MAX_EXPORT_GRADE_FOREGROUND_TO_TABLETOP_AREA_RATIO:
        reasons.append("blocked_tabletop_occlusion")
    if report.get("exceeds_reference_bounds") is True:
        reasons.append("table_collision_exceeds_reference_bounds")

    blocking_reasons = _dedupe(reasons)
    weak_status = "diagnostic_pass" if report.get("status") == "passed" and blocking_reasons else report.get("status", "unknown")
    strict_report = {
        **report,
        "version": 2,
        "status": "passed" if not blocking_reasons else "blocked",
        "weak_status": weak_status,
        "blocking_reasons": blocking_reasons,
        "camera_intrinsics_source": camera_intrinsics_source,
        "camera_extrinsics_source": camera_extrinsics_source,
        "visible_projection_iou": projection_iou,
        "export_grade_tabletop_iou_threshold": EXPORT_GRADE_TABLETOP_IOU,
        "export_grade_visible_iou_threshold": EXPORT_GRADE_VISIBLE_IOU,
        "overreach_ratio": overreach_ratio,
        "max_overreach_ratio": MAX_EXPORT_GRADE_OVERREACH_RATIO,
        "undercoverage_ratio": undercoverage_ratio,
        "max_undercoverage_ratio": MAX_EXPORT_GRADE_UNDERCOVERAGE_RATIO,
        "max_foreground_to_tabletop_area_ratio": MAX_EXPORT_GRADE_FOREGROUND_TO_TABLETOP_AREA_RATIO,
        "legacy_report_path": "qa/table_collision_report.json",
    }
    report_path = run / "qa" / "table_collision_report_v2.json"
    report_path.write_text(json.dumps(strict_report, indent=2), encoding="utf-8")
    return TableCollisionQAResult(report_path=report_path, overlay_path=legacy.overlay_path, report=strict_report)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_grayscale_mask(path: Path, shape: tuple[int, int] | None) -> np.ndarray | None:
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if shape is not None and mask.shape != shape:
        return None
    return mask


def _camera_matrix(camera: dict[str, Any]) -> np.ndarray | None:
    value = camera.get("K") or camera.get("intrinsics")
    if value is None and all(key in camera for key in ("fx", "fy", "cx", "cy")):
        try:
            return np.array(
                [[float(camera["fx"]), 0.0, float(camera["cx"])], [0.0, float(camera["fy"]), float(camera["cy"])], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        try:
            fx = float(value["fx"])
            fy = float(value["fy"])
            cx = float(value["cx"])
            cy = float(value["cy"])
        except (KeyError, TypeError, ValueError):
            return None
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    try:
        k = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if k.shape != (3, 3) or not np.all(np.isfinite(k)):
        return None
    return k


def _world_to_camera(camera: dict[str, Any], *, support_height_m: float = 0.0) -> np.ndarray | None:
    if "T_world_to_camera" in camera:
        return _transform(camera.get("T_world_to_camera"))
    if "T_camera_to_world" in camera:
        t_camera_to_world = _transform(camera.get("T_camera_to_world"))
        if t_camera_to_world is None:
            return None
        try:
            return np.linalg.inv(t_camera_to_world)
        except np.linalg.LinAlgError:
            return None
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -float(support_height_m)], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_extrinsics_source(camera: dict[str, Any]) -> str:
    if "T_world_to_camera" in camera or "T_camera_to_world" in camera:
        return "explicit" if _world_to_camera(camera) is not None else "invalid"
    return "fallback"


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(max(0.0, numerator) / denominator)


def _intersection_from_iou(iou: float | None, area_a: float | None, area_b: float | None) -> float | None:
    if iou is None or area_a is None or area_b is None or iou < 0.0:
        return None
    return float(iou * (area_a + area_b) / (1.0 + iou))


def _transform(value: Any) -> np.ndarray | None:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    return transform


def _vertices(value: Any) -> np.ndarray | None:
    try:
        vertices = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.all(np.isfinite(vertices)):
        return None
    return vertices


def _vertices_from_world_xy(value: Any, *, z: float) -> np.ndarray | None:
    try:
        xy = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.all(np.isfinite(xy)):
        return None
    return np.column_stack([xy, np.full(len(xy), float(z), dtype=np.float64)])


def _polygon_vertex_sets(polygon: dict[str, Any]) -> list[np.ndarray]:
    z = float(polygon.get("top_z_m", 0.0) or 0.0)
    polygons_world = _vertices_3d_set(polygon.get("polygons_world"))
    if polygons_world:
        return polygons_world
    polygons_world_xy = _vertices_xy_set(polygon.get("polygons_world_xy"), z=z)
    if polygons_world_xy:
        return polygons_world_xy
    vertices = polygon.get("vertices_world") or polygon.get("polygon_world") or polygon.get("convex_hull_world")
    world_vertices = _vertices(vertices)
    if world_vertices is not None:
        return [world_vertices]
    world_vertices = _vertices_from_world_xy(polygon.get("polygon_world_xy"), z=z)
    if world_vertices is not None:
        return [world_vertices]
    return []


def _vertices_3d_set(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    out: list[np.ndarray] = []
    for item in value:
        vertices = _vertices(item)
        if vertices is not None:
            out.append(vertices)
    return out


def _vertices_xy_set(value: Any, *, z: float) -> list[np.ndarray]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    out: list[np.ndarray] = []
    for item in value:
        vertices = _vertices_from_world_xy(item, z=z)
        if vertices is not None:
            out.append(vertices)
    return out


def _polygon_extents(vertices: np.ndarray | None) -> tuple[float | None, float | None]:
    if vertices is None:
        return None, None
    return (
        float(np.max(vertices[:, 0]) - np.min(vertices[:, 0])),
        float(np.max(vertices[:, 1]) - np.min(vertices[:, 1])),
    )


def _metric_value(primary: dict[str, Any], secondary: dict[str, Any], key: str, default: float | None) -> float | None:
    value = primary.get(key, secondary.get(key, default))
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(metric):
        return default
    return metric


def _project_vertices(vertices_world: np.ndarray, k: np.ndarray, t_world_to_camera: np.ndarray) -> np.ndarray | None:
    homogeneous = np.concatenate([vertices_world, np.ones((len(vertices_world), 1), dtype=np.float64)], axis=1)
    vertices_camera = (t_world_to_camera @ homogeneous.T).T[:, :3]
    if np.any(vertices_camera[:, 2] <= 1e-6):
        return None
    pixels_h = (k @ vertices_camera.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def _mask_iou(projected_mask: np.ndarray, tabletop_mask: np.ndarray) -> float:
    projected = projected_mask > 0
    target = tabletop_mask > 0
    union = np.logical_or(projected, target).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(projected, target).sum() / union)


def _write_overlay(path: Path, tabletop_mask: np.ndarray, projected_mask: np.ndarray) -> None:
    overlay = np.zeros((*tabletop_mask.shape, 3), dtype=np.uint8)
    overlay[..., 1] = tabletop_mask
    overlay[..., 2] = projected_mask
    both = (tabletop_mask > 0) & (projected_mask > 0)
    overlay[both] = [255, 255, 255]
    cv2.imwrite(str(path), overlay)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
