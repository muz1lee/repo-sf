"""Phone RGB-D support-plane alignment into simulator world coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_PLANE_DISTANCE_THRESHOLD_M = 0.02
DEFAULT_MAX_FRAMES = 8
DEFAULT_MAX_POINTS_PER_FRAME = 5000
MIN_PLANE_INLIERS = 128


def align_phone_sim_world(
    run_dir: str | Path,
    *,
    force: bool = False,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_points_per_frame: int = DEFAULT_MAX_POINTS_PER_FRAME,
    plane_distance_threshold_m: float = DEFAULT_PLANE_DISTANCE_THRESHOLD_M,
) -> dict[str, Any]:
    """Fit a support plane from phone depth and rewrite poses into sim world."""
    run = Path(run_dir)
    background_dir = run / "background"
    background_dir.mkdir(parents=True, exist_ok=True)
    report_path = background_dir / "phone_sim_alignment.json"
    if report_path.is_file() and not force:
        existing = _load_json(report_path)
        if existing.get("status") == "passed":
            return existing

    camera_path = run / "camera.json"
    trajectory_path = run / "trajectory.json"
    camera = _source_camera(run)
    trajectory = _source_trajectory(run)
    reasons = _explicit_phone_reasons(camera, trajectory)
    if reasons:
        report = _blocked_report(run, reasons)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    frames = [item for item in trajectory.get("frames", []) if isinstance(item, dict)]
    selected_frames = _select_depth_frames(run, frames, max_frames=max_frames)
    if not selected_frames:
        report = _blocked_report(run, ["phone_depth_frames_missing"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    k = _camera_matrix(camera)
    if k is None:
        report = _blocked_report(run, ["camera_intrinsics_invalid"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    rng = np.random.default_rng(13)
    point_batches = []
    camera_centers = []
    for frame in selected_frames:
        transform = _transform(frame.get("T_camera_to_world"))
        if transform is None:
            continue
        camera_centers.append(transform[:3, 3])
        points = _points_from_frame(run, frame, k, transform, max_points=max_points_per_frame, rng=rng)
        if points.size:
            point_batches.append(points)
    if not point_batches:
        report = _blocked_report(run, ["insufficient_phone_depth_points"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    points_world = np.concatenate(point_batches, axis=0)
    plane = _fit_plane_ransac(
        points_world,
        camera_centers=np.asarray(camera_centers, dtype=np.float64),
        distance_threshold_m=float(plane_distance_threshold_m),
    )
    if plane is None:
        report = _blocked_report(run, ["support_plane_ransac_failed"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    normal, offset, inliers = plane
    if int(np.count_nonzero(inliers)) < MIN_PLANE_INLIERS:
        report = _blocked_report(run, ["insufficient_support_plane_inliers"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    transform_arkit_to_sim = _alignment_transform(normal, offset)
    transform_sim_to_arkit = np.linalg.inv(transform_arkit_to_sim)
    inlier_points_sim = _transform_points(transform_arkit_to_sim, points_world[inliers])
    polygon_xy = _convex_hull_xy(inlier_points_sim[:, :2])
    if polygon_xy is None:
        report = _blocked_report(run, ["support_plane_polygon_failed"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    first_frame_mask, first_frame_samples = _first_frame_plane_mask(
        run,
        selected_frames[0],
        k,
        transform_arkit_to_sim,
        plane_distance_threshold_m=float(plane_distance_threshold_m),
    )
    Image.fromarray(first_frame_mask.astype(np.uint8) * 255).save(background_dir / "tabletop_mask.png")

    _write_original_pose_backups(run, camera, trajectory)
    sim_camera, sim_trajectory = _rewrite_poses_to_sim(camera, trajectory, transform_arkit_to_sim, report_path=report_path)
    camera_path.write_text(json.dumps(sim_camera, indent=2), encoding="utf-8")
    trajectory_path.write_text(json.dumps(sim_trajectory, indent=2), encoding="utf-8")
    _write_3dgs_anchor(run, trajectory)

    residuals = np.abs(points_world @ normal + offset)
    extent = np.max(polygon_xy, axis=0) - np.min(polygon_xy, axis=0)
    table_polygon = {
        "version": 1,
        "status": "passed",
        "source_backend": "arkit_depth_ransac_plane",
        "geometry_type": "support_plane_polygon",
        "coordinate_frame": "sim_world_z_up_meters",
        "polygon_world_xy": [[float(x), float(y)] for x, y in polygon_xy.tolist()],
        "polygons_world_xy": [[[float(x), float(y)] for x, y in polygon_xy.tolist()]],
        "polygon_count": 1,
        "top_z_m": 0.0,
        "support_height_m": 0.0,
        "polygon_extent_x_m": float(extent[0]),
        "polygon_extent_y_m": float(extent[1]),
        "input_point_count": int(points_world.shape[0]),
        "inlier_count": int(np.count_nonzero(inliers)),
    }
    (background_dir / "table_polygon_world.json").write_text(json.dumps(table_polygon, indent=2), encoding="utf-8")

    report = {
        "version": 1,
        "status": "passed",
        "source_backend": "arkit_depth_ransac_plane",
        "pose_world_before": "arkit",
        "pose_world_after": "sim",
        "T_arkit_world_to_sim_world": transform_arkit_to_sim.tolist(),
        "T_sim_world_to_arkit_world": transform_sim_to_arkit.tolist(),
        "plane_arkit_world": {
            "normal": [float(v) for v in normal.tolist()],
            "offset": float(offset),
            "equation": "normal dot X + offset = 0",
        },
        "metrics": {
            "input_point_count": int(points_world.shape[0]),
            "inlier_count": int(np.count_nonzero(inliers)),
            "inlier_ratio": float(np.count_nonzero(inliers) / max(1, points_world.shape[0])),
            "plane_residual_median_m": float(np.median(residuals[inliers])),
            "plane_residual_p90_m": float(np.percentile(residuals[inliers], 90.0)),
            "plane_distance_threshold_m": float(plane_distance_threshold_m),
            "frame_count_used": int(len(selected_frames)),
        },
        "artifacts": {
            "camera_arkit_backup": "camera.arkit.json",
            "trajectory_arkit_backup": "trajectory.arkit.json",
            "tabletop_mask_path": "background/tabletop_mask.png",
            "table_polygon_world_path": "background/table_polygon_world.json",
            "3dgs_anchor_path": "background/3dgs_camera_pose.json",
        },
        "identity_3dgs_to_sim_allowed_after_retraining_with_sim_poses": True,
        "existing_arkit_trained_3dgs_requires_camera_sim3_registration": True,
        "qa_samples_sim_xyz": [[float(v) for v in row] for row in first_frame_samples[:32].tolist()],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def has_phone_sim_alignment(run_dir: str | Path) -> bool:
    run = Path(run_dir)
    camera = _load_json(run / "camera.json")
    trajectory = _load_json(run / "trajectory.json")
    alignment = _load_json(run / "background" / "phone_sim_alignment.json")
    return (
        camera.get("pose_world") == "sim"
        and trajectory.get("pose_world") == "sim"
        and alignment.get("status") == "passed"
        and _transform(alignment.get("T_arkit_world_to_sim_world")) is not None
    )


def _source_camera(run: Path) -> dict[str, Any]:
    backup = run / "camera.arkit.json"
    if backup.is_file():
        return _load_json(backup)
    return _load_json(run / "camera.json")


def _source_trajectory(run: Path) -> dict[str, Any]:
    backup = run / "trajectory.arkit.json"
    if backup.is_file():
        return _load_json(backup)
    return _load_json(run / "trajectory.json")


def _explicit_phone_reasons(camera: dict[str, Any], trajectory: dict[str, Any]) -> list[str]:
    reasons = []
    if camera.get("intrinsics_source") != "arkit_explicit":
        reasons.append("phone_intrinsics_not_arkit_explicit")
    if camera.get("extrinsics_source") != "arkit_explicit":
        reasons.append("phone_extrinsics_not_arkit_explicit")
    if camera.get("scale_source") != "arkit_sceneDepth_meters":
        reasons.append("phone_scale_not_arkit_sceneDepth_meters")
    if trajectory.get("pose_world") != "arkit":
        reasons.append("phone_trajectory_pose_world_not_arkit")
    if not isinstance(trajectory.get("frames"), list) or not trajectory.get("frames"):
        reasons.append("phone_trajectory_frames_missing")
    return reasons


def _select_depth_frames(run: Path, frames: list[dict[str, Any]], *, max_frames: int) -> list[dict[str, Any]]:
    selected = []
    step = max(1, len(frames) // max(1, int(max_frames)))
    for frame in frames[::step]:
        if len(selected) >= int(max_frames):
            break
        if (run / str(frame.get("depth_path", ""))).is_file() and (run / str(frame.get("confidence_path", ""))).is_file():
            selected.append(frame)
    return selected


def _points_from_frame(
    run: Path,
    frame: dict[str, Any],
    k: np.ndarray,
    t_camera_to_world: np.ndarray,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    depth = np.load(run / str(frame["depth_path"]))
    confidence = np.asarray(Image.open(run / str(frame["confidence_path"])).convert("L"))
    if depth.ndim != 2 or confidence.shape != depth.shape:
        return np.zeros((0, 3), dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0) & (confidence > 0)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if rows.size > int(max_points):
        choice = rng.choice(rows.size, size=int(max_points), replace=False)
        rows = rows[choice]
        cols = cols[choice]
    return _backproject_pixels(depth, rows, cols, k, t_camera_to_world)


def _backproject_pixels(
    depth: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    k: np.ndarray,
    t_camera_to_world: np.ndarray,
) -> np.ndarray:
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - float(k[0, 2])) * z / float(k[0, 0])
    y = (rows.astype(np.float64) - float(k[1, 2])) * z / float(k[1, 1])
    points_camera = np.column_stack([x, y, z, np.ones_like(z)])
    points_world = (t_camera_to_world @ points_camera.T).T[:, :3]
    finite = np.all(np.isfinite(points_world), axis=1)
    return points_world[finite]


def _fit_plane_ransac(
    points: np.ndarray,
    *,
    camera_centers: np.ndarray,
    distance_threshold_m: float,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    if points.shape[0] < MIN_PLANE_INLIERS:
        return None
    rng = np.random.default_rng(17)
    best_inliers = None
    best_count = -1
    iterations = min(512, max(128, points.shape[0] // 32))
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
    offset = -float(normal @ centroid)
    if camera_centers.size:
        signed = camera_centers @ normal + offset
        if float(np.median(signed)) < 0.0:
            normal = -normal
            offset = -offset
    distances = np.abs(points @ normal + offset)
    inliers = distances <= float(distance_threshold_m)
    return normal, offset, inliers


def _alignment_transform(normal: np.ndarray, offset: float) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(reference @ n)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = reference - float(reference @ n) * n
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(n, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    rotation = np.vstack([x_axis, y_axis, n])
    point_on_plane = -float(offset) * n
    translation = -(rotation @ point_on_plane)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _first_frame_plane_mask(
    run: Path,
    frame: dict[str, Any],
    k: np.ndarray,
    t_arkit_to_sim: np.ndarray,
    *,
    plane_distance_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_camera_to_world = _transform(frame.get("T_camera_to_world"))
    if t_camera_to_world is None:
        return np.zeros((1, 1), dtype=bool), np.zeros((0, 3), dtype=np.float64)
    depth = np.load(run / str(frame["depth_path"]))
    confidence = np.asarray(Image.open(run / str(frame["confidence_path"])).convert("L"))
    valid = np.isfinite(depth) & (depth > 0.0) & (confidence > 0)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return np.zeros(depth.shape, dtype=bool), np.zeros((0, 3), dtype=np.float64)
    points_world = _backproject_pixels(depth, rows, cols, k, t_camera_to_world)
    points_sim = _transform_points(t_arkit_to_sim, points_world)
    near = np.abs(points_sim[:, 2]) <= float(plane_distance_threshold_m)
    mask = np.zeros(depth.shape, dtype=bool)
    valid_rows = rows[: points_sim.shape[0]]
    valid_cols = cols[: points_sim.shape[0]]
    mask[valid_rows[near], valid_cols[near]] = True
    return mask, points_sim[near]


def _rewrite_poses_to_sim(
    camera: dict[str, Any],
    trajectory: dict[str, Any],
    t_arkit_to_sim: np.ndarray,
    *,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sim_camera = dict(camera)
    first = _transform(camera.get("T_camera_to_world"))
    if first is not None:
        t_camera_to_sim = t_arkit_to_sim @ first
        sim_camera["T_camera_to_world"] = t_camera_to_sim.tolist()
        sim_camera["T_world_to_camera"] = np.linalg.inv(t_camera_to_sim).tolist()
    sim_camera["pose_world"] = "sim"
    sim_camera["pose_world_source"] = "arkit_depth_ransac_plane"
    sim_camera["phone_sim_alignment_path"] = str(report_path.relative_to(report_path.parents[1]))
    sim_camera["T_arkit_world_to_sim_world"] = t_arkit_to_sim.tolist()

    sim_trajectory = dict(trajectory)
    frames = []
    for frame in trajectory.get("frames", []):
        if not isinstance(frame, dict):
            continue
        out = dict(frame)
        transform = _transform(frame.get("T_camera_to_world"))
        if transform is not None:
            out["T_camera_to_world"] = (t_arkit_to_sim @ transform).tolist()
        frames.append(out)
    sim_trajectory["frames"] = frames
    sim_trajectory["pose_world"] = "sim"
    sim_trajectory["pose_world_source"] = "arkit_depth_ransac_plane"
    sim_trajectory["phone_sim_alignment_path"] = str(report_path.relative_to(report_path.parents[1]))
    sim_trajectory["T_arkit_world_to_sim_world"] = t_arkit_to_sim.tolist()
    return sim_camera, sim_trajectory


def _write_original_pose_backups(run: Path, camera: dict[str, Any], trajectory: dict[str, Any]) -> None:
    camera_backup = run / "camera.arkit.json"
    trajectory_backup = run / "trajectory.arkit.json"
    if not camera_backup.is_file():
        camera_backup.write_text(json.dumps(camera, indent=2), encoding="utf-8")
    if not trajectory_backup.is_file():
        trajectory_backup.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")


def _write_3dgs_anchor(run: Path, arkit_trajectory: dict[str, Any]) -> None:
    frames = [item for item in arkit_trajectory.get("frames", []) if isinstance(item, dict)]
    if not frames:
        return
    anchor = {
        "version": 1,
        "anchor_frame": frames[0].get("frame_id", "frame_000000"),
        "coordinate_convention": "arkit_world_camera_to_world",
        "T_3dgs_camera_to_world": frames[0].get("T_camera_to_world"),
        "source": "phone_arkit_pose_before_sim_alignment",
    }
    (run / "background" / "3dgs_camera_pose.json").write_text(json.dumps(anchor, indent=2), encoding="utf-8")


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return (transform @ homogeneous.T).T[:, :3]


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray | None:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or points_xy.shape[0] < 3:
        return None
    finite = np.all(np.isfinite(points_xy), axis=1)
    points = points_xy[finite]
    if points.shape[0] < 3:
        return None
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    return hull if hull.shape[0] >= 3 else None


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


def _transform(value: Any) -> np.ndarray | None:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    return transform


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _blocked_report(run: Path, reasons: list[str]) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "blocked",
        "source_backend": "arkit_depth_ransac_plane",
        "run_dir": str(run),
        "blocking_reasons": _dedupe(reasons),
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
