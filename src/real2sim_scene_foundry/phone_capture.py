"""Phone RGB-D capture bundle validation and import utilities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MIN_CONFIDENCE_COVERAGE = 0.6
MIN_TRAJECTORY_BASELINE_M = 0.05
VALID_TRACKING_STATES = {"normal", "tracking", "mapped"}


def validate_phone_capture(capture_dir: str | Path) -> dict[str, Any]:
    capture = Path(capture_dir)
    reasons: list[str] = []
    if not capture.is_dir():
        reasons.extend(["phone_capture_bundle_must_be_directory", "ordinary_video_is_not_rgbd_capture"])
        report = _validation_report(capture, reasons)
        return report

    metadata = _load_json(capture / "metadata.json")
    intrinsics = _load_json(capture / "camera" / "intrinsics.json")
    poses_doc = _load_json(capture / "camera" / "poses.json")
    poses = _pose_entries(poses_doc)
    rgb = _stemmed_files(capture / "rgb", "*.jpg")
    depth = _stemmed_files(capture / "depth", "*.npy")
    confidence = _stemmed_files(capture / "confidence", "*.png")

    if not metadata:
        reasons.append("metadata_missing")
    if metadata and str(metadata.get("depth_unit", "")).lower() not in {"meter", "meters", "m"}:
        reasons.append("depth_unit_not_meter")
    if str(metadata.get("source_type", "")).lower() == "rgb_video":
        reasons.append("ordinary_video_is_not_rgbd_capture")
    if not _intrinsics_complete(intrinsics):
        reasons.append("intrinsics_missing_or_incomplete")
    if not poses:
        reasons.append("poses_missing_or_empty")

    pose_by_stem = {str(item.get("frame_id") or Path(str(item.get("rgb_path", ""))).stem): item for item in poses}
    stems = sorted(set(rgb) | set(depth) | set(confidence) | set(pose_by_stem))
    if not stems:
        reasons.append("no_capture_frames")
    missing = {
        "rgb": sorted(set(stems) - set(rgb)),
        "depth": sorted(set(stems) - set(depth)),
        "confidence": sorted(set(stems) - set(confidence)),
        "pose": sorted(set(stems) - set(pose_by_stem)),
    }
    for key, values in missing.items():
        if values:
            reasons.append(f"{key}_frame_count_mismatch")

    invalid_pose_frames = []
    bad_tracking = []
    camera_centers = []
    for stem in stems:
        pose = pose_by_stem.get(stem)
        if not pose:
            continue
        transform = _transform(pose.get("T_camera_to_world") or pose.get("transform_matrix"))
        if transform is None:
            invalid_pose_frames.append(stem)
            continue
        camera_centers.append(transform[:3, 3])
        state = str(pose.get("tracking_state", "")).lower()
        if state not in VALID_TRACKING_STATES:
            bad_tracking.append(stem)
    if invalid_pose_frames:
        reasons.append("camera_pose_transform_invalid")
    if bad_tracking:
        reasons.append("tracking_state_unusable")

    coverage = _confidence_coverage(confidence.values())
    if coverage is None:
        reasons.append("confidence_unreadable")
    elif coverage < MIN_CONFIDENCE_COVERAGE:
        reasons.append("confidence_coverage_too_low")

    baseline = _trajectory_baseline(camera_centers)
    if baseline < MIN_TRAJECTORY_BASELINE_M:
        reasons.append("trajectory_baseline_too_small")

    report = _validation_report(
        capture,
        _dedupe(reasons),
        frame_count=len(stems),
        rgb_count=len(rgb),
        depth_count=len(depth),
        confidence_count=len(confidence),
        pose_count=len(poses),
        confidence_coverage=coverage,
        trajectory_baseline_m=baseline,
        missing_frames=missing,
        metadata=metadata,
    )
    (capture / "capture_contract.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def import_phone_capture(capture_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    capture = Path(capture_dir)
    run = Path(run_dir)
    contract = validate_phone_capture(capture)
    if contract.get("status") != "passed":
        run.mkdir(parents=True, exist_ok=True)
        (run / "capture_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        return {"status": "blocked", "blocking_reasons": contract.get("blocking_reasons", []), "capture_contract": "capture_contract.json"}

    run.mkdir(parents=True, exist_ok=True)
    for dirname in ("frames", "depth", "confidence"):
        (run / dirname).mkdir(parents=True, exist_ok=True)
    intrinsics = _load_json(capture / "camera" / "intrinsics.json")
    poses_doc = _load_json(capture / "camera" / "poses.json")
    poses = _pose_entries(poses_doc)
    frames = []
    for pose in poses:
        stem = str(pose.get("frame_id") or Path(str(pose.get("rgb_path", ""))).stem)
        rgb_src = capture / str(pose.get("rgb_path") or f"rgb/{stem}.jpg")
        depth_src = capture / str(pose.get("depth_path") or f"depth/{stem}.npy")
        confidence_src = capture / str(pose.get("confidence_path") or f"confidence/{stem}.png")
        rgb_dst = run / "frames" / f"{stem}.jpg"
        depth_dst = run / "depth" / f"{stem}.npy"
        confidence_dst = run / "confidence" / f"{stem}.png"
        shutil.copyfile(rgb_src, rgb_dst)
        shutil.copyfile(depth_src, depth_dst)
        shutil.copyfile(confidence_src, confidence_dst)
        transform = _transform(pose.get("T_camera_to_world") or pose.get("transform_matrix"))
        frames.append(
            {
                "frame_id": stem,
                "rgb_path": str(rgb_dst.relative_to(run)),
                "depth_path": str(depth_dst.relative_to(run)),
                "confidence_path": str(confidence_dst.relative_to(run)),
                "T_camera_to_world": transform.tolist(),
                "tracking_state": pose.get("tracking_state", "normal"),
            }
        )

    first_pose = _transform(frames[0]["T_camera_to_world"])
    camera = {
        "width": int(intrinsics["width"]),
        "height": int(intrinsics["height"]),
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
        "K": [[float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])], [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])], [0.0, 0.0, 1.0]],
        "T_camera_to_world": first_pose.tolist(),
        "T_world_to_camera": np.linalg.inv(first_pose).tolist(),
        "intrinsics_source": "arkit_explicit",
        "extrinsics_source": "arkit_explicit",
        "scale_source": "arkit_sceneDepth_meters",
        "depth_unit": "meter",
        "pose_frame_id": frames[0]["frame_id"],
    }
    trajectory = {
        "version": 1,
        "source": "phone_capture_bundle",
        "pose_world": str(poses_doc.get("coordinate_frame", "arkit_world")).replace("_world", ""),
        "intrinsics_source": "arkit_explicit",
        "extrinsics_source": "arkit_explicit",
        "scale_source": "arkit_sceneDepth_meters",
        "frames": frames,
    }
    (run / "camera.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")
    (run / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    (run / "capture_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return {"status": "imported", "frame_count": len(frames), "camera_path": "camera.json", "trajectory_path": "trajectory.json"}


def export_nerfstudio_from_phone_capture(run_dir: str | Path, *, pose_world: str = "arkit") -> dict[str, Any]:
    run = Path(run_dir)
    if pose_world not in {"arkit", "sim"}:
        raise ValueError("pose_world must be 'arkit' or 'sim'")
    trajectory = _load_json(run / "trajectory.json")
    camera = _load_json(run / "camera.json")
    frames = trajectory.get("frames", []) if isinstance(trajectory.get("frames"), list) else []
    out_dir = run / "video" / "nerfstudio_phone"
    images_dir = out_dir / "images"
    if pose_world == "sim" and not _has_phone_sim_alignment(run, camera, trajectory):
        out_dir.mkdir(parents=True, exist_ok=True)
        transforms_path = out_dir / "transforms.json"
        if transforms_path.is_file():
            transforms_path.unlink()
        report = {
            "version": 1,
            "status": "blocked",
            "pose_world": pose_world,
            "camera_pose_world": pose_world,
            "blocking_reasons": ["phone_sim_world_alignment_missing"],
            "required_alignment_path": "background/phone_sim_alignment.json",
            "transforms_path": None,
            "image_count": 0,
        }
        (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    images_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = run / "clean_background" / "rgb"
    capture_root = _capture_root_from_contract(run)
    used_clean = False
    ns_frames = []
    for frame in frames:
        stem = str(frame["frame_id"])
        clean_src = capture_root / "clean_background" / "rgb" / f"{stem}.jpg" if capture_root else clean_dir / f"{stem}.jpg"
        src = clean_src if clean_src.is_file() else run / str(frame["rgb_path"])
        used_clean = used_clean or clean_src.is_file()
        dst = images_dir / f"{stem}.jpg"
        shutil.copyfile(src, dst)
        ns_frames.append({"file_path": f"images/{stem}.jpg", "transform_matrix": frame["T_camera_to_world"]})
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": camera.get("fx"),
        "fl_y": camera.get("fy"),
        "cx": camera.get("cx"),
        "cy": camera.get("cy"),
        "w": camera.get("width"),
        "h": camera.get("height"),
        "frames": ns_frames,
        "phone_capture": {
            "pose_world": pose_world,
            "camera_pose_world": pose_world,
            "source": "arkit_explicit_rgbd_capture",
            "depth_unit": "meter",
            "intrinsics_source": camera.get("intrinsics_source"),
            "extrinsics_source": camera.get("extrinsics_source"),
            "scale_source": camera.get("scale_source"),
            "identity_3dgs_to_sim_allowed_if_trained_with_pose_world": pose_world == "sim",
        },
    }
    transforms_path = out_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    report = {
        "version": 1,
        "status": "exported",
        "pose_world": pose_world,
        "camera_pose_world": pose_world,
        "used_clean_background": used_clean,
        "transforms_path": str(transforms_path.relative_to(run)),
        "image_count": len(ns_frames),
    }
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _has_phone_sim_alignment(run: Path, camera: dict[str, Any], trajectory: dict[str, Any]) -> bool:
    if camera.get("pose_world") != "sim" or trajectory.get("pose_world") != "sim":
        return False
    alignment = _load_json(run / "background" / "phone_sim_alignment.json")
    transform = _transform(alignment.get("T_arkit_world_to_sim_world"))
    return alignment.get("status") == "passed" and transform is not None


def _validation_report(capture: Path, reasons: list[str], **kwargs: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "passed" if not reasons else "blocked",
        "blocking_reasons": _dedupe(reasons),
        "capture_dir": str(capture),
        "intrinsics_source": "arkit_explicit" if not reasons or "intrinsics_missing_or_incomplete" not in reasons else "missing",
        "extrinsics_source": "arkit_explicit" if not reasons or "poses_missing_or_empty" not in reasons else "missing",
        "scale_source": "arkit_sceneDepth_meters" if "depth_unit_not_meter" not in reasons else "unknown",
        **kwargs,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _pose_entries(poses_doc: dict[str, Any]) -> list[dict[str, Any]]:
    poses = poses_doc.get("poses", [])
    return [item for item in poses if isinstance(item, dict)] if isinstance(poses, list) else []


def _stemmed_files(directory: Path, pattern: str) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in sorted(directory.glob(pattern))}


def _intrinsics_complete(intrinsics: dict[str, Any]) -> bool:
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            return False
        try:
            float(intrinsics[key])
        except (TypeError, ValueError):
            return False
    return True


def _transform(value: Any) -> np.ndarray | None:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    return transform


def _confidence_coverage(paths: Any) -> float | None:
    coverages = []
    for path in paths:
        try:
            arr = np.asarray(Image.open(path))
        except OSError:
            return None
        if arr.size == 0:
            return None
        coverages.append(float(np.count_nonzero(arr > 0) / arr.size))
    if not coverages:
        return None
    return float(np.mean(coverages))


def _trajectory_baseline(centers: list[np.ndarray]) -> float:
    if len(centers) < 2:
        return 0.0
    stacked = np.asarray(centers, dtype=np.float64)
    max_distance = 0.0
    for idx in range(len(stacked)):
        distances = np.linalg.norm(stacked[idx + 1 :] - stacked[idx], axis=1)
        if distances.size:
            max_distance = max(max_distance, float(np.max(distances)))
    return max_distance


def _capture_root_from_contract(run: Path) -> Path | None:
    contract = _load_json(run / "capture_contract.json")
    capture_dir = contract.get("capture_dir")
    if not capture_dir:
        return None
    path = Path(str(capture_dir))
    return path if path.is_dir() else None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
