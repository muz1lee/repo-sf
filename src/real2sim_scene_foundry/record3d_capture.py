"""Record3D EXR+JPG export conversion into the phone capture bundle contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .phone_capture import validate_phone_capture


os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402


def convert_record3d_export(
    record3d_dir: str | Path,
    out_dir: str | Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    depth_channel: int = 2,
) -> dict[str, Any]:
    """Convert Record3D's EXR+JPG sequence into rsf's phone_capture_bundle."""
    source = Path(record3d_dir)
    out = Path(out_dir)
    metadata = _load_metadata(source)
    frame_indices = _select_frame_indices(source, metadata, frame_stride=frame_stride, max_frames=max_frames)
    if not frame_indices:
        raise ValueError(f"{source} has no matching Record3D rgb/depth frames")

    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    confidence_dir = out / "confidence"
    camera_dir = out / "camera"
    for directory in (rgb_dir, depth_dir, confidence_dir, camera_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_width = int(metadata.get("w") or Image.open(source / "rgb" / f"{frame_indices[0]}.jpg").width)
    source_height = int(metadata.get("h") or Image.open(source / "rgb" / f"{frame_indices[0]}.jpg").height)
    depth_width = int(metadata.get("dw") or _read_record3d_depth(source / "depth" / f"{frame_indices[0]}.exr", depth_channel=depth_channel).shape[1])
    depth_height = int(metadata.get("dh") or _read_record3d_depth(source / "depth" / f"{frame_indices[0]}.exr", depth_channel=depth_channel).shape[0])
    first_fx, first_fy, first_cx, first_cy = _scaled_intrinsics(
        _intrinsics_for_frame(metadata, frame_indices[0]),
        source_size=(source_width, source_height),
        target_size=(depth_width, depth_height),
    )
    intrinsics = {
        "width": depth_width,
        "height": depth_height,
        "fx": first_fx,
        "fy": first_fy,
        "cx": first_cx,
        "cy": first_cy,
        "source": "record3d_metadata_scaled_to_depth_resolution",
        "source_width": source_width,
        "source_height": source_height,
        "depth_width": depth_width,
        "depth_height": depth_height,
    }
    (camera_dir / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")

    poses = []
    confidence_coverages = []
    for output_idx, source_idx in enumerate(frame_indices):
        stem = f"frame_{output_idx:06d}"
        rgb_src = source / "rgb" / f"{source_idx}.jpg"
        depth_src = source / "depth" / f"{source_idx}.exr"
        rgb_dst = rgb_dir / f"{stem}.jpg"
        depth_dst = depth_dir / f"{stem}.npy"
        confidence_dst = confidence_dir / f"{stem}.png"

        _write_resized_rgb(rgb_src, rgb_dst, size=(depth_width, depth_height))
        depth = _read_record3d_depth(depth_src, depth_channel=depth_channel)
        if depth.shape != (depth_height, depth_width):
            depth = cv2.resize(depth, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
        depth = np.asarray(depth, dtype=np.float32)
        np.save(depth_dst, depth)
        confidence = np.where(np.isfinite(depth) & (depth > 0.0), 2, 0).astype(np.uint8)
        confidence_coverages.append(float(np.count_nonzero(confidence) / confidence.size))
        Image.fromarray(confidence).save(confidence_dst)

        fx, fy, cx, cy = _scaled_intrinsics(
            _intrinsics_for_frame(metadata, source_idx),
            source_size=(source_width, source_height),
            target_size=(depth_width, depth_height),
        )
        poses.append(
            {
                "frame_id": stem,
                "source_frame_index": int(source_idx),
                "timestamp_s": _timestamp_for_frame(metadata, source_idx),
                "rgb_path": f"rgb/{stem}.jpg",
                "depth_path": f"depth/{stem}.npy",
                "confidence_path": f"confidence/{stem}.png",
                "T_camera_to_world": _record3d_pose_to_matrix(metadata["poses"][source_idx]).tolist(),
                "tracking_state": "normal",
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            }
        )

    poses_doc = {
        "coordinate_frame": "arkit_world",
        "pose_source": "record3d_metadata_poses",
        "pose_format": "quaternion_xyzw_translation_xyz_to_camera_to_world",
        "poses": poses,
    }
    (camera_dir / "poses.json").write_text(json.dumps(poses_doc, indent=2), encoding="utf-8")
    bundle_metadata = {
        "capture_kind": "phone_capture_bundle",
        "source_format": "record3d_exr_jpg_sequence",
        "source_record3d_dir": str(source),
        "device": "Record3D iPhone LiDAR capture",
        "depth_unit": "meter",
        "depth_source": "record3d_exr_channel",
        "depth_channel": int(depth_channel),
        "confidence_source": "depth_validity_derived",
        "confidence_coverage_mean": float(np.mean(confidence_coverages)),
        "original_frame_count": int(len(metadata.get("poses", []))),
        "selected_frame_count": int(len(frame_indices)),
        "frame_stride": int(frame_stride),
        "max_frames": max_frames,
        "source_rgb_resolution": [source_width, source_height],
        "converted_resolution": [depth_width, depth_height],
    }
    (out / "metadata.json").write_text(json.dumps(bundle_metadata, indent=2), encoding="utf-8")
    contract = validate_phone_capture(out)
    report = {
        "version": 1,
        "status": "converted" if contract.get("status") == "passed" else "blocked",
        "record3d_dir": str(source),
        "out_dir": str(out),
        "frame_count": len(frame_indices),
        "frame_indices": [int(idx) for idx in frame_indices],
        "intrinsics_path": "camera/intrinsics.json",
        "poses_path": "camera/poses.json",
        "confidence_source": "depth_validity_derived",
        "capture_contract": contract,
    }
    (out / "record3d_import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_metadata(source: Path) -> dict[str, Any]:
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Record3D metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("poses"), list):
        raise ValueError("Record3D metadata must contain a poses list")
    return metadata


def _select_frame_indices(source: Path, metadata: dict[str, Any], *, frame_stride: int, max_frames: int | None) -> list[int]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    count = len(metadata.get("poses", []))
    indices = []
    for idx in range(0, count, int(frame_stride)):
        if (source / "rgb" / f"{idx}.jpg").is_file() and (source / "depth" / f"{idx}.exr").is_file():
            indices.append(idx)
        if max_frames is not None and len(indices) >= int(max_frames):
            break
    return indices


def _write_resized_rgb(src: Path, dst: Path, *, size: tuple[int, int]) -> None:
    image = Image.open(src).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(dst)


def _read_record3d_depth(path: Path, *, depth_channel: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read Record3D EXR depth: {path}")
    if image.ndim == 2:
        return np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or not (0 <= int(depth_channel) < image.shape[2]):
        raise ValueError(f"invalid depth_channel={depth_channel} for {path} shape={image.shape}")
    return np.asarray(image[..., int(depth_channel)], dtype=np.float32)


def _intrinsics_for_frame(metadata: dict[str, Any], frame_idx: int) -> tuple[float, float, float, float]:
    per_frame = metadata.get("perFrameIntrinsicCoeffs")
    if isinstance(per_frame, list) and frame_idx < len(per_frame) and isinstance(per_frame[frame_idx], list) and len(per_frame[frame_idx]) >= 4:
        fx, fy, cx, cy = per_frame[frame_idx][:4]
        return float(fx), float(fy), float(cx), float(cy)
    k = metadata.get("K")
    if isinstance(k, list) and len(k) >= 9:
        return float(k[0]), float(k[4]), float(k[6]), float(k[7])
    raise ValueError("Record3D metadata has no usable intrinsics")


def _scaled_intrinsics(
    intrinsics: tuple[float, float, float, float],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    fx, fy, cx, cy = intrinsics
    sx = float(target_size[0]) / float(source_size[0])
    sy = float(target_size[1]) / float(source_size[1])
    return float(fx * sx), float(fy * sy), float(cx * sx), float(cy * sy)


def _record3d_pose_to_matrix(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Record3D pose must be [qx,qy,qz,qw,tx,ty,tz], got {value!r}")
    qx, qy, qz, qw, tx, ty, tz = pose.tolist()
    rotation = _quat_xyzw_to_rotation(qx, qy, qz, qw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [float(tx), float(ty), float(tz)]
    return transform


def _quat_xyzw_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        raise ValueError("Record3D pose quaternion has zero norm")
    qx, qy, qz, qw = (quat / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _timestamp_for_frame(metadata: dict[str, Any], frame_idx: int) -> float | None:
    timestamps = metadata.get("frameTimestamps")
    if isinstance(timestamps, list) and frame_idx < len(timestamps):
        try:
            return float(timestamps[frame_idx])
        except (TypeError, ValueError):
            return None
    return None
