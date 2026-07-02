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


_EXR_JPG_SEQUENCE = "record3d_exr_jpg_sequence"
_NATIVE_R3D_BUNDLE = "record3d_native_r3d_bundle"


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
    source_format = _detect_record3d_layout(source)
    frame_indices = _select_frame_indices(source, metadata, source_format=source_format, frame_stride=frame_stride, max_frames=max_frames)
    if not frame_indices:
        raise ValueError(f"{source} has no matching Record3D rgb/depth frames")

    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    confidence_dir = out / "confidence"
    camera_dir = out / "camera"
    for directory in (rgb_dir, depth_dir, confidence_dir, camera_dir):
        directory.mkdir(parents=True, exist_ok=True)

    first_rgb_path = _record3d_rgb_path(source, source_format, frame_indices[0])
    source_width = int(metadata.get("w") or Image.open(first_rgb_path).width)
    source_height = int(metadata.get("h") or Image.open(first_rgb_path).height)
    depth_width_value = metadata.get("dw")
    depth_height_value = metadata.get("dh")
    if depth_width_value is None or depth_height_value is None:
        if source_format == _NATIVE_R3D_BUNDLE:
            raise ValueError("Record3D native bundle metadata must contain dw/dh depth resolution")
        first_depth = _read_record3d_depth(_record3d_depth_path(source, source_format, frame_indices[0]), depth_channel=depth_channel, depth_shape=None)
        depth_width = first_depth.shape[1]
        depth_height = first_depth.shape[0]
    else:
        depth_width = int(depth_width_value)
        depth_height = int(depth_height_value)
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
        rgb_src = _record3d_rgb_path(source, source_format, source_idx)
        depth_src = _record3d_depth_path(source, source_format, source_idx)
        rgb_dst = rgb_dir / f"{stem}.jpg"
        depth_dst = depth_dir / f"{stem}.npy"
        confidence_dst = confidence_dir / f"{stem}.png"

        _write_resized_rgb(rgb_src, rgb_dst, size=(depth_width, depth_height))
        depth = _read_record3d_depth(depth_src, depth_channel=depth_channel, depth_shape=(depth_height, depth_width))
        if depth.shape != (depth_height, depth_width):
            depth = cv2.resize(depth, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
        depth = np.asarray(depth, dtype=np.float32)
        np.save(depth_dst, depth)
        confidence = _read_record3d_confidence(_record3d_confidence_path(source, source_format, source_idx), depth=depth)
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
        "depth_camera_to_pose_camera_bridge": "opencv_to_arkit_camera",
        "depth_camera_convention": "opencv_x_right_y_down_z_forward",
        "pose_camera_convention": "arkit_x_right_y_up_z_backward",
        "poses": poses,
    }
    (camera_dir / "poses.json").write_text(json.dumps(poses_doc, indent=2), encoding="utf-8")
    bundle_metadata = {
        "capture_kind": "phone_capture_bundle",
        "source_format": source_format,
        "source_record3d_dir": str(source),
        "device": "Record3D iPhone LiDAR capture",
        "depth_unit": "meter",
        "depth_source": "record3d_exr_channel" if source_format == _EXR_JPG_SEQUENCE else "record3d_lzfse_depth",
        "depth_channel": int(depth_channel),
        "depth_camera_to_pose_camera_bridge": "opencv_to_arkit_camera",
        "depth_camera_convention": "opencv_x_right_y_down_z_forward",
        "pose_camera_convention": "arkit_x_right_y_up_z_backward",
        "confidence_source": "depth_validity_derived" if source_format == _EXR_JPG_SEQUENCE else "record3d_confidence",
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
        "confidence_source": "depth_validity_derived" if source_format == _EXR_JPG_SEQUENCE else "record3d_confidence",
        "capture_contract": contract,
    }
    (out / "record3d_import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_metadata(source: Path) -> dict[str, Any]:
    candidates = [source / "metadata.json", source / "metadata"]
    metadata_path = next((path for path in candidates if path.is_file()), None)
    if metadata_path is None:
        expected = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Record3D metadata missing; expected one of: {expected}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("poses"), list):
        raise ValueError("Record3D metadata must contain a poses list")
    return metadata


def _detect_record3d_layout(source: Path) -> str:
    if (source / "rgb").is_dir() and (source / "depth").is_dir():
        return _EXR_JPG_SEQUENCE
    if (source / "rgbd").is_dir():
        return _NATIVE_R3D_BUNDLE
    raise ValueError(f"unsupported Record3D layout: {source}")


def _record3d_rgb_path(source: Path, source_format: str, frame_idx: int) -> Path:
    if source_format == _EXR_JPG_SEQUENCE:
        return source / "rgb" / f"{frame_idx}.jpg"
    return source / "rgbd" / f"{frame_idx}.jpg"


def _record3d_depth_path(source: Path, source_format: str, frame_idx: int) -> Path:
    if source_format == _EXR_JPG_SEQUENCE:
        return source / "depth" / f"{frame_idx}.exr"
    return source / "rgbd" / f"{frame_idx}.depth"


def _record3d_confidence_path(source: Path, source_format: str, frame_idx: int) -> Path | None:
    if source_format == _NATIVE_R3D_BUNDLE:
        return source / "rgbd" / f"{frame_idx}.conf"
    return None


def _select_frame_indices(source: Path, metadata: dict[str, Any], *, source_format: str, frame_stride: int, max_frames: int | None) -> list[int]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    count = len(metadata.get("poses", []))
    indices = []
    for idx in range(0, count, int(frame_stride)):
        if _record3d_rgb_path(source, source_format, idx).is_file() and _record3d_depth_path(source, source_format, idx).is_file():
            indices.append(idx)
        if max_frames is not None and len(indices) >= int(max_frames):
            break
    return indices


def _write_resized_rgb(src: Path, dst: Path, *, size: tuple[int, int]) -> None:
    image = Image.open(src).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(dst)


def _read_record3d_depth(path: Path, *, depth_channel: int, depth_shape: tuple[int, int] | None = None) -> np.ndarray:
    if path.suffix == ".exr":
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV could not read Record3D EXR depth: {path}")
        if image.ndim == 2:
            return np.asarray(image, dtype=np.float32)
        if image.ndim != 3 or not (0 <= int(depth_channel) < image.shape[2]):
            raise ValueError(f"invalid depth_channel={depth_channel} for {path} shape={image.shape}")
        return np.asarray(image[..., int(depth_channel)], dtype=np.float32)
    if path.suffix != ".depth" or depth_shape is None:
        raise ValueError(f"unsupported Record3D depth file: {path}")
    payload = _decompress_record3d_payload(path.read_bytes(), path=path, expected_raw_bytes=int(np.prod(depth_shape)) * 4)
    depth = np.frombuffer(payload, dtype=np.float32)
    if depth.size != int(np.prod(depth_shape)):
        raise ValueError(f"Record3D depth payload has {depth.size} float32 values, expected {int(np.prod(depth_shape))}: {path}")
    return depth.reshape(depth_shape)


def _read_record3d_confidence(path: Path | None, *, depth: np.ndarray) -> np.ndarray:
    if path is not None and path.is_file():
        payload = _decompress_record3d_payload(path.read_bytes(), path=path, expected_raw_bytes=int(depth.size))
        confidence = np.frombuffer(payload, dtype=np.uint8)
        if confidence.size != int(depth.size):
            raise ValueError(f"Record3D confidence payload has {confidence.size} values, expected {int(depth.size)}: {path}")
        return confidence.reshape(depth.shape).astype(np.uint8, copy=False)
    return np.where(np.isfinite(depth) & (depth > 0.0), 2, 0).astype(np.uint8)


def _decompress_record3d_payload(payload: bytes, *, path: Path, expected_raw_bytes: int) -> bytes:
    if len(payload) == expected_raw_bytes:
        return payload
    try:
        import liblzfse
    except ImportError as exc:
        raise RuntimeError("Record3D native .depth/.conf files require the pyliblzfse dependency") from exc
    decoded = liblzfse.decompress(payload)
    if len(decoded) != expected_raw_bytes:
        raise ValueError(f"decoded Record3D payload has {len(decoded)} bytes, expected {expected_raw_bytes}: {path}")
    return decoded


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
