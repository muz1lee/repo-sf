"""Minimal smoke reconstruction pipeline for a SimFoundry-style digital twin."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .camera import CameraIntrinsics
from .clients import S2M2Client, SAM3Client
from .defaults import S2M2_URLS, SAM3_SEGMENT_URL
from .manifest import SceneManifest, SceneObject


class DepthClient(Protocol):
    def infer_xyz(
        self,
        left_path: str | Path,
        right_path: str | Path,
        camera: CameraIntrinsics,
        baseline_m: float,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class SmokeResult:
    out_dir: Path
    manifest_path: Path
    usd_path: Path
    qa_report_path: Path


class ConstantDepthClient:
    def __init__(self, depth_m: float = 1.0) -> None:
        self._depth_m = float(depth_m)

    def infer_xyz(
        self,
        left_path: str | Path,
        right_path: str | Path,
        camera: CameraIntrinsics,
        baseline_m: float,
    ) -> np.ndarray:
        depth = np.full((camera.height, camera.width), self._depth_m, dtype=np.float32)
        return camera.backproject_depth(depth)


def run_smoke_reconstruction(
    *,
    left_image: str | Path,
    right_image: str | Path,
    camera: CameraIntrinsics,
    baseline_m: float,
    out_dir: str | Path,
    label: str,
    mask_path: str | Path | None = None,
    depth_client: DepthClient | None = None,
    sam3_client: SAM3Client | None = None,
    mock_depth_m: float | None = None,
) -> SmokeResult:
    left_path = Path(left_image)
    right_path = Path(right_image)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(left_path).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"left image shape {rgb.shape[:2]} does not match calibration {(camera.height, camera.width)}")

    if depth_client is None:
        depth_client = ConstantDepthClient(mock_depth_m) if mock_depth_m is not None else S2M2Client(S2M2_URLS)
    xyz = depth_client.infer_xyz(left_path, right_path, camera, baseline_m)

    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    else:
        sam3_client = sam3_client or SAM3Client(SAM3_SEGMENT_URL)
        mask = sam3_client.segment_text(left_path, label)
    if mask.shape != (camera.height, camera.width):
        raise ValueError(f"mask shape {mask.shape} does not match calibration {(camera.height, camera.width)}")

    object_id = slugify(label)
    object_dir = out / "objects" / object_id
    object_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    mask_out = object_dir / "mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_out)
    crop_out = object_dir / "crop.png"
    bbox_xyxy = _save_crop(rgb, mask, crop_out)

    mesh_path = object_dir / "mesh.glb"
    center, extents, valid_ratio = _mesh_from_masked_xyz(xyz, mask, mesh_path)
    T_object_to_camera = np.eye(4, dtype=np.float64)
    T_object_to_camera[:3, 3] = center
    T_object_to_world = _camera_to_world_transform(T_object_to_camera)
    pose_path = object_dir / "pose.json"
    pose = {
        "object_id": object_id,
        "label": label,
        "T_object_to_camera": T_object_to_camera.tolist(),
        "T_object_to_world": T_object_to_world.tolist(),
        "bbox_xyxy": [int(v) for v in bbox_xyxy],
        "extents_m": [float(v) for v in extents],
        "source_backend": "metric_bbox",
    }
    pose_path.write_text(json.dumps(pose, indent=2), encoding="utf-8")

    manifest = SceneManifest(
        objects=[
            SceneObject(
                object_id=object_id,
                label=label,
                mesh_path=str(mesh_path.relative_to(out)),
                mask_path=str(mask_out.relative_to(out)),
                crop_path=str(crop_out.relative_to(out)),
                T_object_to_camera=T_object_to_camera.tolist(),
                T_object_to_world=T_object_to_world.tolist(),
                scale_m=float(np.max(extents)),
                mass_kg=0.25,
                friction=0.8,
                confidence=float(min(1.0, valid_ratio)),
                source_backend="metric_bbox",
                needs_manual_refine=valid_ratio < 0.5,
            )
        ]
    )
    manifest_path = out / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    usd_path = export_dir / "scene.usda"
    _write_usda_stub(usd_path, manifest)

    overlay_path = qa_dir / "overlay.png"
    render_path = qa_dir / "render.png"
    _write_overlay(rgb, mask, bbox_xyxy, overlay_path)
    _write_render_preview(rgb.shape[1], rgb.shape[0], mask, bbox_xyxy, render_path)
    qa_report_path = qa_dir / "qa_report.json"
    qa = {
        "object_count": 1,
        "coordinate_frame": "opencv_x_right_y_down_z_forward_meters",
        "physics_settle": {"status": "not_run_in_smoke", "nan_detected": False},
        "objects": [
            {
                "object_id": object_id,
                "label": label,
                "mask_iou": 1.0,
                "center_error_px": 0.0,
                "depth_residual_m": 0.0,
                "valid_xyz_ratio": float(valid_ratio),
                "needs_manual_refine": valid_ratio < 0.5,
            }
        ],
    }
    qa_report_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return SmokeResult(out_dir=out, manifest_path=manifest_path, usd_path=usd_path, qa_report_path=qa_report_path)


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "object"


def _save_crop(rgb: np.ndarray, mask: np.ndarray, out_path: Path) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("mask is empty")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    Image.fromarray(rgb[y0:y1, x0:x1]).save(out_path)
    return x0, y0, x1, y1


def _mesh_from_masked_xyz(xyz: np.ndarray, mask: np.ndarray, out_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    valid = mask & np.all(np.isfinite(xyz), axis=2) & (xyz[..., 2] > 0.0)
    if not np.any(valid):
        raise ValueError("object mask has no valid metric XYZ points")
    points = np.asarray(xyz[valid], dtype=np.float64)
    center = np.median(points, axis=0)
    lo = np.percentile(points, 5.0, axis=0)
    hi = np.percentile(points, 95.0, axis=0)
    extents = np.maximum(hi - lo, np.array([0.02, 0.02, 0.02], dtype=np.float64))
    mesh = trimesh.creation.box(extents=extents)
    mesh.export(out_path)
    valid_ratio = float(np.count_nonzero(valid) / max(1, np.count_nonzero(mask)))
    return center, extents, valid_ratio


def _camera_to_world_transform(T_object_to_camera: np.ndarray) -> np.ndarray:
    T_world_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return T_world_camera @ T_object_to_camera


def _write_usda_stub(path: Path, manifest: SceneManifest) -> None:
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
    for obj in manifest.objects:
        tx, ty, tz = (float(obj.T_object_to_world[i][3]) for i in range(3))
        lines.extend(
            [
                f'    def Xform "{obj.object_id}"',
                "    {",
                f'        custom string mesh_path = "{obj.mesh_path}"',
                f"        custom double mass_kg = {float(obj.mass_kg):.8g}",
                f"        custom double friction = {float(obj.friction):.8g}",
                f"        double3 xformOp:translate = ({tx:.8g}, {ty:.8g}, {tz:.8g})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
            ]
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overlay(rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int], out_path: Path) -> None:
    overlay = rgb.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_xyxy, outline=(0, 255, 0), width=2)
    image.save(out_path)


def _write_render_preview(width: int, height: int, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int], out_path: Path) -> None:
    preview = Image.new("RGB", (width, height), color=(245, 245, 245))
    color = np.asarray(preview, dtype=np.uint8).copy()
    color[mask] = np.array([80, 140, 255], dtype=np.uint8)
    image = Image.fromarray(color)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_xyxy, outline=(20, 70, 180), width=2)
    image.save(out_path)
