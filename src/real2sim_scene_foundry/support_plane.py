"""Support-plane estimation and world-frame normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image


def estimate_and_apply_support_plane(run_dir: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Estimate the support height, shift world poses so that support plane is z=0."""
    run = Path(run_dir)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = manifest.get("support_plane")
    if not force and isinstance(existing, dict) and existing.get("applied_to_world_frame") is True:
        _merge_qa_support_plane(run, existing)
        return existing

    objects = manifest.get("objects", [])
    if not objects:
        raise ValueError("scene_manifest.json has no objects for support-plane estimation")
    before_bottoms = _object_bottoms(run, objects)
    height, source = _estimate_height_from_point_rings(run, objects)
    if height is None:
        height = float(np.median(list(before_bottoms.values())))
        source = "object_mesh_bottom_median"

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
    }
    manifest["support_plane"] = support_plane
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_usda_stub_from_manifest(run / "exports" / "scene.usda", manifest)
    _merge_qa_support_plane(run, support_plane)
    return support_plane


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


def _object_bottoms(run: Path, objects: list[dict[str, Any]]) -> dict[str, float]:
    bottoms: dict[str, float] = {}
    for item in objects:
        mesh_path = run / str(item["mesh_path"])
        vertices = _mesh_vertices(mesh_path)
        transform = _transform_array(item["T_object_to_world"])
        homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
        world_vertices = (transform @ homogeneous.T).T[:, :3]
        bottoms[str(item["object_id"])] = float(np.min(world_vertices[:, 2]))
    return bottoms


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
    for obj in manifest.get("objects", []):
        transform = _transform_array(obj["T_object_to_world"])
        tx, ty, tz = (float(transform[i, 3]) for i in range(3))
        lines.extend(
            [
                f'    def Xform "{obj["object_id"]}"',
                "    {",
                f'        custom string mesh_path = "{obj["mesh_path"]}"',
                f"        custom double mass_kg = {float(obj['mass_kg']):.8g}",
                f"        custom double friction = {float(obj['friction']):.8g}",
                f"        double3 xformOp:translate = ({tx:.8g}, {ty:.8g}, {tz:.8g})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
            ]
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
