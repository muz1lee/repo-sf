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
    if not force and _support_plane_is_complete(run, existing):
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
    table_report = _write_table_collision_mesh(run, objects, support_height_m=height)
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
    manifest["support_plane"] = support_plane
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_usda_stub_from_manifest(run / "exports" / "scene.usda", manifest)
    _merge_qa_support_plane(run, support_plane)
    return support_plane


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


def _write_table_collision_mesh(run: Path, objects: list[dict[str, Any]], *, support_height_m: float) -> dict[str, Any]:
    bounds, source = _table_bounds_from_background_points(run, objects, support_height_m=support_height_m)
    if bounds is None:
        bounds = _table_bounds_from_objects(run, objects)
        source = "object_bounds_rect"
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
    return {
        "table_collision_mesh_path": str(table_path.relative_to(run)),
        "table_collision_source_backend": source,
        "table_bounds_world_xy": [[float(x0), float(y0)], [float(x1), float(y1)]],
        "table_collision_pos_world": center,
        "table_collision_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "table_collision_size_xyz": size,
        "table_top_z_m": 0.0,
        "table_thickness_m": float(thickness),
    }


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
    return _pad_bounds(float(x0), float(y0), float(x1), float(y1), padding=0.08), "background_support_points_rect"


def _table_bounds_from_objects(run: Path, objects: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    vertices = []
    for item in objects:
        mesh_vertices = _mesh_vertices(run / str(item["mesh_path"]))
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
                f"        custom double mass_kg = {float(obj['mass_kg']):.8g}",
                f"        custom double friction = {float(obj['friction']):.8g}",
                f"        double3 xformOp:translate = ({tx:.8g}, {ty:.8g}, {tz:.8g})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
            ]
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
