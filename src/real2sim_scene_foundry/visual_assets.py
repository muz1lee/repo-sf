"""Visual asset role and provenance helpers."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

SAM3D_MESH_BASE64_KEYS = ("mesh_glb_base64", "glb_base64", "mesh")


@dataclass(frozen=True)
class VisualAssetPlan:
    object_id: str
    label: str
    status: str
    source_backend: str
    alignment_source_path: Path
    aligned_output_path: Path
    final_visual: bool
    visual_path: Path | None
    visual_point_cloud_path: Path | None
    debug_bbox_path: Path | None
    blocked_reason: str | None
    service_response: dict[str, Any]


def extract_mesh_base64(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    for key in SAM3D_MESH_BASE64_KEYS:
        value = metadata.get(key)
        if not value:
            continue
        if isinstance(value, str):
            return key, value
        if isinstance(value, dict):
            for nested_key in ("base64", "data", "content", "uri", "mesh_glb_base64", "glb_base64"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value:
                    return key, nested_value
    return None, None


def export_debug_bbox_from_ply(ply_path: Path, out_path: Path) -> None:
    loaded = trimesh.load(ply_path, process=False)
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
        loaded.export(out_path)
        return

    points = _vertices_from_loaded_geometry(loaded)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        out_path.write_bytes(b"")
        return
    center = np.median(points, axis=0)
    lo = np.percentile(points, 5.0, axis=0)
    hi = np.percentile(points, 95.0, axis=0)
    extents = np.maximum(hi - lo, np.array([0.01, 0.01, 0.01], dtype=np.float64))
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    mesh.export(out_path)


def plan_sam3d_visual_assets(
    *,
    object_id: str,
    label: str,
    object_dir: Path,
    sam3d_result: Any,
    run_dir: Path,
) -> VisualAssetPlan:
    metadata = dict(getattr(sam3d_result, "metadata", {}) or {})
    service_response = summarize_sam3d_response(metadata)
    mesh_path = _existing_nonempty_path(getattr(sam3d_result, "mesh_path", None))
    if mesh_path is not None:
        visual_path = object_dir / "visual.glb"
        source_key = getattr(sam3d_result, "mesh_source_key", None) or service_response.get("mesh_source_key") or "mesh_path"
        return VisualAssetPlan(
            object_id=object_id,
            label=label,
            status="visual_glb_available",
            source_backend="sam3d_aligned",
            alignment_source_path=mesh_path,
            aligned_output_path=visual_path,
            final_visual=True,
            visual_path=visual_path,
            visual_point_cloud_path=None,
            debug_bbox_path=None,
            blocked_reason=None,
            service_response={**service_response, "mesh_source_key": source_key},
        )

    point_cloud_path = _existing_nonempty_path(getattr(sam3d_result, "point_cloud_path", None))
    if point_cloud_path is None:
        candidate = object_dir / "sam3d" / "point_cloud.ply"
        point_cloud_path = _existing_nonempty_path(candidate)
    if point_cloud_path is None:
        raise ValueError(f"{object_id}: SAM3D returned neither mesh nor point cloud geometry")

    visual_point_cloud_path = object_dir / "visual_point_cloud.ply"
    if point_cloud_path.resolve() != visual_point_cloud_path.resolve():
        shutil.copyfile(point_cloud_path, visual_point_cloud_path)

    debug_bbox_source = _existing_nonempty_path(getattr(sam3d_result, "debug_bbox_path", None))
    if debug_bbox_source is None:
        debug_bbox_source = object_dir / "sam3d" / "debug_bbox.glb"
        export_debug_bbox_from_ply(point_cloud_path, debug_bbox_source)
    if not debug_bbox_source.exists() or debug_bbox_source.stat().st_size == 0:
        raise ValueError(f"{object_id}: could not create debug bbox from SAM3D point cloud")

    debug_bbox_path = object_dir / "debug_bbox.glb"
    if debug_bbox_source.resolve() != debug_bbox_path.resolve():
        shutil.copyfile(debug_bbox_source, debug_bbox_path)

    return VisualAssetPlan(
        object_id=object_id,
        label=label,
        status="blocked_no_mesh_returned",
        source_backend="sam3d_point_cloud_debug_bbox_alignment",
        alignment_source_path=debug_bbox_source,
        aligned_output_path=debug_bbox_path,
        final_visual=False,
        visual_path=None,
        visual_point_cloud_path=visual_point_cloud_path,
        debug_bbox_path=debug_bbox_path,
        blocked_reason="sam3d_response_missing_mesh",
        service_response=service_response,
    )


def build_visual_asset_report(
    plan: VisualAssetPlan,
    *,
    run_dir: Path,
    alignment: dict[str, Any],
) -> dict[str, Any]:
    visual_path = plan.visual_path if plan.final_visual else plan.visual_point_cloud_path
    visual_role = "visual_mesh" if plan.final_visual else "visual_point_cloud_candidate"
    report: dict[str, Any] = {
        "object_id": plan.object_id,
        "label": plan.label,
        "status": plan.status,
        "source_backend": plan.source_backend,
        "blocked_reason": plan.blocked_reason,
        "service_response": plan.service_response,
        "visual_asset": {
            "path": _relative_path(visual_path, run_dir) if visual_path is not None else None,
            "asset_role": visual_role,
            "source": "sam3d_mesh_base64" if plan.final_visual else "sam3d_ply_camera_base64",
            "final_visual": bool(plan.final_visual),
        },
        "alignment_asset": {
            "path": _relative_path(plan.aligned_output_path, run_dir),
            "asset_role": "visual_mesh" if plan.final_visual else "debug_proxy_alignment_mesh",
            "source": "sam3d_mesh_base64" if plan.final_visual else "bbox_from_sam3d_point_cloud",
            "final_visual": bool(plan.final_visual),
        },
        "debug_proxy": None,
        "alignment": {
            "backend": alignment.get("backend"),
            "scale_m": alignment.get("scale_m"),
            "bbox_iou": alignment.get("bbox_iou"),
            "center_error_px": alignment.get("center_error_px"),
            "depth_residual_m": alignment.get("depth_residual_m"),
        },
    }
    if plan.debug_bbox_path is not None:
        report["debug_proxy"] = {
            "path": _relative_path(plan.debug_bbox_path, run_dir),
            "asset_role": "debug_proxy",
            "source": "bbox_from_sam3d_point_cloud",
            "final_visual": False,
        }
    return report


def write_visual_asset_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def summarize_sam3d_response(metadata: dict[str, Any]) -> dict[str, Any]:
    mesh_source_key, _encoded = extract_mesh_base64(metadata)
    return {
        "keys": sorted(metadata.keys()),
        "mesh_fields": {key: bool(metadata.get(key)) for key in SAM3D_MESH_BASE64_KEYS},
        "mesh_source_key": mesh_source_key,
        "has_ply_camera_base64": bool(metadata.get("ply_camera_base64")),
        "has_mask_base64": bool(metadata.get("mask_base64")),
    }


def _relative_path(path: Path | None, run_dir: Path) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(run_dir))


def _existing_nonempty_path(value: object) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def _vertices_from_loaded_geometry(loaded: Any) -> np.ndarray:
    if isinstance(loaded, trimesh.points.PointCloud):
        return np.asarray(loaded.vertices, dtype=np.float64)
    if isinstance(loaded, trimesh.Trimesh):
        return np.asarray(loaded.vertices, dtype=np.float64)
    if isinstance(loaded, trimesh.Scene):
        vertices = []
        for geom in loaded.geometry.values():
            if hasattr(geom, "vertices"):
                vertices.append(np.asarray(geom.vertices, dtype=np.float64))
        if vertices:
            return np.concatenate(vertices, axis=0)
    return np.zeros((0, 3), dtype=np.float64)
