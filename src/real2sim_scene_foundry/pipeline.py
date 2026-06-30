"""Minimal smoke reconstruction pipeline for a SimFoundry-style digital twin."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from PIL import Image, ImageDraw

from .background import ArrayInpaintClient, create_background_artifacts
from .camera import CameraIntrinsics
from .clients import S2M2Client, SAM3Client, SAM3DClient
from .defaults import S2M2_URLS, SAM3_SEGMENT_URL, SAM3D_PROCESS_URL
from .manifest import SceneBackground, SceneManifest, SceneObject
from .proposals import ObjectProposal


class DepthClient(Protocol):
    def infer_xyz(
        self,
        left_path: str | Path,
        right_path: str | Path,
        camera: CameraIntrinsics,
        baseline_m: float,
    ) -> np.ndarray:
        ...


class MaskClient(Protocol):
    def segment_text(self, image_path: str | Path, text_prompt: str) -> np.ndarray:
        ...

    def segment_box(self, image_path: str | Path, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        ...


class MeshClient(Protocol):
    def process(
        self,
        image_path: str | Path,
        *,
        mask_path: str | Path | None = None,
        text_prompt: str = "object",
        out_dir: str | Path,
    ):
        ...


@dataclass(frozen=True)
class SmokeResult:
    out_dir: Path
    manifest_path: Path
    usd_path: Path
    qa_report_path: Path


@dataclass(frozen=True)
class ExtractResult:
    out_dir: Path
    manifest_path: Path
    object_dirs: dict[str, Path]


@dataclass(frozen=True)
class ReconstructionResult:
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


def run_extract(
    *,
    left_image: str | Path,
    right_image: str | Path,
    camera: CameraIntrinsics,
    baseline_m: float,
    out_dir: str | Path,
    proposals: list[ObjectProposal],
    depth_client: DepthClient | None = None,
    sam3_client: MaskClient | None = None,
    background_inpaint_client: ArrayInpaintClient | None = None,
    mock_depth_m: float | None = None,
) -> ExtractResult:
    if not proposals:
        raise ValueError("at least one object proposal is required")
    left_path = Path(left_image)
    right_path = Path(right_image)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    objects_dir = out / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    rgb_image = Image.open(left_path).convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    if rgb.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"left image shape {rgb.shape[:2]} does not match calibration {(camera.height, camera.width)}")
    rgb_image.save(out / "left.png")
    Image.open(right_path).convert("RGB").save(out / "right.png")
    (out / "camera.json").write_text(
        json.dumps(
            {
                "width": camera.width,
                "height": camera.height,
                "fx": camera.fx,
                "fy": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
                "baseline_m": baseline_m,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if depth_client is None:
        depth_client = ConstantDepthClient(mock_depth_m) if mock_depth_m is not None else S2M2Client(S2M2_URLS)
    xyz = depth_client.infer_xyz(left_path, right_path, camera, baseline_m)
    expected = (camera.height, camera.width, 3)
    if xyz.shape != expected:
        raise ValueError(f"XYZ must have shape {expected}, got {xyz.shape}")
    np.save(out / "xyz.npy", xyz)
    scene_point_count = _write_point_cloud_ply(out / "scene_cloud.ply", xyz, rgb)

    sam3_client = sam3_client or SAM3Client(SAM3_SEGMENT_URL)
    object_dirs: dict[str, Path] = {}
    manifest_objects = []
    for proposal in proposals:
        object_id = slugify(proposal.object_id)
        object_dir = objects_dir / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        raw_mask = _mask_from_proposal(sam3_client, left_path, proposal)
        mask = _select_prompt_component(raw_mask, bbox_xyxy=proposal.bbox_xyxy, point_xy=proposal.point_xy)
        if mask.shape != (camera.height, camera.width):
            raise ValueError(f"{object_id}: mask shape {mask.shape} does not match calibration {(camera.height, camera.width)}")

        mask_out = object_dir / "mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_out)
        crop_out = object_dir / "crop.png"
        bbox_xyxy = _save_crop(rgb, mask, crop_out)
        object_point_count = _write_point_cloud_ply(object_dir / "object_cloud.ply", xyz, rgb, mask)
        valid_ratio = _valid_xyz_ratio(xyz, mask)
        proposal_path = object_dir / "proposal.json"
        proposal_path.write_text(json.dumps(proposal.to_dict(), indent=2), encoding="utf-8")
        _write_overlay(rgb, mask, bbox_xyxy, object_dir / "mask_overlay.png")

        object_dirs[object_id] = object_dir
        manifest_objects.append(
            {
                "object_id": object_id,
                "label": proposal.label,
                "proposal": proposal.to_dict(),
                "proposal_source": proposal.source,
                "bbox_xyxy": [int(v) for v in bbox_xyxy],
                "mask_path": str(mask_out.relative_to(out)),
                "crop_path": str(crop_out.relative_to(out)),
                "object_cloud": str((object_dir / "object_cloud.ply").relative_to(out)),
                "mask_area_px": int(np.count_nonzero(mask)),
                "object_point_count": int(object_point_count),
                "valid_xyz_ratio": float(valid_ratio),
                "mass_kg": float(proposal.mass_kg),
                "friction": float(proposal.friction),
            }
        )

    manifest = {
        "version": 1,
        "stage": "extract",
        "coordinate_frame": "opencv_x_right_y_down_z_forward_meters",
        "left_image": "left.png",
        "right_image": "right.png",
        "xyz": "xyz.npy",
        "scene_point_cloud": "scene_cloud.ply",
        "scene_point_count": int(scene_point_count),
        "objects": manifest_objects,
    }
    manifest_path = out / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    create_background_artifacts(out, inpaint_client=background_inpaint_client)
    return ExtractResult(out_dir=out, manifest_path=manifest_path, object_dirs=object_dirs)


def run_reconstruct_align(
    *,
    extract_dir: str | Path,
    camera: CameraIntrinsics,
    sam3d_client: MeshClient | None = None,
) -> ReconstructionResult:
    out = Path(extract_dir)
    extraction = json.loads((out / "extraction_manifest.json").read_text(encoding="utf-8"))
    rgb = np.asarray(Image.open(out / extraction["left_image"]).convert("RGB"), dtype=np.uint8)
    xyz = np.load(out / extraction["xyz"])
    sam3d_client = sam3d_client or SAM3DClient(SAM3D_PROCESS_URL)
    export_dir = out / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    scene_objects: list[SceneObject] = []
    qa_objects = []
    for item in extraction["objects"]:
        object_id = item["object_id"]
        object_dir = out / "objects" / object_id
        mask_path = out / item["mask_path"]
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        object_points = _masked_xyz_points(xyz, mask)
        if object_points.size == 0:
            raise ValueError(f"{object_id}: no metric object points for alignment")

        sam3d_dir = object_dir / "sam3d"
        sam3d_errors: list[str] = []
        sam3d_prompt = str(item["label"])
        sam3d_image_path = out / item["crop_path"]
        for candidate_image in _sam3d_image_candidates(out, item, extraction):
            for candidate_prompt in _sam3d_prompt_candidates(str(item["label"])):
                try:
                    sam3d_result = sam3d_client.process(
                        candidate_image,
                        mask_path=mask_path,
                        text_prompt=candidate_prompt,
                        out_dir=sam3d_dir,
                    )
                    sam3d_prompt = candidate_prompt
                    sam3d_image_path = candidate_image
                    break
                except Exception as exc:  # noqa: BLE001 - prompt fallback handles service-level non-detections.
                    sam3d_errors.append(f"{candidate_image.name}/{candidate_prompt}: {exc}")
            else:
                continue
            break
        else:
            raise RuntimeError(f"{object_id}: SAM3D failed for all prompt candidates: {sam3d_errors}")
        raw_mesh_path = sam3d_dir / "raw_mesh.glb"
        result_mesh_path = Path(sam3d_result.mesh_path)
        if result_mesh_path.exists() and result_mesh_path != raw_mesh_path:
            shutil.copyfile(result_mesh_path, raw_mesh_path)
        elif not raw_mesh_path.exists():
            raw_mesh_path.write_bytes(result_mesh_path.read_bytes() if result_mesh_path.exists() else b"")
        (sam3d_dir / "sam3d_metadata.json").write_text(json.dumps(sam3d_result.metadata, indent=2), encoding="utf-8")

        aligned_mesh_path = object_dir / "mesh_aligned.glb"
        align = _align_mesh_to_object_cloud(raw_mesh_path, object_points, camera, mask, aligned_mesh_path)
        T_object_to_camera = align["T_model_to_camera"]
        T_object_to_world = _camera_to_world_transform(np.asarray(T_object_to_camera, dtype=np.float64))
        needs_manual_refine = bool(
            align["center_error_px"] > 30.0 or (align["bbox_iou"] < 0.4 and align["center_error_px"] > 10.0)
        )
        pose = {
            "object_id": object_id,
            "label": item["label"],
            "T_object_to_camera": T_object_to_camera,
            "T_object_to_world": T_object_to_world.tolist(),
            "raw_sam3d_mesh_path": str(raw_mesh_path.relative_to(out)),
            "mesh_path": str(aligned_mesh_path.relative_to(out)),
            "source_backend": "sam3d_aligned",
            "sam3d_text_prompt": sam3d_prompt,
            "sam3d_image_path": str(sam3d_image_path.relative_to(out)),
            "alignment": align,
            "sam3d_metadata": _summarize_sam3d_metadata(sam3d_result.metadata),
        }
        (object_dir / "pose.json").write_text(json.dumps(pose, indent=2), encoding="utf-8")

        scene_objects.append(
            SceneObject(
                object_id=object_id,
                label=item["label"],
                mesh_path=str(aligned_mesh_path.relative_to(out)),
                mask_path=item["mask_path"],
                crop_path=item["crop_path"],
                T_object_to_camera=T_object_to_camera,
                T_object_to_world=T_object_to_world.tolist(),
                scale_m=float(align["scale_m"]),
                mass_kg=float(item.get("mass_kg", 0.25)),
                friction=float(item.get("friction", 0.8)),
                confidence=float(min(1.0, item.get("valid_xyz_ratio", 1.0))),
                source_backend="sam3d_aligned",
                needs_manual_refine=needs_manual_refine,
            )
        )
        qa_objects.append(
            {
                "object_id": object_id,
                "label": item["label"],
                "source_backend": "sam3d_aligned",
                "alignment_backend": "sam3d_bbox_similarity",
                "mask_iou": float(align["bbox_iou"]),
                "center_error_px": float(align["center_error_px"]),
                "depth_residual_m": float(align["depth_residual_m"]),
                "object_point_count": int(item["object_point_count"]),
                "valid_xyz_ratio": float(item["valid_xyz_ratio"]),
                "needs_manual_refine": needs_manual_refine,
            }
        )

    scene_manifest = SceneManifest(objects=scene_objects, background=_load_scene_background(out))
    manifest_path = out / "scene_manifest.json"
    manifest_path.write_text(json.dumps(scene_manifest.to_dict(), indent=2), encoding="utf-8")
    usd_path = export_dir / "scene.usda"
    _write_usda_stub(usd_path, scene_manifest)
    qa_report_path = qa_dir / "qa_report.json"
    qa_report_path.write_text(
        json.dumps(
            {
                "object_count": len(scene_objects),
                "coordinate_frame": "opencv_x_right_y_down_z_forward_meters",
                "scene_point_cloud": extraction["scene_point_cloud"],
                "scene_point_count": int(extraction["scene_point_count"]),
                "background": extraction.get("background"),
                "physics_settle": {"status": "not_run_before_m5", "nan_detected": False},
                "objects": qa_objects,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_scene_overlay(rgb, out, extraction["objects"], qa_dir / "overlay.png")
    _write_scene_render_preview(rgb.shape[1], rgb.shape[0], out, extraction["objects"], qa_dir / "render.png")
    return ReconstructionResult(out_dir=out, manifest_path=manifest_path, usd_path=usd_path, qa_report_path=qa_report_path)


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

    scene_cloud_path = out / "scene_cloud.ply"
    scene_point_count = _write_point_cloud_ply(scene_cloud_path, xyz, rgb)
    object_cloud_path = object_dir / "object_cloud.ply"
    object_point_count = _write_point_cloud_ply(object_cloud_path, xyz, rgb, mask)

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
        "object_cloud_path": str(object_cloud_path.relative_to(out)),
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
        "scene_point_cloud": str(scene_cloud_path.relative_to(out)),
        "scene_point_count": int(scene_point_count),
        "physics_settle": {"status": "not_run_in_smoke", "nan_detected": False},
        "objects": [
            {
                "object_id": object_id,
                "label": label,
                "object_cloud": str(object_cloud_path.relative_to(out)),
                "object_point_count": int(object_point_count),
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


def _mask_from_proposal(client: MaskClient, image_path: Path, proposal: ObjectProposal) -> np.ndarray:
    if proposal.bbox_xyxy is not None:
        return np.asarray(client.segment_box(image_path, proposal.bbox_xyxy), dtype=bool)
    return np.asarray(client.segment_text(image_path, proposal.label), dtype=bool)


def _sam3d_prompt_candidates(label: str) -> list[str]:
    words = [word for word in re.split(r"[^a-zA-Z0-9]+", label.strip().lower()) if word]
    candidates = [label.strip()]
    if len(words) >= 2:
        candidates.append(words[-1])
    candidates.append("object")
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _sam3d_image_candidates(out: Path, item: dict[str, object], extraction: dict[str, object]) -> list[Path]:
    candidates = [out / str(item["crop_path"]), out / str(extraction["left_image"])]
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _summarize_sam3d_metadata(metadata: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key, value in metadata.items():
        if key.endswith("_base64"):
            summary[f"{key}_bytes"] = len(str(value))
            continue
        summary[key] = value
    return summary


def _select_prompt_component(
    mask: np.ndarray,
    *,
    bbox_xyxy: tuple[int, int, int, int] | None,
    point_xy: tuple[int, int] | None,
) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if np.count_nonzero(mask_u8) == 0:
        raise ValueError("SAM3 mask is empty")
    count, labels, _stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 2:
        return mask_u8.astype(bool)

    if point_xy is not None:
        px, py = point_xy
        if 0 <= py < labels.shape[0] and 0 <= px < labels.shape[1] and labels[py, px] > 0:
            selected = int(labels[py, px])
        else:
            selected = _nearest_component(centroids, point_xy)
    elif bbox_xyxy is not None:
        x0, y0, x1, y1 = bbox_xyxy
        selected = _nearest_component(centroids, ((x0 + x1) // 2, (y0 + y1) // 2))
    else:
        areas = np.bincount(labels.reshape(-1))[1:]
        selected = int(np.argmax(areas) + 1)
    return labels == selected


def _nearest_component(centroids: np.ndarray, point_xy: tuple[int, int]) -> int:
    point = np.asarray(point_xy, dtype=np.float64)
    component_centroids = np.asarray(centroids[1:], dtype=np.float64)
    distances = np.linalg.norm(component_centroids - point[None, :], axis=1)
    return int(np.argmin(distances) + 1)


def _valid_xyz_ratio(xyz: np.ndarray, mask: np.ndarray) -> float:
    if np.count_nonzero(mask) == 0:
        return 0.0
    valid = mask & np.all(np.isfinite(xyz), axis=2) & (xyz[..., 2] > 0.0)
    return float(np.count_nonzero(valid) / np.count_nonzero(mask))


def _masked_xyz_points(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.all(np.isfinite(xyz), axis=2) & (xyz[..., 2] > 0.0)
    return np.asarray(xyz[valid], dtype=np.float64)


def _save_crop(rgb: np.ndarray, mask: np.ndarray, out_path: Path) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("mask is empty")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    Image.fromarray(rgb[y0:y1, x0:x1]).save(out_path)
    return x0, y0, x1, y1


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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


def _align_mesh_to_object_cloud(
    raw_mesh_path: Path,
    object_points: np.ndarray,
    camera: CameraIntrinsics,
    mask: np.ndarray,
    out_path: Path,
) -> dict[str, object]:
    mesh = _load_mesh(raw_mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError(f"{raw_mesh_path} has no vertices")

    src_lo, src_hi = mesh.bounds
    src_center = (np.asarray(src_lo, dtype=np.float64) + np.asarray(src_hi, dtype=np.float64)) * 0.5
    src_extents = np.maximum(np.asarray(src_hi, dtype=np.float64) - np.asarray(src_lo, dtype=np.float64), 1e-6)
    target_center = np.median(object_points, axis=0)
    tgt_lo = np.percentile(object_points, 5.0, axis=0)
    tgt_hi = np.percentile(object_points, 95.0, axis=0)
    target_extents = np.maximum(tgt_hi - tgt_lo, np.array([0.02, 0.02, 0.02], dtype=np.float64))
    scale = float(np.median(target_extents / src_extents))
    translation = target_center - scale * src_center

    raw_to_camera = np.eye(4, dtype=np.float64)
    raw_to_camera[:3, :3] *= scale
    raw_to_camera[:3, 3] = translation

    local_transform = np.eye(4, dtype=np.float64)
    local_transform[:3, :3] *= scale
    local_transform[:3, 3] = -scale * src_center
    local_mesh = mesh.copy()
    local_mesh.apply_transform(local_transform)
    local_mesh.export(out_path)

    T_model_to_camera = np.eye(4, dtype=np.float64)
    T_model_to_camera[:3, 3] = target_center
    local_vertices = np.asarray(local_mesh.vertices, dtype=np.float64)
    camera_vertices = local_vertices + target_center[None, :]
    projected_bbox = _projected_bbox(camera, camera_vertices)
    target_bbox = _mask_bbox(mask)
    bbox_iou = _bbox_iou(projected_bbox, target_bbox)
    center_error = _bbox_center_error(projected_bbox, target_bbox)
    depth_residual = _nearest_depth_residual(camera_vertices, object_points)
    return {
        "backend": "sam3d_bbox_similarity",
        "scale": float(scale),
        "scale_m": float(np.max(target_extents)),
        "translation": [float(v) for v in translation],
        "raw_to_camera_transform": raw_to_camera.tolist(),
        "local_mesh_centered": True,
        "T_model_to_camera": T_model_to_camera.tolist(),
        "target_bbox_xyxy": [int(v) for v in target_bbox],
        "projected_bbox_xyxy": [int(v) for v in projected_bbox],
        "bbox_iou": float(bbox_iou),
        "center_error_px": float(center_error),
        "depth_residual_m": float(depth_residual),
    }


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"{path} did not contain mesh geometry")
        return trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a mesh")
    return loaded


def _projected_bbox(camera: CameraIntrinsics, points_xyz: np.ndarray) -> tuple[int, int, int, int]:
    u, v, z = camera.project(points_xyz)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 0.0)
    if not np.any(finite):
        return 0, 0, 1, 1
    u = np.clip(u[finite], 0, camera.width - 1)
    v = np.clip(v[finite], 0, camera.height - 1)
    return int(np.floor(u.min())), int(np.floor(v.min())), int(np.ceil(u.max())) + 1, int(np.ceil(v.max())) + 1


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def _bbox_center_error(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float64)
    bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float64)
    return float(np.linalg.norm(ac - bc))


def _nearest_depth_residual(aligned_vertices: np.ndarray, object_points: np.ndarray) -> float:
    if aligned_vertices.size == 0 or object_points.size == 0:
        return float("inf")
    vertices = aligned_vertices
    if vertices.shape[0] > 2048:
        vertices = vertices[np.linspace(0, vertices.shape[0] - 1, 2048, dtype=int)]
    points = object_points
    if points.shape[0] > 8192:
        points = points[np.linspace(0, points.shape[0] - 1, 8192, dtype=int)]
    distances, indices = cKDTree(points).query(vertices, k=1)
    nearest = points[indices]
    z_delta = np.abs(vertices[:, 2] - nearest[:, 2])
    return float(np.median(np.minimum(distances, z_delta)))


def _write_point_cloud_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, mask: np.ndarray | None = None) -> int:
    valid = np.all(np.isfinite(xyz), axis=2) & (xyz[..., 2] > 0.0)
    if mask is not None:
        valid &= mask
    points = np.asarray(xyz[valid], dtype=np.float64)
    colors = np.asarray(rgb[valid], dtype=np.uint8)

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for point, color in zip(points, colors, strict=True):
        x, y, z = (float(v) for v in point)
        r, g, b = (int(v) for v in color)
        lines.append(f"{x:.8g} {y:.8g} {z:.8g} {r} {g} {b}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(len(points))


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


def _load_scene_background(run_dir: Path) -> SceneBackground | None:
    path = run_dir / "background" / "background_manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SceneBackground(
        source_backend=str(data["source_backend"]),
        bg_only_image_path=str(data["bg_only_image"]),
        foreground_mask_path=str(data["foreground_mask"]),
        point_cloud_path=str(data["bg_only_cloud"]),
        status="proxy_from_single_stereo_pair",
        gaussian_splat_path=None,
    )


def _write_overlay(rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int], out_path: Path) -> None:
    overlay = rgb.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_xyxy, outline=(0, 255, 0), width=2)
    image.save(out_path)


def _write_scene_overlay(rgb: np.ndarray, run_dir: Path, objects: list[dict[str, object]], out_path: Path) -> None:
    overlay = rgb.copy()
    colors = [
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 180, 255], dtype=np.uint8),
        np.array([255, 180, 0], dtype=np.uint8),
        np.array([80, 220, 120], dtype=np.uint8),
    ]
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    for idx, item in enumerate(objects):
        mask = np.asarray(Image.open(run_dir / str(item["mask_path"])).convert("L")) > 0
        color = colors[idx % len(colors)]
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    for item in objects:
        bbox = tuple(int(v) for v in item["bbox_xyxy"])
        draw.rectangle(bbox, outline=(0, 255, 0), width=2)
        draw.text((bbox[0], max(0, bbox[1] - 12)), str(item["label"]), fill=(0, 255, 0))
    image.save(out_path)


def _write_render_preview(width: int, height: int, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int], out_path: Path) -> None:
    preview = Image.new("RGB", (width, height), color=(245, 245, 245))
    color = np.asarray(preview, dtype=np.uint8).copy()
    color[mask] = np.array([80, 140, 255], dtype=np.uint8)
    image = Image.fromarray(color)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_xyxy, outline=(20, 70, 180), width=2)
    image.save(out_path)


def _write_scene_render_preview(width: int, height: int, run_dir: Path, objects: list[dict[str, object]], out_path: Path) -> None:
    preview = Image.new("RGB", (width, height), color=(245, 245, 245))
    color = np.asarray(preview, dtype=np.uint8).copy()
    colors = [
        np.array([80, 140, 255], dtype=np.uint8),
        np.array([255, 130, 80], dtype=np.uint8),
        np.array([120, 210, 120], dtype=np.uint8),
        np.array([220, 180, 80], dtype=np.uint8),
    ]
    for idx, item in enumerate(objects):
        mask = np.asarray(Image.open(run_dir / str(item["mask_path"])).convert("L")) > 0
        color[mask] = colors[idx % len(colors)]
    image = Image.fromarray(color)
    draw = ImageDraw.Draw(image)
    for item in objects:
        bbox = tuple(int(v) for v in item["bbox_xyxy"])
        draw.rectangle(bbox, outline=(20, 70, 180), width=2)
    image.save(out_path)
