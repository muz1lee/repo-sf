"""Foreground removal and background artifacts for SimFoundry-style extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import requests
from PIL import Image

from .background_registration import write_background_registration


class ArrayInpaintClient(Protocol):
    backend_name: str

    def inpaint(self, rgb: np.ndarray, remove_mask: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class BackgroundResult:
    background_dir: Path
    manifest_path: Path
    foreground_mask_path: Path
    bg_only_path: Path
    bg_only_cloud_path: Path


class OpenCVInpaintClient:
    backend_name = "opencv_inpaint_single_frame"

    def __init__(self, *, radius_px: int = 5, method: int = cv2.INPAINT_TELEA) -> None:
        self._radius_px = int(radius_px)
        self._method = int(method)

    def inpaint(self, rgb: np.ndarray, remove_mask: np.ndarray) -> np.ndarray:
        mask_u8 = np.asarray(remove_mask, dtype=np.uint8) * 255
        return cv2.inpaint(np.asarray(rgb, dtype=np.uint8), mask_u8, self._radius_px, self._method)


class HTTPInpaintClient:
    def __init__(self, inpaint_url: str, *, timeout_s: float = 300.0) -> None:
        self._inpaint_url = str(inpaint_url)
        self._timeout_s = float(timeout_s)
        self.backend_name = f"http_inpaint:{self._inpaint_url}"

    def inpaint(self, rgb: np.ndarray, remove_mask: np.ndarray) -> np.ndarray:
        image_buf = _png_bytes(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))
        mask_buf = _png_bytes(Image.fromarray(np.asarray(remove_mask, dtype=np.uint8) * 255))
        files = {
            "image": ("image.png", image_buf, "image/png"),
            "mask": ("mask.png", mask_buf, "image/png"),
        }
        response = requests.post(self._inpaint_url, files=files, timeout=self._timeout_s)
        response.raise_for_status()
        import io

        return np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"), dtype=np.uint8)


def create_background_artifacts(
    run_dir: str | Path,
    *,
    inpaint_client: ArrayInpaintClient | None = None,
    dilation_px: int = 8,
) -> BackgroundResult:
    run = Path(run_dir)
    extraction_path = run / "extraction_manifest.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    rgb = np.asarray(Image.open(run / extraction["left_image"]).convert("RGB"), dtype=np.uint8)
    xyz = np.load(run / extraction["xyz"])
    foreground_mask = _union_object_masks(run, extraction["objects"], rgb.shape[:2])
    foreground_mask = _dilate_mask(foreground_mask, dilation_px)

    client = inpaint_client or OpenCVInpaintClient()
    bg_only = client.inpaint(rgb, foreground_mask)
    if bg_only.shape != rgb.shape:
        raise ValueError(f"inpaint result must have shape {rgb.shape}, got {bg_only.shape}")

    bg_dir = run / "background"
    bg_dir.mkdir(parents=True, exist_ok=True)
    foreground_mask_path = bg_dir / "foreground_mask.png"
    bg_only_path = bg_dir / "bg_only.png"
    bg_only_cloud_path = bg_dir / "bg_only_cloud.ply"
    Image.fromarray(foreground_mask.astype(np.uint8) * 255).save(foreground_mask_path)
    Image.fromarray(np.asarray(bg_only, dtype=np.uint8)).save(bg_only_path)
    bg_point_count = _write_point_cloud_ply(bg_only_cloud_path, xyz, bg_only, mask=~foreground_mask)

    registration = write_background_registration(
        run,
        source_kind="bg_only_cloud",
        status="registered_bg_only_cloud",
        scale_source="input_metric_depth",
    )
    manifest = {
        "version": 1,
        "stage": "background",
        "source_backend": client.backend_name,
        "inpaint_backend": client.backend_name,
        "foreground_mask": str(foreground_mask_path.relative_to(run)),
        "bg_only_image": str(bg_only_path.relative_to(run)),
        "bg_only_cloud": str(bg_only_cloud_path.relative_to(run)),
        "registration_path": str(registration.path.relative_to(run)),
        "registration_status": registration.data["status"],
        "foreground_coverage_px": int(np.count_nonzero(foreground_mask)),
        "background_point_count": int(bg_point_count),
        "background_3dgs": {
            "status": "requires_video_or_multiview",
            "reason": "SimFoundry trains background 3DGS from a BG-only video; a single stereo pair only supports a proxy background.",
            "input_frame_count": 1,
        },
    }
    manifest_path = bg_dir / "background_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _attach_background_to_extraction(extraction_path, extraction, manifest)
    return BackgroundResult(
        background_dir=bg_dir,
        manifest_path=manifest_path,
        foreground_mask_path=foreground_mask_path,
        bg_only_path=bg_only_path,
        bg_only_cloud_path=bg_only_cloud_path,
    )


def _union_object_masks(run_dir: Path, objects: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for item in objects:
        obj_mask = np.asarray(Image.open(run_dir / str(item["mask_path"])).convert("L")) > 0
        if obj_mask.shape != shape:
            raise ValueError(f"{item['object_id']}: mask shape {obj_mask.shape} does not match image shape {shape}")
        mask |= obj_mask
    return mask


def _dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((int(dilation_px) * 2 + 1, int(dilation_px) * 2 + 1), dtype=np.uint8)
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def _attach_background_to_extraction(
    extraction_path: Path,
    extraction: dict[str, object],
    background_manifest: dict[str, object],
) -> None:
    extraction["background"] = {
        "manifest_path": "background/background_manifest.json",
        "foreground_mask": background_manifest["foreground_mask"],
        "bg_only_image": background_manifest["bg_only_image"],
        "bg_only_cloud": background_manifest["bg_only_cloud"],
        "registration_path": background_manifest["registration_path"],
        "source_backend": background_manifest["source_backend"],
        "background_3dgs": background_manifest["background_3dgs"],
    }
    extraction_path.write_text(json.dumps(extraction, indent=2), encoding="utf-8")


def _write_point_cloud_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, mask: np.ndarray) -> int:
    valid = mask & np.all(np.isfinite(xyz), axis=2) & (xyz[..., 2] > 0.0)
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


def _png_bytes(image: Image.Image):
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf
