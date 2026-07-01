"""HTTP clients for external perception services."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import requests
from PIL import Image

from .camera import CameraIntrinsics
from .visual_assets import export_debug_bbox_from_ply, extract_mesh_base64


@dataclass(frozen=True)
class SAM3DResult:
    mesh_path: Path | None
    metadata: dict[str, Any]
    point_cloud_path: Path | None = None
    debug_bbox_path: Path | None = None
    mesh_source_key: str | None = None


class S2M2Client:
    def __init__(self, endpoints: Sequence[str], *, timeout_s: float = 120.0) -> None:
        if not endpoints:
            raise ValueError("at least one S2M2 endpoint is required")
        self._endpoints = [str(endpoint) for endpoint in endpoints]
        self._timeout_s = float(timeout_s)
        self._next_endpoint = 0

    def infer_xyz(
        self,
        left_path: str | Path,
        right_path: str | Path,
        camera: CameraIntrinsics,
        baseline_m: float,
    ) -> np.ndarray:
        errors: list[Exception] = []
        for _ in range(len(self._endpoints)):
            endpoint = self._pick_endpoint()
            try:
                return self._infer_xyz_once(endpoint, Path(left_path), Path(right_path), camera, baseline_m)
            except Exception as exc:  # noqa: BLE001 - service fallback should catch endpoint-local failures.
                errors.append(exc)
        raise RuntimeError(f"S2M2 inference failed for all endpoints: {errors}") from errors[-1]

    def _pick_endpoint(self) -> str:
        endpoint = self._endpoints[self._next_endpoint % len(self._endpoints)]
        self._next_endpoint += 1
        return endpoint

    def _infer_xyz_once(
        self,
        endpoint: str,
        left_path: Path,
        right_path: Path,
        camera: CameraIntrinsics,
        baseline_m: float,
    ) -> np.ndarray:
        data = {
            "K": camera.k_string(),
            "baseline": str(float(baseline_m)),
            "width": str(int(camera.width)),
        }
        with left_path.open("rb") as left_f, right_path.open("rb") as right_f:
            files = {
                "left_file": (left_path.name, left_f),
                "right_file": (right_path.name, right_f),
            }
            response = requests.post(endpoint, files=files, data=data, timeout=self._timeout_s)
        response.raise_for_status()
        xyz = np.load(io.BytesIO(response.content))
        expected = (int(camera.height), int(camera.width), 3)
        if xyz.shape != expected:
            raise ValueError(f"S2M2 response must have shape {expected}, got {xyz.shape}")
        return np.asarray(xyz, dtype=np.float32)


class MoGeClient:
    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)

    def infer_intrinsics(self, image_path: str | Path) -> CameraIntrinsics:
        path = Path(image_path)
        with path.open("rb") as f:
            response = requests.post(
                f"{self._base_url}/api/focal",
                files={"image": (path.name, f, "image/png")},
                timeout=self._timeout_s,
            )
        response.raise_for_status()
        data = response.json()
        return CameraIntrinsics(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data.get("fy", data["fx"])),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
        )


class SAM3Client:
    def __init__(self, segment_url: str, *, timeout_s: float = 120.0) -> None:
        self._segment_url = str(segment_url)
        self._timeout_s = float(timeout_s)

    def segment_text(self, image_path: str | Path, text_prompt: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        payload = _encode_image_json(image)
        payload["text_prompt"] = text_prompt
        response = requests.post(self._segment_url, json=payload, timeout=self._timeout_s)
        response.raise_for_status()
        return decode_sam3_mask(response.json(), size=image.size, label=text_prompt)

    def segment_box(self, image_path: str | Path, box_xyxy: Sequence[int]) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        payload = _encode_image_json(image)
        payload["box_prompts"] = [int(v) for v in box_xyxy]
        response = requests.post(self._segment_url, json=payload, timeout=self._timeout_s)
        response.raise_for_status()
        return decode_sam3_mask(response.json(), size=image.size, label=f"box {box_xyxy}")


class SAM3DClient:
    def __init__(self, process_url: str, *, timeout_s: float = 300.0) -> None:
        self._process_url = str(process_url)
        self._timeout_s = float(timeout_s)

    def process(
        self,
        image_path: str | Path,
        *,
        mask_path: str | Path | None = None,
        text_prompt: str = "object",
        out_dir: str | Path,
    ) -> SAM3DResult:
        image = Path(image_path)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        handles = []
        try:
            image_f = image.open("rb")
            handles.append(image_f)
            files["image_file"] = (image.name, image_f, "image/png")
            response = requests.post(
                self._process_url,
                files=files,
                data={"text_prompt": str(text_prompt)},
                timeout=self._timeout_s,
            )
        finally:
            for handle in handles:
                handle.close()
        response.raise_for_status()
        metadata = response.json()
        if mask_path is not None:
            metadata.setdefault("external_mask_path", str(mask_path))
        mesh_path: Path | None = None
        point_cloud_path: Path | None = None
        debug_bbox_path: Path | None = None
        mesh_source_key, encoded_mesh = extract_mesh_base64(metadata)
        if encoded_mesh:
            mesh_path = out / "mesh.glb"
            mesh_path.write_bytes(_decode_base64_bytes(encoded_mesh))
        elif metadata.get("ply_camera_base64"):
            point_cloud_path = out / "point_cloud.ply"
            point_cloud_path.write_bytes(_decode_base64_bytes(str(metadata["ply_camera_base64"])))
            debug_bbox_path = out / "debug_bbox.glb"
            export_debug_bbox_from_ply(point_cloud_path, debug_bbox_path)
        (out / "sam3d_metadata.json").write_text(json_dumps(metadata), encoding="utf-8")
        return SAM3DResult(
            mesh_path=mesh_path,
            metadata=metadata,
            point_cloud_path=point_cloud_path,
            debug_bbox_path=debug_bbox_path,
            mesh_source_key=mesh_source_key,
        )


class SAM3DMeshJobClient:
    """Client for the persistent SAM3D mesh job API.

    The service accepts an RGBA foreground image plus an optional mask, then
    exposes the generated GLB through a poll/download job contract.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 900.0,
        poll_interval_s: float = 2.0,
        seed: int = 12345,
        alpha_threshold: int = 127,
        generate_texture: bool = False,
    ) -> None:
        url = str(base_url).rstrip("/")
        if url.endswith("/api/jobs"):
            self._base_url = url[: -len("/api/jobs")]
            self._jobs_url = url
        else:
            self._base_url = url
            self._jobs_url = f"{url}/api/jobs"
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._seed = int(seed)
        self._alpha_threshold = int(alpha_threshold)
        self._generate_texture = bool(generate_texture)

    def process(
        self,
        image_path: str | Path,
        *,
        mask_path: str | Path | None = None,
        text_prompt: str = "object",
        out_dir: str | Path,
    ) -> SAM3DResult:
        image = Path(image_path)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        front_path = out / "mesh_job_front.png"
        upload_mask_path = out / "mesh_job_mask.png"
        _write_mesh_job_inputs(
            image,
            Path(mask_path) if mask_path is not None else None,
            front_path=front_path,
            mask_path=upload_mask_path,
            alpha_threshold=self._alpha_threshold,
        )

        data = {
            "seed": str(self._seed),
            "alpha_threshold": str(self._alpha_threshold),
            "generate_texture": "true" if self._generate_texture else "false",
        }
        with front_path.open("rb") as front_f, upload_mask_path.open("rb") as mask_f:
            response = requests.post(
                self._jobs_url,
                files={
                    "front": ("front.png", front_f, "image/png"),
                    "mask": ("mask.png", mask_f, "image/png"),
                },
                data=data,
                timeout=self._timeout_s,
            )
        response.raise_for_status()
        create_payload = response.json()
        job_id = str(create_payload.get("job_id") or "")
        if not job_id:
            raise RuntimeError(f"SAM3D mesh job response missing job_id: {create_payload}")

        final_status = self._wait_for_job(job_id)
        download_url = f"{self._base_url}/download/{job_id}"
        download_response = requests.get(download_url, timeout=self._timeout_s)
        download_response.raise_for_status()
        if not download_response.content:
            raise RuntimeError(f"SAM3D mesh job {job_id} returned empty GLB")
        mesh_path = out / "mesh.glb"
        mesh_path.write_bytes(download_response.content)
        metadata = {
            "source_backend": "sam3d_mesh_job",
            "mesh_job": final_status,
            "mesh_job_endpoint": self._jobs_url,
            "download_url": download_url,
            "text_prompt": str(text_prompt),
            "input_image_path": str(image),
            "external_mask_path": str(mask_path) if mask_path is not None else None,
            "prepared_front_path": str(front_path),
            "prepared_mask_path": str(upload_mask_path),
        }
        (out / "sam3d_metadata.json").write_text(json_dumps(metadata), encoding="utf-8")
        return SAM3DResult(mesh_path=mesh_path, metadata=metadata, mesh_source_key="download_glb")

    def _wait_for_job(self, job_id: str) -> dict[str, Any]:
        status_url = f"{self._jobs_url}/{job_id}"
        deadline = time.monotonic() + self._timeout_s
        last_payload: dict[str, Any] = {}
        while True:
            response = requests.get(status_url, timeout=self._timeout_s)
            response.raise_for_status()
            payload = response.json()
            last_payload = dict(payload)
            status = str(payload.get("status", "")).lower()
            if status == "done":
                return last_payload
            if status in {"error", "failed", "cancelled", "canceled"}:
                raise RuntimeError(f"SAM3D mesh job {job_id} failed: {payload}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"SAM3D mesh job {job_id} timed out; last status: {last_payload}")
            if self._poll_interval_s > 0.0:
                time.sleep(self._poll_interval_s)



def decode_sam3_mask(data: dict[str, Any], *, size: tuple[int, int], label: str) -> np.ndarray:
    if not data.get("success"):
        raise RuntimeError(f"SAM3 segmentation failed for {label}: {data}")
    masks: list[np.ndarray] = []
    for detection in data.get("detections") or []:
        encoded = detection.get("mask")
        if not encoded:
            continue
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        masks.append(np.asarray(image) > 0)
    if not masks:
        raise RuntimeError(f"SAM3 returned no masks for {label}: {data}")
    return np.logical_or.reduce(masks).astype(bool)



def _write_mesh_job_inputs(
    image_path: Path,
    source_mask_path: Path | None,
    *,
    front_path: Path,
    mask_path: Path,
    alpha_threshold: int,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    if source_mask_path is None:
        mask = image.getchannel("A")
    else:
        mask = _mask_for_image_size(source_mask_path, image.size, alpha_threshold=alpha_threshold)
    rgb = image.convert("RGB")
    front = Image.merge("RGBA", (*rgb.split(), mask))
    front.save(front_path)
    mask.save(mask_path)


def _mask_for_image_size(mask_path: Path, image_size: tuple[int, int], *, alpha_threshold: int) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    if mask.size == image_size:
        return mask
    mask_array = np.asarray(mask) > alpha_threshold
    ys, xs = np.where(mask_array)
    if xs.size == 0:
        raise ValueError(f"foreground mask is empty: {mask_path}")
    crop = mask.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    if crop.size != image_size:
        raise ValueError(
            f"mask size {mask.size} does not match image size {image_size}, "
            f"and foreground bbox crop size {crop.size} does not match image"
        )
    return crop


def _encode_image_json(image: Image.Image) -> dict[str, str]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}


def _decode_base64_bytes(value: str) -> bytes:
    if "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def json_dumps(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2)
