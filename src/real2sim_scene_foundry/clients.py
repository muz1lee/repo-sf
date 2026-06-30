"""HTTP clients for external perception services."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import requests
from PIL import Image

from .camera import CameraIntrinsics


@dataclass(frozen=True)
class SAM3DResult:
    mesh_path: Path
    metadata: dict[str, Any]


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
            files["image"] = (image.name, image_f, "image/png")
            if mask_path is not None:
                mask = Path(mask_path)
                mask_f = mask.open("rb")
                handles.append(mask_f)
                files["mask"] = (mask.name, mask_f, "image/png")
            response = requests.post(self._process_url, files=files, timeout=self._timeout_s)
        finally:
            for handle in handles:
                handle.close()
        response.raise_for_status()
        metadata = response.json()
        mesh_path = out / "mesh.glb"
        encoded_mesh = metadata.get("mesh_glb_base64") or metadata.get("glb_base64") or metadata.get("mesh")
        if encoded_mesh:
            if "," in encoded_mesh:
                encoded_mesh = encoded_mesh.split(",", 1)[1]
            mesh_path.write_bytes(base64.b64decode(encoded_mesh))
        else:
            mesh_path.write_bytes(b"")
        (out / "sam3d_metadata.json").write_text(json_dumps(metadata), encoding="utf-8")
        return SAM3DResult(mesh_path=mesh_path, metadata=metadata)


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


def _encode_image_json(image: Image.Image) -> dict[str, str]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}


def json_dumps(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2)
