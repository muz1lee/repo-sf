"""Video-reference object scene composition for the SimFoundry reproduction path."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from .background_registration import BackgroundRegistrationResult, write_background_registration
from .camera import CameraIntrinsics
from .clients import SAM3Client, SAM3DClient
from .defaults import SAM3D_PROCESS_URL, SAM3_SEGMENT_URL
from .interactive import export_interactive_scene
from .manifest import SceneBackground
from .pipeline import MaskClient, MeshClient, ReconstructionResult, run_extract, run_reconstruct_align
from .proposals import ObjectProposal, QwenProposalClient
from .support_plane import estimate_and_apply_support_plane


class ProposalClient:
    def propose(self, image_path: str | Path, *, target_labels: Sequence[str] | None = None) -> list[ObjectProposal]:
        raise NotImplementedError


@dataclass(frozen=True)
class VideoReferenceSceneResult:
    run_dir: Path
    manifest_path: Path
    usd_path: Path
    qa_report_path: Path
    interactive_script_path: Path
    interaction_report_path: Path


class MoGePointMapDepthClient:
    def __init__(self, points_path: str | Path) -> None:
        self.points_path = Path(points_path)

    def infer_xyz(
        self,
        left_path: str | Path,  # noqa: ARG002 - kept for DepthClient compatibility.
        right_path: str | Path,  # noqa: ARG002 - kept for DepthClient compatibility.
        camera: CameraIntrinsics | None,
        baseline_m: float,  # noqa: ARG002 - MoGe points are already metric-ish XYZ.
    ) -> np.ndarray:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        points = cv2.imread(str(self.points_path), cv2.IMREAD_UNCHANGED)
        if points is None:
            raise ValueError(f"failed to read MoGe points EXR: {self.points_path}")
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(f"MoGe points must have shape HxWx3, got {points.shape}")
        points = points[..., [2, 1, 0]]
        if camera is not None and points.shape[:2] != (camera.height, camera.width):
            raise ValueError(f"MoGe points shape {points.shape[:2]} does not match camera {(camera.height, camera.width)}")
        points[~np.isfinite(points)] = 0.0
        return points


def camera_from_moge_fov(fov: dict[str, float], *, width: int, height: int) -> CameraIntrinsics:
    fx = _focal_from_fov(width, float(fov["fov_x"]))
    fy = _focal_from_fov(height, float(fov.get("fov_y", fov["fov_x"])))
    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=(int(width) - 1) / 2.0,
        cy=(int(height) - 1) / 2.0,
    )


def load_default_video_object_labels(run_dir: str | Path) -> list[str]:
    status_path = Path(run_dir) / "video" / "bg_only_status.json"
    if not status_path.is_file():
        return []
    status = json.loads(status_path.read_text(encoding="utf-8"))
    labels: list[str] = []
    for label in status.get("observed_labels", []):
        clean = str(label).strip()
        if clean and clean not in labels:
            labels.append(clean)
    return labels


def run_video_reference_scene(
    *,
    run_dir: str | Path,
    proposals: list[ObjectProposal] | None = None,
    target_labels: Sequence[str] | None = None,
    max_objects: int | None = None,
    proposal_client: ProposalClient | None = None,
    sam3_client: MaskClient | None = None,
    sam3d_client: MeshClient | None = None,
    settle_steps: int = 100,
) -> VideoReferenceSceneResult:
    run = Path(run_dir)
    reference_dir = run / "video" / "moge_reference" / "reference"
    image_path = reference_dir / "image.jpg"
    points_path = reference_dir / "points.exr"
    fov_path = reference_dir / "fov.json"
    if not image_path.is_file():
        raise FileNotFoundError(f"MoGe reference image not found: {image_path}")
    if not points_path.is_file():
        raise FileNotFoundError(f"MoGe points EXR not found: {points_path}")
    if not fov_path.is_file():
        raise FileNotFoundError(f"MoGe fov.json not found: {fov_path}")

    width, height = Image.open(image_path).size
    camera = camera_from_moge_fov(json.loads(fov_path.read_text(encoding="utf-8")), width=width, height=height)
    if proposals is None:
        labels = list(target_labels or load_default_video_object_labels(run))
        proposal_client = proposal_client or QwenProposalClient()
        proposals = proposal_client.propose(image_path, target_labels=labels or None)
    if max_objects is not None:
        proposals = proposals[: int(max_objects)]
    if not proposals:
        raise ValueError("no video reference object proposals available")

    extract = run_extract(
        left_image=image_path,
        right_image=image_path,
        camera=camera,
        baseline_m=0.0,
        out_dir=run,
        proposals=proposals,
        depth_client=MoGePointMapDepthClient(points_path),
        sam3_client=sam3_client or SAM3Client(SAM3_SEGMENT_URL),
    )
    reconstruction = run_reconstruct_align(
        extract_dir=extract.out_dir,
        camera=camera,
        sam3d_client=sam3d_client or SAM3DClient(SAM3D_PROCESS_URL),
    )
    attach_video_3dgs_background(run)
    estimate_and_apply_support_plane(run)
    write_background_registration(run)
    interactive = export_interactive_scene(run, settle_steps=settle_steps)
    return VideoReferenceSceneResult(
        run_dir=run,
        manifest_path=reconstruction.manifest_path,
        usd_path=reconstruction.usd_path,
        qa_report_path=reconstruction.qa_report_path,
        interactive_script_path=interactive.script_path,
        interaction_report_path=interactive.report_path,
    )


def attach_video_3dgs_background(run_dir: str | Path) -> SceneBackground:
    run = Path(run_dir)
    status_path = run / "video" / "3dgs_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"3DGS status not found: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError(f"3DGS status must be completed, got {status.get('status')!r}")
    outputs = status.get("outputs", {})
    background = SceneBackground(
        source_backend="video_3dgs_splatfacto",
        bg_only_image_path=_first_existing_bg_only_frame(run),
        foreground_mask_path="background/foreground_mask.png",
        point_cloud_path="video/moge_reference/reference/pointcloud.ply",
        status="trained_video_3dgs",
        gaussian_splat_path=str(outputs.get("latest_run_dir") or outputs.get("output_dir") or "video/3dgs"),
        gaussian_splat_config_path=str(outputs.get("latest_config")) if outputs.get("latest_config") else None,
        gaussian_splat_checkpoint_path=str(outputs.get("latest_checkpoint")) if outputs.get("latest_checkpoint") else None,
    )
    registration = write_background_registration(run)
    _merge_manifest_background(run, background, registration)
    _merge_qa_background(run, background, registration)
    return background


def _merge_manifest_background(run: Path, background: SceneBackground, registration: BackgroundRegistrationResult) -> None:
    manifest_path = run / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    background_data = background.to_dict()
    background_data["registration_path"] = str(registration.path.relative_to(run))
    background_data["registration_status"] = registration.data["status"]
    background_data["source_kind"] = registration.data["source_kind"]
    manifest["background"] = background_data
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _merge_qa_background(run: Path, background: SceneBackground, registration: BackgroundRegistrationResult) -> None:
    qa_path = run / "qa" / "qa_report.json"
    if not qa_path.is_file():
        return
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    background_data = background.to_dict()
    background_data["registration_path"] = str(registration.path.relative_to(run))
    background_data["registration_status"] = registration.data["status"]
    background_data["source_kind"] = registration.data["source_kind"]
    qa["background"] = background_data
    qa["background_registration"] = {
        "path": str(registration.path.relative_to(run)),
        "status": registration.data["status"],
        "source_kind": registration.data["source_kind"],
        "scale_source": registration.data["scale_source"],
        "registrations": registration.data.get("registrations", {}),
        "gaussian_splat": registration.data.get("gaussian_splat", {}),
    }
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")


def _first_existing_bg_only_frame(run: Path) -> str:
    frames = sorted((run / "video" / "bg_only" / "frames").glob("*.png"))
    if frames:
        return str(frames[0].relative_to(run))
    return "background/bg_only.png"


def _focal_from_fov(size_px: int, fov_deg: float) -> float:
    return float(size_px) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
