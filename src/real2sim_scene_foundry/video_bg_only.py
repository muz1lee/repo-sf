"""BG-only frame generation for the SimFoundry video background path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from .background import ArrayInpaintClient, OpenCVInpaintClient
from .proposals import ObjectProposal


class ProposalClient(Protocol):
    def propose(self, image_path: str | Path, *, target_labels: Sequence[str] | None = None) -> list[ObjectProposal]:
        ...


class MaskClient(Protocol):
    def segment_text(self, image_path: str | Path, text_prompt: str) -> np.ndarray:
        ...

    def segment_box(self, image_path: str | Path, box_xyxy: Sequence[int]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class VideoBgOnlyResult:
    run_dir: Path
    status_path: Path
    frames_dir: Path
    masks_dir: Path
    proposals_dir: Path


def create_video_bg_only_artifacts(
    *,
    run_dir: str | Path,
    proposal_client: ProposalClient,
    mask_client: MaskClient,
    inpaint_client: ArrayInpaintClient | None = None,
    target_labels: Sequence[str] | None = None,
    max_frames: int | None = None,
    dilation_px: int = 10,
    reuse_last_mask_on_failure: bool = True,
) -> VideoBgOnlyResult:
    run = Path(run_dir)
    video_dir = run / "video"
    source_frames = sorted((video_dir / "frames").glob("*.png"))
    if not source_frames:
        raise ValueError(f"no sampled video frames found in {video_dir / 'frames'}")
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        source_frames = source_frames[:max_frames]

    bg_dir = video_dir / "bg_only"
    frames_dir = bg_dir / "frames"
    masks_dir = bg_dir / "masks"
    proposals_dir = bg_dir / "proposals"
    for directory in (frames_dir, masks_dir, proposals_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _clear_previous_outputs(frames_dir, masks_dir, proposals_dir)

    client = inpaint_client or OpenCVInpaintClient()
    requested_labels = [str(label) for label in target_labels or [] if str(label).strip()]
    active_labels = requested_labels
    observed_labels: list[str] = []
    frame_reports: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    last_mask: np.ndarray | None = None

    for frame_path in source_frames:
        rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
        report: dict[str, object] = {
            "frame": frame_path.name,
            "image_path": str(frame_path.relative_to(run)),
        }
        try:
            proposals = proposal_client.propose(frame_path, target_labels=active_labels or None)
            if not proposals:
                raise RuntimeError("proposal client returned no objects")
            for proposal in proposals:
                if proposal.label not in observed_labels:
                    observed_labels.append(proposal.label)
            raw_mask = _union_masks(frame_path, proposals, mask_client, rgb.shape[:2])
            mask = _dilate_mask(raw_mask, dilation_px=dilation_px)
            report["proposal_count"] = len(proposals)
            report["mask_area_px"] = int(np.count_nonzero(mask))
            report["reused_previous_mask"] = False
        except Exception as exc:  # noqa: BLE001 - video stage records per-frame service failures.
            if last_mask is None or not reuse_last_mask_on_failure:
                report["error"] = str(exc)
                failures.append(report)
                raise RuntimeError(f"{frame_path.name}: BG-only frame generation failed: {exc}") from exc
            mask = last_mask
            proposals = []
            report["proposal_count"] = 0
            report["mask_area_px"] = int(np.count_nonzero(mask))
            report["reused_previous_mask"] = True
            report["error"] = str(exc)
            failures.append(dict(report))

        bg_only = client.inpaint(rgb, mask)
        if bg_only.shape != rgb.shape:
            raise ValueError(f"inpaint result must have shape {rgb.shape}, got {bg_only.shape}")
        out_frame = frames_dir / frame_path.name
        out_mask = masks_dir / f"{frame_path.stem}_mask.png"
        out_proposals = proposals_dir / f"{frame_path.stem}.json"
        Image.fromarray(np.asarray(bg_only, dtype=np.uint8)).save(out_frame)
        Image.fromarray(mask.astype(np.uint8) * 255).save(out_mask)
        out_proposals.write_text(
            json.dumps([proposal.to_dict() for proposal in proposals], indent=2),
            encoding="utf-8",
        )
        report["bg_only_path"] = str(out_frame.relative_to(run))
        report["mask_path"] = str(out_mask.relative_to(run))
        report["proposals_path"] = str(out_proposals.relative_to(run))
        frame_reports.append(report)
        last_mask = mask

    status = {
        "version": 1,
        "stage": "video_bg_only",
        "status": "completed",
        "run_dir": str(run),
        "frame_count": len(frame_reports),
        "target_labels": requested_labels,
        "observed_labels": observed_labels,
        "source_frame_dir": "video/frames",
        "inpaint_backend": client.backend_name,
        "dilation_px": int(dilation_px),
        "failures": failures,
        "frames": frame_reports,
        "outputs": {
            "frames_dir": "video/bg_only/frames",
            "masks_dir": "video/bg_only/masks",
            "proposals_dir": "video/bg_only/proposals",
        },
    }
    status_path = video_dir / "bg_only_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    _update_video_manifest(video_dir / "video_manifest.json")
    return VideoBgOnlyResult(
        run_dir=run,
        status_path=status_path,
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        proposals_dir=proposals_dir,
    )


def _union_masks(
    image_path: Path,
    proposals: list[ObjectProposal],
    mask_client: MaskClient,
    shape: tuple[int, int],
) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for proposal in proposals:
        if proposal.bbox_xyxy is not None:
            mask = mask_client.segment_box(image_path, proposal.bbox_xyxy)
        else:
            mask = mask_client.segment_text(image_path, proposal.label)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"{proposal.object_id}: mask shape {mask.shape} does not match frame shape {shape}")
        union |= _select_prompt_component(mask, bbox_xyxy=proposal.bbox_xyxy, point_xy=proposal.point_xy)
    return union


def _select_prompt_component(
    mask: np.ndarray,
    *,
    bbox_xyxy: tuple[int, int, int, int] | None,
    point_xy: tuple[int, int] | None,
) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if count <= 2:
        return mask.astype(bool)
    selected_label = None
    if point_xy is not None:
        x, y = point_xy
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and labels[y, x] > 0:
            selected_label = int(labels[y, x])
    if selected_label is None and bbox_xyxy is not None:
        x0, y0, x1, y1 = bbox_xyxy
        region = labels[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        values, counts = np.unique(region[region > 0], return_counts=True)
        if len(values):
            selected_label = int(values[np.argmax(counts)])
    if selected_label is None:
        areas = np.bincount(labels.reshape(-1))
        areas[0] = 0
        selected_label = int(np.argmax(areas))
    return labels == selected_label


def _dilate_mask(mask: np.ndarray, *, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((int(dilation_px) * 2 + 1, int(dilation_px) * 2 + 1), dtype=np.uint8)
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def _clear_previous_outputs(frames_dir: Path, masks_dir: Path, proposals_dir: Path) -> None:
    for directory, pattern in ((frames_dir, "*.png"), (masks_dir, "*.png"), (proposals_dir, "*.json")):
        for path in directory.glob(pattern):
            path.unlink()


def _update_video_manifest(path: Path) -> None:
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["foreground_removal"] = {
        "status": "completed",
        "bg_only_status": "video/bg_only_status.json",
        "bg_only_frame_dir": "video/bg_only/frames",
        "mask_dir": "video/bg_only/masks",
    }
    manifest["background_3dgs"] = {
        "status": "ready_for_training",
        "required_inputs": {
            "bg_only_frames": "video/bg_only/frames",
            "camera_poses": "video/colmap/sparse/0",
            "dense_reference": "video/moge_reference/reference/pointcloud.ply",
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
