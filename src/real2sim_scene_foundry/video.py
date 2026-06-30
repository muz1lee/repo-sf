"""RGB video preparation for the SimFoundry-style background path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


@dataclass(frozen=True)
class VideoPrepResult:
    out_dir: Path
    manifest_path: Path
    frames_dir: Path
    reference_path: Path


def prepare_rgb_video(
    *,
    video_path: str | Path,
    out_dir: str | Path,
    frame_stride: int = 10,
    max_frames: int | None = None,
    reference_frame_index: int = 0,
) -> VideoPrepResult:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    if reference_frame_index < 0:
        raise ValueError("reference_frame_index must be non-negative")

    video = Path(video_path)
    out = Path(out_dir)
    video_dir = out / "video"
    frames_dir = video_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    reference_path = video_dir / "reference.png"

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    sampled_frames: list[dict[str, object]] = []
    total_frames = 0
    reference_written = False
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index = total_frames
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_index == reference_frame_index:
                Image.fromarray(frame_rgb).save(reference_path)
                reference_written = True
            if frame_index % frame_stride == 0 and (max_frames is None or len(sampled_frames) < max_frames):
                rel_path = Path("video") / "frames" / f"frame_{frame_index:06d}.png"
                Image.fromarray(frame_rgb).save(out / rel_path)
                sampled_frames.append(
                    {
                        "frame_index": int(frame_index),
                        "time_s": float(frame_index / fps) if fps > 0.0 else None,
                        "image_path": rel_path.as_posix(),
                    }
                )
    finally:
        capture.release()

    if total_frames == 0:
        raise ValueError(f"video has no readable frames: {video}")
    if not reference_written:
        raise ValueError(f"reference_frame_index {reference_frame_index} was not present in {video}")
    if not sampled_frames:
        raise ValueError("no frames were sampled from the video")

    manifest = {
        "version": 1,
        "stage": "video_prep",
        "source_type": "rgb_video",
        "input_video": str(video),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frame_count": int(total_frames),
        "sampled_frame_count": len(sampled_frames),
        "frame_stride": int(frame_stride),
        "frames": sampled_frames,
        "reference_frame": {
            "frame_index": int(reference_frame_index),
            "image_path": "video/reference.png",
        },
        "camera_poses": {
            "status": "not_estimated",
            "required_for": "background_3dgs",
            "recommended_next_step": "Run COLMAP or another RGB-SfM/SLAM estimator on video/frames.",
        },
        "foreground_removal": {
            "status": "not_started",
            "recommended_next_step": "Run Qwen/SAM on sampled frames and inpaint masks to create a BG-only frame set.",
        },
        "background_3dgs": {
            "status": "not_started_requires_camera_poses",
            "recommended_next_step": "Fit a per-scene 3DGS after BG-only frames and camera poses are available.",
        },
    }
    manifest_path = video_dir / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return VideoPrepResult(out_dir=out, manifest_path=manifest_path, frames_dir=frames_dir, reference_path=reference_path)
