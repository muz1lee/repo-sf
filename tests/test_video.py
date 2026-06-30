import json

import cv2
import numpy as np
from PIL import Image

from real2sim_scene_foundry.video import prepare_rgb_video


def _write_test_video(path, *, size=(8, 6), frame_count=4, fps=5.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((size[1], size[0], 3), idx * 40, dtype=np.uint8)
        frame[:, :, 1] = 255 - idx * 20
        writer.write(frame)
    writer.release()


def test_prepare_rgb_video_writes_frames_reference_and_manifest(tmp_path):
    video = tmp_path / "phone_capture.avi"
    _write_test_video(video)

    result = prepare_rgb_video(
        video_path=video,
        out_dir=tmp_path / "run",
        frame_stride=2,
        max_frames=2,
        reference_frame_index=1,
    )

    assert result.manifest_path == tmp_path / "run" / "video" / "video_manifest.json"
    assert (tmp_path / "run" / "video" / "frames" / "frame_000000.png").is_file()
    assert (tmp_path / "run" / "video" / "frames" / "frame_000002.png").is_file()
    assert (tmp_path / "run" / "video" / "reference.png").is_file()
    assert Image.open(tmp_path / "run" / "video" / "reference.png").size == (8, 6)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "video_prep"
    assert manifest["source_type"] == "rgb_video"
    assert manifest["input_video"] == str(video)
    assert manifest["total_frame_count"] == 4
    assert manifest["sampled_frame_count"] == 2
    assert manifest["reference_frame"]["frame_index"] == 1
    assert manifest["camera_poses"]["status"] == "not_estimated"
    assert manifest["background_3dgs"]["status"] == "not_started_requires_camera_poses"
