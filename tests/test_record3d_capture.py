import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from real2sim_scene_foundry.cli import main
from real2sim_scene_foundry.record3d_capture import convert_record3d_export


os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402


def _write_record3d_export(root: Path, *, frame_count: int = 3) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    poses = []
    intrinsics = []
    timestamps = []
    for idx in range(frame_count):
        Image.new("RGB", (8, 6), color=(idx * 40, 30, 50)).save(root / "rgb" / f"{idx}.jpg")
        depth = np.zeros((3, 4, 3), dtype=np.float32)
        depth[..., 2] = 1.0 + idx * 0.01
        assert cv2.imwrite(str(root / "depth" / f"{idx}.exr"), depth)
        poses.append([0.0, 0.0, 0.0, 1.0, 0.06 * idx, 0.0, 0.0])
        intrinsics.append([8.0, 8.0, 4.0, 3.0])
        timestamps.append(float(idx) / 30.0)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "w": 8,
                "h": 6,
                "dw": 4,
                "dh": 3,
                "K": [8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 4.0, 3.0, 1.0],
                "poses": poses,
                "perFrameIntrinsicCoeffs": intrinsics,
                "frameTimestamps": timestamps,
                "fps": 30,
                "cameraType": 1,
            }
        ),
        encoding="utf-8",
    )


def test_convert_record3d_export_writes_valid_phone_capture_bundle(tmp_path):
    record3d = tmp_path / "record3d"
    bundle = tmp_path / "bundle"
    _write_record3d_export(record3d)

    report = convert_record3d_export(record3d, bundle)

    assert report["status"] == "converted"
    assert report["frame_count"] == 3
    assert report["capture_contract"]["status"] == "passed"
    intrinsics = json.loads((bundle / "camera" / "intrinsics.json").read_text(encoding="utf-8"))
    assert intrinsics["width"] == 4
    assert intrinsics["height"] == 3
    assert intrinsics["fx"] == 4.0
    assert intrinsics["fy"] == 4.0
    assert intrinsics["cx"] == 2.0
    assert intrinsics["cy"] == 1.5
    depth = np.load(bundle / "depth" / "frame_000000.npy")
    assert depth.shape == (3, 4)
    assert float(depth[0, 0]) == 1.0
    confidence = np.asarray(Image.open(bundle / "confidence" / "frame_000000.png").convert("L"))
    assert confidence.shape == (3, 4)
    assert np.all(confidence == 2)
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_format"] == "record3d_exr_jpg_sequence"
    assert metadata["confidence_source"] == "depth_validity_derived"
    poses = json.loads((bundle / "camera" / "poses.json").read_text(encoding="utf-8"))
    assert poses["coordinate_frame"] == "arkit_world"
    assert poses["poses"][2]["T_camera_to_world"][0][3] == 0.12


def test_convert_record3d_export_supports_stride_and_max_frames(tmp_path):
    record3d = tmp_path / "record3d"
    bundle = tmp_path / "bundle"
    _write_record3d_export(record3d, frame_count=6)

    report = convert_record3d_export(record3d, bundle, frame_stride=2, max_frames=2)

    assert report["status"] == "converted"
    assert report["frame_count"] == 2
    poses = json.loads((bundle / "camera" / "poses.json").read_text(encoding="utf-8"))
    assert [item["source_frame_index"] for item in poses["poses"]] == [0, 2]
    assert sorted(path.name for path in (bundle / "rgb").glob("*.jpg")) == ["frame_000000.jpg", "frame_000001.jpg"]


def test_import_record3d_cli_command(tmp_path):
    record3d = tmp_path / "record3d"
    bundle = tmp_path / "bundle"
    _write_record3d_export(record3d)

    assert main(["import-record3d", "--record3d-dir", str(record3d), "--out", str(bundle)]) == 0
    assert (bundle / "capture_contract.json").is_file()
    assert (bundle / "record3d_import_report.json").is_file()
