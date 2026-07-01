import json
from pathlib import Path

import numpy as np
from PIL import Image

from real2sim_scene_foundry.cli import main
from real2sim_scene_foundry.background_registration import register_3dgs_background
from real2sim_scene_foundry.phone_capture import (
    export_nerfstudio_from_phone_capture,
    import_phone_capture,
    validate_phone_capture,
)
from real2sim_scene_foundry.phone_sim_alignment import align_phone_sim_world
from real2sim_scene_foundry.support_plane import estimate_and_apply_support_plane
from real2sim_scene_foundry.table_collision_qa import write_table_collision_projection_qa_v2


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _pose(tx: float) -> list[list[float]]:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = tx
    return transform.tolist()


def _write_phone_bundle(root: Path, *, frame_count: int = 3, baseline_m: float = 0.12) -> None:
    width = 16
    height = 16
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "confidence").mkdir()
    (root / "camera").mkdir()
    (root / "clean_background" / "rgb").mkdir(parents=True)
    poses = []
    for idx in range(frame_count):
        stem = f"frame_{idx:06d}"
        Image.new("RGB", (width, height), color=(idx * 20, 10, 30)).save(root / "rgb" / f"{stem}.jpg")
        Image.new("RGB", (width, height), color=(idx * 20, 40, 60)).save(root / "clean_background" / "rgb" / f"{stem}.jpg")
        np.save(root / "depth" / f"{stem}.npy", np.full((height, width), 1.0 + idx * 0.01, dtype=np.float32))
        Image.fromarray(np.full((height, width), 2, dtype=np.uint8)).save(root / "confidence" / f"{stem}.png")
        poses.append(
            {
                "frame_id": stem,
                "rgb_path": f"rgb/{stem}.jpg",
                "depth_path": f"depth/{stem}.npy",
                "confidence_path": f"confidence/{stem}.png",
                "T_camera_to_world": _pose(baseline_m * idx / max(1, frame_count - 1)),
                "tracking_state": "normal",
            }
        )
    (root / "camera" / "intrinsics.json").write_text(
        json.dumps({"width": width, "height": height, "fx": 80.0, "fy": 80.0, "cx": 7.5, "cy": 7.5}),
        encoding="utf-8",
    )
    (root / "camera" / "poses.json").write_text(
        json.dumps({"coordinate_frame": "arkit_world", "poses": poses}),
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps({"capture_kind": "phone_capture_bundle", "device": "iPhone 17 Pro", "depth_unit": "meter"}),
        encoding="utf-8",
    )


def test_validate_phone_capture_accepts_explicit_arkit_rgbd_bundle(tmp_path):
    capture = tmp_path / "capture"
    _write_phone_bundle(capture)

    report = validate_phone_capture(capture)

    assert report["status"] == "passed"
    assert report["frame_count"] == 3
    assert report["intrinsics_source"] == "arkit_explicit"
    assert report["extrinsics_source"] == "arkit_explicit"
    assert report["scale_source"] == "arkit_sceneDepth_meters"
    assert report["trajectory_baseline_m"] >= 0.1
    assert report["confidence_coverage"] == 1.0
    assert (capture / "capture_contract.json").is_file()


def test_validate_phone_capture_rejects_plain_video_file(tmp_path):
    video = tmp_path / "plain.mp4"
    video.write_bytes(b"not a capture bundle")

    report = validate_phone_capture(video)

    assert report["status"] == "blocked"
    assert "phone_capture_bundle_must_be_directory" in report["blocking_reasons"]
    assert "ordinary_video_is_not_rgbd_capture" in report["blocking_reasons"]


def test_import_phone_capture_writes_explicit_camera_and_trajectory(tmp_path):
    capture = tmp_path / "capture"
    run = tmp_path / "run"
    _write_phone_bundle(capture)

    result = import_phone_capture(capture, run)

    camera = json.loads((run / "camera.json").read_text(encoding="utf-8"))
    trajectory = json.loads((run / "trajectory.json").read_text(encoding="utf-8"))
    assert result["status"] == "imported"
    assert camera["intrinsics_source"] == "arkit_explicit"
    assert camera["extrinsics_source"] == "arkit_explicit"
    assert camera["scale_source"] == "arkit_sceneDepth_meters"
    assert camera["T_camera_to_world"] == IDENTITY
    assert trajectory["pose_world"] == "arkit"
    assert len(trajectory["frames"]) == 3
    assert (run / "frames" / "frame_000000.jpg").is_file()
    assert (run / "depth" / "frame_000000.npy").is_file()
    assert (run / "confidence" / "frame_000000.png").is_file()
    assert (run / "capture_contract.json").is_file()


def test_export_nerfstudio_from_phone_capture_blocks_sim_world_without_alignment(tmp_path):
    capture = tmp_path / "capture"
    run = tmp_path / "run"
    _write_phone_bundle(capture)
    import_phone_capture(capture, run)

    report = export_nerfstudio_from_phone_capture(run, pose_world="sim")

    assert report["status"] == "blocked"
    assert "phone_sim_world_alignment_missing" in report["blocking_reasons"]
    assert not (run / "video" / "nerfstudio_phone" / "transforms.json").is_file()


def test_align_phone_sim_world_fits_depth_plane_exports_sim_poses_and_registers_3dgs(tmp_path):
    capture = tmp_path / "capture"
    run = tmp_path / "run"
    _write_phone_bundle(capture, frame_count=4, baseline_m=0.15)
    import_phone_capture(capture, run)
    for depth_path in (run / "depth").glob("*.npy"):
        np.save(depth_path, np.full((16, 16), 1.0, dtype=np.float32))
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir(exist_ok=True)
    (run / "video" / "3dgs_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    alignment = align_phone_sim_world(run, force=True)
    export_report = export_nerfstudio_from_phone_capture(run, pose_world="sim")
    registration = register_3dgs_background(run, method="camera-sim3", write=True)

    trajectory = json.loads((run / "trajectory.json").read_text(encoding="utf-8"))
    transforms = json.loads((run / "video" / "nerfstudio_phone" / "transforms.json").read_text(encoding="utf-8"))
    table_polygon = json.loads((run / "background" / "table_polygon_world.json").read_text(encoding="utf-8"))
    t_arkit_to_sim = np.asarray(alignment["T_arkit_world_to_sim_world"], dtype=float)
    assert alignment["status"] == "passed"
    assert alignment["source_backend"] == "arkit_depth_ransac_plane"
    assert trajectory["pose_world"] == "sim"
    assert export_report["status"] == "exported"
    assert transforms["phone_capture"]["pose_world"] == "sim"
    assert transforms["phone_capture"]["identity_3dgs_to_sim_allowed_if_trained_with_pose_world"] is True
    assert registration.data["status"] == "registered"
    assert registration.data["method"] == "camera-sim3"
    assert np.allclose(np.asarray(registration.data["T_3dgs_world_to_sim_world"], dtype=float), t_arkit_to_sim)
    assert table_polygon["source_backend"] == "arkit_depth_ransac_plane"
    assert max(abs(point[2]) for point in alignment["qa_samples_sim_xyz"]) < 1e-5


def test_phone_capture_cli_commands(tmp_path):
    capture = tmp_path / "capture"
    run = tmp_path / "run"
    _write_phone_bundle(capture)

    assert main(["validate-phone-capture", "--capture-dir", str(capture)]) == 0
    assert main(["import-phone-capture", "--capture-dir", str(capture), "--run-dir", str(run)]) == 0
    assert main(["export-nerfstudio-from-phone-capture", "--run-dir", str(run), "--pose-world", "arkit"]) == 0
    assert (capture / "capture_contract.json").is_file()
    assert (run / "camera.json").is_file()
    assert (run / "video" / "nerfstudio_phone" / "transforms.json").is_file()


def test_support_plane_uses_phone_depth_point_cloud_and_strict_table_qa(tmp_path):
    import trimesh

    capture = tmp_path / "capture"
    run = tmp_path / "run"
    _write_phone_bundle(capture)
    import_phone_capture(capture, run)
    object_dir = run / "objects" / "cup"
    object_dir.mkdir(parents=True)
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(object_dir / "visual.glb")
    (run / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/visual.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.1], [0, 0, 0, 1]],
                        "T_object_to_camera": IDENTITY,
                        "scale_m": 0.1,
                        "mass_kg": 0.2,
                        "friction": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    support = estimate_and_apply_support_plane(run, force=True)
    qa = write_table_collision_projection_qa_v2(run).report

    assert support["source_backend"] == "arkit_depth_point_cloud_plane"
    assert support["table_collision_mesh_path"] == "table/collision_polygon_slab.glb"
    assert support["support_surface_status"] == "passed"
    assert (run / "table" / "collision_polygon_slab.glb").is_file()
    assert qa["camera_intrinsics_source"] == "explicit"
    assert qa["camera_extrinsics_source"] == "explicit"
    assert qa["status"] == "passed"
