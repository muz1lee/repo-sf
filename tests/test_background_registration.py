import json

import numpy as np


def test_write_background_registration_records_bg_only_transform(tmp_path):
    from real2sim_scene_foundry.background_registration import write_background_registration

    run = tmp_path / "run"
    (run / "background").mkdir(parents=True)
    (run / "background" / "bg_only_cloud.ply").write_text("ply\n", encoding="utf-8")
    (run / "scene_manifest.json").write_text(
        json.dumps({"support_plane": {"original_height_world_m": 0.12}}),
        encoding="utf-8",
    )

    result = write_background_registration(
        run,
        source_kind="bg_only_cloud",
        status="registered_bg_only_cloud",
        scale_source="input_metric_depth",
    )

    data = json.loads(result.path.read_text(encoding="utf-8"))
    assert data["source_kind"] == "bg_only_cloud"
    assert data["status"] == "registered_bg_only_cloud"
    assert data["scale_source"] == "input_metric_depth"
    assert data["transforms"]["T_bg_only_cloud_to_sim_world"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, -0.12],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_write_background_registration_registers_bg_only_even_when_3dgs_is_sidecar(tmp_path):
    from real2sim_scene_foundry.background_registration import write_background_registration

    run = tmp_path / "run"
    (run / "background").mkdir(parents=True)
    (run / "background" / "bg_only_cloud.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps({"status": "completed", "outputs": {"latest_config": "video/3dgs/run/config.yml"}}),
        encoding="utf-8",
    )

    result = write_background_registration(run, source_kind="bg_only_cloud")

    data = json.loads(result.path.read_text(encoding="utf-8"))
    assert data["status"] == "partial_external_render_only"
    assert data["scale_source"] == "input_metric_depth"
    assert data["T_3dgs_world_to_sim_world"] is None
    assert data["transforms"]["T_3dgs_world_to_sim_world"] is None
    assert (
        data["transform_sources"]["T_3dgs_world_to_sim_world"]
        == "blocked_missing_camera_pose_scale_evidence"
    )
    assert data["registrations"]["bg_only_cloud"]["status"] == "registered"
    assert data["registrations"]["3dgs"]["status"] == "blocked_missing_camera_pose_scale_evidence"
    assert data["gaussian_splat"]["status"] == "sidecar_not_native"
    assert data["gaussian_splat"]["registration_status"] == "blocked_missing_camera_pose_scale_evidence"
    assert data["gaussian_splat"]["config_path"] == "video/3dgs/run/config.yml"


def test_write_background_registration_marks_native_3dgs_asset_candidate_as_blocked(tmp_path):
    from real2sim_scene_foundry.background_registration import write_background_registration

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "scene_cloud.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps({"status": "completed", "outputs": {"latest_config": "video/3dgs/run/config.yml"}}),
        encoding="utf-8",
    )

    result = write_background_registration(run)

    data = json.loads(result.path.read_text(encoding="utf-8"))
    assert data["source_kind"] == "3dgs_native_asset_candidate"
    assert data["status"] == "blocked_native_3dgs_not_integrated"
    assert data["status"] != "passed"
    assert data["assets"]["3dgs_native_asset_candidate"] == "background/3dgs_native/splat_rgb.ply"
    assert data["assets"]["full_scene_cloud_debug"] == "scene_cloud.ply"
    assert data["gaussian_splat"]["status"] == "native_asset_candidate"
    assert data["gaussian_splat"]["native_rendering"] is False


def test_write_background_registration_blocks_registered_3dgs_without_transform_evidence(tmp_path):
    from real2sim_scene_foundry.background_registration import write_background_registration

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps({"status": "completed", "outputs": {"latest_config": "video/3dgs/run/config.yml"}}),
        encoding="utf-8",
    )

    result = write_background_registration(run, source_kind="registered_3dgs")

    data = json.loads(result.path.read_text(encoding="utf-8"))
    assert data["source_kind"] == "registered_3dgs"
    assert data["status"] == "blocked_missing_3dgs_transform_evidence"
    assert data["T_3dgs_world_to_sim_world"] is None
    assert data["transforms"]["T_3dgs_world_to_sim_world"] is None
    assert data["registrations"]["3dgs"]["status"] == "blocked_missing_camera_pose_scale_evidence"
    assert data["gaussian_splat"]["native_rendering"] is False


def test_register_3dgs_background_blocks_when_anchor_pose_evidence_is_missing(tmp_path):
    from real2sim_scene_foundry.background_registration import register_3dgs_background

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    result = register_3dgs_background(run, method="camera-sim3", write=True)

    assert result.data["status"] == "blocked_missing_camera_pose_scale_evidence"
    assert result.data["T_3dgs_world_to_sim_world"] is None
    assert "missing_3dgs_anchor_camera_pose" in result.data["blocking_reasons"]
    assert result.path == run / "background" / "registration.json"


def test_register_3dgs_background_writes_camera_bridge_transform_and_qa(tmp_path):
    from real2sim_scene_foundry.background_registration import qa_background_registration, register_3dgs_background

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    t_gs_cam = [[1, 0, 0, 0.5], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]]
    t_sim_cam = [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]]
    (run / "background" / "3dgs_camera_pose.json").write_text(
        json.dumps({"anchor_frame": 7, "T_3dgs_camera_to_world": t_gs_cam}),
        encoding="utf-8",
    )
    (run / "camera.json").write_text(json.dumps({"T_camera_to_world": t_sim_cam}), encoding="utf-8")

    result = register_3dgs_background(run, method="camera-sim3", write=True)
    qa = qa_background_registration(run, heldout_frames=16)

    expected = np.asarray(t_sim_cam) @ np.linalg.inv(np.asarray(t_gs_cam))
    assert result.data["status"] == "registered"
    assert result.data["method"] == "camera-sim3"
    assert result.data["anchor_frame"] == 7
    assert result.data["scale"] == 1.0
    assert np.allclose(np.asarray(result.data["T_3dgs_world_to_sim_world"]), expected)
    assert result.data["transform_sources"]["T_3dgs_world_to_sim_world"] == "camera_anchor_bridge"
    assert result.data["registrations"]["3dgs"]["status"] == "registered"
    assert qa["status"] == "passed"
    assert qa["heldout_frames_requested"] == 16
    assert (run / "qa" / "background_registration_report.json").is_file()
    assert (run / "qa" / "background_registration_overlay.png").is_file()
