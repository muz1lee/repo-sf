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
    assert qa["visual_registration_status"] == "not_checked"
    assert qa["overlay_status"] == "placeholder_not_visual_evidence"
    assert qa["heldout_frames_requested"] == 16
    assert (run / "qa" / "background_registration_report.json").is_file()
    assert qa["overlay_path"] is None
    assert not (run / "qa" / "background_registration_overlay.png").is_file()


def test_register_3dgs_background_closes_arkit_world_with_phone_alignment_bridge(tmp_path):
    from real2sim_scene_foundry.background_registration import qa_background_registration, register_3dgs_background

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "inputs": {"pose_world": "arkit"},
                "claim": {"sim_world_registered": False, "identity_transform_allowed": False},
            }
        ),
        encoding="utf-8",
    )
    t_arkit_to_sim = np.asarray(
        [[1.0, 0.0, 0.0, 0.25], [0.0, 0.0, 1.0, 0.5], [0.0, -1.0, 0.0, 1.25], [0.0, 0.0, 0.0, 1.0]]
    )
    t_gs_cam = np.asarray([[1, 0, 0, 0.5], [0, 1, 0, 0.1], [0, 0, 1, 0.2], [0, 0, 0, 1]], dtype=float)
    t_sim_cam = t_arkit_to_sim @ t_gs_cam
    (run / "background" / "3dgs_camera_pose.json").write_text(
        json.dumps(
            {
                "anchor_frame": 3,
                "coordinate_convention": "arkit_world_camera_to_world",
                "T_3dgs_camera_to_world": t_gs_cam.tolist(),
            }
        ),
        encoding="utf-8",
    )
    (run / "camera.json").write_text(json.dumps({"T_camera_to_world": t_sim_cam.tolist()}), encoding="utf-8")
    (run / "background" / "phone_sim_alignment.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "source_backend": "arkit_depth_ransac_plane",
                "pose_world_before": "arkit",
                "pose_world_after": "sim",
                "T_arkit_world_to_sim_world": t_arkit_to_sim.tolist(),
                "metrics": {"plane_residual_median_m": 0.004, "plane_residual_p90_m": 0.015},
            }
        ),
        encoding="utf-8",
    )

    result = register_3dgs_background(run, method="camera-sim3", write=True)
    qa = qa_background_registration(run, heldout_frames=8)

    assert result.data["status"] == "registered"
    assert np.allclose(np.asarray(result.data["T_3dgs_world_to_sim_world"]), t_arkit_to_sim)
    assert result.data["transform_sources"]["T_3dgs_world_to_sim_world"] == "phone_sim_alignment_arkit_to_sim_world"
    assert result.data["registrations"]["3dgs"]["transform_source"] == "phone_sim_alignment_arkit_to_sim_world"
    assert result.data["scale_source"] == "arkit_sceneDepth_meters"
    assert result.data["coordinate_bridge"]["status"] == "closed"
    assert result.data["coordinate_bridge"]["source_world"] == "arkit"
    assert result.data["coordinate_bridge"]["target_world"] == "sim_world"
    assert result.data["coordinate_bridge"]["phone_sim_alignment_path"] == "background/phone_sim_alignment.json"
    assert result.data["metrics"]["phone_alignment_plane_residual_median_m"] == 0.004
    assert qa["status"] == "passed"
    assert qa["coordinate_bridge_status"] == "closed"
    assert qa["visual_registration_status"] == "not_checked"


def test_register_3dgs_background_blocks_arkit_world_without_phone_alignment_bridge(tmp_path):
    from real2sim_scene_foundry.background_registration import qa_background_registration, register_3dgs_background

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps({"status": "completed", "inputs": {"pose_world": "arkit"}}),
        encoding="utf-8",
    )
    t_gs_cam = [[1, 0, 0, 0.5], [0, 1, 0, 0.1], [0, 0, 1, 0.2], [0, 0, 0, 1]]
    t_sim_cam = [[1, 0, 0, 0.75], [0, 0, 1, 0.7], [0, -1, 0, 1.05], [0, 0, 0, 1]]
    (run / "background" / "3dgs_camera_pose.json").write_text(
        json.dumps(
            {
                "anchor_frame": 3,
                "coordinate_convention": "arkit_world_camera_to_world",
                "T_3dgs_camera_to_world": t_gs_cam,
            }
        ),
        encoding="utf-8",
    )
    (run / "camera.json").write_text(json.dumps({"T_camera_to_world": t_sim_cam}), encoding="utf-8")

    result = register_3dgs_background(run, method="camera-sim3", write=True)
    qa = qa_background_registration(run, heldout_frames=8)

    assert result.data["status"] == "blocked_missing_arkit_to_sim_bridge"
    assert result.data["T_3dgs_world_to_sim_world"] is None
    assert "missing_phone_sim_alignment_arkit_to_sim_world" in result.data["blocking_reasons"]
    assert result.data["coordinate_bridge"]["status"] == "blocked"
    assert result.data["registrations"]["3dgs"]["transform_source"] == "blocked_missing_arkit_to_sim_bridge"
    assert qa["status"] == "blocked"
    assert "background_unregistered" in qa["blocking_reasons"]


def test_qa_background_registration_blocks_arkit_camera_anchor_bridge_without_closed_coordinate_bridge(tmp_path):
    from real2sim_scene_foundry.background_registration import qa_background_registration

    run = tmp_path / "run"
    (run / "background").mkdir(parents=True)
    (run / "background" / "registration.json").write_text(
        json.dumps(
            {
                "status": "registered",
                "source_kind": "registered_3dgs",
                "coordinate_convention": {"3dgs": "arkit_world_camera_to_world", "sim": "z_up_meter_camera_to_world"},
                "T_3dgs_world_to_sim_world": np.eye(4).tolist(),
                "transform_sources": {"T_3dgs_world_to_sim_world": "camera_anchor_bridge"},
            }
        ),
        encoding="utf-8",
    )

    qa = qa_background_registration(run, heldout_frames=8)

    assert qa["status"] == "blocked"
    assert qa["coordinate_bridge_status"] == "blocked"
    assert "arkit_3dgs_coordinate_bridge_not_closed" in qa["blocking_reasons"]


def test_register_3dgs_known_phone_sim_world_allows_identity_with_evidence(tmp_path):
    from real2sim_scene_foundry.background_registration import register_3dgs_background

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "inputs": {"pose_world": "sim"},
                "claim": {"identity_transform_allowed": True},
            }
        ),
        encoding="utf-8",
    )
    (run / "video" / "nerfstudio_phone").mkdir(parents=True)
    (run / "video" / "nerfstudio_phone" / "transforms.json").write_text(
        json.dumps(
            {
                "phone_capture": {
                    "pose_world": "sim",
                    "camera_pose_world": "sim",
                    "identity_3dgs_to_sim_allowed_if_trained_with_pose_world": True,
                    "intrinsics_source": "arkit_explicit",
                    "extrinsics_source": "arkit_explicit",
                    "scale_source": "arkit_sceneDepth_meters",
                },
                "frames": [{"file_path": "images/frame_000000.jpg", "transform_matrix": np.eye(4).tolist()}],
            }
        ),
        encoding="utf-8",
    )

    result = register_3dgs_background(run, method="known-phone-sim-world", write=True)

    assert result.data["status"] == "registered"
    assert result.data["method"] == "known-phone-sim-world"
    assert result.data["identity_allowed"] is True
    assert result.data["T_3dgs_world_to_sim_world"] == np.eye(4).tolist()
    assert result.data["transform_sources"]["T_3dgs_world_to_sim_world"] == "identity_allowed_by_phone_sim_world_training"
    assert result.data["evidence"]["phone_capture_pose_world"] == "sim"
    assert result.data["evidence"]["transforms_path"] == "video/nerfstudio_phone/transforms.json"


def test_register_3dgs_known_phone_sim_world_blocks_identity_when_training_used_arkit_poses(tmp_path):
    from real2sim_scene_foundry.background_registration import register_3dgs_background

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video" / "nerfstudio_phone").mkdir(parents=True)
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "inputs": {"pose_world": "arkit"},
                "claim": {"identity_transform_allowed": False},
            }
        ),
        encoding="utf-8",
    )
    (run / "video" / "nerfstudio_phone" / "transforms.json").write_text(
        json.dumps(
            {
                "phone_capture": {
                    "pose_world": "sim",
                    "camera_pose_world": "sim",
                    "identity_3dgs_to_sim_allowed_if_trained_with_pose_world": True,
                    "intrinsics_source": "arkit_explicit",
                    "extrinsics_source": "arkit_explicit",
                    "scale_source": "arkit_sceneDepth_meters",
                },
                "frames": [{"file_path": "images/frame_000000.jpg", "transform_matrix": np.eye(4).tolist()}],
            }
        ),
        encoding="utf-8",
    )

    result = register_3dgs_background(run, method="known-phone-sim-world", write=True)

    assert result.data["status"] == "blocked_identity_transform_without_phone_sim_world_evidence"
    assert result.data["identity_allowed"] is False
    assert "3dgs_training_pose_world_not_sim" in result.data["blocking_reasons"]
    assert "3dgs_identity_training_claim_missing" in result.data["blocking_reasons"]


def test_register_3dgs_known_phone_sim_world_blocks_identity_without_evidence(tmp_path):
    from real2sim_scene_foundry.background_registration import register_3dgs_background

    run = tmp_path / "run"
    (run / "background" / "3dgs_native").mkdir(parents=True)
    (run / "background" / "3dgs_native" / "splat_rgb.ply").write_text("ply\n", encoding="utf-8")
    (run / "video").mkdir()
    (run / "video" / "3dgs_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    result = register_3dgs_background(run, method="known-phone-sim-world", write=True)

    assert result.data["status"] == "blocked_identity_transform_without_phone_sim_world_evidence"
    assert result.data["identity_allowed"] is False
    assert result.data["T_3dgs_world_to_sim_world"] is None
    assert "missing_phone_sim_world_3dgs_training_evidence" in result.data["blocking_reasons"]
