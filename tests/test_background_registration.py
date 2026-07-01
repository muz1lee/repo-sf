import json


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
