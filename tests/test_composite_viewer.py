import json

import trimesh

from real2sim_scene_foundry.cli import main
from real2sim_scene_foundry.composite_viewer import export_composite_viewer


def _ply_text(rgb_line):
    return (
        "\\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                rgb_line,
            ]
        )
        + "\\n"
    )

def _identity4():
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _write_run(run_dir, *, include_bg_only_cloud=True, include_3dgs=True, include_registration=True):
    object_dir = run_dir / "objects" / "cup"
    object_dir.mkdir(parents=True)
    table_dir = run_dir / "background"
    table_dir.mkdir()
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(object_dir / "mesh_aligned.glb")
    trimesh.creation.uv_sphere(radius=0.05).export(object_dir / "visual.glb")
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(object_dir / "collision.glb")
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(object_dir / "debug_bbox.glb")
    trimesh.creation.box(extents=(1.0, 0.8, 0.04)).export(table_dir / "table_collision.glb")
    (run_dir / "scene_cloud.ply").write_text(_ply_text("0 0 1 120 130 140"), encoding="utf-8")
    if include_bg_only_cloud:
        (table_dir / "bg_only_cloud.ply").write_text(_ply_text("0 0 1 10 20 30"), encoding="utf-8")
    if include_registration:
        (table_dir / "registration.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "source_kind": "bg_only_cloud",
                    "status": "blocked_3dgs_world_to_sim_world_not_estimated",
                    "scale_source": "moge_reference_metric_points",
                    "transforms": {
                        "T_bg_only_cloud_to_sim_world": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, -0.12], [0, 0, 0, 1]],
                        "T_3dgs_world_to_sim_world": _identity4(),
                    },
                    "blocked_reason": "Native 3DGS rendering is not integrated into the browser viewer.",
                }
            ),
            encoding="utf-8",
        )
    if include_3dgs:
        (run_dir / "video").mkdir()
        (run_dir / "video" / "3dgs_status.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "outputs": {
                        "latest_run_dir": "video/3dgs/run",
                        "latest_config": "video/3dgs/run/config.yml",
                        "latest_checkpoint": "video/3dgs/run/nerfstudio_models/step.ckpt",
                    },
                }
            ),
            encoding="utf-8",
        )
    (run_dir / "qa").mkdir()
    (run_dir / "qa" / "qa_report.json").write_text(
        json.dumps(
            {
                "physics_settle": {
                    "status": "proxy_checked",
                    "settle_steps": 100,
                    "nan_detected": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "passed",
                "settle_steps": 100,
                "max_penetration_depth_m": 0.001,
                "max_displacement_m": 0.02,
                "fall_out_detected": False,
                "nan_detected": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "support_plane": {
                    "status": "estimated",
                    "source_backend": "background_point_ring_median",
                    "geometry_type": "support_box_proxy",
                    "artifact_role": "collision_proxy",
                    "top_z": 0.0,
                    "thickness": 0.04,
                    "qa_status": "passed",
                    "label": "estimated support box proxy",
                    "original_height_world_m": 0.12,
                    "table_collision_mesh_path": "background/table_collision.glb",
                    "table_collision_pos_world": [0.0, 1.0, -0.02],
                    "table_collision_size_xyz": [1.0, 0.8, 0.04],
                    "table_bounds_world_xy": [[-0.5, 0.6], [0.5, 1.4]],
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/mesh_aligned.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        "T_object_to_world": [[0, -1, 0, 0.1], [1, 0, 0, 1.0], [0, 0, 1, 0.2], [0, 0, 0, 1]],
                        "asset_scale": 0.5,
                        "scale_m": 0.2,
                        "mass_kg": 0.2,
                        "friction": 0.8,
                        "confidence": 0.9,
                    }
                ],
                "background": {
                    "source_backend": "video_3dgs_splatfacto",
                    "status": "trained_video_3dgs",
                    "point_cloud_path": "video/moge_reference/reference/pointcloud.ply",
                    "bg_only_image_path": "background/bg_only.png",
                    "foreground_mask_path": "background/foreground_mask.png",
                    "gaussian_splat_config_path": "video/3dgs/run/config.yml",
                    "registration_path": "background/registration.json",
                },
            }
        ),
        encoding="utf-8",
    )
    if not include_3dgs:
        manifest_path = run_dir / "scene_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["background"].pop("gaussian_splat_config_path", None)
        manifest["background"]["source_backend"] = "opencv_inpaint_single_frame"
        manifest["background"]["status"] = "proxy_from_single_stereo_pair"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_export_composite_viewer_blocks_final_visual_when_external_3dgs_render_is_missing(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run)

    result = export_composite_viewer(run)

    assert result.index_path == run / "exports" / "composite_viewer" / "index.html"
    assert result.config_path == run / "exports" / "composite_viewer" / "viewer_config.json"
    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["background"]["source_kind"] == "external_3dgs_render_missing"
    assert config["background"]["status"] == "blocked_missing_external_3dgs_render"
    assert config["background"]["is_final_visual"] is False
    assert config["background"]["diagnostic_only"] is True
    assert config["background"]["point_cloud_path"] == "../../background/bg_only_cloud.ply"
    assert config["background"]["point_cloud_diagnostic_only"] is True
    assert config["background"]["debug_full_scene_cloud_path"] == "../../scene_cloud.ply"
    assert config["background"]["registration_path"] == "../../background/registration.json"
    assert config["background"]["registration"]["status"] == "blocked_3dgs_world_to_sim_world_not_estimated"
    assert config["background"]["gaussian_splat"]["status"] == "blocked_viewer_not_integrated"
    assert config["background"]["gaussian_splat"]["config_path"] == "video/3dgs/run/config.yml"
    assert config["support_plane"]["table_collision_mesh_path"] == "../../background/table_collision.glb"
    assert config["support_plane"]["source_backend"] == "background_point_ring_median"
    assert config["support_plane"]["geometry_type"] == "support_box_proxy"
    assert config["support_plane"]["final_or_proxy"] == "proxy"
    assert config["support_plane"]["top_z"] == 0.0
    assert config["support_plane"]["thickness"] == 0.04
    assert config["support_plane"]["mesh_path"] == "../../background/table_collision.glb"
    assert config["support_plane"]["qa_status"] == "passed"
    assert config["support_plane"]["label"] == "estimated support box proxy"
    assert config["objects"][0]["mesh_path"] == "../../objects/cup/visual.glb"
    assert config["objects"][0]["visual_mesh_path"] == "../../objects/cup/visual.glb"
    assert config["objects"][0]["collision_mesh_path"] == "../../objects/cup/collision.glb"
    assert config["objects"][0]["debug_proxy_path"] == "../../objects/cup/debug_bbox.glb"
    assert config["objects"][0]["final_visual_is_proxy"] is False
    assert config["objects"][0]["world_position"] == [0.1, 1.0, 0.2]
    assert config["objects"][0]["world_quat_wxyz"] == [0.7071067811865476, 0.0, 0.0, 0.7071067811865475]
    assert config["objects"][0]["asset_scale"] == 0.5
    assert config["reference_camera"]["mode"] == "opencv_reference_overlay"
    assert config["reference_camera"]["camera_world_position"] == [0.0, 0.0, -0.12]
    assert config["reference_camera"]["camera_world_lookat"] == [0.0, 1.0, -0.12]
    assert config["reference_camera"]["camera_world_up"] == [0.0, 0.0, 1.0]
    assert config["qa"]["stability_status"] == "passed"
    assert config["qa"]["max_penetration_depth_m"] == 0.001
    html = result.index_path.read_text(encoding="utf-8")
    assert "PLYLoader" in html
    assert "GLTFLoader" in html
    assert "viewer_config.json" in html
    assert "applyReferenceCamera" in html
    assert "resetReferenceCamera" in html
    assert "lockReferenceCamera" in html
    assert "Reference Camera" in html
    assert "Table Collision" in html
    assert "toggleTableCollision" in html
    assert "stage" in html
    assert "setScalar(item.asset_scale || 1)" in html


def test_export_composite_viewer_uses_external_3dgs_render_as_final_background(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run)
    (run / "qa" / "background_3dgs_render.png").write_text("png", encoding="utf-8")
    (run / "qa" / "background_3dgs_render_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "rendered",
                "backend": "external_3dgs_renderer",
                "source_kind": "external_3dgs_renderer",
                "registered_3dgs_rendered": True,
                "simulator_native": False,
                "render_path": "qa/background_3dgs_render.png",
                "image_size": [1280, 720],
            }
        ),
        encoding="utf-8",
    )

    result = export_composite_viewer(run)

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["status"] == "partial"
    assert config["status_reason"] == "partial_external_render_only"
    assert config["background"]["source_kind"] == "external_3dgs_renderer"
    assert config["background"]["status"] == "external_3dgs_rendered"
    assert config["background"]["viewer_status"] == "partial_external_render_only"
    assert config["background"]["render_mode"] == "external_3dgs_png_sidecar"
    assert config["background"]["live_3dgs_runtime"] is False
    assert config["background"]["orbit_policy"] == "locked_to_reference_camera"
    assert config["background"]["camera_lock_required"] is True
    assert config["background"]["provenance_label"] == "external 3DGS PNG sidecar (reference view only)"
    assert config["background"]["background_layers"]["external_3dgs_png_sidecar"]["status"] == "active_reference_view_only"
    assert config["background"]["background_layers"]["bg_only_diagnostic_cloud"]["status"] == "available_diagnostic_only"
    assert config["background"]["background_layers"]["full_scene_debug_cloud"]["status"] == "available_debug_only"
    assert config["background"]["image_path"] == "../../qa/background_3dgs_render.png"
    assert config["background"]["image_size"] == [1280, 720]
    assert config["background"]["is_final_visual"] is True
    assert config["background"]["simulator_native"] is False
    assert config["background"]["point_cloud_path"] == "../../background/bg_only_cloud.ply"
    assert config["background"]["point_cloud_diagnostic_only"] is True
    assert config["objects"][0]["mesh_path"] == "../../objects/cup/visual.glb"
    assert config["objects"][0]["final_visual_is_proxy"] is False
    assert config["objects"][0]["debug_proxy_path"] == "../../objects/cup/debug_bbox.glb"
    assert config["pose_display"]["status"] == "settled_pose_unavailable"
    assert config["pose_display"]["toggle_enabled"] is False
    assert config["objects"][0]["settled_pose_available"] is False
    assert config["audits"]["object_path_audit"] == "object_path_audit.json"
    assert config["audits"]["background_provenance_audit"] == "background_provenance_audit.json"
    object_audit = json.loads((result.index_path.parent / "object_path_audit.json").read_text(encoding="utf-8"))
    assert object_audit["objects"][0]["final_visual_path"] == "objects/cup/visual.glb"
    assert object_audit["objects"][0]["final_visual_is_proxy"] is False
    assert object_audit["objects"][0]["legacy_mesh_path"] == "objects/cup/mesh_aligned.glb"
    bg_audit = json.loads((result.index_path.parent / "background_provenance_audit.json").read_text(encoding="utf-8"))
    assert bg_audit["source_kind"] == "external_3dgs_render_sidecar"
    assert bg_audit["viewer_status"] == "partial_external_render_only"
    assert bg_audit["render_mode"] == "external_3dgs_png_sidecar"
    assert bg_audit["live_3dgs_runtime"] is False
    assert bg_audit["orbit_policy"] == "locked_to_reference_camera"
    assert bg_audit["simulator_native"] is False
    assert bg_audit["final_visual_path"] == "qa/background_3dgs_render.png"
    html = result.index_path.read_text(encoding="utf-8")
    assert "toggleBackground" in html
    assert "backgroundImage" in html
    assert "new THREE.TextureLoader" in html
    assert "scene.background = texture" in html
    assert "external_3dgs_render_sidecar" in html
    assert "toggleCollision" in html
    assert "toggleTableCollision" in html
    assert "toggleDebug" in html
    assert "inspectPanel" in html
    assert "inspectTableCollision" in html
    assert "source_backend" in html
    assert "geometry_type" in html
    assert "final_or_proxy" in html
    assert "top_z" in html
    assert "thickness" in html
    assert "qa_status" in html
    assert "background is 2D sidecar render; projection alignment only valid in reference camera." in html
    assert "partial_external_render_only" in html
    assert "Native/live 3DGS runtime unsupported" in html
    assert "provenancePanel" in html
    assert "Initial Pose" in html
    assert "Settled Pose" in html
    assert "applyPoseMode" in html


def test_export_composite_viewer_consumes_settled_pose_delta_report(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run, include_3dgs=False)
    settled_transform = [[0, -1, 0, 0.1], [1, 0, 0, 1.0], [0, 0, 1, 0.25], [0, 0, 0, 1]]
    (run / "qa" / "settled_pose_delta_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "source": "genesis_settle",
                "objects": [
                    {
                        "object_id": "cup",
                        "settled_T_object_to_world": settled_transform,
                        "translation_delta_m": [0.0, 0.0, 0.05],
                        "rotation_delta_deg": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = export_composite_viewer(run)

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["pose_display"]["status"] == "completed"
    assert config["pose_display"]["toggle_enabled"] is True
    assert config["pose_display"]["report_path"] == "../../qa/settled_pose_delta_report.json"
    assert config["objects"][0]["initial_world_position"] == [0.1, 1.0, 0.2]
    assert config["objects"][0]["settled_world_position"] == [0.1, 1.0, 0.25]
    assert config["objects"][0]["settled_pose_available"] is True
    assert config["objects"][0]["settled_pose_delta"]["translation_delta_m"] == [0.0, 0.0, 0.05]


def test_export_composite_viewer_consumes_settled_pose_delta_final_pos_without_matrix(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run, include_3dgs=False)
    (run / "qa" / "settled_pose_delta_report.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "objects": [
                    {
                        "object_id": "cup",
                        "initial_pos": [0.1, 1.0, 0.2],
                        "final_pos": [0.11, 1.02, 0.23],
                        "translation_delta_m": [0.01, 0.02, 0.03],
                        "viewer_export_action": "display_delta_only",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = export_composite_viewer(run)

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["pose_display"]["status"] == "partial"
    assert config["pose_display"]["toggle_enabled"] is True
    assert config["objects"][0]["settled_world_position"] == [0.11, 1.02, 0.23]
    assert config["objects"][0]["settled_world_quat_wxyz"] == config["objects"][0]["initial_world_quat_wxyz"]
    assert config["objects"][0]["settled_pose_available"] is True
    assert config["objects"][0]["settled_pose_delta"]["translation_delta_m"] == [0.01, 0.02, 0.03]



def test_export_composite_viewer_uses_final_table_collision_provenance(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run, include_3dgs=False)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["support_plane"].update(
        {
            "table_collision_source_backend": "tabletop_mask_polygon_slab",
            "table_collision_geometry_type": "polygon_slab",
            "table_collision_final": True,
            "support_surface_status": "passed",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = export_composite_viewer(run)

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["support_plane"]["source_backend"] == "tabletop_mask_polygon_slab"
    assert config["support_plane"]["geometry_type"] == "polygon_slab"
    assert config["support_plane"]["final_or_proxy"] == "final"
    assert config["support_plane"]["qa_status"] == "passed"


def test_export_composite_viewer_uses_strict_manifest_over_stale_scene_manifest(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run, include_3dgs=False)
    manifest_path = run / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["support_plane"].update(
        {
            "table_collision_source_backend": "tabletop_mask_polygon_slab",
            "table_collision_geometry_type": "polygon_slab",
            "table_collision_final": True,
            "support_surface_status": "passed",
            "table_collision_visual_qa_path": "qa/table_collision_overlay.png",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run / "sim_export_manifest.json").write_text(
        json.dumps(
            {
                "support_surface": {
                    "support_surface_status": "blocked",
                    "table_collision_final": False,
                    "table_collision_source_backend": "tabletop_mask_polygon_slab",
                    "table_collision_geometry_type": "polygon_slab",
                    "table_collision_visual_qa_path": "qa/table_collision_overlay.png",
                    "blocking_reasons": ["table_collision_support_coverage_too_small"],
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "qa" / "table_collision_report.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "blocking_reasons": ["table_collision_support_coverage_too_small"],
                "projection_iou": 0.9,
            }
        ),
        encoding="utf-8",
    )

    result = export_composite_viewer(run)

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["support_plane"]["source_backend"] == "tabletop_mask_polygon_slab"
    assert config["support_plane"]["geometry_type"] == "polygon_slab"
    assert config["support_plane"]["final_or_proxy"] == "proxy"
    assert config["support_plane"]["qa_status"] == "blocked"
    assert config["support_plane"]["blocking_reasons"] == ["table_collision_support_coverage_too_small"]
    assert config["support_plane"]["projection_iou"] == 0.9


def test_export_composite_viewer_prefers_latest_genesis_settle_report_over_stale_qa_report(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run, include_3dgs=False)
    (run / "qa" / "qa_report.json").write_text(
        json.dumps(
            {
                "physics_settle": {
                    "status": "completed",
                    "stability_status": "passed",
                    "settle_steps": 50,
                    "max_penetration_depth_m": 0.123,
                    "max_displacement_m": 0.456,
                    "fall_out_detected": True,
                    "nan_detected": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "passed",
                "settle_steps": 100,
                "max_penetration_depth_m": 0.0,
                "max_displacement_m": 0.175,
                "fall_out_detected": False,
                "nan_detected": False,
            }
        ),
        encoding="utf-8",
    )

    result = export_composite_viewer(run)

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["qa"]["settle_steps"] == 100
    assert config["qa"]["max_penetration_depth_m"] == 0.0
    assert config["qa"]["max_displacement_m"] == 0.175
    assert config["qa"]["fall_out_detected"] is False
    assert config["qa"]["nan_detected"] is False


def test_export_composite_viewer_labels_full_scene_cloud_as_debug_only(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run, include_bg_only_cloud=False, include_3dgs=False, include_registration=False)

    result = export_composite_viewer(run)

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["background"]["source_kind"] == "full_scene_cloud_debug"
    assert config["background"]["diagnostic_only"] is True
    assert config["background"]["status"] == "blocked_missing_bg_only_background"
    assert config["background"]["point_cloud_path"] == "../../scene_cloud.ply"


def test_export_composite_viewer_marks_native_3dgs_candidate_as_not_passed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run, include_bg_only_cloud=False, include_3dgs=True, include_registration=False)
    native_dir = run / "background" / "3dgs_native"
    native_dir.mkdir(parents=True)
    (native_dir / "splat_rgb.ply").write_text(_ply_text("0 0 1 50 60 70"), encoding="utf-8")

    result = export_composite_viewer(run)

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["background"]["source_kind"] == "3dgs_native_asset_candidate"
    assert config["background"]["status"] == "blocked_native_3dgs_not_integrated"
    assert config["background"]["status"] != "passed"
    assert config["background"]["point_cloud_path"] is None
    assert config["background"]["native_asset_candidate_path"] == "../../background/3dgs_native/splat_rgb.ply"
    assert config["background"]["debug_full_scene_cloud_path"] == "../../scene_cloud.ply"
    assert config["background"]["gaussian_splat"]["status"] == "native_asset_candidate"
    assert config["background"]["gaussian_splat"]["native_rendering"] is False


def test_cli_composite_viewer_export_only(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run)

    code = main(["composite-viewer", "--run-dir", str(run), "--export-only"])

    assert code == 0
    assert (run / "exports" / "composite_viewer" / "index.html").is_file()
    assert (run / "exports" / "composite_viewer" / "viewer_config.json").is_file()
