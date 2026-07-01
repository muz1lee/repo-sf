import json

import numpy as np
import trimesh
from PIL import Image

from real2sim_scene_foundry.export_qa import write_sim_export_report
from real2sim_scene_foundry.genesis_export import export_genesis_scene
from real2sim_scene_foundry.isaac_export import export_isaac_scene
from real2sim_scene_foundry.sim_export_manifest import write_sim_export_manifest
from real2sim_scene_foundry.usd_export import export_usd_scene


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]]


def _write_text(path, text="placeholder"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_ply(path):
    _write_text(
        path,
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
            ]
        )
        + "\n",
    )


def _write_ready_run(run_dir):
    obj_dir = run_dir / "objects" / "cup"
    bg_dir = run_dir / "background"
    obj_dir.mkdir(parents=True)
    bg_dir.mkdir()
    trimesh.creation.uv_sphere(radius=0.05).export(obj_dir / "visual.glb")
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(obj_dir / "collision.glb")
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(obj_dir / "debug_bbox.glb")
    trimesh.creation.cylinder(radius=0.5, height=0.04, sections=6).export(bg_dir / "table_collision.glb")
    _write_text(obj_dir / "mask.png")
    _write_text(obj_dir / "crop.png")
    _write_text(bg_dir / "bg_only.png")
    _write_text(bg_dir / "foreground_mask.png")
    _write_text(bg_dir / "tabletop_mask.png")
    _write_ply(bg_dir / "bg_only_cloud.ply")
    _write_text(run_dir / "qa" / "table_collision_overlay.png", "png")
    (run_dir / "qa" / "table_collision_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "passed",
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "derived_from_tabletop_mask": True,
                "visual_qa_path": "qa/table_collision_overlay.png",
            }
        ),
        encoding="utf-8",
    )
    (obj_dir / "physics.json").write_text(
        json.dumps({"mass_kg": 0.25, "friction": 0.8, "restitution": 0.01, "source": "class_default"}),
        encoding="utf-8",
    )
    (obj_dir / "pose_refinement_report.json").write_text(
        json.dumps(
            {
                "status": "accepted",
                "rotation_source": "service_pose_refined",
                "translation_source": "rgbd_alignment",
                "scale_source": "rgbd_alignment",
            }
        ),
        encoding="utf-8",
    )
    (bg_dir / "registration.json").write_text(
        json.dumps(
            {
                "status": "registered",
                "T_3dgs_world_to_sim_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "scale_source": "metric_reference_camera",
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
                "background": {
                    "source_backend": "video_3dgs_splatfacto",
                    "status": "trained_video_3dgs",
                    "bg_only_image_path": "background/bg_only.png",
                    "foreground_mask_path": "background/foreground_mask.png",
                    "point_cloud_path": "background/bg_only_cloud.ply",
                    "gaussian_splat_config_path": "video/3dgs/run/config.yml",
                },
                "support_plane": {
                    "status": "estimated",
                    "source_backend": "tabletop_mask",
                    "height_world_m": 0.0,
                    "normal_world": [0.0, 0.0, 1.0],
                    "applied_to_world_frame": True,
                    "tabletop_mask_path": "background/tabletop_mask.png",
                    "table_collision_mesh_path": "background/table_collision.glb",
                    "table_collision_pos_world": [0.0, 0.0, -0.02],
                    "table_collision_size_xyz": [1.0, 0.8, 0.04],
                    "table_bounds_world_xy": [[-0.5, -0.4], [0.5, 0.4]],
                    "table_collision_source_backend": "tabletop_mask",
                    "table_collision_geometry_type": "polygon_slab",
                    "table_collision_final": True,
                    "table_collision_visual_qa_path": "qa/table_collision_overlay.png",
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/visual.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": IDENTITY,
                        "T_object_to_world": IDENTITY,
                        "scale_m": 0.2,
                        "mass_kg": 0.25,
                        "friction": 0.8,
                        "confidence": 0.9,
                        "source_backend": "sam3d_aligned",
                        "needs_manual_refine": False,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_proxy_run(run_dir):
    obj_dir = run_dir / "objects" / "cup"
    bg_dir = run_dir / "background"
    obj_dir.mkdir(parents=True)
    bg_dir.mkdir()
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(obj_dir / "mesh_aligned.glb")
    trimesh.creation.cylinder(radius=0.5, height=0.04, sections=6).export(bg_dir / "table_collision.glb")
    _write_text(obj_dir / "mask.png")
    _write_text(obj_dir / "crop.png")
    _write_text(bg_dir / "bg_only.png")
    _write_text(bg_dir / "foreground_mask.png")
    _write_text(bg_dir / "tabletop_mask.png")
    _write_ply(bg_dir / "bg_only_cloud.ply")
    _write_text(run_dir / "qa" / "table_collision_overlay.png", "png")
    (run_dir / "qa" / "table_collision_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "passed",
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "derived_from_tabletop_mask": True,
                "visual_qa_path": "qa/table_collision_overlay.png",
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "coordinate_frames": {"camera": "opencv_x_right_y_down_z_forward_meters", "world": "z_up_ground_plane_meters"},
        "background": {
            "source_backend": "opencv_inpaint_single_frame",
            "status": "proxy_from_single_stereo_pair",
            "bg_only_image_path": "background/bg_only.png",
            "foreground_mask_path": "background/foreground_mask.png",
            "point_cloud_path": "background/bg_only_cloud.ply",
        },
        "support_plane": {"status": "estimated", "table_collision_mesh_path": "background/table_collision.glb"},
        "objects": [
            {
                "object_id": "cup",
                "label": "cup",
                "mesh_path": "objects/cup/mesh_aligned.glb",
                "mask_path": "objects/cup/mask.png",
                "crop_path": "objects/cup/crop.png",
                "T_object_to_camera": IDENTITY,
                "T_object_to_world": IDENTITY,
                "scale_m": 0.2,
                "mass_kg": 0.25,
                "friction": 0.8,
                "confidence": 0.9,
                "source_backend": "sam3d_aligned",
                "needs_manual_refine": False,
            }
        ],
    }
    (run_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_export_qa_blocks_proxy_visuals_with_exact_reasons(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_proxy_run(run_dir)
    write_sim_export_manifest(run_dir)

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "blocked"
    assert report["sections"]["pose_alignment"]["status"] == "passed"
    assert report["sections"]["collision_assets"]["status"] == "blocked"
    assert "blocked_proxy_visual_asset:cup" in report["blocking_reasons"]
    assert "missing_collision_mesh:cup" in report["blocking_reasons"]


def test_export_qa_blocks_bg_only_cloud_when_3dgs_is_sidecar_not_runtime_verified(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_text(run_dir / "video" / "3dgs" / "run" / "config.yml", "viewer: splatfacto")
    (run_dir / "video" / "3dgs_status.json").write_text(
        json.dumps({"status": "completed", "outputs": {"latest_config": "video/3dgs/run/config.yml"}}),
        encoding="utf-8",
    )
    export_usd_scene(run_dir)
    export_genesis_scene(run_dir)
    _write_text(run_dir / "exports" / "isaac_scene.py")
    (run_dir / "qa" / "isaac_load_report.json").write_text(
        json.dumps({"status": "loaded", "checked_rigid_collision_metadata": True}),
        encoding="utf-8",
    )
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "passed",
                "nan_detected": False,
                "fall_out_detected": False,
                "max_penetration_depth_m": 0.001,
            }
        ),
        encoding="utf-8",
    )

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["sections"]["usd_export"]["status"] == "passed"
    assert report["sections"]["genesis_export"]["status"] == "passed"
    assert report["sections"]["physics_settle"]["status"] == "passed"
    assert report["sections"]["isaac_export"]["status"] == "passed"
    assert report["sections"]["background_registration"]["status"] == "blocked"
    assert report["overall_status"] == "blocked"
    assert report["overall_status"] != "passed"
    assert "background_3dgs_runtime_not_verified" in report["blocking_reasons"]
    assert "blocked_background_proxy_visual:bg_only_cloud" in report["blocking_reasons"]


def test_export_qa_blocks_background_when_3dgs_is_external_sidecar_only(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_text(run_dir / "video" / "3dgs" / "run" / "config.yml", "viewer: splatfacto")
    _write_text(run_dir / "video" / "3dgs" / "run" / "nerfstudio_models" / "step.ckpt")
    (run_dir / "video" / "3dgs_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "outputs": {
                    "latest_config": "video/3dgs/run/config.yml",
                    "latest_checkpoint": "video/3dgs/run/nerfstudio_models/step.ckpt",
                },
            }
        ),
        encoding="utf-8",
    )
    _write_text(run_dir / "qa" / "background_3dgs_render.png", "png")
    (run_dir / "qa" / "background_3dgs_render_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "rendered",
                "backend": "external_3dgs_renderer",
                "source_kind": "external_3dgs_renderer",
                "registered_3dgs_rendered": True,
                "simulator_native": False,
                "native_rendering_proven": False,
                "config_path": "video/3dgs/run/config.yml",
                "checkpoint_path": "video/3dgs/run/nerfstudio_models/step.ckpt",
                "render_path": "qa/background_3dgs_render.png",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "background" / "registration.json").write_text(
        json.dumps(
            {
                "status": "registered",
                "T_3dgs_world_to_sim_world": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "transform_sources": {"T_3dgs_world_to_sim_world": "manual_splat_alignment"},
                "scale_source": "manual_splat_alignment",
            }
        ),
        encoding="utf-8",
    )
    export_usd_scene(run_dir)
    export_genesis_scene(run_dir)
    _write_text(run_dir / "exports" / "isaac_scene.py")
    (run_dir / "qa" / "isaac_load_report.json").write_text(
        json.dumps(
            {
                "status": "loaded",
                "checked_rigid_collision_metadata": True,
                "native_3dgs_required": False,
                "native_3dgs_supported": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "passed",
                "nan_detected": False,
                "fall_out_detected": False,
                "max_penetration_depth_m": 0.001,
            }
        ),
        encoding="utf-8",
    )

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["sections"]["background_registration"]["status"] == "blocked"
    assert report["sections"]["background_registration"]["visual_asset"]["source_kind"] == "external_3dgs_renderer"
    assert report["sections"]["background_registration"]["visual_asset"]["simulator_native"] is False
    assert "background_external_render_only" in report["sections"]["background_registration"]["blocking_reasons"]
    assert "blocked_background_proxy_visual:bg_only_cloud" not in report["blocking_reasons"]
    assert "background_3dgs_runtime_not_verified" not in report["blocking_reasons"]
    assert report["overall_status"] == "blocked"


def test_export_qa_blocks_external_3dgs_render_with_placeholder_registration(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_text(run_dir / "video" / "3dgs" / "run" / "config.yml", "viewer: splatfacto")
    _write_text(run_dir / "video" / "3dgs" / "run" / "nerfstudio_models" / "step.ckpt")
    _write_text(run_dir / "qa" / "background_3dgs_render.png", "png")
    (run_dir / "background" / "registration.json").write_text(
        json.dumps(
            {
                "status": "registered",
                "T_3dgs_world_to_sim_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "transform_sources": {"T_3dgs_world_to_sim_world": "identity_placeholder_not_registered"},
                "scale_source": "input_metric_depth",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "qa" / "background_3dgs_render_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "rendered",
                "backend": "external_3dgs_renderer",
                "source_kind": "external_3dgs_renderer",
                "registered_3dgs_rendered": True,
                "simulator_native": False,
                "native_rendering_proven": False,
                "config_path": "video/3dgs/run/config.yml",
                "checkpoint_path": "video/3dgs/run/nerfstudio_models/step.ckpt",
                "render_path": "qa/background_3dgs_render.png",
            }
        ),
        encoding="utf-8",
    )
    export_usd_scene(run_dir)
    export_genesis_scene(run_dir)
    _write_text(run_dir / "exports" / "isaac_scene.py")
    (run_dir / "qa" / "isaac_load_report.json").write_text(
        json.dumps({"status": "loaded", "checked_rigid_collision_metadata": True}),
        encoding="utf-8",
    )
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "passed",
                "nan_detected": False,
                "fall_out_detected": False,
                "max_penetration_depth_m": 0.001,
            }
        ),
        encoding="utf-8",
    )

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["sections"]["background_registration"]["status"] == "blocked"
    assert "background_3dgs_registration_placeholder" in report["blocking_reasons"]
    assert report["overall_status"] == "blocked"


def test_export_qa_reports_vertical_fall_and_excessive_displacement(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_text(run_dir / "qa" / "background_3dgs_render.png", "png")
    (run_dir / "qa" / "background_3dgs_render_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "rendered",
                "backend": "external_3dgs_renderer",
                "registered_3dgs_rendered": True,
                "render_path": "qa/background_3dgs_render.png",
                "simulator_native": False,
            }
        ),
        encoding="utf-8",
    )
    export_usd_scene(run_dir)
    export_genesis_scene(run_dir)
    _write_text(run_dir / "exports" / "isaac_scene.py")
    (run_dir / "qa" / "isaac_load_report.json").write_text(json.dumps({"status": "loaded"}), encoding="utf-8")
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "failed",
                "nan_detected": False,
                "fall_out_detected": False,
                "fall_below_support_detected": True,
                "excessive_displacement_detected": True,
                "max_displacement_m": 4.9,
                "max_penetration_depth_m": 0.0,
            }
        ),
        encoding="utf-8",
    )

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    reasons = report["sections"]["physics_settle"]["blocking_reasons"]
    assert report["overall_status"] == "blocked"
    assert "genesis_settle_fall_below_support" in reasons
    assert "genesis_settle_excessive_displacement" in reasons



def test_export_qa_marks_isaac_runtime_unavailable_as_blocking_not_passed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    write_sim_export_manifest(run_dir)
    export_usd_scene(run_dir)
    export_genesis_scene(run_dir)
    export_isaac_scene(run_dir, python_executable=None)
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "passed",
                "nan_detected": False,
                "fall_out_detected": False,
                "max_penetration_depth_m": 0.001,
            }
        ),
        encoding="utf-8",
    )

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["sections"]["usd_export"]["status"] == "passed"
    assert report["sections"]["genesis_export"]["status"] == "passed"
    assert report["sections"]["physics_settle"]["status"] == "passed"
    assert report["sections"]["isaac_export"]["status"] == "runtime_unavailable"
    assert report["overall_status"] == "blocked"
    assert "isaac_runtime_unavailable" in report["blocking_reasons"]
    assert report["overall_status"] != "passed"


def _write_passing_backend_reports(run_dir):
    _write_text(run_dir / "qa" / "background_3dgs_render.png", "png")
    (run_dir / "qa" / "background_3dgs_render_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "rendered",
                "backend": "external_3dgs_renderer",
                "registered_3dgs_rendered": True,
                "render_path": "qa/background_3dgs_render.png",
                "simulator_native": False,
            }
        ),
        encoding="utf-8",
    )
    export_usd_scene(run_dir)
    export_genesis_scene(run_dir)
    _write_text(run_dir / "exports" / "isaac_scene.py")
    (run_dir / "qa" / "isaac_load_report.json").write_text(json.dumps({"status": "loaded"}), encoding="utf-8")
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "passed",
                "nan_detected": False,
                "fall_out_detected": False,
                "max_penetration_depth_m": 0.001,
            }
        ),
        encoding="utf-8",
    )


def test_export_qa_fails_closed_when_table_collision_projection_report_is_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    (run_dir / "qa" / "table_collision_report.json").unlink()
    _write_passing_backend_reports(run_dir)

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    section = report["sections"]["table_collision_projection"]
    assert section["status"] == "blocked"
    assert "table_collision_missing_visual_qa" in report["blocking_reasons"]
    assert report["overall_status"] == "blocked"


def test_export_qa_blocks_failed_table_collision_projection_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    (run_dir / "qa" / "table_collision_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "failed",
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "derived_from_tabletop_mask": True,
                "visual_qa_path": "qa/table_collision_overlay.png",
                "blocking_reasons": ["low_tabletop_projection_iou"],
            }
        ),
        encoding="utf-8",
    )
    _write_passing_backend_reports(run_dir)

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    section = report["sections"]["table_collision_projection"]
    assert section["status"] == "blocked"
    assert "table_collision_projection_failed" in report["blocking_reasons"]
    assert "low_tabletop_projection_iou" in section["blocking_reasons"]
    assert report["overall_status"] == "blocked"


def test_export_qa_generates_missing_table_collision_projection_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:71, 30:71] = 255
    Image.fromarray(mask).save(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "background" / "table_polygon_world.json").write_text(
        json.dumps(
            {
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "vertices_world": [[-0.2, -0.2, 1.0], [0.2, -0.2, 1.0], [0.2, 0.2, 1.0], [-0.2, 0.2, 1.0]],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "camera.json").write_text(
        json.dumps(
            {
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                "T_world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": 100,
                "height": 100,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "qa" / "table_collision_report.json").unlink()
    (run_dir / "qa" / "table_collision_overlay.png").unlink()

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert (run_dir / "qa" / "table_collision_report.json").is_file()
    assert (run_dir / "qa" / "table_collision_overlay.png").is_file()
    assert report["sections"]["table_collision_projection"]["status"] == "passed"


def test_export_qa_refreshes_stale_usd_report_after_table_collision_qa_passes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    (run_dir / "qa" / "table_collision_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "blocked",
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "derived_from_tabletop_mask": True,
                "visual_qa_path": "qa/table_collision_overlay.png",
                "blocking_reasons": ["low_tabletop_projection_iou"],
            }
        ),
        encoding="utf-8",
    )
    export_usd_scene(run_dir)
    stale_report = json.loads((run_dir / "qa" / "usd_export_report.json").read_text(encoding="utf-8"))
    assert stale_report["support_surface_status"] == "blocked"

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:71, 30:71] = 255
    Image.fromarray(mask).save(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "background" / "table_polygon_world.json").write_text(
        json.dumps(
            {
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "vertices_world": [[-0.2, -0.2, 1.0], [0.2, -0.2, 1.0], [0.2, 0.2, 1.0], [-0.2, 0.2, 1.0]],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "camera.json").write_text(
        json.dumps(
            {
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                "T_world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": 100,
                "height": 100,
            }
        ),
        encoding="utf-8",
    )
    export_genesis_scene(run_dir)
    _write_text(run_dir / "exports" / "isaac_scene.py")
    (run_dir / "qa" / "isaac_load_report.json").write_text(json.dumps({"status": "loaded"}), encoding="utf-8")
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stability_status": "passed",
                "nan_detected": False,
                "fall_out_detected": False,
                "max_penetration_depth_m": 0.001,
            }
        ),
        encoding="utf-8",
    )

    result = write_sim_export_report(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    refreshed_usd_report = json.loads((run_dir / "qa" / "usd_export_report.json").read_text(encoding="utf-8"))
    assert report["sections"]["table_collision_projection"]["status"] == "passed"
    assert refreshed_usd_report["support_surface_status"] == "ready"


def test_strict_claims_write_honest_milestone_and_keep_external_3dgs_partial(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_passing_backend_reports(run_dir)
    (run_dir / "background" / "registration.json").write_text(
        json.dumps(
            {
                "status": "registered",
                "T_3dgs_world_to_sim_world": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "transform_sources": {"T_3dgs_world_to_sim_world": "manual_splat_alignment"},
                "scale_source": "manual_splat_alignment",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "qa" / "isaac_load_report.json").write_text(
        json.dumps(
            {
                "status": "loaded",
                "report_source": "preserved_existing_worker_report",
                "validation_reused": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "qa" / "collision_rebuild_report.json").write_text(
        json.dumps({"status": "blocked", "paper_equivalence_status": "blocked"}),
        encoding="utf-8",
    )
    (run_dir / "qa" / "physics_property_report.json").write_text(
        json.dumps({"status": "partial", "paper_equivalence_status": "partial"}),
        encoding="utf-8",
    )

    result = write_sim_export_report(run_dir, strict_claims=True)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    milestone = json.loads((run_dir / "qa" / "honest_milestone.json").read_text(encoding="utf-8"))
    isaac_fresh = json.loads((run_dir / "qa" / "isaac_fresh_validation_report.json").read_text(encoding="utf-8"))
    assert report["strict_claims"] is True
    assert report["claim_gate"]["engineering_interactive_scene_status"] == "blocked"
    assert report["claim_gate"]["simfoundry_upper_reproduction_status"] == "partial"
    assert report["sections"]["background_registration"]["status"] == "blocked"
    assert "background_external_render_only" in report["sections"]["background_registration"]["blocking_reasons"]
    assert report["sections"]["isaac_export"]["status"] == "partial"
    assert report["sections"]["isaac_export"]["repeatability_status"] == "partial_preserved_report"
    assert isaac_fresh["status"] == "partial_preserved_report"
    assert isaac_fresh["fresh_direct_validation"] is False
    assert "isaac_validation_reused" in isaac_fresh["blocking_reasons"]
    assert "isaac_validation_reused" in milestone["blocking_reasons"]
    assert "external_3dgs_render_sidecar" in milestone["remaining_gaps"]
    assert "coacd_collision_decomposition" in milestone["remaining_gaps"]
    assert "physics_property_inference" in milestone["remaining_gaps"]
    assert milestone["simfoundry_full_pipeline_status"] == "out_of_scope"


def test_strict_claims_blocks_weak_table_collision_v2(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_passing_backend_reports(run_dir)
    (run_dir / "qa" / "table_collision_report_v2.json").write_text(
        json.dumps(
            {
                "version": 2,
                "status": "blocked",
                "weak_status": "diagnostic_pass",
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "derived_from_tabletop_mask": True,
                "visual_qa_path": "qa/table_collision_overlay.png",
                "projection_iou": 0.531,
                "projection_iou_with_tabletop_mask": 0.531,
                "blocking_reasons": ["below_export_grade_tabletop_iou"],
            }
        ),
        encoding="utf-8",
    )

    result = write_sim_export_report(run_dir, strict_claims=True)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["sections"]["table_collision_projection"]["status"] == "blocked"
    assert "below_export_grade_tabletop_iou" in report["blocking_reasons"]
    milestone = json.loads((run_dir / "qa" / "honest_milestone.json").read_text(encoding="utf-8"))
    assert "table_collision_export_grade_blocked" in milestone["blocking_reasons"]
