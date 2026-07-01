import json

import pytest
import trimesh

from real2sim_scene_foundry.sim_export_manifest import build_sim_export_manifest, write_sim_export_manifest


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]]


def _write_placeholder_file(path, text="placeholder"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_ply(path):
    _write_placeholder_file(
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


def _base_scene_manifest(
    *,
    mesh_path="objects/cup/mesh_aligned.glb",
    needs_manual_refine=False,
    support_source_backend="tabletop_mask",
    table_collision_geometry_type="polygon_slab",
    table_collision_final=True,
):
    return {
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
            "source_backend": support_source_backend,
            "height_world_m": 0.0,
            "normal_world": [0.0, 0.0, 1.0],
            "applied_to_world_frame": True,
            "tabletop_mask_path": "background/tabletop_mask.png",
            "table_collision_mesh_path": "background/table_collision.glb",
            "table_collision_pos_world": [0.0, 0.0, -0.02],
            "table_collision_size_xyz": [1.0, 0.8, 0.04],
            "table_bounds_world_xy": [[-0.5, -0.4], [0.5, 0.4]],
            "table_collision_source_backend": support_source_backend,
            "table_collision_geometry_type": table_collision_geometry_type,
            "table_collision_final": table_collision_final,
            "table_collision_visual_qa_path": "qa/table_collision_overlay.png",
        },
        "objects": [
            {
                "object_id": "cup",
                "label": "cup",
                "mesh_path": mesh_path,
                "mask_path": "objects/cup/mask.png",
                "crop_path": "objects/cup/crop.png",
                "T_object_to_camera": IDENTITY,
                "T_object_to_world": IDENTITY,
                "asset_scale": 0.5,
                "scale_m": 0.2,
                "mass_kg": 0.25,
                "friction": 0.8,
                "confidence": 0.9,
                "source_backend": "sam3d_aligned",
                "needs_manual_refine": needs_manual_refine,
            }
        ],
    }


def _write_common_inputs(run_dir):
    _write_placeholder_file(run_dir / "objects" / "cup" / "mask.png")
    _write_placeholder_file(run_dir / "objects" / "cup" / "crop.png")
    _write_placeholder_file(run_dir / "background" / "bg_only.png")
    _write_placeholder_file(run_dir / "background" / "foreground_mask.png")
    _write_placeholder_file(run_dir / "background" / "tabletop_mask.png")
    _write_ply(run_dir / "background" / "bg_only_cloud.ply")
    trimesh.creation.cylinder(radius=0.5, height=0.04, sections=6).export(run_dir / "background" / "table_collision.glb")
    _write_placeholder_file(run_dir / "qa" / "table_collision_overlay.png", "png")
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


def _write_proxy_run(run_dir):
    _write_common_inputs(run_dir)
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(run_dir / "objects" / "cup" / "mesh_aligned.glb")
    (run_dir / "scene_manifest.json").write_text(json.dumps(_base_scene_manifest(), indent=2), encoding="utf-8")


def _write_ready_run(run_dir):
    _write_common_inputs(run_dir)
    obj_dir = run_dir / "objects" / "cup"
    trimesh.creation.uv_sphere(radius=0.05).export(obj_dir / "visual.glb")
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(obj_dir / "collision.glb")
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(obj_dir / "debug_bbox.glb")
    (obj_dir / "physics.json").write_text(
        json.dumps(
            {
                "mass_kg": 0.25,
                "friction": 0.8,
                "restitution": 0.01,
                "source": "class_default",
            }
        ),
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
    (run_dir / "background" / "registration.json").write_text(
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
        json.dumps(_base_scene_manifest(mesh_path="objects/cup/visual.glb"), indent=2), encoding="utf-8"
    )


def test_build_manifest_blocks_bbox_proxy_as_visual_asset(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_proxy_run(run_dir)

    manifest = build_sim_export_manifest(run_dir)
    data = manifest.to_dict()

    assert data["status"] == "blocked"
    assert "blocked_proxy_visual_asset:cup" in data["blocking_reasons"]
    assert "missing_collision_mesh:cup" in data["blocking_reasons"]
    assert data["objects"][0]["visual_asset"]["status"] == "blocked_proxy_visual_asset"
    assert data["objects"][0]["visual_asset"]["role"] == "debug_proxy"
    assert data["objects"][0]["debug_proxy"]["path"] == "objects/cup/mesh_aligned.glb"


def test_build_manifest_records_separated_assets_and_export_blockers(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    manifest = build_sim_export_manifest(run_dir)
    data = manifest.to_dict()

    assert data["schema_name"] == "real2sim_scene_foundry.sim_export_manifest"
    assert data["objects"][0]["visual_asset"] == {
        "role": "visual_asset",
        "path": "objects/cup/visual.glb",
        "source": "sam3d_or_visual_reconstruction",
        "status": "ready",
        "diagnostic_only": False,
    }
    assert data["objects"][0]["collision_asset"]["path"] == "objects/cup/collision.glb"
    assert data["objects"][0]["physics"]["restitution"] == pytest.approx(0.01)
    assert data["objects"][0]["pose"]["asset_scale"] == pytest.approx(0.5)
    assert data["background"]["registration"]["status"] == "blocked_3dgs_registration_placeholder"
    assert "background_3dgs_registration_placeholder" in data["blocking_reasons"]
    assert data["exports"]["usd"]["status"] == "missing"
    assert data["status"] == "blocked"
    assert "missing_backend_export:usd" in data["blocking_reasons"]
    assert "missing_backend_export:genesis" in data["blocking_reasons"]


def test_build_manifest_blocks_proxy_support_surface_sources(tmp_path):
    for source in ["background_support_points_rect", "object_bounds_rect", "estimated_support_box_proxy"]:
        run_dir = tmp_path / source
        run_dir.mkdir()
        _write_common_inputs(run_dir)
        obj_dir = run_dir / "objects" / "cup"
        trimesh.creation.uv_sphere(radius=0.05).export(obj_dir / "visual.glb")
        trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(obj_dir / "collision.glb")
        (obj_dir / "physics.json").write_text(
            json.dumps({"mass_kg": 0.25, "friction": 0.8, "restitution": 0.01}),
            encoding="utf-8",
        )
        (obj_dir / "pose_refinement_report.json").write_text(
            json.dumps({"status": "accepted", "source": "service_pose_refined"}),
            encoding="utf-8",
        )
        (run_dir / "background" / "registration.json").write_text(
            json.dumps({"status": "registered", "T_3dgs_world_to_sim_world": IDENTITY}),
            encoding="utf-8",
        )
        (run_dir / "scene_manifest.json").write_text(
            json.dumps(
                _base_scene_manifest(
                    mesh_path="objects/cup/visual.glb",
                    support_source_backend=source,
                    table_collision_final=False,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        data = build_sim_export_manifest(run_dir).to_dict()

        assert data["support_surface"]["support_surface_status"] == "blocked"
        assert data["support_surface"]["table_collision_final"] is False
        assert "support_surface_is_proxy" in data["blocking_reasons"]
        assert "table_collision_not_tabletop_mask_derived" in data["blocking_reasons"]


def test_build_manifest_blocks_8_vertex_table_collision_as_final_surface(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    trimesh.creation.box(extents=(1.0, 0.8, 0.04)).export(run_dir / "background" / "table_collision.glb")

    data = build_sim_export_manifest(run_dir).to_dict()

    assert data["support_surface"]["support_surface_status"] == "blocked"
    assert data["support_surface"]["table_collision_final"] is False
    assert "table_collision_box_proxy" in data["blocking_reasons"]


def test_build_manifest_accepts_tabletop_mask_polygon_slab_with_projection_qa(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    data = build_sim_export_manifest(run_dir).to_dict()

    support = data["support_surface"]
    assert support["status"] == "ready"
    assert support["support_surface_status"] == "passed"
    assert support["table_collision_source_backend"] == "tabletop_mask"
    assert support["table_collision_geometry_type"] == "polygon_slab"
    assert support["table_collision_final"] is True
    assert support["table_collision_visual_qa_path"] == "qa/table_collision_overlay.png"
    assert "support_surface_is_proxy" not in data["blocking_reasons"]
    assert "table_collision_projection_failed" not in data["blocking_reasons"]


def test_build_manifest_accepts_tabletop_collision_when_support_height_source_is_background_points(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    scene["support_plane"]["source_backend"] = "background_point_ring_median"
    scene["support_plane"]["table_collision_source_backend"] = "tabletop_mask_polygon_slab"
    scene["support_plane"]["table_collision_geometry_type"] = "polygon_slab"
    scene["support_plane"]["table_collision_final"] = True
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")

    data = build_sim_export_manifest(run_dir).to_dict()

    support = data["support_surface"]
    assert support["support_surface_status"] == "passed"
    assert support["source_backend"] == "background_point_ring_median"
    assert support["table_collision_source_backend"] == "tabletop_mask_polygon_slab"
    assert support["table_collision_final"] is True
    assert "support_surface_is_proxy" not in data["blocking_reasons"]
    assert "table_collision_box_proxy" not in data["blocking_reasons"]


def test_build_manifest_blocks_tiny_tabletop_patch_even_with_passed_projection_qa(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    scene["support_plane"]["table_collision_size_xyz"] = [0.06, 0.07, 0.04]
    scene["support_plane"]["table_bounds_world_xy"] = [[0.0, 0.0], [0.06, 0.07]]
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    (run_dir / "qa" / "table_collision_report.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "passed",
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "derived_from_tabletop_mask": True,
                "visual_qa_path": "qa/table_collision_overlay.png",
                "projection_iou": 0.9,
                "selected_candidate_ratio": 0.003,
                "polygon_extent_x_m": 0.06,
                "polygon_extent_y_m": 0.07,
            }
        ),
        encoding="utf-8",
    )

    data = build_sim_export_manifest(run_dir).to_dict()

    assert data["support_surface"]["support_surface_status"] == "blocked"
    assert data["support_surface"]["table_collision_final"] is False
    assert "table_collision_support_coverage_too_small" in data["blocking_reasons"]
    assert "table_collision_extent_too_small" in data["blocking_reasons"]


def test_build_manifest_blocks_legacy_bbox_mesh_path_even_with_real_visual_asset(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(run_dir / "objects" / "cup" / "mesh_aligned.glb")
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    scene["objects"][0]["mesh_path"] = "objects/cup/mesh_aligned.glb"
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")

    data = build_sim_export_manifest(run_dir).to_dict()

    assert data["status"] == "blocked"
    assert "blocked_legacy_bbox_visual_reference:cup" in data["blocking_reasons"]
    assert data["objects"][0]["visual_asset"]["path"] == "objects/cup/visual.glb"
    assert data["objects"][0]["visual_asset"]["status"] == "ready"



def test_build_manifest_blocks_visual_glb_when_it_is_bbox_proxy(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(run_dir / "objects" / "cup" / "visual.glb")

    data = build_sim_export_manifest(run_dir).to_dict()

    assert data["status"] == "blocked"
    assert "blocked_proxy_visual_asset:cup" in data["blocking_reasons"]
    assert data["objects"][0]["visual_asset"]["status"] == "blocked_proxy_visual_asset"
    assert data["objects"][0]["visual_asset"]["diagnostic_only"] is True


def test_build_manifest_accepts_external_3dgs_background_render(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_placeholder_file(run_dir / "video" / "3dgs" / "run" / "config.yml", "viewer: splatfacto")
    _write_placeholder_file(run_dir / "video" / "3dgs" / "run" / "nerfstudio_models" / "step.ckpt")
    _write_placeholder_file(run_dir / "qa" / "background_3dgs_render.png", "png")
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

    data = build_sim_export_manifest(run_dir).to_dict()

    assert data["background"]["visual_asset"]["status"] == "ready"
    assert data["background"]["visual_asset"]["source_kind"] == "external_3dgs_renderer"
    assert data["background"]["visual_asset"]["path"] == "qa/background_3dgs_render.png"
    assert data["background"]["visual_asset"]["simulator_native"] is False
    assert data["background"]["visual_asset"]["external_render_verified"] is True
    assert "blocked_background_proxy_visual:bg_only_cloud" not in data["blocking_reasons"]
    assert "background_3dgs_runtime_not_verified" not in data["blocking_reasons"]


def test_build_manifest_blocks_external_3dgs_render_with_placeholder_registration(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_placeholder_file(run_dir / "video" / "3dgs" / "run" / "config.yml", "viewer: splatfacto")
    _write_placeholder_file(run_dir / "video" / "3dgs" / "run" / "nerfstudio_models" / "step.ckpt")
    _write_placeholder_file(run_dir / "qa" / "background_3dgs_render.png", "png")
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

    data = build_sim_export_manifest(run_dir).to_dict()

    assert data["background"]["registration"]["status"] == "blocked_3dgs_registration_placeholder"
    assert "background_3dgs_registration_placeholder" in data["blocking_reasons"]


def test_build_manifest_accepts_manual_refined_pose_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    obj_dir = run_dir / "objects" / "cup"
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    scene["objects"][0]["needs_manual_refine"] = True
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    (obj_dir / "pose_refinement_report.json").write_text(
        json.dumps(
            {
                "status": "accepted",
                "source": "manual_refined_from_auto",
                "rotation_source": "manual_refined_from_auto",
                "translation_source": "manual_refined_from_auto",
                "scale_source": "rgbd_alignment",
            }
        ),
        encoding="utf-8",
    )

    data = build_sim_export_manifest(run_dir).to_dict()

    assert "pose_needs_manual_refine:cup" not in data["blocking_reasons"]
    assert data["objects"][0]["pose"]["status"] == "ready"
    assert data["objects"][0]["pose"]["needs_manual_refine"] is False


def test_build_manifest_blocks_visual_mesh_penetrating_support_plane(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    scene["objects"][0]["T_object_to_world"][2][3] = -0.10
    scene["objects"][0]["T_object_to_camera"][2][3] = -0.10
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    (run_dir / "exports").mkdir()
    (run_dir / "exports" / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (run_dir / "exports" / "genesis_scene.py").write_text("# script\n", encoding="utf-8")
    (run_dir / "exports" / "isaac_scene.py").write_text("# script\n", encoding="utf-8")

    data = build_sim_export_manifest(run_dir).to_dict()

    assert data["status"] == "blocked"
    assert "pose_support_penetration:cup" in data["blocking_reasons"]
    assert data["objects"][0]["pose"]["status"] == "blocked_pose_support_penetration"
    assert data["objects"][0]["pose"]["support_alignment"]["status"] == "penetrating_support"
    assert data["objects"][0]["pose"]["support_alignment"]["bottom_z_m"] < 0.0


def test_write_manifest_persists_schema_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    path = write_sim_export_manifest(run_dir)

    assert path == run_dir / "sim_export_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["source_scene_manifest"] == "scene_manifest.json"
    assert data["coordinate_frames"]["world"] == "z_up_ground_plane_meters"
    assert data["objects"][0]["pose"]["status"] == "ready"
    assert data["qa"]["sim_export_report"] == "qa/sim_export_report.json"
