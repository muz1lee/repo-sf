import json

import pytest
import trimesh

from real2sim_scene_foundry.interactive import export_interactive_scene


def _write_manifest(run_dir):
    mesh_path = run_dir / "objects" / "cup" / "mesh_aligned.glb"
    mesh_path.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(mesh_path)
    table_path = run_dir / "background" / "table_collision.glb"
    table_path.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.8, 0.6, 0.04), transform=trimesh.transformations.translation_matrix((0.0, 0.0, -0.02))).export(table_path)
    (run_dir / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "background": {
                    "source_backend": "opencv_inpaint_single_frame",
                    "bg_only_image_path": "background/bg_only.png",
                    "foreground_mask_path": "background/foreground_mask.png",
                    "point_cloud_path": "background/bg_only_cloud.ply",
                    "status": "proxy_from_single_stereo_pair",
                },
                "support_plane": {
                    "status": "estimated",
                    "source_backend": "object_mesh_bottom_median",
                    "height_world_m": 0.0,
                    "normal_world": [0.0, 0.0, 1.0],
                    "applied_to_world_frame": True,
                    "table_collision_mesh_path": "background/table_collision.glb",
                    "table_collision_source_backend": "tabletop_mask_polygon_slab",
                    "table_collision_geometry_type": "polygon_slab",
                    "table_collision_final": True,
                    "table_bounds_world_xy": [[-0.4, -0.3], [0.4, 0.3]],
                    "table_collision_pos_world": [0.0, 0.0, -0.02],
                    "table_collision_size_xyz": [0.8, 0.6, 0.04],
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/mesh_aligned.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        "T_object_to_world": [[0, -1, 0, 0.2], [1, 0, 0, 0.3], [0, 0, 1, 0.4], [0, 0, 0, 1]],
                        "scale_m": 0.2,
                        "mass_kg": 0.25,
                        "friction": 0.8,
                        "confidence": 0.9,
                        "source_backend": "sam3d_aligned",
                        "needs_manual_refine": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_export_interactive_scene_writes_genesis_launcher_and_proxy_settle(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)

    result = export_interactive_scene(run_dir, settle_steps=25)

    script = result.script_path.read_text(encoding="utf-8")
    compile(script, str(result.script_path), "exec")
    assert "gs.morphs.Mesh" in script
    assert "gs.materials.Rigid" in script
    assert "set_mass" in script
    assert "genesis_settle_report.json" in script
    assert "qa_report.json" in script
    assert "--backend" in script
    assert "gs.cpu" in script
    assert "table_collision_mesh_path" in script
    assert "gs.morphs.Box" in script
    assert "quat=quat" in script
    assert "align=False" in script
    assert "final_pose_by_object" in script
    assert "penetration_depth_m" in script
    assert "fall_out" in script
    assert "get_AABB" in script
    assert "stability_status" in script
    assert "settled_pose_delta_report.json" in script
    assert "translation_delta_m" in script
    assert "pose_written_back" in script
    assert "viewer_export_action" in script
    assert "visual_physics_coherence" in script
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["support_plane"]["status"] == "estimated"
    assert report["support_plane"]["table_collision_mesh_path"] == "background/table_collision.glb"
    assert report["support_plane"]["table_collision_geometry_type"] == "polygon_slab"
    assert report["support_plane"]["table_collision_final"] is True
    assert report["support_plane"]["table_collision_size_xyz"] == [0.8, 0.6, 0.04]
    assert report["physics_settle"]["status"] == "proxy_checked"
    assert report["physics_settle"]["settle_steps"] == 25
    assert report["physics_settle"]["settled_pose_delta_report"] == "qa/settled_pose_delta_report.json"
    assert report["physics_settle"]["visual_physics_coherence"]["status"] == "partial"
    assert report["objects"][0]["object_id"] == "cup"
    assert report["objects"][0]["mesh_loadable"] is True
    assert report["objects"][0]["world_quat_wxyz"] == pytest.approx([0.70710678, 0.0, 0.0, 0.70710678])


def test_export_interactive_scene_prioritizes_polygon_slab_support_mesh_over_box_proxy(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)

    result = export_interactive_scene(run_dir, settle_steps=25)

    script = result.script_path.read_text(encoding="utf-8")
    support_block = script[script.index("support_plane = manifest.get") : script.index("entities = []")]
    assert "table_collision_geometry_type" in support_block
    assert 'support_geometry_type in {"polygon_slab", "convex_hull_slab"}' in support_block
    assert support_block.index("gs.morphs.Mesh") < support_block.index("gs.morphs.Box")
    mesh_branch = support_block[support_block.index("if use_support_mesh:") : support_block.index("elif table_collision_size:")]
    assert "gs.morphs.Mesh" in mesh_branch
    assert "fixed=True" in mesh_branch
    assert "collision=True" in mesh_branch
    assert "convexify=False" in mesh_branch
    assert "file_meshes_are_zup=True" in mesh_branch
    assert "gs.morphs.Box" not in mesh_branch
    assert "support_surface_runtime" in script
    assert "used_polygon_slab_mesh" in script
    assert "used_box_proxy" in script
    assert "support_mesh_file_meshes_are_zup" in script


def test_export_interactive_scene_reports_collision_assets_used_by_genesis(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)

    result = export_interactive_scene(run_dir, settle_steps=25)

    script = result.script_path.read_text(encoding="utf-8")
    compile(script, str(result.script_path), "exec")
    assert 'collision_asset = obj.get("collision_asset", {})' in script
    assert 'collision_mesh_file = args.run_dir / collision_asset["path"]' in script
    assert 'genesis_collision_source' in script
    assert 'obj["mesh_path"]' not in script
    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["objects"][0]["collision_asset"]["path"] == "objects/cup/collision.glb"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    obj_report = report["objects"][0]
    assert obj_report["collision_asset"]["path"] == "objects/cup/collision.glb"
    assert obj_report["collision_source"] == "bbox_from_legacy_mesh_proxy"
    assert obj_report["physics"]["path"] == "objects/cup/physics.json"
    assert report["physics_settle"]["collision_sources_by_object"] == {"cup": "bbox_from_legacy_mesh_proxy"}
