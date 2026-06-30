import json

import trimesh

from real2sim_scene_foundry.cli import main
from real2sim_scene_foundry.composite_viewer import export_composite_viewer


def _write_run(run_dir):
    object_dir = run_dir / "objects" / "cup"
    object_dir.mkdir(parents=True)
    table_dir = run_dir / "background"
    table_dir.mkdir()
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(object_dir / "mesh_aligned.glb")
    trimesh.creation.box(extents=(1.0, 0.8, 0.04)).export(table_dir / "table_collision.glb")
    (run_dir / "scene_cloud.ply").write_text(
        "\n".join(
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
                "0 0 1 120 130 140",
            ]
        )
        + "\n",
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
                },
            }
        ),
        encoding="utf-8",
    )


def test_export_composite_viewer_writes_browser_assets(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run)

    result = export_composite_viewer(run)

    assert result.index_path == run / "exports" / "composite_viewer" / "index.html"
    assert result.config_path == run / "exports" / "composite_viewer" / "viewer_config.json"
    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["background"]["point_cloud_path"] == "../../scene_cloud.ply"
    assert config["background"]["source_backend"] == "video_3dgs_splatfacto_proxy_point_cloud"
    assert config["support_plane"]["table_collision_mesh_path"] == "../../background/table_collision.glb"
    assert config["objects"][0]["mesh_path"] == "../../objects/cup/mesh_aligned.glb"
    assert config["objects"][0]["world_position"] == [0.1, 1.0, 0.2]
    assert config["objects"][0]["world_quat_wxyz"] == [0.7071067811865476, 0.0, 0.0, 0.7071067811865475]
    assert config["qa"]["stability_status"] == "passed"
    assert config["qa"]["max_penetration_depth_m"] == 0.001
    html = result.index_path.read_text(encoding="utf-8")
    assert "PLYLoader" in html
    assert "GLTFLoader" in html
    assert "viewer_config.json" in html


def test_cli_composite_viewer_export_only(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run)

    code = main(["composite-viewer", "--run-dir", str(run), "--export-only"])

    assert code == 0
    assert (run / "exports" / "composite_viewer" / "index.html").is_file()
    assert (run / "exports" / "composite_viewer" / "viewer_config.json").is_file()
