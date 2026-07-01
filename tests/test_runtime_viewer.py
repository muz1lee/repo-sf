import json

import trimesh

from real2sim_scene_foundry.runtime_viewer import export_runtime_viewer
from real2sim_scene_foundry.sim_export_manifest import write_sim_export_manifest


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]]


def _write_file(path, text="placeholder"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_ply(path):
    _write_file(
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
    trimesh.creation.box(extents=(1.0, 0.8, 0.04)).export(bg_dir / "table_collision.glb")
    _write_file(obj_dir / "mask.png")
    _write_file(obj_dir / "crop.png")
    _write_file(bg_dir / "bg_only.png")
    _write_file(bg_dir / "foreground_mask.png")
    _write_ply(bg_dir / "bg_only_cloud.ply")
    _write_file(run_dir / "qa" / "background_3dgs_render.png", "png")
    _write_file(
        run_dir / "qa" / "background_3dgs_render_report.json",
        json.dumps(
            {
                "status": "rendered",
                "backend": "external_3dgs_renderer",
                "registered_3dgs_rendered": True,
                "simulator_native": False,
                "render_path": "qa/background_3dgs_render.png",
                "image_size": [640, 480],
            }
        ),
    )
    _write_file(
        obj_dir / "physics.json",
        json.dumps({"mass_kg": 0.25, "friction": 0.8, "restitution": 0.01, "source": "class_default"}),
    )
    _write_file(
        obj_dir / "pose_refinement_report.json",
        json.dumps({"status": "accepted", "rotation_source": "service_pose_refined", "translation_source": "rgbd_alignment"}),
    )
    _write_file(bg_dir / "registration.json", json.dumps({"status": "registered", "scale_source": "metric_reference_camera"}))
    _write_file(
        run_dir / "scene_manifest.json",
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {"camera": "opencv_x_right_y_down_z_forward_meters", "world": "z_up_ground_plane_meters"},
                "background": {
                    "source_backend": "video_3dgs_splatfacto",
                    "status": "trained_video_3dgs",
                    "point_cloud_path": "background/bg_only_cloud.ply",
                    "gaussian_splat_config_path": "video/3dgs/run/config.yml",
                },
                "support_plane": {
                    "status": "estimated",
                    "source_backend": "background_support_points_rect",
                    "height_world_m": 0.0,
                    "table_collision_mesh_path": "background/table_collision.glb",
                    "table_collision_pos_world": [0.0, 0.0, -0.02],
                    "table_collision_size_xyz": [1.0, 0.8, 0.04],
                    "table_bounds_world_xy": [[-0.5, -0.4], [0.5, 0.4]],
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
                        "asset_scale": 0.5,
                        "scale_m": 0.2,
                        "mass_kg": 0.25,
                        "friction": 0.8,
                        "confidence": 0.9,
                        "source_backend": "sam3d_aligned",
                    }
                ],
            },
            indent=2,
        ),
    )
    write_sim_export_manifest(run_dir)


def test_export_genesis_runtime_viewer_writes_real_runtime_api_and_click_ui(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    result = export_runtime_viewer(run_dir, backend="genesis")

    script = result.script_path.read_text(encoding="utf-8")
    html = result.index_path.read_text(encoding="utf-8")
    runtime_manifest = json.loads(result.runtime_manifest_path.read_text(encoding="utf-8"))
    compile(script, str(result.script_path), "exec")
    assert runtime_manifest["backend"] == "genesis"
    assert runtime_manifest["status"] == "runtime_script_written"
    assert runtime_manifest["background"]["viewer_status"] == "partial_external_render_only"
    assert runtime_manifest["background"]["render_mode"] == "external_3dgs_png_sidecar"
    assert runtime_manifest["background"]["live_3dgs_runtime"] is False
    assert runtime_manifest["background"]["simulator_native"] is False
    assert runtime_manifest["api"]["state"] == "/api/state"
    assert "import genesis as gs" in script
    assert "ThreadingHTTPServer" in script
    assert '"/api/state"' in script
    assert '"/api/apply-force"' in script
    assert '"/api/reset"' in script
    assert '"/api/replay"' in script
    assert "control_dofs_force" in script or "set_pos" in script
    assert "fixed=False" in script
    assert "get_pos" in script
    assert "Raycaster" in html
    assert "apply-force" in html
    assert "reset" in html
    assert "replay" in html
    assert "runtime backend: genesis" in html
    assert "partial_external_render_only" in html
    assert "Native/live 3DGS background is unavailable in this runtime viewer." in html
    assert result.report["background"]["viewer_status"] == "partial_external_render_only"


def test_export_isaac_runtime_viewer_is_explicitly_unavailable_without_runtime(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    result = export_runtime_viewer(run_dir, backend="isaac")

    runtime_manifest = json.loads(result.runtime_manifest_path.read_text(encoding="utf-8"))
    html = result.index_path.read_text(encoding="utf-8")
    assert result.script_path is None
    assert runtime_manifest["backend"] == "isaac"
    assert runtime_manifest["status"] == "runtime_unavailable"
    assert "isaac_runtime_unavailable" in runtime_manifest["blocked_reason"]
    assert "runtime_unavailable" in html
