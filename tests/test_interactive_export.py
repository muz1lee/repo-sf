import json

import trimesh

from real2sim_scene_foundry.interactive import export_interactive_scene


def _write_manifest(run_dir):
    mesh_path = run_dir / "objects" / "cup" / "mesh_aligned.glb"
    mesh_path.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(mesh_path)
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
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/mesh_aligned.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        "T_object_to_world": [[1, 0, 0, 0.2], [0, 1, 0, 0.3], [0, 0, 1, 0.4], [0, 0, 0, 1]],
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
    assert "gs.morphs.Mesh" in script
    assert "gs.materials.Rigid" in script
    assert "set_mass" in script
    assert "genesis_settle_report.json" in script
    assert "qa_report.json" in script
    assert "--backend" in script
    assert "gs.cpu" in script
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["physics_settle"]["status"] == "proxy_checked"
    assert report["physics_settle"]["settle_steps"] == 25
    assert report["objects"][0]["object_id"] == "cup"
    assert report["objects"][0]["mesh_loadable"] is True
