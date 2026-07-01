import json

import trimesh

from real2sim_scene_foundry.genesis_export import apply_genesis_settle_writeback, export_genesis_scene
from real2sim_scene_foundry.isaac_export import export_isaac_scene
from real2sim_scene_foundry.export_qa import write_sim_export_report
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
    trimesh.creation.box(extents=(1.0, 0.8, 0.04)).export(bg_dir / "table_collision.glb")
    _write_text(obj_dir / "mask.png")
    _write_text(obj_dir / "crop.png")
    _write_text(bg_dir / "bg_only.png")
    _write_text(bg_dir / "foreground_mask.png")
    _write_ply(bg_dir / "bg_only_cloud.ply")
    _write_text(run_dir / "camera.json", json.dumps({"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480}))
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
                    "source_backend": "background_support_points_rect",
                    "height_world_m": 0.0,
                    "normal_world": [0.0, 0.0, 1.0],
                    "applied_to_world_frame": True,
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
                        "needs_manual_refine": False,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_sim_export_manifest(run_dir)


def _write_external_unregistered_background(run_dir):
    bg_dir = run_dir / "background"
    qa_dir = run_dir / "qa"
    _write_text(qa_dir / "background_3dgs_render.png")
    (qa_dir / "background_3dgs_render_report.json").write_text(
        json.dumps(
            {
                "status": "rendered",
                "backend": "external_3dgs_renderer",
                "render_path": "qa/background_3dgs_render.png",
                "registered_3dgs_rendered": True,
                "simulator_native": False,
            }
        ),
        encoding="utf-8",
    )
    (bg_dir / "registration.json").write_text(
        json.dumps(
            {
                "status": "partial_external_render_only",
                "T_3dgs_world_to_sim_world": None,
                "scale_source": "input_metric_depth",
                "transform_sources": {
                    "T_3dgs_world_to_sim_world": "blocked_missing_camera_pose_scale_evidence"
                },
            }
        ),
        encoding="utf-8",
    )


def test_usd_export_writes_mesh_references_physics_collision_and_camera(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    result = export_usd_scene(run_dir)

    text = result.scene_path.read_text(encoding="utf-8")
    assert result.report["status"] == "exported"
    assert 'def Xform "World"' in text
    assert 'def Xform "SupportSurface"' in text
    assert 'def Xform "cup"' in text
    assert 'prepend references = @../objects/cup/visual.glb@' in text
    assert 'prepend references = @../objects/cup/collision.glb@' in text
    assert "PhysicsRigidBodyAPI" in text
    assert "PhysicsCollisionAPI" in text
    assert "physics:mass = 0.25" in text
    assert "rsf:friction = 0.8" in text
    assert "rsf:asset_scale = 0.5" in text
    assert 'def Camera "ReferenceCamera"' in text
    assert result.report["object_visual_refs"] == [
        {
            "object_id": "cup",
            "path": "objects/cup/visual.glb",
            "status": "ready",
            "source": "sam3d_or_visual_reconstruction",
            "diagnostic_only": False,
            "referenced_by_runtime": True,
        }
    ]
    assert result.report["object_collision_refs"][0]["path"] == "objects/cup/collision.glb"
    assert result.report["object_collision_refs"][0]["referenced_by_runtime"] is True
    assert result.report["background_asset"]["path"] == "background/bg_only_cloud.ply"
    assert result.report["background_asset"]["gaussian_splat_config_path"] == "video/3dgs/run/config.yml"
    assert result.report["background_runtime"]["native_3dgs_status"] == "unsupported_unverified"
    assert result.report["background_runtime"]["native_3dgs_supported"] is False
    assert result.report["debug_proxy_refs"][0]["path"] == "objects/cup/debug_bbox.glb"
    assert result.report["debug_proxy_visible"] is False


def test_usd_export_marks_external_unregistered_background_as_incomplete(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_external_unregistered_background(run_dir)

    result = export_usd_scene(run_dir)

    assert result.report["background_asset"]["path"] == "qa/background_3dgs_render.png"
    assert result.report["background_asset"]["source_kind"] == "external_3dgs_renderer"
    assert result.report["background_asset"]["registration_status"] == "partial_external_render_only"
    assert result.report["background_runtime"]["native_3dgs_supported"] is False
    assert result.report["background_completion"]["status"] == "blocked"
    assert result.report["background_completion"]["complete"] is False
    assert result.report["background_completion"]["simulator_native"] is False
    assert "background_unregistered" in result.report["background_completion"]["blocking_reasons"]
    assert "external_3dgs_render_sidecar_not_native_simulator_background" in result.report["background_completion"]["caveats"]


def test_genesis_export_writes_loader_that_uses_manifest_physics_and_settle(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    result = export_genesis_scene(run_dir, settle_steps=25)

    script = result.script_path.read_text(encoding="utf-8")
    compile(script, str(result.script_path), "exec")
    assert result.report["status"] == "script_written"
    assert "sim_export_manifest.json" in script
    assert "collision_asset" in script
    assert "gs.Scene" in script
    assert "gs.morphs.Mesh" in script
    assert "scale=asset_scale" in script
    assert "set_mass" in script
    assert script.index("scene.build()") < script.index("set_mass")
    assert "genesis_settle_report.json" in script
    assert "stability_status" in script
    assert "object_visual_refs" in script
    assert "object_collision_refs" in script
    assert "background_runtime" in script
    assert "background_completion" in script
    assert "debug_proxy_visible" in script
    assert result.report["object_visual_refs"][0]["path"] == "objects/cup/visual.glb"
    assert result.report["object_visual_refs"][0]["referenced_by_runtime"] is False
    assert result.report["object_collision_refs"][0]["path"] == "objects/cup/collision.glb"
    assert result.report["object_collision_refs"][0]["referenced_by_runtime"] is True
    assert result.report["background_asset"]["path"] == "background/bg_only_cloud.ply"
    assert result.report["background_asset"]["referenced_by_runtime"] is False
    assert result.report["background_runtime"]["native_3dgs_status"] == "unsupported_unverified"
    assert result.report["debug_proxy_visible"] is False


def test_genesis_export_marks_external_unregistered_background_as_incomplete(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    _write_external_unregistered_background(run_dir)

    result = export_genesis_scene(run_dir, settle_steps=25)

    assert result.report["background_asset"]["path"] == "qa/background_3dgs_render.png"
    assert result.report["background_asset"]["referenced_by_runtime"] is False
    assert result.report["background_completion"]["status"] == "blocked"
    assert result.report["background_completion"]["complete"] is False
    assert result.report["background_completion"]["simulator_native"] is False
    assert "background_unregistered" in result.report["background_completion"]["blocking_reasons"]


def test_genesis_export_settle_delta_report_contract_exposes_initial_vs_settled_pose(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    result = export_genesis_scene(run_dir, settle_steps=25)

    script = result.script_path.read_text(encoding="utf-8")
    assert result.report["writes_settled_pose_delta_report"] == "qa/settled_pose_delta_report.json"
    assert result.report["visual_physics_coherence"]["status"] == "partial"
    assert result.report["visual_physics_coherence"]["visual_assets_referenced_by_runtime"] is False
    assert result.report["visual_physics_coherence"]["collision_assets_referenced_by_runtime"] is True
    assert "settled_pose_delta_report.json" in script
    assert "settled_pose_delta_report" in script
    assert "translation_delta_m" in script
    assert "pose_written_back" in script
    assert "viewer_export_action" in script
    assert "requires_viewer_export_delta_display" in script
    assert "visual_physics_coherence" in script


def test_apply_genesis_settle_writeback_updates_settled_manifest_and_pose_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    settled = [[1, 0, 0, 0.2], [0, 1, 0, 0.1], [0, 0, 1, 0.3], [0, 0, 0, 1]]
    (run_dir / "objects" / "cup" / "pose.json").write_text(
        json.dumps({"object_id": "cup", "T_object_to_world": IDENTITY, "T_object_to_camera": IDENTITY}),
        encoding="utf-8",
    )
    (run_dir / "qa").mkdir(exist_ok=True)
    (run_dir / "qa" / "genesis_settle_report.json").write_text(
        json.dumps({"status": "completed", "stability_status": "passed"}),
        encoding="utf-8",
    )
    (run_dir / "qa" / "settled_pose_delta_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "objects": [
                    {
                        "object_id": "cup",
                        "settled_T_object_to_world": settled,
                        "translation_delta_m": [0.2, 0.1, 0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = apply_genesis_settle_writeback(run_dir, writeback=True)

    assert result["status"] == "completed"
    assert result["pose_written_back"] is True
    settled_manifest = json.loads((run_dir / "scene_manifest.settled.json").read_text(encoding="utf-8"))
    live_manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    pose = json.loads((run_dir / "objects" / "cup" / "pose.json").read_text(encoding="utf-8"))
    assert settled_manifest["objects"][0]["T_object_to_world"] == settled
    assert live_manifest["objects"][0]["T_object_to_world"] == settled
    assert pose["T_object_to_world"] == settled


def test_genesis_export_prioritizes_polygon_slab_support_mesh_over_box_proxy(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    result = export_genesis_scene(run_dir, settle_steps=25)

    script = result.script_path.read_text(encoding="utf-8")
    support_block = script[script.index('support = manifest.get("support_surface", {})') : script.index("entities = {}")]
    assert "table_collision_geometry_type" in support_block
    assert 'support_geometry_type in {"polygon_slab", "convex_hull_slab"}' in support_block
    assert support_block.index("gs.morphs.Mesh") < support_block.index("gs.morphs.Box")
    mesh_branch = support_block[support_block.index("if use_support_mesh:") : support_block.index("elif support_size and support_pos:")]
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


def test_genesis_export_settle_report_detects_vertical_fall_and_large_displacement(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    result = export_genesis_scene(run_dir, settle_steps=25)

    script = result.script_path.read_text(encoding="utf-8")
    assert "gs.morphs.Box" in script
    assert "support_top_z" in script
    assert "fall_below_support" in script
    assert "max_allowed_displacement_m" in script
    assert 'fall_below_support_detected": bool(fall_below_support_detected)' in script
    assert 'stability_status = "passed" if not nan_detected and not fall_out_detected and not fall_below_support_detected and not excessive_displacement_detected else "failed"' in script


def test_isaac_export_marks_runtime_unavailable_instead_of_passed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)

    result = export_isaac_scene(run_dir, python_executable=None)

    script = result.script_path.read_text(encoding="utf-8")
    compile(script, str(result.script_path), "exec")
    assert "SimulationApp" in script
    assert "Usd.Stage.Open" in script
    assert 'isaac_load_report.json' in script
    assert 'emit_report' in script
    assert "_probe_native_3dgs_runtime" in script
    assert script.index("SimulationApp({") < script.index("native_3dgs_report = _probe_native_3dgs_runtime")
    assert "get_extension_manager" in script
    assert "native_3dgs_status" in script
    assert "native_3dgs_evidence" in script
    assert "debug_proxy_visible" in script
    assert result.report_path == run_dir / "qa" / "isaac_load_report.json"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "runtime_unavailable"
    assert report["status"] != "passed"
    assert report["native_3dgs_required"] is True
    assert report["native_3dgs_supported"] is False
    assert report["native_3dgs_status"] == "runtime_unavailable"
    assert report["native_3dgs_evidence"]["probe_phase"] == "not_run"
    assert report["native_3dgs_evidence"]["reason"] == "no_isaac_python_executable_provided"
    assert report["object_visual_refs"][0]["path"] == "objects/cup/visual.glb"
    assert report["object_collision_refs"][0]["path"] == "objects/cup/collision.glb"
    assert report["background_asset"]["path"] == "background/bg_only_cloud.ply"
    assert report["background_runtime"]["native_3dgs_required"] is True
    assert report["background_runtime"]["native_3dgs_status"] == "runtime_unavailable"
    assert report["background_runtime"]["native_3dgs_supported"] is False
    assert report["background_runtime"]["native_3dgs_evidence"]["probe_phase"] == "not_run"
    assert report["report_source"] == "direct_runtime_unavailable"
    assert report["repeatability"]["status"] == "direct_runtime_unavailable"
    assert report["repeatability"]["fresh_direct_validation"] is False
    assert report["repeatability"]["direct_worker_access_available"] is False
    assert report["debug_proxy_refs"][0]["path"] == "objects/cup/debug_bbox.glb"
    assert report["debug_proxy_visible"] is False


def test_isaac_export_preserves_existing_loaded_worker_report_without_local_runtime(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    existing = {
        "status": "loaded",
        "object_count": 1,
        "missing_prims": [],
        "native_3dgs_required": False,
        "native_3dgs_supported": False,
        "native_3dgs_status": "not_requested",
    }
    _write_text(run_dir / "qa" / "isaac_load_report.json", json.dumps(existing))

    result = export_isaac_scene(run_dir, python_executable=None)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "loaded"
    assert report["object_count"] == 1
    assert report["report_source"] == "preserved_existing_worker_report"
    assert report["validation_reused"] is True
    assert report["repeatability"]["status"] == "preserved_existing_worker_report"
    assert report["repeatability"]["fresh_direct_validation"] is False
    assert report["repeatability"]["direct_worker_access_available"] is False
    assert "not a fresh direct Isaac validation" in report["repeatability"]["caveat"]
    assert report["object_visual_refs"][0]["path"] == "objects/cup/visual.glb"
    assert report["object_collision_refs"][0]["path"] == "objects/cup/collision.glb"
    assert report["debug_proxy_visible"] is False

    qa = write_sim_export_report(run_dir).report
    isaac_section = qa["sections"]["isaac_export"]
    assert isaac_section["status"] == "passed"
    assert isaac_section["report_source"] == "preserved_existing_worker_report"
    assert isaac_section["validation_reused"] is True
    assert isaac_section["repeatability"]["status"] == "preserved_existing_worker_report"
    assert isaac_section["repeatability"]["fresh_direct_validation"] is False
    assert isaac_section["repeatability_status"] == "partial_preserved_report"


def test_isaac_export_parses_json_report_after_runtime_logs(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    executable = tmp_path / "python.sh"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    class Completed:
        returncode = 0
        stdout = (
            'runtime log\n'
            '{"status":"loaded","object_count":1,"missing_prims":[],'
            '"native_3dgs_supported":true,'
            '"native_3dgs_status":"native_3dgs_loaded",'
            '"native_3dgs_evidence":{"probe_phase":"after_simulation_app","extension_matches":["omni.example.splat"]},'
            '"background_runtime_verified":true}\n'
        )
        stderr = "warning log"

    monkeypatch.setattr("real2sim_scene_foundry.isaac_export.subprocess.run", lambda *args, **kwargs: Completed())

    result = export_isaac_scene(run_dir, python_executable=executable)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "loaded"
    assert report["object_count"] == 1
    assert report["stderr_tail"] == "warning log"
    assert report["native_3dgs_required"] is True
    assert report["native_3dgs_supported"] is True
    assert report["native_3dgs_status"] == "native_3dgs_loaded"
    assert report["native_3dgs_evidence"]["probe_phase"] == "after_simulation_app"
    assert report["background_runtime_verified"] is True
    assert report["object_visual_refs"][0]["path"] == "objects/cup/visual.glb"
    assert report["object_collision_refs"][0]["path"] == "objects/cup/collision.glb"
    assert report["background_asset"]["path"] == "background/bg_only_cloud.ply"
    assert report["background_runtime"]["native_3dgs_required"] is True
    assert report["background_runtime"]["native_3dgs_status"] == "native_3dgs_loaded"
    assert report["background_runtime"]["native_3dgs_supported"] is True
    assert report["background_runtime"]["native_3dgs_evidence"]["extension_matches"] == ["omni.example.splat"]
    assert report["debug_proxy_visible"] is False



def test_isaac_export_reads_report_file_when_stdout_has_no_json(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready_run(run_dir)
    executable = tmp_path / "python.sh"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    class Completed:
        returncode = 0
        stdout = "runtime log without json"
        stderr = "warning log"

    def fake_run(*args, **kwargs):
        report_path = run_dir / "qa" / "isaac_load_report.json"
        report_path.write_text(
            json.dumps({"status": "loaded", "object_count": 1, "missing_prims": []}),
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr("real2sim_scene_foundry.isaac_export.subprocess.run", fake_run)

    result = export_isaac_scene(run_dir, python_executable=executable)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert result.report["status"] == "loaded"
    assert report["object_count"] == 1
    assert report["stdout_tail"] == "runtime log without json"
    assert report["stderr_tail"] == "warning log"
    assert report["object_visual_refs"][0]["path"] == "objects/cup/visual.glb"
    assert report["object_collision_refs"][0]["path"] == "objects/cup/collision.glb"
    assert report["background_asset"]["path"] == "background/bg_only_cloud.ply"
    assert report["background_runtime"]["native_3dgs_status"] == "unsupported_unverified"
    assert report["debug_proxy_visible"] is False
