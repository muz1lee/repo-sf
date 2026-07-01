import json
from pathlib import Path

from real2sim_scene_foundry.cli import main


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_export_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    obj_dir = run / "objects" / "cup"
    bg_dir = run / "background"
    exports_dir = run / "exports"
    qa_dir = run / "qa"
    obj_dir.mkdir(parents=True)
    bg_dir.mkdir(parents=True)
    exports_dir.mkdir(parents=True)
    qa_dir.mkdir(parents=True)

    _write(obj_dir / "visual.glb")
    _write(obj_dir / "collision.glb")
    _write(obj_dir / "debug_bbox.glb")
    _write(obj_dir / "mask.png")
    _write(obj_dir / "crop.png")
    _write(bg_dir / "bg_only_cloud.ply", "ply\n")
    _write(bg_dir / "table_collision.glb")
    _write(exports_dir / "scene.usda", "#usda 1.0\n")
    _write(exports_dir / "isaac_scene.py", "print('isaac loader')\n")
    (obj_dir / "physics.json").write_text(
        json.dumps({"mass_kg": 0.2, "friction": 0.8, "restitution": 0.0}),
        encoding="utf-8",
    )
    (bg_dir / "registration.json").write_text(
        json.dumps(
            {
                "status": "registered",
                "scale_source": "input_metric_depth",
                "T_3dgs_world_to_sim_world": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        ),
        encoding="utf-8",
    )
    transform = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]]
    (run / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "background": {
                    "source_backend": "opencv_inpaint_single_frame",
                    "point_cloud_path": "background/bg_only_cloud.ply",
                },
                "support_plane": {
                    "status": "estimated",
                    "table_collision_mesh_path": "background/table_collision.glb",
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/visual.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": transform,
                        "T_object_to_world": transform,
                        "scale_m": 0.2,
                        "mass_kg": 0.2,
                        "friction": 0.8,
                        "confidence": 0.9,
                        "needs_manual_refine": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_cli_isaac_worker_bundle_writes_minimal_file_list_and_operator_commands(tmp_path):
    run = _minimal_export_run(tmp_path)

    code = main(["isaac-worker-bundle", "--run-dir", str(run), "--worker-run-dir", "/worker/run"])

    assert code == 0
    bundle = json.loads((run / "exports" / "isaac_worker_bundle.json").read_text(encoding="utf-8"))
    files = bundle["files"]
    assert "scene_manifest.json" in files
    assert "sim_export_manifest.json" in files
    assert "exports/scene.usda" in files
    assert "exports/isaac_scene.py" in files
    assert "objects/cup/visual.glb" in files
    assert "objects/cup/collision.glb" in files
    assert "background/bg_only_cloud.ply" in files
    assert "scene_cloud.ply" not in files
    assert bundle["commands"]["transfer_to_worker"].startswith("ssh wenqian_h200 ")
    assert "| ssh -p 1024 root@101.132.143.105 " in bundle["commands"]["transfer_to_worker"]
    assert "/isaac-sim/python.sh exports/isaac_scene.py --run-dir /worker/run" in bundle["commands"]["run_on_worker"]
    assert "rsf isaac-worker-report" in bundle["commands"]["ingest_report_to_h200"]
    assert (run / "exports" / "isaac_worker_bundle_files.txt").read_text(encoding="utf-8").splitlines() == files


def test_cli_isaac_worker_report_missing_marks_runtime_unavailable_not_passed(tmp_path):
    run = _minimal_export_run(tmp_path)

    code = main(["isaac-worker-report", "--run-dir", str(run), "--report", str(run / "qa" / "missing.json")])

    assert code == 0
    report = json.loads((run / "qa" / "isaac_load_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "runtime_unavailable"
    assert report["reason"] == "isaac_worker_report_missing"
    assert report["status"] != "passed"
    manifest = json.loads((run / "sim_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["exports"]["isaac"]["status"] == "runtime_unavailable"
    assert "isaac_runtime_unavailable" in manifest["blocking_reasons"]

    assert main(["qa-sim", "--run-dir", str(run)]) == 0
    qa = json.loads((run / "qa" / "sim_export_report.json").read_text(encoding="utf-8"))
    assert qa["sections"]["isaac_export"]["status"] == "runtime_unavailable"
    assert qa["overall_status"] == "blocked"


def test_cli_isaac_worker_report_ingests_and_enriches_loaded_report(tmp_path):
    run = _minimal_export_run(tmp_path)
    worker_report = run / "qa" / "worker_report.json"
    worker_report.write_text(json.dumps({"status": "loaded", "prim_count": 7}), encoding="utf-8")

    code = main(["isaac-worker-report", "--run-dir", str(run), "--report", str(worker_report)])

    assert code == 0
    report = json.loads((run / "qa" / "isaac_load_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "loaded"
    assert report["prim_count"] == 7
    assert report["worker_bridge"]["status"] == "ingested"
    assert report["worker_bridge"]["ingested_from"] == str(worker_report)
    assert report["object_visual_refs"][0]["object_id"] == "cup"
