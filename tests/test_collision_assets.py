import json
from pathlib import Path

import trimesh

from real2sim_scene_foundry.collision_assets import ensure_collision_assets, qa_physics


def _write_legacy_run(run_dir: Path, *, mesh_name: str = "mesh_aligned.glb") -> None:
    object_dir = run_dir / "objects" / "cup"
    object_dir.mkdir(parents=True)
    mesh_path = object_dir / mesh_name
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(mesh_path)
    (run_dir / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": str(mesh_path.relative_to(run_dir)),
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        "T_object_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1]],
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


def test_ensure_collision_assets_separates_roles_and_records_provenance(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_legacy_run(run_dir)

    report = ensure_collision_assets(run_dir)

    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    obj = manifest["objects"][0]
    assert obj["visual_asset"]["path"] == "objects/cup/mesh_aligned.glb"
    assert obj["visual_asset"]["status"] == "blocked_proxy_visual_asset"
    assert obj["visual_asset"]["final_visual"] is False
    assert obj["collision_asset"]["path"] == "objects/cup/collision.glb"
    assert obj["collision_asset"]["source"] == "bbox_from_legacy_mesh_proxy"
    assert obj["collision_asset"]["path"] != obj["visual_asset"]["path"]
    assert obj["debug_proxy"]["path"] == "objects/cup/debug_bbox.glb"
    assert obj["physics"]["path"] == "objects/cup/physics.json"
    assert (run_dir / obj["collision_asset"]["path"]).is_file()
    assert (run_dir / obj["debug_proxy"]["path"]).is_file()
    physics = json.loads((run_dir / obj["physics"]["path"]).read_text(encoding="utf-8"))
    assert physics["mass_kg"] == 0.25
    assert physics["friction"] == 0.8
    assert physics["restitution"] == 0.0
    assert physics["collision_source"] == "bbox_from_legacy_mesh_proxy"
    assert report["objects"][0]["collision_source"] == "bbox_from_legacy_mesh_proxy"


def test_ensure_collision_assets_preserves_existing_visual_mesh_and_uses_convex_hull(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    object_dir = run_dir / "objects" / "cup"
    object_dir.mkdir(parents=True)
    visual_path = object_dir / "visual.glb"
    trimesh.creation.icosphere(subdivisions=1, radius=0.1).export(visual_path)
    (run_dir / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/visual.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "visual_asset": {
                            "path": "objects/cup/visual.glb",
                            "source": "sam3d_mesh_glb_base64",
                            "status": "ready",
                            "final_visual": True,
                        },
                        "T_object_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        "T_object_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1]],
                        "scale_m": 0.2,
                        "mass_kg": 0.25,
                        "friction": 0.8,
                        "confidence": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ensure_collision_assets(run_dir)

    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    obj = manifest["objects"][0]
    assert obj["visual_asset"]["path"] == "objects/cup/visual.glb"
    assert obj["visual_asset"]["final_visual"] is True
    assert obj["collision_asset"]["path"] == "objects/cup/collision.glb"
    assert obj["collision_asset"]["source"] == "convex_hull_from_visual_mesh"
    assert obj["collision_asset"]["path"] != obj["visual_asset"]["path"]
    assert (run_dir / "objects/cup/physics.json").is_file()
    collision_report = json.loads((run_dir / "objects/cup/collision_report.json").read_text(encoding="utf-8"))
    assert collision_report["status"] == "partial"
    assert collision_report["reproduction_status"] == "partial"
    assert collision_report["paper_equivalent"] is False
    assert collision_report["collision_source"] == "convex_hull_from_visual_mesh"
    assert collision_report["decomposition_backend"] == "trimesh_convex_hull"
    assert collision_report["simfoundry_target_backend"] == "coacd"
    assert collision_report["availability"]["coacd"]["available"] is False
    assert collision_report["availability"]["vhacd"]["available"] is False
    assert "coacd_unavailable" in collision_report["not_paper_equivalent_reasons"]
    qa_report = json.loads((run_dir / "qa/collision_rebuild_report.json").read_text(encoding="utf-8"))
    assert qa_report["status"] == "partial"
    assert qa_report["reproduction_status"] == "partial"
    assert qa_report["paper_equivalent"] is False
    assert qa_report["objects"][0]["decomposition_backend"] == "trimesh_convex_hull"


def test_ensure_collision_assets_rebuilds_stale_collision_when_visual_changes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    object_dir = run_dir / "objects" / "cup"
    object_dir.mkdir(parents=True)
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(object_dir / "mesh_aligned.glb")
    trimesh.creation.icosphere(subdivisions=1, radius=0.1).export(object_dir / "visual.glb")
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(object_dir / "collision.glb")
    (run_dir / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/visual.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "visual_asset": {
                            "path": "objects/cup/visual.glb",
                            "source": "sam3d_mesh_job_download_glb_aligned",
                            "status": "ready",
                            "final_visual": True,
                        },
                        "collision_asset": {
                            "role": "collision_asset",
                            "path": "objects/cup/collision.glb",
                            "source": "bbox_from_legacy_mesh_proxy",
                            "status": "ready",
                            "provenance": {
                                "method": "bbox_proxy",
                                "visual_asset_path": "objects/cup/mesh_aligned.glb",
                            },
                        },
                        "T_object_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        "T_object_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1]],
                        "scale_m": 0.2,
                        "mass_kg": 0.25,
                        "friction": 0.8,
                        "confidence": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ensure_collision_assets(run_dir)

    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    collision = manifest["objects"][0]["collision_asset"]
    assert collision["source"] == "convex_hull_from_visual_mesh"
    assert collision["provenance"]["visual_asset_path"] == "objects/cup/visual.glb"


def test_scene_manifest_physics_fields_are_not_reported_as_inferred_or_reproduced(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_legacy_run(run_dir)

    ensure_collision_assets(run_dir)

    physics = json.loads((run_dir / "objects/cup/physics.json").read_text(encoding="utf-8"))
    assert physics["source"] == "scene_manifest_physics_fields"
    assert physics["source_category"] == "scene_manifest_physics_fields"
    assert physics["reproduction_status"] == "partial"
    assert physics["paper_equivalent"] is False
    assert physics["property_sources"]["mass_kg"]["category"] == "scene_manifest_physics_fields"
    assert physics["property_sources"]["friction"]["category"] == "scene_manifest_physics_fields"
    assert physics["property_sources"]["restitution"]["category"] == "heuristic"
    report = json.loads((run_dir / "qa/physics_property_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["reproduction_status"] == "partial"
    assert report["paper_equivalent"] is False
    obj = report["objects"][0]
    assert obj["mass_source_category"] == "scene_manifest_physics_fields"
    assert obj["friction_source_category"] == "scene_manifest_physics_fields"
    assert obj["restitution_source_category"] == "heuristic"
    encoded = json.dumps(report).lower()
    assert "inferred" not in encoded
    assert '"reproduced"' not in encoded


def test_explicit_heuristic_physics_source_is_not_promoted_to_inference(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_legacy_run(run_dir)
    manifest_path = run_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objects"][0]["physics_source_category"] = "heuristic"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ensure_collision_assets(run_dir)

    physics = json.loads((run_dir / "objects/cup/physics.json").read_text(encoding="utf-8"))
    assert physics["source"] == "heuristic"
    assert physics["source_category"] == "heuristic"
    assert physics["property_sources"]["mass_kg"]["category"] == "heuristic"
    assert physics["property_sources"]["friction"]["category"] == "heuristic"
    assert physics["reproduction_status"] == "partial"
    assert physics["paper_equivalent"] is False
    report = json.loads((run_dir / "qa/physics_property_report.json").read_text(encoding="utf-8"))
    assert report["objects"][0]["mass_source_category"] == "heuristic"
    assert report["objects"][0]["friction_source_category"] == "heuristic"
    encoded = json.dumps(report).lower()
    assert "inferred" not in encoded
    assert '"reproduced"' not in encoded


def test_strict_coacd_request_blocks_paper_equivalence_without_backend_integration(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_legacy_run(run_dir, mesh_name="visual.glb")
    trimesh.creation.icosphere(subdivisions=1, radius=0.1).export(run_dir / "objects/cup/visual.glb")

    report = ensure_collision_assets(run_dir, backend="coacd", strict_provenance=True)

    assert report["backend_requested"] == "coacd"
    assert report["strict_provenance"] is True
    assert report["status"] == "blocked"
    assert report["interactive_status"] == "usable"
    assert report["paper_equivalence_status"] == "blocked"
    assert any(reason.startswith("coacd_") for reason in report["blocking_for_paper"])
    object_report = json.loads((run_dir / "objects/cup/collision_report.json").read_text(encoding="utf-8"))
    assert object_report["backend_requested"] == "coacd"
    assert object_report["status"] == "blocked"
    assert object_report["interactive_status"] == "usable"
    assert object_report["paper_equivalence_status"] == "blocked"
    assert object_report["decomposition_backend"] == "trimesh_convex_hull"


def test_qa_physics_reports_interactive_usable_but_paper_partial_for_manifest_fields(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_legacy_run(run_dir)
    ensure_collision_assets(run_dir)

    report = qa_physics(run_dir)

    assert report["status"] == "partial"
    assert report["interactive_status"] == "usable"
    assert report["paper_equivalence_status"] == "partial"
    assert report["report_path"] == "qa/physics_property_report.json"
    assert "mass_kg_source_not_vlm_inference:scene_manifest_physics_fields" in report["blocking_for_paper"]
    obj = report["objects"][0]
    assert obj["interactive_status"] == "usable"
    assert obj["paper_equivalence_status"] == "partial"
    assert obj["source_category"] == "scene_manifest_physics_fields"
