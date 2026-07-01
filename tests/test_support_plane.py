import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from real2sim_scene_foundry.support_plane import estimate_and_apply_support_plane


def _write_object(run_dir: Path, object_id: str, *, extents: tuple[float, float, float], world_z: float) -> None:
    object_dir = run_dir / "objects" / object_id
    object_dir.mkdir(parents=True)
    mesh_path = object_dir / "mesh_aligned.glb"
    trimesh.creation.box(extents=extents).export(mesh_path)
    transform = [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, world_z], [0, 0, 0, 1]]
    (object_dir / "pose.json").write_text(
        json.dumps(
            {
                "object_id": object_id,
                "label": object_id,
                "T_object_to_camera": transform,
                "T_object_to_world": transform,
                "mesh_path": str(mesh_path.relative_to(run_dir)),
            }
        ),
        encoding="utf-8",
    )


def _write_run(run_dir: Path) -> None:
    _write_object(run_dir, "cup", extents=(0.2, 0.2, 0.2), world_z=0.35)
    _write_object(run_dir, "bottle", extents=(0.2, 0.2, 0.4), world_z=0.45)
    objects = []
    for object_id in ("cup", "bottle"):
        pose = json.loads((run_dir / "objects" / object_id / "pose.json").read_text(encoding="utf-8"))
        objects.append(
            {
                "object_id": object_id,
                "label": object_id,
                "mesh_path": f"objects/{object_id}/mesh_aligned.glb",
                "mask_path": f"objects/{object_id}/mask.png",
                "crop_path": f"objects/{object_id}/crop.png",
                "T_object_to_camera": pose["T_object_to_camera"],
                "T_object_to_world": pose["T_object_to_world"],
                "scale_m": 0.2,
                "mass_kg": 0.25,
                "friction": 0.8,
                "confidence": 0.9,
                "source_backend": "sam3d_aligned",
                "needs_manual_refine": False,
            }
        )
    (run_dir / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "objects": objects,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "exports").mkdir()
    (run_dir / "exports" / "scene.usda").write_text(
        '#usda 1.0\n'
        'def Xform "World"\n'
        "{\n"
        '    def Xform "cup"\n'
        "    {\n"
        '        custom string mesh_path = "objects/cup/mesh_aligned.glb"\n'
        "        double3 xformOp:translate = (0, 0, 0.35)\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (run_dir / "qa").mkdir()
    (run_dir / "qa" / "qa_report.json").write_text(json.dumps({"objects": []}), encoding="utf-8")


def test_estimate_and_apply_support_plane_normalizes_object_bottoms_and_reports(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run(run_dir)

    report = estimate_and_apply_support_plane(run_dir)

    assert report["status"] == "estimated"
    assert report["source_backend"] == "object_mesh_bottom_median"
    assert report["original_height_world_m"] == pytest.approx(0.25)
    assert report["height_world_m"] == 0.0
    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["support_plane"]["applied_to_world_frame"] is True
    by_id = {obj["object_id"]: obj for obj in manifest["objects"]}
    assert by_id["cup"]["T_object_to_world"][2][3] == pytest.approx(0.10)
    assert by_id["bottle"]["T_object_to_world"][2][3] == pytest.approx(0.20)
    for object_id in ("cup", "bottle"):
        pose = json.loads((run_dir / "objects" / object_id / "pose.json").read_text(encoding="utf-8"))
        assert pose["T_object_to_world"][2][3] == pytest.approx(by_id[object_id]["T_object_to_world"][2][3])
    qa = json.loads((run_dir / "qa" / "qa_report.json").read_text(encoding="utf-8"))
    assert qa["support_plane"]["object_bottoms_after_m"]["cup"] == pytest.approx(0.0, abs=1e-8)
    assert qa["support_plane"]["object_bottoms_after_m"]["bottle"] == pytest.approx(0.0, abs=1e-8)
    assert qa["support_plane"]["object_vertical_corrections_m"]["cup"] == pytest.approx(0.0, abs=1e-8)
    assert qa["support_plane"]["object_vertical_corrections_m"]["bottle"] == pytest.approx(0.0, abs=1e-8)
    table_path = run_dir / qa["support_plane"]["table_collision_mesh_path"]
    assert table_path.is_file()
    assert qa["support_plane"]["table_collision_size_xyz"][2] == pytest.approx(0.04)
    assert "0.1" in (run_dir / "exports" / "scene.usda").read_text(encoding="utf-8")


def test_estimate_and_apply_support_plane_is_idempotent(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run(run_dir)

    estimate_and_apply_support_plane(run_dir)
    first = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    estimate_and_apply_support_plane(run_dir)
    second = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))

    assert second["objects"] == first["objects"]
    assert second["support_plane"] == first["support_plane"]


def test_estimate_and_apply_support_plane_force_recomputes_existing_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run(run_dir)
    estimate_and_apply_support_plane(run_dir)
    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    manifest["objects"][0]["T_object_to_world"][2][3] += 0.5
    (run_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    skipped = estimate_and_apply_support_plane(run_dir)
    skipped_manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    forced = estimate_and_apply_support_plane(run_dir, force=True)

    assert skipped == manifest["support_plane"]
    assert skipped_manifest["objects"][0]["T_object_to_world"][2][3] == pytest.approx(0.6)
    assert forced["object_bottoms_after_m"]["cup"] == pytest.approx(0.0, abs=1e-8)


def test_estimate_and_apply_support_plane_prefers_background_point_ring(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run(run_dir)
    xyz = np.zeros((7, 7, 3), dtype=np.float32)
    xyz[..., 0] = np.linspace(-0.3, 0.3, 7)[None, :]
    xyz[..., 1] = -0.12
    xyz[..., 2] = np.linspace(0.8, 1.4, 7)[:, None]
    np.save(run_dir / "xyz.npy", xyz)
    for object_id, xy in {"cup": (3, 3), "bottle": (1, 1)}.items():
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[xy[1], xy[0]] = 255
        Image.fromarray(mask).save(run_dir / "objects" / object_id / "mask.png")

    report = estimate_and_apply_support_plane(run_dir)

    assert report["source_backend"] == "background_point_ring_median"
    assert report["original_height_world_m"] == pytest.approx(0.12)
    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    by_id = {obj["object_id"]: obj for obj in manifest["objects"]}
    assert by_id["cup"]["T_object_to_world"][2][3] == pytest.approx(0.10)
    assert by_id["bottle"]["T_object_to_world"][2][3] == pytest.approx(0.20)
    assert manifest["support_plane"]["object_bottoms_after_m"]["cup"] == pytest.approx(0.0, abs=1e-8)
    assert manifest["support_plane"]["object_bottoms_after_m"]["bottle"] == pytest.approx(0.0, abs=1e-8)
    assert manifest["support_plane"]["object_vertical_corrections_m"]["cup"] == pytest.approx(-0.13)
    assert manifest["support_plane"]["object_vertical_corrections_m"]["bottle"] == pytest.approx(-0.13)
    assert manifest["support_plane"]["support_surface_status"] == "passed"
    assert manifest["support_plane"]["table_collision_final"] is True
    assert manifest["support_plane"]["table_collision_source_backend"] == "tabletop_mask_polygon_slab"
    assert manifest["support_plane"]["table_collision_geometry_type"] == "polygon_slab"
    assert (run_dir / "background" / "tabletop_mask.png").is_file()
    assert (run_dir / "background" / "table_polygon_world.json").is_file()
    assert (run_dir / "background" / "table_collision_report.json").is_file()
    table_path = run_dir / manifest["support_plane"]["table_collision_mesh_path"]
    assert table_path.is_file()
    table = trimesh.load(table_path, force="mesh")
    pos = manifest["support_plane"]["table_collision_pos_world"]
    assert pos[2] + table.bounds[1, 2] == pytest.approx(0.0, abs=1e-8)
    assert table.extents[0] >= 0.55
    assert table.extents[1] >= 0.55
    assert manifest["support_plane"]["table_collision_size_xyz"] == pytest.approx(table.extents)
    usda = (run_dir / "exports" / "scene.usda").read_text(encoding="utf-8")
    assert "table_collision_mesh_path" in usda
    assert "table_collision_pos_world" in usda


def test_estimate_and_apply_support_plane_prefers_collision_asset_geometry(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    object_dir = run_dir / "objects" / "cup"
    object_dir.mkdir(parents=True)
    visual_path = object_dir / "mesh_aligned.glb"
    collision_path = object_dir / "collision.glb"
    trimesh.creation.box(extents=(0.2, 0.2, 1.0)).export(visual_path)
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(collision_path)
    transform = [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.35], [0, 0, 0, 1]]
    (object_dir / "pose.json").write_text(
        json.dumps(
            {
                "object_id": "cup",
                "label": "cup",
                "T_object_to_camera": transform,
                "T_object_to_world": transform,
                "mesh_path": str(visual_path.relative_to(run_dir)),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "exports").mkdir()
    (run_dir / "qa").mkdir()
    (run_dir / "qa" / "qa_report.json").write_text(json.dumps({"objects": []}), encoding="utf-8")
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
                        "mesh_path": str(visual_path.relative_to(run_dir)),
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "collision_asset": {
                            "path": str(collision_path.relative_to(run_dir)),
                            "source": "bbox_from_legacy_mesh_proxy",
                            "status": "ready",
                        },
                        "T_object_to_camera": transform,
                        "T_object_to_world": transform,
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

    report = estimate_and_apply_support_plane(run_dir)

    assert report["original_height_world_m"] == pytest.approx(0.25)
    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["objects"][0]["T_object_to_world"][2][3] == pytest.approx(0.10)
