import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from real2sim_scene_foundry.support_plane import build_tabletop_support_mask, estimate_and_apply_support_plane


def _write_object(run_dir: Path, object_id: str) -> None:
    object_dir = run_dir / "objects" / object_id
    object_dir.mkdir(parents=True)
    mesh_path = object_dir / "mesh_aligned.glb"
    trimesh.creation.box(extents=(0.2, 0.2, 0.24)).export(mesh_path)
    transform = [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.24], [0, 0, 0, 1]]
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


def _write_manifest(run_dir: Path) -> None:
    _write_object(run_dir, "cup")
    pose = json.loads((run_dir / "objects" / "cup" / "pose.json").read_text(encoding="utf-8"))
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
                        "mesh_path": "objects/cup/mesh_aligned.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": pose["T_object_to_camera"],
                        "T_object_to_world": pose["T_object_to_world"],
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
    (run_dir / "exports").mkdir()
    (run_dir / "qa").mkdir()
    (run_dir / "qa" / "qa_report.json").write_text(json.dumps({"objects": []}), encoding="utf-8")


def _fill_xyz_component(
    xyz: np.ndarray,
    rows: slice,
    cols: slice,
    *,
    x_center: float,
    y_center: float,
    support_height: float,
) -> None:
    rr, cc = np.mgrid[rows, cols]
    xyz[rows, cols, 0] = x_center + (cc - float(np.mean(cc))) * 0.02
    xyz[rows, cols, 1] = -support_height
    xyz[rows, cols, 2] = y_center + (rr - float(np.mean(rr))) * 0.02


def _write_disconnected_near_height_table_points(run_dir: Path) -> None:
    shape = (64, 64)
    support_height = 0.12
    xyz = np.full((*shape, 3), np.nan, dtype=np.float32)
    _fill_xyz_component(
        xyz,
        slice(3, 29),
        slice(3, 24),
        x_center=2.5,
        y_center=2.0,
        support_height=support_height,
    )
    _fill_xyz_component(
        xyz,
        slice(34, 54),
        slice(28, 48),
        x_center=0.0,
        y_center=1.0,
        support_height=support_height,
    )
    np.save(run_dir / "xyz.npy", xyz)

    object_mask = np.zeros(shape, dtype=np.uint8)
    object_mask[42:48, 36:42] = 255
    Image.fromarray(object_mask).save(run_dir / "objects" / "cup" / "mask.png")
    background_dir = run_dir / "background"
    background_dir.mkdir()
    Image.fromarray(object_mask).save(background_dir / "foreground_mask.png")


def test_table_collision_uses_contact_tabletop_component_not_global_near_height_rect(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)
    _write_disconnected_near_height_table_points(run_dir)

    report = estimate_and_apply_support_plane(run_dir)

    assert report["support_surface_status"] == "passed"
    assert report["table_collision_final"] is True
    assert report["table_collision_source_backend"] == "tabletop_mask_polygon_slab"
    assert report["table_collision_geometry_type"] == "polygon_slab"
    assert report["table_collision_size_xyz"][0] < 0.8


def test_support_plane_writes_reference_camera_table_collision_qa(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)
    _write_disconnected_near_height_table_points(run_dir)
    transform = [[1, 0, 0, 0.0], [0, 1, 0, 1.0], [0, 0, 1, 0.24], [0, 0, 0, 1]]
    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    manifest["objects"][0]["T_object_to_camera"] = transform
    manifest["objects"][0]["T_object_to_world"] = transform
    (run_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "objects" / "cup" / "pose.json").write_text(
        json.dumps({"object_id": "cup", "T_object_to_world": transform}),
        encoding="utf-8",
    )
    (run_dir / "camera.json").write_text(
        json.dumps({"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 42.0, "width": 64, "height": 64}),
        encoding="utf-8",
    )

    report = estimate_and_apply_support_plane(run_dir)

    qa_report_path = run_dir / "qa" / "table_collision_report.json"
    overlay_path = run_dir / "qa" / "table_collision_overlay.png"
    qa_report = json.loads(qa_report_path.read_text(encoding="utf-8"))
    assert qa_report_path.is_file()
    assert overlay_path.is_file()
    assert report["table_collision_visual_qa_path"] == "qa/table_collision_overlay.png"
    assert qa_report["iou_threshold"] == 0.5
    assert qa_report["source_backend"] == "tabletop_mask_polygon_slab"


def test_tabletop_support_mask_filters_near_height_thick_band(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "objects" / "cup").mkdir(parents=True)
    shape = (20, 20)
    support_height = 0.12
    xyz = np.full((*shape, 3), np.nan, dtype=np.float32)
    for row in range(5, 15):
        residual = -0.05 + (row - 5) * (0.10 / 9.0)
        for col in range(5, 15):
            xyz[row, col] = [(col - 10) * 0.01, -(support_height + residual), 1.0 + (row - 10) * 0.01]
    np.save(run_dir / "xyz.npy", xyz)
    object_mask = np.zeros(shape, dtype=np.uint8)
    object_mask[9:11, 9:11] = 255
    Image.fromarray(object_mask).save(run_dir / "objects" / "cup" / "mask.png")
    (run_dir / "background").mkdir()
    Image.fromarray(object_mask).save(run_dir / "background" / "foreground_mask.png")
    (run_dir / "scene_manifest.json").write_text(
        json.dumps(
            {
                "support_plane": {"original_height_world_m": support_height},
                "objects": [{"object_id": "cup", "mask_path": "objects/cup/mask.png"}],
            }
        ),
        encoding="utf-8",
    )

    report = build_tabletop_support_mask(run_dir)

    assert report["status"] == "passed"
    assert report["candidate_count"] > report["selected_component_area"]
    assert report["selected_component_area"] <= 44
    assert report["plane_residual_tolerance_m"] == pytest.approx(0.02)


def test_tabletop_support_mask_uses_same_row_visible_table_not_tiny_contact_patch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "objects" / "cup").mkdir(parents=True)
    (run_dir / "background").mkdir()
    shape = (80, 120)
    support_height = 0.12
    xyz = np.full((*shape, 3), np.nan, dtype=np.float32)
    _fill_xyz_component(
        xyz,
        slice(30, 46),
        slice(4, 48),
        x_center=-0.55,
        y_center=1.0,
        support_height=support_height,
    )
    _fill_xyz_component(
        xyz,
        slice(30, 46),
        slice(72, 116),
        x_center=0.55,
        y_center=1.0,
        support_height=support_height,
    )
    _fill_xyz_component(
        xyz,
        slice(34, 40),
        slice(58, 64),
        x_center=0.0,
        y_center=1.0,
        support_height=support_height,
    )
    np.save(run_dir / "xyz.npy", xyz)
    object_mask = np.zeros(shape, dtype=np.uint8)
    object_mask[40:52, 54:68] = 255
    Image.fromarray(object_mask).save(run_dir / "objects" / "cup" / "mask.png")
    Image.fromarray(object_mask).save(run_dir / "background" / "foreground_mask.png")
    (run_dir / "scene_manifest.json").write_text(
        json.dumps(
            {
                "support_plane": {"original_height_world_m": support_height},
                "objects": [{"object_id": "cup", "mask_path": "objects/cup/mask.png"}],
            }
        ),
        encoding="utf-8",
    )

    report = build_tabletop_support_mask(run_dir)

    mask = np.asarray(Image.open(run_dir / "background" / "tabletop_mask.png").convert("L")) > 0
    assert report["status"] == "passed"
    assert report["source"] == "contact_row_components"
    assert report["selected_component_area"] > 1200
    assert report["selected_candidate_ratio"] > 0.7
    assert np.count_nonzero(mask[:, :50]) > 300
    assert np.count_nonzero(mask[:, 70:]) > 300


def test_table_collision_preserves_multiple_tabletop_components_instead_of_single_gap_hull(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)
    (run_dir / "background").mkdir()
    shape = (80, 120)
    support_height = 0.10
    xyz = np.full((*shape, 3), np.nan, dtype=np.float32)
    _fill_xyz_component(
        xyz,
        slice(30, 46),
        slice(4, 48),
        x_center=-0.55,
        y_center=1.0,
        support_height=support_height,
    )
    _fill_xyz_component(
        xyz,
        slice(30, 46),
        slice(72, 116),
        x_center=0.55,
        y_center=1.0,
        support_height=support_height,
    )
    _fill_xyz_component(
        xyz,
        slice(34, 40),
        slice(58, 64),
        x_center=0.0,
        y_center=1.0,
        support_height=support_height,
    )
    np.save(run_dir / "xyz.npy", xyz)
    object_mask = np.zeros(shape, dtype=np.uint8)
    object_mask[40:52, 54:68] = 255
    Image.fromarray(object_mask).save(run_dir / "objects" / "cup" / "mask.png")
    Image.fromarray(object_mask).save(run_dir / "background" / "foreground_mask.png")

    report = estimate_and_apply_support_plane(run_dir)

    polygon = json.loads((run_dir / "background" / "table_polygon_world.json").read_text(encoding="utf-8"))
    assert report["table_collision_final"] is True
    assert polygon["polygon_count"] >= 3
    assert len(polygon["polygons_world_xy"]) == polygon["polygon_count"]
    assert polygon["source_backend"] == "tabletop_mask_polygon_slab"


def test_table_collision_extends_under_object_support_footprint_when_tabletop_is_occluded(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)
    (run_dir / "background").mkdir()
    transform = [[1, 0, 0, 0.0], [0, 1, 0, 1.0], [0, 0, 1, 0.24], [0, 0, 0, 1]]
    manifest = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    manifest["objects"][0]["T_object_to_camera"] = transform
    manifest["objects"][0]["T_object_to_world"] = transform
    (run_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "objects" / "cup" / "pose.json").write_text(
        json.dumps({"object_id": "cup", "T_object_to_world": transform}),
        encoding="utf-8",
    )
    shape = (80, 120)
    support_height = 0.12
    xyz = np.full((*shape, 3), np.nan, dtype=np.float32)
    _fill_xyz_component(
        xyz,
        slice(30, 46),
        slice(4, 48),
        x_center=-0.55,
        y_center=1.45,
        support_height=support_height,
    )
    _fill_xyz_component(
        xyz,
        slice(30, 46),
        slice(72, 116),
        x_center=0.55,
        y_center=1.45,
        support_height=support_height,
    )
    np.save(run_dir / "xyz.npy", xyz)
    object_mask = np.zeros(shape, dtype=np.uint8)
    object_mask[40:52, 54:68] = 255
    Image.fromarray(object_mask).save(run_dir / "objects" / "cup" / "mask.png")
    Image.fromarray(object_mask).save(run_dir / "background" / "foreground_mask.png")

    report = estimate_and_apply_support_plane(run_dir)

    polygon = json.loads((run_dir / "background" / "table_polygon_world.json").read_text(encoding="utf-8"))
    all_y = [point[1] for poly in polygon["polygons_world_xy"] for point in poly]
    assert report["table_collision_final"] is True
    assert polygon["object_support_footprint_polygon_count"] == 1
    assert min(all_y) <= 0.91
