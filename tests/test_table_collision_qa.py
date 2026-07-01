import json

import numpy as np
from PIL import Image

from real2sim_scene_foundry.table_collision_qa import write_table_collision_projection_qa, write_table_collision_projection_qa_v2


def _write_mask(path, *, fill=True):
    mask = np.zeros((100, 100), dtype=np.uint8)
    if fill:
        mask[30:71, 30:71] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def _write_polygon(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_backend": "tabletop_mask",
                "geometry_type": "polygon_slab",
                "vertices_world": [[-0.2, -0.2, 1.0], [0.2, -0.2, 1.0], [0.2, 0.2, 1.0], [-0.2, 0.2, 1.0]],
            }
        ),
        encoding="utf-8",
    )


def test_table_collision_projection_qa_blocks_when_camera_intrinsics_missing(tmp_path):
    run_dir = tmp_path / "run"
    _write_polygon(run_dir / "background" / "table_polygon_world.json")
    _write_mask(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "camera.json").write_text(json.dumps({"width": 100, "height": 100}), encoding="utf-8")

    result = write_table_collision_projection_qa(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "camera_intrinsics_missing" in report["blocking_reasons"]
    assert report["overlay_path"] is None


def test_table_collision_projection_qa_writes_overlay_and_passed_report(tmp_path):
    run_dir = tmp_path / "run"
    _write_polygon(run_dir / "background" / "table_polygon_world.json")
    _write_mask(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "camera.json").write_text(
        json.dumps(
            {
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                "T_world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": 100,
                "height": 100,
            }
        ),
        encoding="utf-8",
    )

    result = write_table_collision_projection_qa(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["derived_from_tabletop_mask"] is True
    assert report["geometry_type"] == "polygon_slab"
    assert report["projection_iou"] > 0.9
    assert (run_dir / "qa" / "table_collision_overlay.png").is_file()


def test_table_collision_projection_qa_blocks_low_default_projection_iou(tmp_path):
    run_dir = tmp_path / "run"
    _write_polygon(run_dir / "background" / "table_polygon_world.json")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[15:85, 15:85] = 255
    mask_path = run_dir / "background" / "tabletop_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    (run_dir / "camera.json").write_text(
        json.dumps(
            {
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                "T_world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": 100,
                "height": 100,
            }
        ),
        encoding="utf-8",
    )

    result = write_table_collision_projection_qa(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "low_tabletop_projection_iou" in report["blocking_reasons"]
    assert 0.3 < report["projection_iou"] < 0.4
    assert report["iou_threshold"] == 0.5


def test_table_collision_projection_qa_v2_blocks_weak_iou_that_legacy_accepts(tmp_path):
    run_dir = tmp_path / "run"
    _write_polygon(run_dir / "background" / "table_polygon_world.json")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[22:78, 22:78] = 255
    mask_path = run_dir / "background" / "tabletop_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    (run_dir / "camera.json").write_text(
        json.dumps(
            {
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                "T_world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": 100,
                "height": 100,
            }
        ),
        encoding="utf-8",
    )

    legacy = write_table_collision_projection_qa(run_dir)
    strict = write_table_collision_projection_qa_v2(run_dir)

    legacy_report = json.loads(legacy.report_path.read_text(encoding="utf-8"))
    strict_report = json.loads(strict.report_path.read_text(encoding="utf-8"))
    assert legacy_report["status"] == "passed"
    assert 0.50 < legacy_report["projection_iou_with_tabletop_mask"] < 0.65
    assert strict.report_path.name == "table_collision_report_v2.json"
    assert strict_report["status"] == "blocked"
    assert strict_report["weak_status"] == "diagnostic_pass"
    assert "below_export_grade_tabletop_iou" in strict_report["blocking_reasons"]
    assert strict_report["camera_intrinsics_source"] == "explicit"
    assert strict_report["camera_extrinsics_source"] == "explicit"


def test_table_collision_projection_qa_v2_blocks_camera_fallback_bridge(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "background").mkdir(parents=True)
    (run_dir / "background" / "table_polygon_world.json").write_text(
        json.dumps(
            {
                "source_backend": "tabletop_mask_polygon_slab",
                "geometry_type": "polygon_slab",
                "polygon_world_xy": [[-0.2, 1.0], [0.2, 1.0], [0.2, 1.4], [-0.2, 1.4]],
                "top_z_m": 0.0,
                "support_height_m": 0.2,
            }
        ),
        encoding="utf-8",
    )
    _write_mask(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "camera.json").write_text(
        json.dumps({"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "width": 100, "height": 100}),
        encoding="utf-8",
    )

    result = write_table_collision_projection_qa_v2(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["camera_intrinsics_source"] == "explicit"
    assert report["camera_extrinsics_source"] == "fallback"
    assert "camera_extrinsics_not_explicit" in report["blocking_reasons"]


def test_table_collision_projection_qa_blocks_foreground_inflated_occlusion(tmp_path):
    run_dir = tmp_path / "run"
    _write_polygon(run_dir / "background" / "table_polygon_world.json")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 45:55] = 255
    mask_path = run_dir / "background" / "tabletop_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    foreground = np.zeros((100, 100), dtype=np.uint8)
    foreground[30:71, 30:71] = 255
    Image.fromarray(foreground).save(run_dir / "background" / "foreground_mask.png")
    (run_dir / "camera.json").write_text(
        json.dumps(
            {
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                "T_world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": 100,
                "height": 100,
            }
        ),
        encoding="utf-8",
    )

    result = write_table_collision_projection_qa(run_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "blocked_tabletop_occlusion" in report["blocking_reasons"]
    assert report["projection_iou"] > 0.9
    assert report["projection_iou_with_tabletop_mask"] < 0.1
    assert report["accepted_foreground_projection_area_px"] > report["tabletop_mask_area_px"]



def test_table_collision_projection_qa_accepts_support_plane_polygon_world_xy_and_camera_bridge(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "background").mkdir(parents=True)
    (run_dir / "background" / "table_polygon_world.json").write_text(
        json.dumps(
            {
                "source_backend": "tabletop_mask_polygon_slab",
                "geometry_type": "polygon_slab",
                "polygon_world_xy": [[-0.2, 1.0], [0.2, 1.0], [0.2, 1.4], [-0.2, 1.4]],
                "top_z_m": 0.0,
                "support_height_m": 0.2,
            }
        ),
        encoding="utf-8",
    )
    _write_mask(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "camera.json").write_text(
        json.dumps({"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "width": 100, "height": 100}),
        encoding="utf-8",
    )

    result = write_table_collision_projection_qa(run_dir, iou_threshold=0.0)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["derived_from_tabletop_mask"] is True
    assert report["source_backend"] == "tabletop_mask_polygon_slab"
    assert (run_dir / "qa" / "table_collision_overlay.png").is_file()


def test_table_collision_projection_qa_uses_polygon_slab_silhouette(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "background").mkdir(parents=True)
    (run_dir / "background" / "table_polygon_world.json").write_text(
        json.dumps(
            {
                "source_backend": "tabletop_mask_polygon_slab",
                "geometry_type": "polygon_slab",
                "polygon_world_xy": [[-0.2, 1.25], [0.2, 1.25], [0.2, 1.65], [-0.2, 1.65]],
                "top_z_m": 0.0,
                "thickness_m": 0.04,
                "support_height_m": 0.094,
            }
        ),
        encoding="utf-8",
    )
    mask = np.zeros((405, 720), dtype=np.uint8)
    mask[156:171, 374:400] = 255
    Image.fromarray(mask).save(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "camera.json").write_text(
        json.dumps({"fx": 598.43, "fy": 598.43, "cx": 359.5, "cy": 202.0, "width": 720, "height": 405}),
        encoding="utf-8",
    )

    result = write_table_collision_projection_qa(run_dir, iou_threshold=0.0)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["projection_surface"] == "slab_silhouette"
    assert report["projected_polygon_area_px"] > 0


def test_table_collision_projection_qa_blocks_tiny_tabletop_support_patch(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "background").mkdir(parents=True)
    (run_dir / "background" / "table_polygon_world.json").write_text(
        json.dumps(
            {
                "source_backend": "tabletop_mask_polygon_slab",
                "geometry_type": "polygon_slab",
                "vertices_world": [[-0.03, -0.03, 1.0], [0.03, -0.03, 1.0], [0.03, 0.03, 1.0], [-0.03, 0.03, 1.0]],
                "thickness_m": 0.04,
            }
        ),
        encoding="utf-8",
    )
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[47:54, 47:54] = 255
    Image.fromarray(mask).save(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "background" / "tabletop_support_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "candidate_count": 16000,
                "selected_component_area": 49,
                "selected_candidate_ratio": 0.003,
                "selected_extent_x_m": 0.06,
                "selected_extent_y_m": 0.06,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "camera.json").write_text(
        json.dumps(
            {
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                "T_world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": 100,
                "height": 100,
            }
        ),
        encoding="utf-8",
    )

    result = write_table_collision_projection_qa(run_dir, iou_threshold=0.0)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "table_collision_support_coverage_too_small" in report["blocking_reasons"]
    assert "table_collision_extent_too_small" in report["blocking_reasons"]
    assert report["tabletop_mask_area_px"] == 49
    assert report["projection_iou_with_tabletop_mask"] == report["projection_iou"]


def test_table_collision_projection_qa_projects_multiple_polygons_without_filling_gap(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "background").mkdir(parents=True)
    (run_dir / "background" / "table_polygon_world.json").write_text(
        json.dumps(
            {
                "source_backend": "tabletop_mask_polygon_slab",
                "geometry_type": "polygon_slab",
                "polygons_world": [
                    [[-0.4, -0.2, 1.0], [-0.2, -0.2, 1.0], [-0.2, 0.2, 1.0], [-0.4, 0.2, 1.0]],
                    [[0.2, -0.2, 1.0], [0.4, -0.2, 1.0], [0.4, 0.2, 1.0], [0.2, 0.2, 1.0]],
                ],
                "polygon_extent_x_m": 0.8,
                "polygon_extent_y_m": 0.4,
                "selected_candidate_ratio": 0.5,
            }
        ),
        encoding="utf-8",
    )
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:71, 10:31] = 255
    mask[30:71, 70:91] = 255
    Image.fromarray(mask).save(run_dir / "background" / "tabletop_mask.png")
    (run_dir / "camera.json").write_text(
        json.dumps(
            {
                "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                "T_world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": 100,
                "height": 100,
            }
        ),
        encoding="utf-8",
    )

    result = write_table_collision_projection_qa(run_dir, iou_threshold=0.8)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["polygon_count"] == 2
    assert report["projection_iou"] > 0.8
    assert report["projected_polygon_area_px"] < 2200
