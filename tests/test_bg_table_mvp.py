import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from real2sim_scene_foundry.cli import main


def _identity4():
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _translate4(x: float, y: float, z: float):
    matrix = _identity4()
    matrix[0][3] = x
    matrix[1][3] = y
    matrix[2][3] = z
    return matrix


def _write_ascii_splat(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    rows = [f"{x} {y} {z} 220 180 120" for x, y, z in points]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def _write_ascii_gaussian_splat(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    rows = [
        f"{x} {y} {z} 0 0 0 220 180 120 1.0 0.01 0.01 0.01 1 0 0 0"
        for x, y, z in points
    ]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def _write_bg_table_run(run: Path) -> None:
    (run / "background").mkdir(parents=True)
    (run / "frames").mkdir()
    (run / "depth").mkdir()
    (run / "qa").mkdir()
    polygon = {
        "version": 1,
        "status": "passed",
        "source_backend": "arkit_depth_ransac_plane",
        "geometry_type": "support_plane_polygon",
        "coordinate_world": "sim_world",
        "coordinate_frame": "sim_world",
        "unit": "meter",
        "polygon_world_xy": [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]],
        "polygons_world_xy": [[[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]],
        "top_z_m": 1.0,
        "support_height_m": 1.0,
        "plane": {"normal_world": [0.0, 0.0, 1.0], "offset_m": -1.0},
    }
    (run / "background" / "table_polygon_world.json").write_text(json.dumps(polygon), encoding="utf-8")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:71, 30:71] = 255
    Image.fromarray(mask).save(run / "background" / "tabletop_mask.png")
    Image.new("RGB", (100, 100), color=(12, 18, 24)).save(run / "frames" / "frame_000000.jpg")
    depth = np.ones((100, 100), dtype=np.float32)
    np.save(run / "depth" / "frame_000000.npy", depth)
    camera = {
        "width": 100,
        "height": 100,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 50.0,
        "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        "T_world_to_camera": _identity4(),
        "T_camera_to_world": _identity4(),
        "intrinsics_source": "arkit_explicit",
        "extrinsics_source": "arkit_explicit",
        "scale_source": "arkit_sceneDepth_meters",
        "depth_unit": "meter",
    }
    (run / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    trajectory = {
        "pose_world": "sim",
        "frames": [
            {
                "frame_id": "frame_000000",
                "rgb_path": "frames/frame_000000.jpg",
                "depth_path": "depth/frame_000000.npy",
                "T_camera_to_world": _identity4(),
            }
        ],
    }
    (run / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (run / "background" / "registration.json").write_text(
        json.dumps(
            {
                "status": "registered",
                "source_kind": "registered_3dgs",
                "coordinate_convention": {"3dgs": "arkit_world_camera_to_world", "sim": "z_up_meter_camera_to_world"},
                "coordinate_bridge": {
                    "status": "closed",
                    "source": "phone_sim_alignment_arkit_to_sim_world",
                    "source_world": "arkit",
                    "target_world": "sim_world",
                    "phone_sim_alignment_path": "background/phone_sim_alignment.json",
                },
                "T_3dgs_world_to_sim_world": _identity4(),
                "transforms": {"T_3dgs_world_to_sim_world": _identity4()},
                "transform_sources": {"T_3dgs_world_to_sim_world": "phone_sim_alignment_arkit_to_sim_world"},
            }
        ),
        encoding="utf-8",
    )
    (run / "background" / "phone_sim_alignment.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "source_backend": "arkit_depth_ransac_plane",
                "pose_world_before": "arkit",
                "pose_world_after": "sim",
                "T_arkit_world_to_sim_world": _identity4(),
                "metrics": {"plane_residual_median_m": 0.0, "plane_residual_p90_m": 0.0},
            }
        ),
        encoding="utf-8",
    )
    _write_ascii_splat(
        run / "background" / "3dgs_native" / "splat_rgb.ply",
        [(-0.1, -0.1, 1.0), (0.0, 0.0, 1.0), (0.1, 0.1, 1.0), (0.5, 0.5, 1.2)],
    )
    (run / "qa" / "registered_3dgs_table_renders.json").write_text(
        json.dumps(
            {
                "status": "rendered",
                "backend": "browser_3dgs_offscreen",
                "runtime": "browser-3dgs",
                "uses_T_3dgs_world_to_sim_world": True,
                "frames": [
                    {
                        "frame_index": 0,
                        "frame_id": "frame_000000",
                        "render_path": "qa/bg3dgs_render_000000.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (100, 100), color=(20, 24, 28)).save(run / "qa" / "bg3dgs_render_000000.png")


def test_build_table_collision_writes_world_polygon_slab_and_scope(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)

    code = main(
        [
            "build-table-collision",
            "--run-dir",
            str(run),
            "--source",
            "background/table_polygon_world.json",
            "--height",
            "0.03",
            "--write",
        ]
    )

    assert code == 0
    mesh_path = run / "table" / "collision_polygon_slab.glb"
    usd_path = run / "table" / "collision_polygon_slab.usd"
    scope = json.loads((run / "qa" / "bg_table_scope.json").read_text(encoding="utf-8"))
    report = json.loads((run / "table" / "collision_polygon_slab_report.json").read_text(encoding="utf-8"))
    mesh = trimesh.load(mesh_path, force="mesh")
    assert mesh_path.is_file()
    assert usd_path.is_file()
    assert scope["scope"] == "background_3dgs_and_table_collision_only"
    assert scope["ignore"]["object_assets"] is True
    assert scope["ignore"]["object_poses"] is True
    assert scope["ignore"]["object_physics"] is True
    assert report["source_backend"] == "arkit_depth_ransac_plane"
    assert report["coordinate_world"] == "sim_world"
    assert report["coordinate_frame"] == "sim_world"
    assert report["unit"] == "meter"
    assert report["top_z_m"] == pytest.approx(1.0)
    assert report["thickness_m"] == pytest.approx(0.03)
    assert report["thickness_direction"] == "down_negative_z"
    assert float(mesh.vertices[:, 2].max()) == pytest.approx(1.0)
    assert float(mesh.vertices[:, 2].min()) == pytest.approx(0.97)


def test_qa_bg_table_passes_without_objects_and_writes_overlays(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["scope"] == "background_3dgs_and_table_collision_only"
    assert report["background"]["registration_status"] == "registered"
    assert report["background"]["coordinate_bridge_status"] == "closed"
    assert report["background"]["transform_source"] == "phone_sim_alignment_arkit_to_sim_world"
    assert report["table_polygon"]["source_backend"] == "arkit_depth_ransac_plane"
    assert report["table_polygon"]["coordinate_frame"] == "sim_world"
    assert report["table_polygon"]["unit"] == "meter"
    assert report["table_collision"]["projection_iou"] >= 0.65
    assert report["table_collision"]["visible_iou"] >= 0.70
    assert report["table_collision"]["overreach_ratio"] <= 0.15
    assert report["table_collision"]["undercoverage_ratio"] <= 0.20
    assert report["table_collision"]["depth_plane_median_abs_residual_m"] <= 0.02
    assert report["object_scope"]["ignored"] is True
    assert report["claim"] == "registered 3DGS background and tabletop collision are aligned in the same sim world"
    assert report["3dgs_overlay"]["status"] == "rendered"
    assert report["3dgs_overlay"]["runtime"] == "browser-3dgs"
    assert report["3dgs_overlay"]["uses_T_3dgs_world_to_sim_world"] is True
    assert (run / "qa" / "bg3dgs_table_overlay_000000.png").is_file()
    assert report["splat_center_diagnostic"]["collision_geometry_source"] == "not_used"


def test_qa_bg_table_applies_nerfstudio_gaussian_asset_axis_bridge_for_center_diagnostic(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    _write_ascii_gaussian_splat(
        run / "background" / "3dgs_native" / "splat_rgb.ply",
        [(-0.1, -0.1, -1.0), (0.0, 0.0, -1.0), (0.1, 0.1, -1.0), (0.5, 0.5, -1.2)],
    )
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    diagnostic = report["splat_center_diagnostic"]
    assert diagnostic["status"] == "computed"
    assert diagnostic["asset_axis_bridge"]["status"] == "applied_for_nerfstudio_gaussian_ply_diagnostic"
    assert diagnostic["asset_axis_bridge"]["matrix"] == [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    assert diagnostic["near_plane_inside_polygon_count"] == 3
    assert diagnostic["near_plane_inside_polygon_ratio"] == pytest.approx(1.0)
    assert diagnostic["transform_source"] == "diagnostic_splat_asset_to_sim_world"


def test_qa_bg_table_selects_identity_asset_bridge_when_it_aligns_better_than_nerfstudio_bridge(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    _write_ascii_gaussian_splat(
        run / "background" / "3dgs_native" / "splat_rgb.ply",
        [(-0.1, -0.1, 1.0), (0.0, 0.0, 1.0), (0.1, 0.1, 1.0), (0.5, 0.5, -1.2)],
    )
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    diagnostic = report["splat_center_diagnostic"]
    assert diagnostic["status"] == "computed"
    assert diagnostic["alignment_status"] == "aligned"
    assert diagnostic["asset_axis_bridge"]["status"] == "not_required_for_registered_splat_diagnostic"
    assert diagnostic["asset_axis_bridge"]["selected_by"] == "table_alignment_score"
    assert diagnostic["near_plane_inside_polygon_count"] == 3
    assert diagnostic["near_plane_inside_polygon_ratio"] == pytest.approx(1.0)
    assert diagnostic["transform_source"] == "registration_T_3dgs_world_to_sim_world"


def test_qa_bg_table_is_partial_when_splat_centers_do_not_overlap_table_polygon(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    _write_ascii_splat(
        run / "background" / "3dgs_native" / "splat_rgb.ply",
        [(0.6, 0.6, 1.0), (0.7, 0.6, 1.0), (0.6, 0.7, 1.0), (0.8, 0.8, 1.0)],
    )
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert "splat_center_table_alignment_below_threshold" in report["blocking_reasons"]
    assert report["claim"] is None
    diagnostic = report["splat_center_diagnostic"]
    assert diagnostic["alignment_status"] == "misregistered"
    assert diagnostic["near_plane_inside_polygon_ratio"] == pytest.approx(0.0)
    assert diagnostic["min_near_plane_inside_polygon_ratio"] == pytest.approx(0.25)


def test_qa_bg_table_blocks_when_registered_transform_is_missing(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    registration_path = run / "background" / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registration["T_3dgs_world_to_sim_world"] = None
    registration["transforms"]["T_3dgs_world_to_sim_world"] = None
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "missing_T_3dgs_world_to_sim_world" in report["blocking_reasons"]
    assert report["claim"] is None


def test_qa_bg_table_blocks_when_table_polygon_projection_overreaches_visible_mask(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    polygon_path = run / "background" / "table_polygon_world.json"
    polygon = json.loads(polygon_path.read_text(encoding="utf-8"))
    polygon["polygon_world_xy"] = [[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]]
    polygon["polygons_world_xy"] = [polygon["polygon_world_xy"]]
    polygon_path.write_text(json.dumps(polygon), encoding="utf-8")
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["table_collision"]["raw_overreach_ratio"] > 0.15
    assert "overreach_ratio_above_threshold" in report["blocking_reasons"]
    assert report["claim"] is None


def test_qa_bg_table_blocks_arkit_3dgs_when_coordinate_bridge_is_not_closed(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    registration_path = run / "background" / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registration["coordinate_bridge"] = {"status": "blocked", "source_world": "arkit", "target_world": "sim_world"}
    registration["transform_sources"]["T_3dgs_world_to_sim_world"] = "camera_anchor_bridge"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["background"]["coordinate_bridge_status"] == "blocked"
    assert "background_registration_coordinate_bridge_not_closed" in report["blocking_reasons"]
    assert report["claim"] is None


def test_qa_bg_table_is_partial_when_3dgs_overlay_runtime_is_missing(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    (run / "qa" / "registered_3dgs_table_renders.json").unlink()
    (run / "qa" / "bg3dgs_render_000000.png").unlink()
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["3dgs_overlay"]["status"] == "blocked"
    assert "registered_3dgs_overlay_not_rendered" in report["blocking_reasons"]
    assert report["claim"] is None
    assert not (run / "qa" / "bg3dgs_table_overlay_000000.png").is_file()


def test_composite_viewer_bg_table_mode_does_not_require_scene_manifest_or_objects(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])
    main(["qa-bg-table", "--run-dir", str(run), "--frames", "0", "--write-overlays"])

    code = main(
        [
            "composite-viewer",
            "--run-dir",
            str(run),
            "--backend",
            "browser-3dgs",
            "--mode",
            "bg-table",
            "--export-only",
        ]
    )

    assert code == 0
    config = json.loads((run / "exports" / "composite_viewer" / "viewer_config.json").read_text(encoding="utf-8"))
    assert config["mode"] == "bg-table"
    assert config["scope"] == "background_3dgs_and_table_collision_only"
    assert config["objects"] == []
    assert config["background"]["source_kind"] == "registered_3dgs"
    assert config["background"]["browser_3dgs"]["asset_path"] == "../../background/3dgs_native/splat_rgb.ply"
    assert config["background"]["browser_3dgs"]["registration_transform"] == _identity4()
    assert config["background"]["live_3dgs_runtime"] is False
    assert config["support_plane"]["table_collision_mesh_path"] == "../../table/collision_polygon_slab.glb"
    assert config["camera_frustums"]["count"] == 1
    assert config["qa"]["overlays"] == ["../../qa/bg3dgs_table_overlay_000000.png"]



def test_fit_tabletop_from_semantic_mask_writes_masked_depth_polygon_and_refined_mask(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    semantic = np.zeros((100, 100), dtype=np.uint8)
    semantic[30:71, 30:71] = 255
    semantic_path = run / "qa" / "manual_tabletop_semantic_mask.png"
    Image.fromarray(semantic).save(semantic_path)
    confidence_dir = run / "confidence"
    confidence_dir.mkdir()
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(confidence_dir / "frame_000000.png")
    trajectory = json.loads((run / "trajectory.json").read_text(encoding="utf-8"))
    trajectory["frames"][0]["confidence_path"] = "confidence/frame_000000.png"
    (run / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (run / "background" / "table_polygon_world.json").unlink()
    (run / "background" / "tabletop_mask.png").unlink()

    code = main(
        [
            "fit-tabletop-from-mask",
            "--run-dir",
            str(run),
            "--mask",
            str(semantic_path),
            "--frame-index",
            "0",
            "--hull",
            "convex-hull",
            "--write",
        ]
    )

    assert code == 0
    report = json.loads((run / "qa" / "tabletop_semantic_fit_report.json").read_text(encoding="utf-8"))
    polygon = json.loads((run / "background" / "table_polygon_world.json").read_text(encoding="utf-8"))
    assert (run / "background" / "tabletop_semantic_mask.png").is_file()
    assert (run / "background" / "tabletop_mask.png").is_file()
    assert report["status"] == "passed"
    assert report["source_backend"] == "semantic_masked_arkit_depth_ransac_plane"
    assert report["semantic_mask_path"] == "background/tabletop_semantic_mask.png"
    assert report["refined_plane_mask_path"] == "background/tabletop_mask.png"
    assert report["hull_method"] == "convex-hull"
    assert report["metrics"]["semantic_mask_area_px"] == int(np.count_nonzero(semantic))
    assert report["metrics"]["near_plane_mask_area_px"] > 0
    assert report["metrics"]["plane_residual_median_m"] <= 0.02
    assert polygon["status"] == "passed"
    assert polygon["source_backend"] == "semantic_masked_arkit_depth_ransac_plane"
    assert polygon["semantic_mask_path"] == "background/tabletop_semantic_mask.png"
    assert polygon["top_z_m"] == pytest.approx(1.0)
    assert polygon["unit"] == "meter"
    assert len(polygon["polygon_world_xy"]) >= 3


def test_fit_tabletop_multiframe_fuses_semantic_depth_from_multiple_poses(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    masks_dir = run / "background" / "tabletop_semantic_masks"
    masks_dir.mkdir()
    confidence_dir = run / "confidence"
    confidence_dir.mkdir()
    semantic = np.zeros((100, 100), dtype=np.uint8)
    semantic[30:71, 30:71] = 255
    Image.fromarray(semantic).save(masks_dir / "frame_000000.png")
    Image.fromarray(semantic).save(masks_dir / "frame_000020.png")
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(confidence_dir / "frame_000000.png")
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(confidence_dir / "frame_000020.png")
    np.save(run / "depth" / "frame_000020.npy", np.ones((100, 100), dtype=np.float32))
    Image.new("RGB", (100, 100), color=(18, 22, 26)).save(run / "frames" / "frame_000020.jpg")
    trajectory = json.loads((run / "trajectory.json").read_text(encoding="utf-8"))
    trajectory["frames"][0]["confidence_path"] = "confidence/frame_000000.png"
    trajectory["frames"].append(
        {
            "frame_id": "frame_000020",
            "rgb_path": "frames/frame_000020.jpg",
            "depth_path": "depth/frame_000020.npy",
            "confidence_path": "confidence/frame_000020.png",
            "T_camera_to_world": _translate4(0.08, 0.0, 0.0),
        }
    )
    (run / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (run / "background" / "table_polygon_world.json").unlink()
    (run / "background" / "tabletop_mask.png").unlink()

    code = main(
        [
            "fit-tabletop-multiframe",
            "--run-dir",
            str(run),
            "--masks",
            "background/tabletop_semantic_masks",
            "--frames",
            "0,20",
            "--hull",
            "convex-hull",
            "--boundary-mode",
            "union-hull",
            "--write",
        ]
    )

    assert code == 0
    report = json.loads((run / "qa" / "table_multiframe_report.json").read_text(encoding="utf-8"))
    plane = json.loads((run / "background" / "table_plane_multiframe.json").read_text(encoding="utf-8"))
    polygon = json.loads((run / "background" / "table_polygon_world.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["source_backend"] == "semantic_multiframe_arkit_depth_ransac_plane"
    assert report["metrics"]["used_frame_count"] == 2
    assert report["metrics"]["trajectory_baseline_m"] == pytest.approx(0.08)
    assert report["metrics"]["plane_inlier_count"] > 0
    assert report["artifacts"]["fused_points_path"] == "background/fused_tabletop_points.ply"
    assert plane["status"] == "passed"
    assert plane["source_backend"] == "semantic_multiframe_arkit_depth_ransac_plane"
    assert polygon["source_backend"] == "semantic_multiframe_arkit_depth_ransac_plane"
    assert polygon["source_frame_count"] == 2
    assert polygon["semantic_mask_collection_path"] == "background/tabletop_semantic_masks"
    assert polygon["top_z_m"] == pytest.approx(1.0)
    assert polygon["polygon_extent_x_m"] > 0.45
    assert (run / "background" / "fused_tabletop_points.ply").is_file()
    assert (run / "background" / "tabletop_semantic_mask.png").is_file()
    assert (run / "background" / "tabletop_mask.png").is_file()


def test_fit_tabletop_multiframe_default_consensus_rejects_single_frame_planar_overreach(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    masks_dir = run / "background" / "tabletop_semantic_masks"
    masks_dir.mkdir()
    confidence_dir = run / "confidence"
    confidence_dir.mkdir()
    anchor = np.zeros((100, 100), dtype=np.uint8)
    anchor[30:71, 30:71] = 255
    overreaching = np.zeros((100, 100), dtype=np.uint8)
    overreaching[10:91, 10:91] = 255
    Image.fromarray(anchor).save(masks_dir / "frame_000000.png")
    Image.fromarray(overreaching).save(masks_dir / "frame_000020.png")
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(confidence_dir / "frame_000000.png")
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(confidence_dir / "frame_000020.png")
    np.save(run / "depth" / "frame_000020.npy", np.ones((100, 100), dtype=np.float32))
    Image.new("RGB", (100, 100), color=(18, 22, 26)).save(run / "frames" / "frame_000020.jpg")
    trajectory = json.loads((run / "trajectory.json").read_text(encoding="utf-8"))
    trajectory["frames"][0]["confidence_path"] = "confidence/frame_000000.png"
    trajectory["frames"].append(
        {
            "frame_id": "frame_000020",
            "rgb_path": "frames/frame_000020.jpg",
            "depth_path": "depth/frame_000020.npy",
            "confidence_path": "confidence/frame_000020.png",
            "T_camera_to_world": _translate4(0.08, 0.0, 0.0),
        }
    )
    (run / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    code = main(
        [
            "fit-tabletop-multiframe",
            "--run-dir",
            str(run),
            "--masks",
            "background/tabletop_semantic_masks",
            "--frames",
            "0,20",
            "--write",
        ]
    )

    assert code == 0
    report = json.loads((run / "qa" / "table_multiframe_report.json").read_text(encoding="utf-8"))
    polygon = json.loads((run / "background" / "table_polygon_world.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["metrics"]["boundary_mode"] == "consensus-hull"
    assert report["metrics"]["boundary_min_frame_support"] == 2
    assert report["metrics"]["boundary_anchor_required"] is True
    assert polygon["boundary_mode"] == "consensus-hull"
    assert polygon["polygon_extent_x_m"] < 0.55
    assert polygon["polygon_extent_y_m"] < 0.55


def test_fit_tabletop_multiframe_blocks_when_only_one_frame_has_a_mask(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    masks_dir = run / "background" / "tabletop_semantic_masks"
    masks_dir.mkdir()
    confidence_dir = run / "confidence"
    confidence_dir.mkdir()
    semantic = np.zeros((100, 100), dtype=np.uint8)
    semantic[30:71, 30:71] = 255
    Image.fromarray(semantic).save(masks_dir / "frame_000000.png")
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(confidence_dir / "frame_000000.png")
    trajectory = json.loads((run / "trajectory.json").read_text(encoding="utf-8"))
    trajectory["frames"][0]["confidence_path"] = "confidence/frame_000000.png"
    trajectory["frames"].append(
        {
            "frame_id": "frame_000020",
            "rgb_path": "frames/frame_000020.jpg",
            "depth_path": "depth/frame_000020.npy",
            "confidence_path": "confidence/frame_000020.png",
            "T_camera_to_world": _translate4(0.08, 0.0, 0.0),
        }
    )
    (run / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    code = main(
        [
            "fit-tabletop-multiframe",
            "--run-dir",
            str(run),
            "--masks",
            "background/tabletop_semantic_masks",
            "--frames",
            "0,20",
            "--write",
        ]
    )

    assert code == 0
    report = json.loads((run / "qa" / "table_multiframe_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["metrics"]["used_frame_count"] == 1
    assert "insufficient_multiframe_tabletop_frames" in report["blocking_reasons"]
    assert report["artifacts"]["polygon_path"] is None


def test_qa_bg_table_uses_semantic_mask_for_raw_projection_gate(tmp_path):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    semantic = np.zeros((100, 100), dtype=np.uint8)
    semantic[38:63, 38:63] = 255
    Image.fromarray(semantic).save(run / "background" / "tabletop_semantic_mask.png")
    main(["build-table-collision", "--run-dir", str(run), "--source", "background/table_polygon_world.json", "--write"])

    code = main(["qa-bg-table", "--run-dir", str(run), "--frames", "0"])

    assert code == 0
    report = json.loads((run / "qa" / "bg_table_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["table_collision"]["projection_mask_source"] == "semantic_tabletop_mask"
    assert report["table_collision"]["projection_mask_path"] == "background/tabletop_semantic_mask.png"
    assert report["table_collision"]["raw_overreach_ratio"] > 0.15
    assert "overreach_ratio_above_threshold" in report["blocking_reasons"]
    assert report["claim"] is None



def test_fit_tabletop_from_mask_blocked_does_not_claim_written_geometry(tmp_path, capsys):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    (run / "background" / "tabletop_semantic_mask.png").unlink(missing_ok=True)

    code = main(["fit-tabletop-from-mask", "--run-dir", str(run), "--write"])

    captured = capsys.readouterr().out
    report = json.loads((run / "qa" / "tabletop_semantic_fit_report.json").read_text(encoding="utf-8"))
    assert code == 0
    assert report["status"] == "blocked"
    assert "tabletop_semantic_mask_missing" in report["blocking_reasons"]
    assert "background/tabletop_semantic_mask.png" not in captured
    assert "background/table_polygon_world.json" not in captured



def test_cli_segment_tabletop_mask_writes_standard_semantic_mask(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _write_bg_table_run(run)
    semantic = np.zeros((100, 100), dtype=bool)
    semantic[30:71, 30:71] = True

    class FakeSAM:
        def segment_text(self, image_path, text_prompt):  # noqa: ANN001
            assert Path(image_path).name == "frame_000000.jpg"
            assert "coffee table top" in text_prompt
            return semantic

        def segment_box(self, image_path, box_xyxy):  # noqa: ANN001
            raise AssertionError("text prompt path expected")

    from real2sim_scene_foundry import cli

    monkeypatch.setattr(cli, "_sam3_client_from_args", lambda args: FakeSAM())

    code = cli.main([
        "segment-tabletop-mask",
        "--run-dir",
        str(run),
        "--frame-index",
        "0",
        "--prompt",
        "tabletop / coffee table top",
    ])

    assert code == 0
    report = json.loads((run / "qa" / "tabletop_semantic_mask_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["source_backend"] == "sam3_text_prompt"
    assert report["semantic_mask_path"] == "background/tabletop_semantic_mask.png"
    assert report["mask_area_px"] == int(np.count_nonzero(semantic))
    assert (run / "background" / "tabletop_semantic_mask.png").is_file()
