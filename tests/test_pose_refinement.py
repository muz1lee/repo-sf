import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from real2sim_scene_foundry.pose_refinement import apply_visual_orientation_overrides, refine_visual_pose_to_masks, snap_object_poses_to_support


CAMERA_TO_WORLD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _write_pose_refinement_run(run_dir):
    object_dir = run_dir / "objects" / "cup"
    background_dir = run_dir / "background"
    object_dir.mkdir(parents=True)
    background_dir.mkdir()
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(object_dir / "visual.glb")
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(object_dir / "collision.glb")
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(object_dir / "debug_bbox.glb")
    trimesh.creation.box(extents=(1.0, 1.0, 0.04)).export(background_dir / "table_collision.glb")
    for path in [object_dir / "mask.png", object_dir / "crop.png", background_dir / "bg_only.png", background_dir / "foreground_mask.png"]:
        path.write_text("placeholder", encoding="utf-8")
    (background_dir / "bg_only_cloud.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (background_dir / "registration.json").write_text(json.dumps({"status": "registered"}), encoding="utf-8")
    T_world = CAMERA_TO_WORLD.copy()
    T_world[:3, 3] = [0.0, 1.0, -0.04]
    T_camera = np.linalg.inv(CAMERA_TO_WORLD) @ T_world
    scene = {
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
        },
        "support_plane": {
            "status": "estimated",
            "height_world_m": 0.0,
            "table_top_z_m": 0.0,
            "table_collision_mesh_path": "background/table_collision.glb",
            "table_collision_pos_world": [0.0, 1.0, -0.02],
            "table_collision_size_xyz": [1.0, 1.0, 0.04],
        },
        "objects": [
            {
                "object_id": "cup",
                "label": "cup",
                "mesh_path": "objects/cup/visual.glb",
                "mask_path": "objects/cup/mask.png",
                "crop_path": "objects/cup/crop.png",
                "T_object_to_camera": T_camera.tolist(),
                "T_object_to_world": T_world.tolist(),
                "scale_m": 0.2,
                "mass_kg": 0.25,
                "friction": 0.8,
                "confidence": 0.9,
                "source_backend": "sam3d_aligned",
                "needs_manual_refine": True,
                "visual_asset": {
                    "role": "visual_asset",
                    "path": "objects/cup/visual.glb",
                    "source": "sam3d_mesh_job_download_glb_aligned",
                    "status": "ready",
                    "final_visual": True,
                },
                "collision_asset": {
                    "role": "collision_asset",
                    "path": "objects/cup/collision.glb",
                    "source": "convex_hull_from_visual_mesh",
                    "status": "ready",
                },
                "debug_proxy": {
                    "role": "debug_proxy",
                    "path": "objects/cup/debug_bbox.glb",
                    "source": "bbox_from_visual_bounds",
                    "status": "ready",
                },
            }
        ],
    }
    (object_dir / "pose.json").write_text(
        json.dumps(
            {
                "object_id": "cup",
                "T_object_to_camera": T_camera.tolist(),
                "T_object_to_world": T_world.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")


def _mesh_counts(path):
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        return len(loaded.vertices), len(loaded.faces)
    vertices = 0
    faces = 0
    for geom in loaded.geometry.values():
        vertices += len(getattr(geom, "vertices", []))
        faces += len(getattr(geom, "faces", []))
    return vertices, faces


def _world_bottom(run_dir):
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    obj = scene["objects"][0]
    mesh = trimesh.load(run_dir / obj["visual_asset"]["path"], force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices * float(obj.get("asset_scale", 1.0))
    transform = np.asarray(obj["T_object_to_world"], dtype=np.float64)
    world = (transform @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]
    return float(world[:, 2].min())


def test_snap_object_poses_to_support_updates_world_and_camera_transforms(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_pose_refinement_run(run_dir)

    assert _world_bottom(run_dir) == pytest.approx(-0.14)

    result = snap_object_poses_to_support(run_dir)

    assert result.report_path == run_dir / "qa" / "pose_support_alignment_report.json"
    assert _world_bottom(run_dir) == pytest.approx(0.0, abs=1e-7)
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    obj = scene["objects"][0]
    assert obj["mesh_path"] == "objects/cup/visual.glb"
    assert obj["visual_asset"]["path"] == "objects/cup/visual.glb"
    assert obj["collision_asset"]["path"] == "objects/cup/collision.glb"
    assert obj["needs_manual_refine"] is True
    T_world = np.asarray(obj["T_object_to_world"], dtype=np.float64)
    T_camera = np.asarray(obj["T_object_to_camera"], dtype=np.float64)
    assert np.allclose(CAMERA_TO_WORLD @ T_camera, T_world)
    pose = json.loads((run_dir / "objects" / "cup" / "pose.json").read_text(encoding="utf-8"))
    assert pose["T_object_to_world"] == obj["T_object_to_world"]
    object_report = json.loads((run_dir / "objects" / "cup" / "pose_refinement_report.json").read_text(encoding="utf-8"))
    assert object_report["status"] == "accepted"
    assert object_report["source"] == "manual_refined_from_auto"
    assert object_report["translation_source"] == "manual_refined_from_viewer_overlay"
    assert object_report["support_alignment"]["status"] == "support_aligned"
    assert object_report["support_alignment"]["before_bottom_z_m"] == pytest.approx(-0.14)
    assert object_report["support_alignment"]["after_bottom_z_m"] == pytest.approx(0.0, abs=1e-7)
    assert result.report["status"] == "accepted"
    assert result.report["objects"][0]["translation_delta_world_m"] == pytest.approx([0.0, 0.0, 0.14])


def test_refine_visual_pose_to_masks_writes_asset_scale_and_keeps_support_contact(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_pose_refinement_run(run_dir)
    mask = Image.new("L", (200, 100), 0)
    pixels = mask.load()
    for y in range(45, 65):
        for x in range(90, 110):
            pixels[x, y] = 255
    mask.save(run_dir / "objects" / "cup" / "mask.png")
    (run_dir / "camera.json").write_text(
        json.dumps({"width": 200, "height": 100, "fx": 100.0, "fy": 100.0, "cx": 100.0, "cy": 50.0}),
        encoding="utf-8",
    )

    result = refine_visual_pose_to_masks(run_dir)

    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    obj = scene["objects"][0]
    assert result.report["status"] == "accepted"
    assert 0.1 < obj["asset_scale"] < 1.0
    assert _world_bottom(run_dir) == pytest.approx(0.0, abs=1e-7)
    object_report = json.loads((run_dir / "objects" / "cup" / "pose_refinement_report.json").read_text(encoding="utf-8"))
    assert object_report["scale_source"] == "reference_camera_bbox_refinement"
    assert object_report["reference_projection"]["mask_bbox_xyxy"] == [90, 45, 109, 64]
    assert object_report["asset_scale"] == pytest.approx(obj["asset_scale"])


def test_snap_after_support_change_records_current_visual_support_contact(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_pose_refinement_run(run_dir)
    snap_object_poses_to_support(run_dir)

    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    obj = scene["objects"][0]
    transform = np.asarray(obj["T_object_to_world"], dtype=np.float64)
    transform[2, 3] -= 0.037
    obj["T_object_to_world"] = transform.tolist()
    obj["T_object_to_camera"] = (np.linalg.inv(CAMERA_TO_WORLD) @ transform).tolist()
    scene["support_plane"]["table_top_z_m"] = 0.025
    scene["support_plane"]["height_world_m"] = 0.025
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    pose_path = run_dir / "objects" / "cup" / "pose.json"
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    pose["T_object_to_world"] = obj["T_object_to_world"]
    pose["T_object_to_camera"] = obj["T_object_to_camera"]
    pose_path.write_text(json.dumps(pose, indent=2), encoding="utf-8")

    assert _world_bottom(run_dir) < 0.025

    result = snap_object_poses_to_support(run_dir)

    assert _world_bottom(run_dir) == pytest.approx(0.025, abs=1e-7)
    object_report = json.loads((run_dir / "objects" / "cup" / "pose_refinement_report.json").read_text(encoding="utf-8"))
    assert object_report["support_alignment"]["support_z_m"] == pytest.approx(0.025)
    assert object_report["support_alignment"]["after_bottom_z_m"] == pytest.approx(0.025, abs=1e-7)
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    assert pose["pose_refinement"]["support_z_m"] == pytest.approx(0.025)
    assert pose["pose_refinement"]["support_alignment"]["after_bottom_z_m"] == pytest.approx(0.025, abs=1e-7)
    assert result.report["objects"][0]["support_alignment"]["after_bottom_z_m"] == pytest.approx(0.025, abs=1e-7)


def test_snap_preserves_final_tabletop_polygon_collision_mesh(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_pose_refinement_run(run_dir)
    table_path = run_dir / "background" / "table_collision.glb"
    trimesh.creation.cylinder(radius=0.6, height=0.04, sections=24).export(table_path)
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    scene["support_plane"]["table_collision_source_backend"] = "tabletop_mask_polygon_slab"
    scene["support_plane"]["table_collision_geometry_type"] = "polygon_slab"
    scene["support_plane"]["table_collision_final"] = True
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    before_counts = _mesh_counts(table_path)

    snap_object_poses_to_support(run_dir)

    after_counts = _mesh_counts(table_path)
    assert before_counts == after_counts
    assert after_counts[0] > 8
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    support = scene["support_plane"]
    assert support["table_collision_source_backend"] == "tabletop_mask_polygon_slab"
    assert support["table_collision_geometry_type"] == "polygon_slab"
    assert support["table_collision_final"] is True


def test_apply_visual_orientation_overrides_records_rotation_and_keeps_support_contact(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_pose_refinement_run(run_dir)
    snap_object_poses_to_support(run_dir)
    scene_before = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    before_R = np.asarray(scene_before["objects"][0]["T_object_to_world"], dtype=np.float64)[:3, :3]

    result = apply_visual_orientation_overrides(run_dir, {"cup": "flip_local_x_180"})

    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    obj = scene["objects"][0]
    after_R = np.asarray(obj["T_object_to_world"], dtype=np.float64)[:3, :3]
    assert result.report["status"] == "accepted"
    assert np.allclose(after_R, before_R @ np.diag([1.0, -1.0, -1.0]))
    assert _world_bottom(run_dir) == pytest.approx(0.0, abs=1e-7)
    object_report = json.loads((run_dir / "objects" / "cup" / "pose_refinement_report.json").read_text(encoding="utf-8"))
    assert object_report["rotation_source"] == "manual_refined_from_auto"
    assert object_report["rotation_override"] == "flip_local_x_180"
    assert np.allclose(np.asarray(object_report["rotation_before_override"], dtype=np.float64), before_R)
    assert np.allclose(np.asarray(object_report["rotation_after_override"], dtype=np.float64), after_R)
    assert object_report["support_alignment"]["status"] == "support_aligned"
    support = scene["support_plane"]
    assert support["table_collision_extent_source"] == "visible_support_bounds_plus_object_footprints"
    assert support["table_collision_size_xyz"][0] >= 1.0
    assert support["table_collision_size_xyz"][1] >= 1.0

    apply_visual_orientation_overrides(run_dir, {"cup": "flip_local_x_180"})
    scene_after_second_run = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    second_R = np.asarray(scene_after_second_run["objects"][0]["T_object_to_world"], dtype=np.float64)[:3, :3]
    assert np.allclose(second_R, after_R)
