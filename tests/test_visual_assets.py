from pathlib import Path

from real2sim_scene_foundry.visual_assets import plan_sam3d_visual_assets


class PointCloudOnlyResult:
    mesh_path = None
    metadata = {"success": True, "ply_camera_base64": "payload"}
    debug_bbox_path = None

    def __init__(self, point_cloud_path: Path):
        self.point_cloud_path = point_cloud_path


def test_point_cloud_only_visual_plan_keeps_bbox_debug_only(tmp_path):
    object_dir = tmp_path / "objects" / "cup"
    sam3d_dir = object_dir / "sam3d"
    sam3d_dir.mkdir(parents=True)
    point_cloud = sam3d_dir / "point_cloud.ply"
    point_cloud.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 1",
                "1 0 1",
                "0 1 1",
                "1 1 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plan = plan_sam3d_visual_assets(
        object_id="cup",
        label="cup",
        object_dir=object_dir,
        sam3d_result=PointCloudOnlyResult(point_cloud),
        run_dir=tmp_path,
    )

    assert plan.final_visual is False
    assert plan.visual_path is None
    assert plan.visual_point_cloud_path == object_dir / "visual_point_cloud.ply"
    assert plan.aligned_output_path == object_dir / "debug_bbox.glb"
    assert plan.debug_bbox_path == object_dir / "debug_bbox.glb"
    assert "mesh_aligned" not in plan.aligned_output_path.name
    assert not (object_dir / "mesh_aligned.glb").exists()
    assert (object_dir / "visual_point_cloud.ply").is_file()
    assert (object_dir / "debug_bbox.glb").is_file()
