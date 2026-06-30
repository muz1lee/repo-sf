import json

import numpy as np
from PIL import Image

from real2sim_scene_foundry.camera import CameraIntrinsics
from real2sim_scene_foundry.pipeline import run_smoke_reconstruction


class FakeDepthClient:
    def __init__(self, xyz):
        self.xyz = xyz
        self.calls = []

    def infer_xyz(self, left_path, right_path, camera, baseline_m):
        self.calls.append((left_path, right_path, camera, baseline_m))
        return self.xyz.copy()


def test_run_smoke_reconstruction_writes_manifest_mesh_export_and_qa(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (6, 4), color=(80, 90, 100)).save(left)
    Image.new("RGB", (6, 4), color=(75, 85, 95)).save(right)
    mask_arr = np.zeros((4, 6), dtype=np.uint8)
    mask_arr[1:3, 2:5] = 255
    Image.fromarray(mask_arr).save(mask)
    camera = CameraIntrinsics(width=6, height=4, fx=50.0, fy=50.0, cx=2.5, cy=1.5)
    depth = np.full((4, 6), 1.2, dtype=np.float32)
    xyz = camera.backproject_depth(depth)
    out = tmp_path / "run"

    result = run_smoke_reconstruction(
        left_image=left,
        right_image=right,
        camera=camera,
        baseline_m=0.08,
        out_dir=out,
        label="blue cup",
        mask_path=mask,
        depth_client=FakeDepthClient(xyz),
    )

    assert result.manifest_path == out / "scene_manifest.json"
    assert (out / "objects" / "blue_cup" / "mask.png").is_file()
    assert (out / "objects" / "blue_cup" / "crop.png").is_file()
    assert (out / "objects" / "blue_cup" / "mesh.glb").is_file()
    assert (out / "objects" / "blue_cup" / "pose.json").is_file()
    assert (out / "exports" / "scene.usda").is_file()
    assert (out / "qa" / "overlay.png").is_file()
    assert (out / "qa" / "render.png").is_file()
    qa = json.loads((out / "qa" / "qa_report.json").read_text(encoding="utf-8"))
    assert qa["object_count"] == 1
    assert qa["objects"][0]["mask_iou"] == 1.0
    assert qa["objects"][0]["needs_manual_refine"] is False
    manifest = json.loads((out / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["objects"][0]["object_id"] == "blue_cup"
    assert manifest["objects"][0]["source_backend"] == "metric_bbox"
