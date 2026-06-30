import json

import numpy as np
from PIL import Image

from real2sim_scene_foundry.camera import CameraIntrinsics
from real2sim_scene_foundry.pipeline import run_extract
from real2sim_scene_foundry.proposals import ObjectProposal


class FakeDepthClient:
    def __init__(self, xyz):
        self.xyz = xyz

    def infer_xyz(self, left_path, right_path, camera, baseline_m):
        return self.xyz.copy()


class FakeSAM3Client:
    def __init__(self):
        self.boxes = []

    def segment_box(self, image_path, box_xyxy):
        self.boxes.append(tuple(box_xyxy))
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:3, 1:3] = True
        mask[4:6, 5:7] = True
        return mask


def test_run_extract_uses_box_prompt_and_keeps_component_near_point(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (8, 6), color=(80, 90, 100)).save(left)
    Image.new("RGB", (8, 6), color=(70, 80, 90)).save(right)
    camera = CameraIntrinsics(width=8, height=6, fx=50.0, fy=50.0, cx=3.5, cy=2.5)
    xyz = camera.backproject_depth(np.full((6, 8), 1.0, dtype=np.float32))
    sam3 = FakeSAM3Client()

    result = run_extract(
        left_image=left,
        right_image=right,
        camera=camera,
        baseline_m=0.08,
        out_dir=tmp_path / "extract",
        proposals=[
            ObjectProposal(
                label="red cup",
                object_id="red_cup",
                bbox_xyxy=(0, 0, 7, 5),
                point_xy=(6, 5),
                confidence=0.9,
                source="qwen",
            )
        ],
        depth_client=FakeDepthClient(xyz),
        sam3_client=sam3,
    )

    out = result.out_dir
    assert sam3.boxes == [(0, 0, 7, 5)]
    mask = np.asarray(Image.open(out / "objects" / "red_cup" / "mask.png").convert("L")) > 0
    assert mask.sum() == 4
    assert mask[4:6, 5:7].all()
    assert not mask[1:3, 1:3].any()
    assert (out / "left.png").is_file()
    assert (out / "xyz.npy").is_file()
    assert (out / "scene_cloud.ply").is_file()
    manifest = json.loads((out / "extraction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["objects"][0]["proposal_source"] == "qwen"
    assert manifest["objects"][0]["mask_area_px"] == 4
    assert result.object_dirs["red_cup"] == out / "objects" / "red_cup"

