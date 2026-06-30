import json

import numpy as np
from PIL import Image

from real2sim_scene_foundry.background import ArrayInpaintClient, create_background_artifacts
from real2sim_scene_foundry.camera import CameraIntrinsics
from real2sim_scene_foundry.pipeline import run_extract
from real2sim_scene_foundry.proposals import ObjectProposal


class FakeDepthClient:
    def __init__(self, xyz):
        self.xyz = xyz

    def infer_xyz(self, left_path, right_path, camera, baseline_m):
        return self.xyz.copy()


class FakeSAM3Client:
    def segment_box(self, image_path, box_xyxy):
        mask = np.zeros((4, 5), dtype=bool)
        x0, y0, x1, y1 = box_xyxy
        mask[y0:y1, x0:x1] = True
        return mask


class FillInpaintClient(ArrayInpaintClient):
    backend_name = "fill-test"

    def inpaint(self, rgb, remove_mask):
        out = rgb.copy()
        out[remove_mask] = np.array([9, 8, 7], dtype=np.uint8)
        return out


def test_create_background_artifacts_unions_masks_and_writes_bg_only_cloud(tmp_path):
    run = tmp_path / "run"
    objects = run / "objects" / "cup"
    objects.mkdir(parents=True)
    rgb = np.full((4, 5, 3), 100, dtype=np.uint8)
    Image.fromarray(rgb).save(run / "left.png")
    xyz = np.dstack(
        [
            np.tile(np.arange(5, dtype=np.float32), (4, 1)),
            np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 5)),
            np.ones((4, 5), dtype=np.float32),
        ]
    )
    np.save(run / "xyz.npy", xyz)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    Image.fromarray(mask).save(objects / "mask.png")
    (run / "extraction_manifest.json").write_text(
        json.dumps(
            {
                "left_image": "left.png",
                "xyz": "xyz.npy",
                "objects": [{"object_id": "cup", "mask_path": "objects/cup/mask.png"}],
            }
        ),
        encoding="utf-8",
    )

    result = create_background_artifacts(
        run,
        inpaint_client=FillInpaintClient(),
        dilation_px=0,
    )

    bg = np.asarray(Image.open(result.bg_only_path).convert("RGB"))
    fg = np.asarray(Image.open(result.foreground_mask_path).convert("L")) > 0
    assert fg.sum() == 4
    assert (bg[fg] == np.array([9, 8, 7], dtype=np.uint8)).all()
    assert "element vertex 16" in result.bg_only_cloud_path.read_text(encoding="utf-8")
    data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert data["stage"] == "background"
    assert data["inpaint_backend"] == "fill-test"
    assert data["background_3dgs"]["status"] == "requires_video_or_multiview"


def test_run_extract_writes_background_branch_by_default(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (5, 4), color=(80, 90, 100)).save(left)
    Image.new("RGB", (5, 4), color=(70, 80, 90)).save(right)
    camera = CameraIntrinsics(width=5, height=4, fx=50.0, fy=50.0, cx=2.0, cy=1.5)
    xyz = camera.backproject_depth(np.full((4, 5), 1.0, dtype=np.float32))

    result = run_extract(
        left_image=left,
        right_image=right,
        camera=camera,
        baseline_m=0.08,
        out_dir=tmp_path / "extract",
        proposals=[
            ObjectProposal(
                label="cup",
                object_id="cup",
                bbox_xyxy=(1, 1, 4, 3),
                point_xy=(2, 2),
                source="qwen",
            )
        ],
        depth_client=FakeDepthClient(xyz),
        sam3_client=FakeSAM3Client(),
    )

    assert (result.out_dir / "background" / "foreground_mask.png").is_file()
    assert (result.out_dir / "background" / "bg_only.png").is_file()
    manifest = json.loads((result.out_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["background"]["manifest_path"] == "background/background_manifest.json"
