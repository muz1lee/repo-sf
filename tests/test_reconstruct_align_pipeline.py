import json

import numpy as np
import trimesh
from PIL import Image

from real2sim_scene_foundry.camera import CameraIntrinsics
from real2sim_scene_foundry.pipeline import run_extract, run_reconstruct_align
from real2sim_scene_foundry.proposals import ObjectProposal


class FakeDepthClient:
    def __init__(self, xyz):
        self.xyz = xyz

    def infer_xyz(self, left_path, right_path, camera, baseline_m):
        return self.xyz.copy()


class FakeSAM3Client:
    def segment_box(self, image_path, box_xyxy):
        mask = np.zeros((5, 6), dtype=bool)
        mask[1:4, 2:5] = True
        return mask


class FakeSAM3DClient:
    def __init__(self):
        self.calls = []

    def process(self, image_path, *, mask_path=None, text_prompt="object", out_dir):
        self.calls.append((image_path, mask_path, text_prompt, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = out_dir / "mesh.glb"
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)
        return type(
            "Result",
            (),
            {
                "mesh_path": mesh_path,
                "metadata": {
                    "success": True,
                    "ply_camera_base64": "large-payload",
                    "T_model_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                },
            },
        )()


class PromptFallbackSAM3DClient(FakeSAM3DClient):
    def process(self, image_path, *, mask_path=None, text_prompt="object", out_dir):
        if text_prompt == "red cup":
            self.calls.append((image_path, mask_path, text_prompt, out_dir))
            raise RuntimeError("not detected")
        return super().process(image_path, mask_path=mask_path, text_prompt=text_prompt, out_dir=out_dir)


class ImagePromptFallbackSAM3DClient(FakeSAM3DClient):
    def process(self, image_path, *, mask_path=None, text_prompt="object", out_dir):
        self.calls.append((image_path, mask_path, text_prompt, out_dir))
        if image_path.name == "crop.png" or text_prompt == "red cup":
            raise RuntimeError("not detected")
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = out_dir / "mesh.glb"
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)
        return type("Result", (), {"mesh_path": mesh_path, "metadata": {"success": True}})()


def test_run_reconstruct_align_writes_sam3d_raw_mesh_aligned_mesh_and_metrics(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (6, 5), color=(50, 60, 70)).save(left)
    Image.new("RGB", (6, 5), color=(45, 55, 65)).save(right)
    camera = CameraIntrinsics(width=6, height=5, fx=60.0, fy=60.0, cx=2.5, cy=2.0)
    depth = np.full((5, 6), 1.2, dtype=np.float32)
    xyz = camera.backproject_depth(depth)
    extract = run_extract(
        left_image=left,
        right_image=right,
        camera=camera,
        baseline_m=0.08,
        out_dir=tmp_path / "run",
        proposals=[
            ObjectProposal(
                label="red cup",
                object_id="red_cup",
                bbox_xyxy=(2, 1, 5, 4),
                point_xy=(3, 2),
                confidence=0.93,
                mass_kg=0.11,
                friction=0.6,
                source="qwen",
            )
        ],
        depth_client=FakeDepthClient(xyz),
        sam3_client=FakeSAM3Client(),
    )
    sam3d = FakeSAM3DClient()

    result = run_reconstruct_align(extract_dir=extract.out_dir, camera=camera, sam3d_client=sam3d)

    obj_dir = extract.out_dir / "objects" / "red_cup"
    assert sam3d.calls
    assert (obj_dir / "sam3d" / "raw_mesh.glb").is_file()
    assert (obj_dir / "mesh_aligned.glb").is_file()
    pose = json.loads((obj_dir / "pose.json").read_text(encoding="utf-8"))
    assert pose["source_backend"] == "sam3d_aligned"
    assert "ply_camera_base64" not in pose["sam3d_metadata"]
    assert pose["alignment"]["scale"] > 0
    assert pose["alignment"]["depth_residual_m"] >= 0
    assert pose["alignment"]["center_error_px"] >= 0
    local_mesh = trimesh.load(obj_dir / "mesh_aligned.glb", force="mesh")
    assert np.linalg.norm(local_mesh.bounds.mean(axis=0)) < 1e-6
    assert pose["T_object_to_camera"][2][3] > 0.0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["objects"][0]["mesh_path"] == "objects/red_cup/mesh_aligned.glb"
    assert manifest["objects"][0]["source_backend"] == "sam3d_aligned"
    assert manifest["background"]["bg_only_image_path"] == "background/bg_only.png"
    qa = json.loads(result.qa_report_path.read_text(encoding="utf-8"))
    assert qa["objects"][0]["alignment_backend"] == "sam3d_bbox_similarity"
    assert qa["objects"][0]["source_backend"] == "sam3d_aligned"


def test_run_reconstruct_align_falls_back_to_noun_prompt_for_sam3d(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (6, 5), color=(50, 60, 70)).save(left)
    Image.new("RGB", (6, 5), color=(45, 55, 65)).save(right)
    camera = CameraIntrinsics(width=6, height=5, fx=60.0, fy=60.0, cx=2.5, cy=2.0)
    xyz = camera.backproject_depth(np.full((5, 6), 1.2, dtype=np.float32))
    extract = run_extract(
        left_image=left,
        right_image=right,
        camera=camera,
        baseline_m=0.08,
        out_dir=tmp_path / "run",
        proposals=[
            ObjectProposal(
                label="red cup",
                object_id="red_cup",
                bbox_xyxy=(2, 1, 5, 4),
                point_xy=(3, 2),
                source="qwen",
            )
        ],
        depth_client=FakeDepthClient(xyz),
        sam3_client=FakeSAM3Client(),
    )
    sam3d = PromptFallbackSAM3DClient()

    run_reconstruct_align(extract_dir=extract.out_dir, camera=camera, sam3d_client=sam3d)

    prompts = [call[2] for call in sam3d.calls]
    assert prompts == ["red cup", "cup"]


def test_run_reconstruct_align_falls_back_from_crop_to_full_frame_for_sam3d(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (6, 5), color=(50, 60, 70)).save(left)
    Image.new("RGB", (6, 5), color=(45, 55, 65)).save(right)
    camera = CameraIntrinsics(width=6, height=5, fx=60.0, fy=60.0, cx=2.5, cy=2.0)
    xyz = camera.backproject_depth(np.full((5, 6), 1.2, dtype=np.float32))
    extract = run_extract(
        left_image=left,
        right_image=right,
        camera=camera,
        baseline_m=0.08,
        out_dir=tmp_path / "run",
        proposals=[
            ObjectProposal(
                label="red cup",
                object_id="red_cup",
                bbox_xyxy=(2, 1, 5, 4),
                point_xy=(3, 2),
                source="qwen",
            )
        ],
        depth_client=FakeDepthClient(xyz),
        sam3_client=FakeSAM3Client(),
    )
    sam3d = ImagePromptFallbackSAM3DClient()

    run_reconstruct_align(extract_dir=extract.out_dir, camera=camera, sam3d_client=sam3d)

    called = [(call[0].name, call[2]) for call in sam3d.calls]
    assert called == [("crop.png", "red cup"), ("crop.png", "cup"), ("crop.png", "object"), ("left.png", "red cup"), ("left.png", "cup")]
