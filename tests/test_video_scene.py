import json
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from real2sim_scene_foundry.manifest import SceneBackground, SceneManifest
from real2sim_scene_foundry.proposals import ObjectProposal
from real2sim_scene_foundry.video_scene import (
    MoGePointMapDepthClient,
    attach_video_3dgs_background,
    camera_from_moge_fov,
    load_default_video_object_labels,
    run_video_reference_scene,
)


class FakeMaskClient:
    def segment_box(self, image_path, box_xyxy):
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 2:5] = True
        return mask

    def segment_text(self, image_path, text_prompt):
        raise AssertionError("box proposal should call segment_box")


class FakeSAM3DClient:
    def process(self, image_path, *, mask_path=None, text_prompt="object", out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh = out_dir / "mesh.glb"
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh)
        return type("Result", (), {"mesh_path": mesh, "metadata": {"success": True}})()


def _write_exr(path: Path, points: np.ndarray) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    ok = cv2.imwrite(str(path), points.astype(np.float32))
    assert ok


def test_camera_from_moge_fov_uses_image_dimensions():
    fov = {"fov_x": 60.0, "fov_y": 40.0}

    camera = camera_from_moge_fov(fov, width=720, height=405)

    assert camera.width == 720
    assert camera.height == 405
    assert camera.cx == 359.5
    assert camera.cy == 202.0
    assert camera.fx > 0
    assert camera.fy > 0


def test_moge_point_map_depth_client_loads_exr(tmp_path):
    points = np.zeros((4, 6, 3), dtype=np.float32)
    points[..., 0] = 1.25
    points[..., 1] = 0.2
    points[..., 2] = -0.1
    exr = tmp_path / "points.exr"
    _write_exr(exr, points)

    loaded = MoGePointMapDepthClient(exr).infer_xyz("left.png", "right.png", camera=None, baseline_m=0.0)

    assert loaded.shape == (4, 6, 3)
    np.testing.assert_allclose(loaded[0, 0], [-0.1, 0.2, 1.25], atol=1e-6)


def test_load_default_video_object_labels_prefers_bg_only_observed_labels(tmp_path):
    run = tmp_path / "run"
    status = run / "video" / "bg_only_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"observed_labels": ["bottle", "cup", "bottle", "cloth"]}),
        encoding="utf-8",
    )

    assert load_default_video_object_labels(run) == ["bottle", "cup", "cloth"]


def test_attach_video_3dgs_background_updates_scene_manifest_and_qa(tmp_path):
    run = tmp_path / "run"
    (run / "qa").mkdir(parents=True)
    (run / "scene_manifest.json").write_text(
        json.dumps(SceneManifest(objects=[]).to_dict()),
        encoding="utf-8",
    )
    (run / "qa" / "qa_report.json").write_text(json.dumps({"background": {}}), encoding="utf-8")
    (run / "video" / "3dgs_status.json").parent.mkdir(parents=True)
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "outputs": {
                    "latest_run_dir": "video/3dgs/run",
                    "latest_config": "video/3dgs/run/config.yml",
                    "latest_checkpoint": "video/3dgs/run/nerfstudio_models/step.ckpt",
                },
            }
        ),
        encoding="utf-8",
    )

    attach_video_3dgs_background(run)

    manifest = json.loads((run / "scene_manifest.json").read_text(encoding="utf-8"))
    background = manifest["background"]
    assert background["source_backend"] == "video_3dgs_splatfacto"
    assert background["gaussian_splat_path"] == "video/3dgs/run"
    assert background["gaussian_splat_config_path"] == "video/3dgs/run/config.yml"
    assert background["gaussian_splat_checkpoint_path"] == "video/3dgs/run/nerfstudio_models/step.ckpt"
    qa = json.loads((run / "qa" / "qa_report.json").read_text(encoding="utf-8"))
    assert qa["background"]["status"] == "trained_video_3dgs"


def test_run_video_reference_scene_builds_objects_and_attaches_3dgs_background(tmp_path):
    run = tmp_path / "run"
    reference_dir = run / "video" / "moge_reference" / "reference"
    reference_dir.mkdir(parents=True)
    Image.new("RGB", (6, 4), color=(10, 20, 30)).save(reference_dir / "image.jpg")
    points = np.zeros((4, 6, 3), dtype=np.float32)
    points[..., 2] = 1.2
    points[..., 0] = np.linspace(-0.1, 0.1, 6)[None, :]
    points[..., 1] = np.linspace(-0.05, 0.05, 4)[:, None]
    _write_exr(reference_dir / "points.exr", points)
    (reference_dir / "fov.json").write_text(json.dumps({"fov_x": 60.0, "fov_y": 40.0}), encoding="utf-8")
    (run / "video" / "3dgs_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "outputs": {
                    "latest_run_dir": "video/3dgs/run",
                    "latest_config": "video/3dgs/run/config.yml",
                    "latest_checkpoint": "video/3dgs/run/nerfstudio_models/step.ckpt",
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_video_reference_scene(
        run_dir=run,
        proposals=[
            ObjectProposal(
                label="cup",
                object_id="cup",
                bbox_xyxy=(2, 1, 5, 3),
                point_xy=(3, 2),
                source="test",
            )
        ],
        sam3_client=FakeMaskClient(),
        sam3d_client=FakeSAM3DClient(),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["objects"][0]["object_id"] == "cup"
    assert manifest["background"]["status"] == "trained_video_3dgs"
    assert manifest["background"]["gaussian_splat_config_path"] == "video/3dgs/run/config.yml"
    assert (run / "exports" / "run_interactive_scene.py").is_file()
    interaction = json.loads((run / "qa" / "interaction_report.json").read_text(encoding="utf-8"))
    assert interaction["background"]["status"] == "trained_video_3dgs"
