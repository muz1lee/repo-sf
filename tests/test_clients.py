import io

import numpy as np
import pytest
from PIL import Image

from real2sim_scene_foundry.camera import CameraIntrinsics
from real2sim_scene_foundry.clients import MoGeClient, S2M2Client, SAM3Client, SAM3DClient, decode_sam3_mask


class DummyResponse:
    def __init__(self, *, json_data=None, content=b"", status_error=None):
        self._json_data = json_data
        self.content = content
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def json(self):
        return self._json_data


def test_s2m2_client_posts_stereo_pair_and_loads_xyz(tmp_path, monkeypatch):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(left)
    Image.new("RGB", (2, 2), color=(4, 5, 6)).save(right)
    xyz = np.ones((2, 2, 3), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, xyz)
    calls = []

    def fake_post(url, files, data, timeout):
        calls.append({"url": url, "data": dict(data), "timeout": timeout, "files": sorted(files)})
        return DummyResponse(content=buf.getvalue())

    monkeypatch.setattr("real2sim_scene_foundry.clients.requests.post", fake_post)

    camera = CameraIntrinsics(width=2, height=2, fx=10.0, fy=11.0, cx=1.0, cy=1.0)
    result = S2M2Client(["http://depth/api/process"], timeout_s=7.0).infer_xyz(left, right, camera, 0.08)

    np.testing.assert_allclose(result, xyz)
    assert calls == [
        {
            "url": "http://depth/api/process",
            "data": {"K": "10.0 0.0 1.0 0.0 11.0 1.0 0.0 0.0 1.0", "baseline": "0.08", "width": "2"},
            "timeout": 7.0,
            "files": ["left_file", "right_file"],
        }
    ]


def test_moge_client_returns_intrinsics(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (4, 3)).save(image)

    def fake_post(url, files, timeout):
        return DummyResponse(json_data={"width": 4, "height": 3, "fx": 100.0, "fy": 101.0, "cx": 2.0, "cy": 1.5})

    monkeypatch.setattr("real2sim_scene_foundry.clients.requests.post", fake_post)

    camera = MoGeClient("http://moge").infer_intrinsics(image)

    assert camera == CameraIntrinsics(width=4, height=3, fx=100.0, fy=101.0, cx=2.0, cy=1.5)


def test_sam3_decode_rejects_empty_detections():
    with pytest.raises(RuntimeError, match="no masks"):
        decode_sam3_mask({"success": True, "detections": []}, size=(8, 6), label="cup")


def test_sam3_client_sends_text_prompt(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (2, 2)).save(image)
    mask = Image.new("L", (2, 2), color=255)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    import base64

    payload = {"success": True, "detections": [{"mask": base64.b64encode(buf.getvalue()).decode("ascii")}]}
    requests_seen = []

    def fake_post(url, json, timeout):
        requests_seen.append((url, json["text_prompt"]))
        return DummyResponse(json_data=payload)

    monkeypatch.setattr("real2sim_scene_foundry.clients.requests.post", fake_post)

    result = SAM3Client("http://sam3/segment").segment_text(image, "blue cup")

    assert requests_seen == [("http://sam3/segment", "blue cup")]
    assert result.shape == (2, 2)
    assert result.dtype == bool
    assert result.all()


def test_sam3d_client_posts_service_contract_and_writes_base64_mesh(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    mask = tmp_path / "mask.png"
    out = tmp_path / "sam3d"
    Image.new("RGB", (2, 2)).save(image)
    Image.new("L", (2, 2), color=255).save(mask)
    import base64

    payload = {
        "success": True,
        "mesh_glb_base64": base64.b64encode(b"glb-bytes").decode("ascii"),
        "T_model_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
    }
    seen = []

    def fake_post(url, files, data, timeout):
        seen.append((url, sorted(files), dict(data)))
        return DummyResponse(json_data=payload)

    monkeypatch.setattr("real2sim_scene_foundry.clients.requests.post", fake_post)

    result = SAM3DClient("http://sam3d/api/process").process(image, mask_path=mask, text_prompt="red cup", out_dir=out)

    assert seen == [("http://sam3d/api/process", ["image_file"], {"text_prompt": "red cup"})]
    assert result.mesh_path.read_bytes() == b"glb-bytes"
    assert result.metadata["T_model_to_camera"][2][3] == 1


def test_sam3d_client_converts_ply_camera_response_to_proxy_mesh(monkeypatch, tmp_path):
    image = tmp_path / "frame.png"
    out = tmp_path / "sam3d"
    Image.new("RGB", (2, 2)).save(image)
    import base64

    ply = b"""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
end_header
0 0 1
1 0 1
0 1 1
1 1 2
"""
    payload = {"success": True, "ply_camera_base64": base64.b64encode(ply).decode("ascii")}

    def fake_post(url, files, data, timeout):
        return DummyResponse(json_data=payload)

    monkeypatch.setattr("real2sim_scene_foundry.clients.requests.post", fake_post)

    result = SAM3DClient("http://sam3d/api/process").process(image, text_prompt="cup", out_dir=out)

    assert (out / "point_cloud.ply").read_bytes() == ply
    assert result.mesh_path.name == "mesh.glb"
    assert result.mesh_path.stat().st_size > 0
