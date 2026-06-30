import numpy as np

from real2sim_scene_foundry.camera import CameraIntrinsics


def test_project_backproject_depth_round_trip():
    camera = CameraIntrinsics(width=4, height=3, fx=100.0, fy=120.0, cx=1.5, cy=1.0)
    depth = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.5, 2.0, 2.5, 3.0],
            [0.0, 1.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )

    xyz = camera.backproject_depth(depth)
    u, v, z = camera.project(xyz.reshape(-1, 3))
    valid = depth.reshape(-1) > 0.0

    expected_u, expected_v = np.meshgrid(np.arange(4), np.arange(3))
    np.testing.assert_allclose(u[valid], expected_u.reshape(-1)[valid], atol=1e-5)
    np.testing.assert_allclose(v[valid], expected_v.reshape(-1)[valid], atol=1e-5)
    np.testing.assert_allclose(z[valid], depth.reshape(-1)[valid], atol=1e-5)


def test_calibration_json_supports_nested_head_schema(tmp_path):
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        '{"head":{"width":1280,"height":720,"fx":441.9,"fy":441.8,'
        '"cx":647.5,"cy":358.4,"baseline":0.07999}}',
        encoding="utf-8",
    )

    camera, baseline = CameraIntrinsics.from_calibration_json(calib_path, camera_name="head")

    assert camera.width == 1280
    assert camera.height == 720
    assert camera.fx == 441.9
    assert camera.fy == 441.8
    assert baseline == 0.07999
