import json

import numpy as np
import pytest
from PIL import Image

from real2sim_scene_foundry.cli import main


def test_cli_smoke_with_mock_depth_and_mask(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    mask = tmp_path / "mask.png"
    calib = tmp_path / "calib.json"
    out = tmp_path / "out"
    Image.new("RGB", (5, 4), color=(10, 20, 30)).save(left)
    Image.new("RGB", (5, 4), color=(11, 21, 31)).save(right)
    mask_arr = np.zeros((4, 5), dtype=np.uint8)
    mask_arr[1:3, 1:4] = 255
    Image.fromarray(mask_arr).save(mask)
    calib.write_text(
        json.dumps({"width": 5, "height": 4, "fx": 40.0, "fy": 40.0, "cx": 2.0, "cy": 1.5, "baseline": 0.08}),
        encoding="utf-8",
    )

    code = main(
        [
            "smoke",
            "--left",
            str(left),
            "--right",
            str(right),
            "--calib",
            str(calib),
            "--mask",
            str(mask),
            "--label",
            "test cup",
            "--mock-depth",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert (out / "scene_manifest.json").is_file()
    assert (out / "qa" / "qa_report.json").is_file()


def test_cli_extract_with_yaml_and_mock_depth(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    objects = tmp_path / "objects.yaml"
    calib = tmp_path / "calib.json"
    out = tmp_path / "extract"
    Image.new("RGB", (6, 5), color=(10, 20, 30)).save(left)
    Image.new("RGB", (6, 5), color=(11, 21, 31)).save(right)
    objects.write_text(
        """
objects:
  - label: red cup
    object_id: red_cup
    bbox_xyxy: [1, 1, 4, 4]
    point_xy: [2, 2]
""",
        encoding="utf-8",
    )
    calib.write_text(
        json.dumps({"width": 6, "height": 5, "fx": 40.0, "fy": 40.0, "cx": 2.5, "cy": 2.0, "baseline": 0.08}),
        encoding="utf-8",
    )

    code = main(
        [
            "extract",
            "--left",
            str(left),
            "--right",
            str(right),
            "--calib",
            str(calib),
            "--objects-yaml",
            str(objects),
            "--mock-depth",
            "--mock-mask-from-bbox",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert (out / "extraction_manifest.json").is_file()
    assert (out / "objects" / "red_cup" / "object_cloud.ply").is_file()


def test_cli_extract_uses_qwen_proposal_before_sam_when_no_manual_proposal(tmp_path, monkeypatch):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    calib = tmp_path / "calib.json"
    out = tmp_path / "extract"
    Image.new("RGB", (6, 5), color=(10, 20, 30)).save(left)
    Image.new("RGB", (6, 5), color=(11, 21, 31)).save(right)
    calib.write_text(
        json.dumps({"width": 6, "height": 5, "fx": 40.0, "fy": 40.0, "cx": 2.5, "cy": 2.0, "baseline": 0.08}),
        encoding="utf-8",
    )
    calls = []

    class FakeQwenProposalClient:
        def __init__(self, *, base_url=None, api_key=None, model=None):  # noqa: ANN001
            calls.append((base_url, api_key, model))

        def propose(self, image_path, *, target_labels=None):  # noqa: ANN001
            from real2sim_scene_foundry.proposals import ObjectProposal

            calls.append(("propose", image_path, target_labels))
            return [
                ObjectProposal(
                    label="red cup",
                    object_id="red_cup",
                    bbox_xyxy=(1, 1, 4, 4),
                    point_xy=(2, 2),
                    source="qwen",
                )
            ]

    monkeypatch.setattr("real2sim_scene_foundry.cli.QwenProposalClient", FakeQwenProposalClient)

    code = main(
        [
            "extract",
            "--left",
            str(left),
            "--right",
            str(right),
            "--calib",
            str(calib),
            "--label",
            "red cup",
            "--mock-depth",
            "--mock-mask-from-bbox",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert calls[1] == ("propose", left, ["red cup"])
    manifest = json.loads((out / "extraction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["objects"][0]["proposal_source"] == "qwen"


def test_cli_interactive_writes_scene_launcher(tmp_path):
    import trimesh

    run = tmp_path / "run"
    mesh = run / "objects" / "cup" / "mesh_aligned.glb"
    mesh.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(mesh)
    (run / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/mesh_aligned.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        "T_object_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        "scale_m": 0.1,
                        "mass_kg": 0.2,
                        "friction": 0.8,
                        "confidence": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main(["interactive", "--run-dir", str(run), "--settle-steps", "10"])

    assert code == 0
    assert (run / "exports" / "run_interactive_scene.py").is_file()
    report = json.loads((run / "qa" / "interaction_report.json").read_text(encoding="utf-8"))
    assert report["physics_settle"]["status"] == "proxy_checked"


def test_cli_support_plane_updates_existing_scene(tmp_path):
    import trimesh

    run = tmp_path / "run"
    mesh = run / "objects" / "cup" / "mesh_aligned.glb"
    mesh.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(mesh)
    transform = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1]]
    (mesh.parent / "pose.json").write_text(
        json.dumps({"object_id": "cup", "T_object_to_world": transform}),
        encoding="utf-8",
    )
    (run / "exports").mkdir(parents=True)
    (run / "qa").mkdir()
    (run / "qa" / "qa_report.json").write_text(json.dumps({}), encoding="utf-8")
    (run / "scene_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_frames": {
                    "camera": "opencv_x_right_y_down_z_forward_meters",
                    "world": "z_up_ground_plane_meters",
                },
                "objects": [
                    {
                        "object_id": "cup",
                        "label": "cup",
                        "mesh_path": "objects/cup/mesh_aligned.glb",
                        "mask_path": "objects/cup/mask.png",
                        "crop_path": "objects/cup/crop.png",
                        "T_object_to_camera": transform,
                        "T_object_to_world": transform,
                        "scale_m": 0.2,
                        "mass_kg": 0.2,
                        "friction": 0.8,
                        "confidence": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main(["support-plane", "--run-dir", str(run)])

    assert code == 0
    manifest = json.loads((run / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["support_plane"]["status"] == "estimated"
    assert manifest["objects"][0]["T_object_to_world"][2][3] == pytest.approx(0.1)


def test_cli_export_manifest_writes_manifest_json(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    run.mkdir()
    calls = []

    def fake_write(run_dir):  # noqa: ANN001
        calls.append(run_dir)
        path = run_dir / "sim_export_manifest.json"
        path.write_text('{"status":"passed"}', encoding="utf-8")
        return path

    def fake_build(run_dir):  # noqa: ANN001
        return run_dir / "not_written_manifest_repr.json"

    monkeypatch.setattr("real2sim_scene_foundry.cli.write_sim_export_manifest", fake_write, raising=False)
    monkeypatch.setattr("real2sim_scene_foundry.cli.build_sim_export_manifest", fake_build)

    code = main(["export-manifest", "--run-dir", str(run)])

    captured = capsys.readouterr()
    assert code == 0
    assert calls == [run]
    assert (run / "sim_export_manifest.json").is_file()
    assert f"wrote {run / 'sim_export_manifest.json'}" in captured.out


def test_cli_render_3dgs_background_calls_external_renderer(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    calls = []

    class Result:
        report_path = run / "qa" / "background_3dgs_render_report.json"
        report = {"status": "rendered"}

    def fake_render(run_dir, **kwargs):  # noqa: ANN001
        calls.append((run_dir, kwargs))
        return Result()

    monkeypatch.setattr("real2sim_scene_foundry.cli.render_external_3dgs_background", fake_render)

    code = main(["render-3dgs-background", "--run-dir", str(run), "--split", "test", "--camera-idx", "0"])

    assert code == 0
    assert calls == [(run, {"split": "test", "camera_idx": 0})]


def test_cli_runtime_viewer_exports_genesis_runtime_bridge(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    seen = []

    class Result:
        index_path = run / "exports" / "runtime_viewer" / "index.html"
        runtime_manifest_path = run / "exports" / "runtime_viewer" / "runtime_manifest.json"
        script_path = run / "exports" / "genesis_runtime_server.py"
        report = {"status": "runtime_script_written", "backend": "genesis"}

    def fake_export(run_dir, *, backend):  # noqa: ANN001
        seen.append((run_dir, backend))
        return Result()

    monkeypatch.setattr("real2sim_scene_foundry.cli.export_runtime_viewer", fake_export)

    code = main(["runtime-viewer", "--run-dir", str(run), "--backend", "genesis", "--port", "7031"])

    assert code == 0
    assert seen == [(run, "genesis")]


def test_cli_extract_passes_http_inpaint_client(tmp_path, monkeypatch):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    objects = tmp_path / "objects.yaml"
    calib = tmp_path / "calib.json"
    out = tmp_path / "extract"
    Image.new("RGB", (6, 5), color=(10, 20, 30)).save(left)
    Image.new("RGB", (6, 5), color=(11, 21, 31)).save(right)
    objects.write_text(
        """
objects:
  - label: red cup
    object_id: red_cup
    bbox_xyxy: [1, 1, 4, 4]
""",
        encoding="utf-8",
    )
    calib.write_text(
        json.dumps({"width": 6, "height": 5, "fx": 40.0, "fy": 40.0, "cx": 2.5, "cy": 2.0, "baseline": 0.08}),
        encoding="utf-8",
    )
    seen = []

    class FakeHTTPInpaintClient:
        def __init__(self, url):
            self.url = url

    def fake_run_extract(**kwargs):
        seen.append(kwargs["background_inpaint_client"].url)
        return type("Result", (), {"manifest_path": out / "extraction_manifest.json"})()

    monkeypatch.setattr("real2sim_scene_foundry.cli.HTTPInpaintClient", FakeHTTPInpaintClient)
    monkeypatch.setattr("real2sim_scene_foundry.cli.run_extract", fake_run_extract)

    code = main(
        [
            "extract",
            "--left",
            str(left),
            "--right",
            str(right),
            "--calib",
            str(calib),
            "--objects-yaml",
            str(objects),
            "--inpaint-url",
            "http://inpaint/inpaint",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert seen == ["http://inpaint/inpaint"]


def test_cli_video_prep_writes_manifest(tmp_path):
    import cv2

    video = tmp_path / "phone.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (6, 4))
    assert writer.isOpened()
    for idx in range(3):
        frame = np.full((4, 6, 3), idx * 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    out = tmp_path / "out"

    code = main(
        [
            "video-prep",
            "--video",
            str(video),
            "--out",
            str(out),
            "--frame-stride",
            "1",
            "--reference-frame-index",
            "0",
        ]
    )

    assert code == 0
    manifest = json.loads((out / "video" / "video_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sampled_frame_count"] == 3


def test_cli_video_scene_runs_reference_scene_builder(tmp_path, monkeypatch):
    run = tmp_path / "run"
    seen = []

    def fake_run_video_reference_scene(**kwargs):
        seen.append(kwargs)
        return type(
            "Result",
            (),
            {
                "manifest_path": run / "scene_manifest.json",
                "usd_path": run / "exports" / "scene.usda",
                "qa_report_path": run / "qa" / "qa_report.json",
                "interactive_script_path": run / "exports" / "run_interactive_scene.py",
                "interaction_report_path": run / "qa" / "interaction_report.json",
            },
        )()

    monkeypatch.setattr("real2sim_scene_foundry.cli.run_video_reference_scene", fake_run_video_reference_scene)

    code = main(
        [
            "video-scene",
            "--run-dir",
            str(run),
            "--target-label",
            "cup",
            "--max-objects",
            "2",
            "--settle-steps",
            "5",
            "--qwen-base-url",
            "http://qwen.example/v1",
            "--qwen-api-key",
            "secret",
        ]
    )

    assert code == 0
    assert seen[0]["run_dir"] == run
    assert seen[0]["target_labels"] == ["cup"]
    assert seen[0]["max_objects"] == 2
    assert seen[0]["settle_steps"] == 5
