import json

import numpy as np
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
