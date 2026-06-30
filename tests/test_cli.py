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
