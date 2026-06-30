import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (8, 6))
    assert writer.isOpened()
    for idx in range(3):
        frame = np.full((6, 8, 3), idx * 60, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_prep_script_runs_with_synthetic_video(tmp_path):
    video = tmp_path / "phone.avi"
    out = tmp_path / "video_run"
    _write_test_video(video)

    env = os.environ.copy()
    env["RSF_PYTHON"] = sys.executable
    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "rsf_video_prep.sh"),
            "--video",
            str(video),
            "--out",
            str(out),
            "--frame-stride",
            "1",
            "--reference-frame-index",
            "0",
        ],
        env=env,
        check=True,
        cwd=PROJECT_ROOT,
    )

    manifest = json.loads((out / "video" / "video_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_type"] == "rgb_video"
    assert manifest["sampled_frame_count"] == 3
    assert manifest["camera_poses"]["status"] == "not_estimated"


def test_stereo_scene_script_uses_qwen_env_and_inpaint_url():
    script = (PROJECT_ROOT / "scripts" / "rsf_stereo_scene.sh").read_text(encoding="utf-8")

    assert ".qwen_env.local" in script
    assert "--inpaint-url" in script
    assert "http://localhost:5092/inpaint" in script
    assert "rsf_genesis_settle.sh" in script


def test_genesis_settle_script_uses_knowin_world_python():
    script = (PROJECT_ROOT / "scripts" / "rsf_genesis_settle.sh").read_text(encoding="utf-8")

    assert "/mnt/workspace/wenqian/knowin-world/.venv/bin/python" in script
    assert "run_interactive_scene.py" in script
    assert "real2sim_scene_foundry.cli" in script


def test_video_colmap_script_writes_missing_colmap_status(tmp_path):
    run = tmp_path / "run"
    frames = run / "video" / "frames"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "frame_000000.png"), np.zeros((4, 6, 3), dtype=np.uint8))

    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "rsf_video_colmap.sh"),
            "--run-dir",
            str(run),
            "--allow-missing",
        ],
        env={**os.environ, "COLMAP_BIN": str(tmp_path / "missing-colmap")},
        check=True,
        cwd=PROJECT_ROOT,
    )

    status = json.loads((run / "video" / "colmap_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "missing_colmap"
    assert status["frames_dir"] == str(frames)


def test_video_m7_script_runs_video_prep_and_records_missing_colmap(tmp_path):
    video = tmp_path / "phone.avi"
    out = tmp_path / "m7"
    _write_test_video(video)

    env = os.environ.copy()
    env["RSF_PYTHON"] = sys.executable
    env["COLMAP_BIN"] = str(tmp_path / "missing-colmap")
    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "rsf_video_m7.sh"),
            "--video",
            str(video),
            "--out",
            str(out),
            "--frame-stride",
            "1",
            "--reference-frame-index",
            "0",
        ],
        env=env,
        check=True,
        cwd=PROJECT_ROOT,
    )

    assert (out / "video" / "video_manifest.json").is_file()
    status = json.loads((out / "video" / "colmap_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "missing_colmap"
