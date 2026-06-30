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


def test_video_colmap_script_defaults_to_headless_qt():
    script = (PROJECT_ROOT / "scripts" / "rsf_video_colmap.sh").read_text(encoding="utf-8")

    assert 'export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"' in script


def test_video_colmap_script_defaults_to_cpu_sift():
    script = (PROJECT_ROOT / "scripts" / "rsf_video_colmap.sh").read_text(encoding="utf-8")

    assert 'COLMAP_USE_GPU="${COLMAP_USE_GPU:-0}"' in script
    assert '--SiftExtraction.use_gpu "$COLMAP_USE_GPU"' in script
    assert '--SiftMatching.use_gpu "$COLMAP_USE_GPU"' in script


def test_video_moge_background_script_writes_missing_runtime_status(tmp_path):
    run = tmp_path / "run"
    video_dir = run / "video"
    video_dir.mkdir(parents=True)
    cv2.imwrite(str(video_dir / "reference.png"), np.zeros((4, 6, 3), dtype=np.uint8))

    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "rsf_video_moge_background.sh"),
            "--run-dir",
            str(run),
            "--allow-missing",
        ],
        env={**os.environ, "MOGE_PYTHON": str(tmp_path / "missing-moge-python")},
        check=True,
        cwd=PROJECT_ROOT,
    )

    status = json.loads((run / "video" / "moge_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "missing_moge_runtime"
    assert status["input_image"] == str(video_dir / "reference.png")
    assert status["outputs"]["pointcloud"].endswith("video/moge_reference/reference/pointcloud.ply")


def test_video_moge_background_script_uses_existing_moge_runtime():
    script = (PROJECT_ROOT / "scripts" / "rsf_video_moge_background.sh").read_text(encoding="utf-8")

    assert "/mnt/workspace/wenqian/hawor_runtime/moge_venv/bin/python" in script
    assert "/mnt/workspace/wenqian/hawor_runtime/MoGe" in script
    assert "PYTHONPATH=\"$MOGE_ROOT\"" in script
    assert "--maps" in script
    assert "--ply" in script


def test_video_bg_only_script_writes_missing_python_status(tmp_path):
    run = tmp_path / "run"
    frames = run / "video" / "frames"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "frame_000000.png"), np.zeros((4, 6, 3), dtype=np.uint8))

    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "rsf_video_bg_only.sh"),
            "--run-dir",
            str(run),
            "--allow-missing",
        ],
        env={**os.environ, "RSF_PYTHON": str(tmp_path / "missing-python")},
        check=True,
        cwd=PROJECT_ROOT,
    )

    status = json.loads((run / "video" / "bg_only_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "missing_python_runtime"
    assert status["outputs"]["frames_dir"] == "video/bg_only/frames"


def test_video_3dgs_script_writes_missing_runtime_status(tmp_path):
    run = tmp_path / "run"
    bg_frames = run / "video" / "bg_only" / "frames"
    sparse = run / "video" / "colmap" / "sparse" / "0"
    bg_frames.mkdir(parents=True)
    sparse.mkdir(parents=True)
    cv2.imwrite(str(bg_frames / "frame_000000.png"), np.zeros((4, 6, 3), dtype=np.uint8))
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(b"x")
    (run / "video" / "video_manifest.json").write_text(
        json.dumps({"background_3dgs": {"status": "ready_for_training"}}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "rsf_video_3dgs.sh"),
            "--run-dir",
            str(run),
            "--allow-missing",
        ],
        env={**os.environ, "NS_TRAIN_BIN": str(tmp_path / "missing-ns-train")},
        check=True,
        cwd=PROJECT_ROOT,
    )

    status = json.loads((run / "video" / "3dgs_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "missing_3dgs_runtime"
    assert status["outputs"]["dataset_dir"] == "video/nerfstudio_bg"
    assert status["outputs"]["latest_config"] is None


def test_video_3dgs_script_records_latest_training_outputs(tmp_path):
    run = tmp_path / "run"
    bg_frames = run / "video" / "bg_only" / "frames"
    sparse = run / "video" / "colmap" / "sparse" / "0"
    bg_frames.mkdir(parents=True)
    sparse.mkdir(parents=True)
    cv2.imwrite(str(bg_frames / "frame_000000.png"), np.zeros((4, 6, 3), dtype=np.uint8))
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(b"x")
    (run / "video" / "video_manifest.json").write_text(
        json.dumps({"background_3dgs": {"status": "ready_for_training"}}),
        encoding="utf-8",
    )

    fake_ns_train = tmp_path / "ns-train"
    fake_ns_train.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$output_dir/unnamed/splatfacto/test-run/nerfstudio_models"
printf "config" > "$output_dir/unnamed/splatfacto/test-run/config.yml"
printf "ckpt" > "$output_dir/unnamed/splatfacto/test-run/nerfstudio_models/step-000000007.ckpt"
""",
        encoding="utf-8",
    )
    fake_ns_train.chmod(0o755)

    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "rsf_video_3dgs.sh"),
            "--run-dir",
            str(run),
            "--max-steps",
            "7",
        ],
        env={**os.environ, "NS_TRAIN_BIN": str(fake_ns_train)},
        check=True,
        cwd=PROJECT_ROOT,
    )

    status = json.loads((run / "video" / "3dgs_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["runtime"]["max_steps"] == 7
    assert status["outputs"]["latest_run_dir"] == "video/3dgs/unnamed/splatfacto/test-run"
    assert status["outputs"]["latest_config"] == "video/3dgs/unnamed/splatfacto/test-run/config.yml"
    assert (
        status["outputs"]["latest_checkpoint"]
        == "video/3dgs/unnamed/splatfacto/test-run/nerfstudio_models/step-000000007.ckpt"
    )
    manifest = json.loads((run / "video" / "video_manifest.json").read_text(encoding="utf-8"))
    assert manifest["background_3dgs"]["status"] == "completed"
    assert manifest["background_3dgs"]["latest_config"] == status["outputs"]["latest_config"]


def test_video_3dgs_script_runs_headless_tensorboard_visualizer():
    script = (PROJECT_ROOT / "scripts" / "rsf_video_3dgs.sh").read_text(encoding="utf-8")

    assert "--vis tensorboard" in script
    assert "colmap \\" in script
    assert "--images-path images" in script
    assert "--colmap-path sparse/0" in script


def test_video_3dgs_viewer_script_loads_latest_config():
    script = (PROJECT_ROOT / "scripts" / "rsf_video_view_3dgs.sh").read_text(encoding="utf-8")

    assert "ns-viewer" in script
    assert "3dgs_status.json" in script
    assert "latest_config" in script
    assert "--load-config" in script
    assert "--viewer.websocket-port" in script


def test_video_m7_script_runs_video_prep_and_records_missing_colmap(tmp_path):
    video = tmp_path / "phone.avi"
    out = tmp_path / "m7"
    _write_test_video(video)

    env = os.environ.copy()
    env["RSF_PYTHON"] = sys.executable
    env["COLMAP_BIN"] = str(tmp_path / "missing-colmap")
    env["MOGE_PYTHON"] = str(tmp_path / "missing-moge-python")
    env["RSF_VIDEO_BG_ONLY_SKIP"] = "1"
    env["RSF_VIDEO_3DGS_SKIP"] = "1"
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
    moge_status = json.loads((out / "video" / "moge_status.json").read_text(encoding="utf-8"))
    assert moge_status["status"] == "missing_moge_runtime"


def test_video_m7_script_runs_moge_background_step():
    script = (PROJECT_ROOT / "scripts" / "rsf_video_m7.sh").read_text(encoding="utf-8")

    assert "rsf_video_moge_background.sh" in script
    assert "moge_status.json" in script
    assert "rsf_video_bg_only.sh" in script
    assert "bg_only_status.json" in script
    assert "rsf_video_3dgs.sh" in script
    assert "3dgs_status.json" in script
