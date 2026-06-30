#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_video_moge_background.sh --run-dir RUN_DIR [--input IMAGE]
                                      [--allow-missing]

Runs MoGe on the representative video frame to produce dense monocular geometry:
  video/reference.png -> video/moge_reference/reference/{pointcloud.ply,depth.exr,...}

The script uses the existing MoGe runtime instead of installing torch/model
dependencies into the real2sim_scene_foundry venv.
EOF
}

RUN_DIR=""
INPUT_IMAGE=""
ALLOW_MISSING="0"
MOGE_PYTHON="${MOGE_PYTHON:-/mnt/workspace/wenqian/hawor_runtime/moge_venv/bin/python}"
MOGE_ROOT="${MOGE_ROOT:-/mnt/workspace/wenqian/hawor_runtime/MoGe}"
MOGE_CHECKPOINT="${MOGE_CHECKPOINT:-/mnt/workspace/RexOmni/models--Ruicheng--moge-vitl/blobs/da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f}"
MOGE_VERSION="${MOGE_VERSION:-v1}"
MOGE_DEVICE="${MOGE_DEVICE:-cuda}"
MOGE_RESIZE="${MOGE_RESIZE:-720}"
MOGE_FP16="${MOGE_FP16:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --input)
      INPUT_IMAGE="$2"
      shift 2
      ;;
    --allow-missing)
      ALLOW_MISSING="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "--run-dir is required" >&2
  usage >&2
  exit 2
fi

VIDEO_DIR="${RUN_DIR}/video"
if [[ -z "$INPUT_IMAGE" ]]; then
  INPUT_IMAGE="${VIDEO_DIR}/reference.png"
fi
OUTPUT_DIR="${VIDEO_DIR}/moge_reference"
STATUS_PATH="${VIDEO_DIR}/moge_status.json"
POINTCLOUD_PATH="${OUTPUT_DIR}/reference/pointcloud.ply"
DEPTH_PATH="${OUTPUT_DIR}/reference/depth.exr"
DEPTH_VIS_PATH="${OUTPUT_DIR}/reference/depth_vis.png"
FOV_PATH="${OUTPUT_DIR}/reference/fov.json"

write_status() {
  local status="$1"
  local message="$2"
  python3 - "$STATUS_PATH" "$status" "$message" "$RUN_DIR" "$INPUT_IMAGE" "$MOGE_PYTHON" "$MOGE_ROOT" "$MOGE_CHECKPOINT" "$MOGE_VERSION" "$MOGE_DEVICE" "$MOGE_RESIZE" "$POINTCLOUD_PATH" "$DEPTH_PATH" "$DEPTH_VIS_PATH" "$FOV_PATH" <<'PY'
import json
import sys
from pathlib import Path

(
    path,
    status,
    message,
    run_dir,
    input_image,
    moge_python,
    moge_root,
    moge_checkpoint,
    moge_version,
    moge_device,
    moge_resize,
    pointcloud_path,
    depth_path,
    depth_vis_path,
    fov_path,
) = sys.argv[1:]
Path(path).parent.mkdir(parents=True, exist_ok=True)
Path(path).write_text(
    json.dumps(
        {
            "version": 1,
            "stage": "video_moge_background",
            "status": status,
            "message": message,
            "run_dir": run_dir,
            "input_image": input_image,
            "runtime": {
                "python": moge_python,
                "root": moge_root,
                "checkpoint": moge_checkpoint,
                "model_version": moge_version,
                "device": moge_device,
                "resize": int(moge_resize),
            },
            "outputs": {
                "output_dir": f"{run_dir}/video/moge_reference",
                "pointcloud": pointcloud_path,
                "depth": depth_path,
                "depth_vis": depth_vis_path,
                "fov": fov_path,
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
}

exit_for_missing() {
  if [[ "$ALLOW_MISSING" == "1" ]]; then
    cat "$STATUS_PATH"
    exit 0
  fi
  cat "$STATUS_PATH" >&2
  exit 3
}

if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "Input image not found: $INPUT_IMAGE" >&2
  exit 2
fi

if [[ ! -x "$MOGE_PYTHON" ]]; then
  write_status "missing_moge_runtime" "MoGe Python runtime was not found or is not executable."
  exit_for_missing
fi

if [[ ! -d "$MOGE_ROOT" ]]; then
  write_status "missing_moge_runtime" "MoGe source root was not found."
  exit_for_missing
fi

if [[ ! -f "$MOGE_CHECKPOINT" ]]; then
  write_status "missing_moge_checkpoint" "MoGe checkpoint was not found."
  exit_for_missing
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

moge_cmd=(
  "$MOGE_PYTHON"
  -m moge.scripts.infer
  -i "$INPUT_IMAGE"
  -o "$OUTPUT_DIR"
  --pretrained "$MOGE_CHECKPOINT"
  --version "$MOGE_VERSION"
  --device "$MOGE_DEVICE"
  --resize "$MOGE_RESIZE"
  --maps
  --ply
)
if [[ "$MOGE_FP16" == "1" ]]; then
  moge_cmd+=(--fp16)
fi

if ! PYTHONPATH="$MOGE_ROOT" "${moge_cmd[@]}"; then
  write_status "failed" "MoGe dense geometry inference failed."
  cat "$STATUS_PATH" >&2
  exit 4
fi

if [[ ! -f "$POINTCLOUD_PATH" ]]; then
  write_status "failed" "MoGe finished but did not produce pointcloud.ply."
  cat "$STATUS_PATH" >&2
  exit 4
fi

write_status "completed" "MoGe dense reference-frame geometry completed."
cat "$STATUS_PATH"
