#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_video_colmap.sh --run-dir RUN_DIR [--allow-missing]

Runs COLMAP on video/frames to estimate camera poses. If COLMAP is not
available and --allow-missing is set, writes video/colmap_status.json with
status=missing_colmap and exits 0.
EOF
}

RUN_DIR=""
ALLOW_MISSING="0"
COLMAP_BIN="${COLMAP_BIN:-colmap}"
COLMAP_USE_GPU="${COLMAP_USE_GPU:-0}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
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

FRAMES_DIR="${RUN_DIR}/video/frames"
STATUS_PATH="${RUN_DIR}/video/colmap_status.json"
COLMAP_DIR="${RUN_DIR}/video/colmap"
DATABASE_PATH="${COLMAP_DIR}/database.db"
SPARSE_DIR="${COLMAP_DIR}/sparse"

if [[ ! -d "$FRAMES_DIR" ]]; then
  echo "Frames directory not found: $FRAMES_DIR" >&2
  exit 2
fi

write_status() {
  local status="$1"
  local message="$2"
  python3 - "$STATUS_PATH" "$status" "$message" "$RUN_DIR" "$FRAMES_DIR" "$COLMAP_BIN" <<'PY'
import json
import sys
from pathlib import Path

path, status, message, run_dir, frames_dir, colmap_bin = sys.argv[1:]
Path(path).parent.mkdir(parents=True, exist_ok=True)
Path(path).write_text(
    json.dumps(
        {
            "version": 1,
            "stage": "video_colmap",
            "status": status,
            "message": message,
            "run_dir": run_dir,
            "frames_dir": frames_dir,
            "colmap_bin": colmap_bin,
            "outputs": {
                "database": f"{run_dir}/video/colmap/database.db",
                "sparse_dir": f"{run_dir}/video/colmap/sparse",
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
}

if ! command -v "$COLMAP_BIN" >/dev/null 2>&1; then
  write_status "missing_colmap" "COLMAP executable was not found. Install COLMAP or set COLMAP_BIN."
  cat "$STATUS_PATH"
  if [[ "$ALLOW_MISSING" == "1" ]]; then
    exit 0
  fi
  exit 3
fi

mkdir -p "$SPARSE_DIR"
"$COLMAP_BIN" feature_extractor \
  --database_path "$DATABASE_PATH" \
  --image_path "$FRAMES_DIR" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu "$COLMAP_USE_GPU"
"$COLMAP_BIN" sequential_matcher \
  --database_path "$DATABASE_PATH" \
  --SiftMatching.use_gpu "$COLMAP_USE_GPU"
"$COLMAP_BIN" mapper \
  --database_path "$DATABASE_PATH" \
  --image_path "$FRAMES_DIR" \
  --output_path "$SPARSE_DIR"

write_status "completed" "COLMAP sparse reconstruction completed."
cat "$STATUS_PATH"
