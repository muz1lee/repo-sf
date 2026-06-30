#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_video_3dgs.sh --run-dir RUN_DIR [--max-steps N] [--allow-missing]

Prepares a Nerfstudio/Splatfacto dataset from BG-only frames and COLMAP poses,
then starts 3DGS training when ns-train is available.
EOF
}

RUN_DIR=""
ALLOW_MISSING="0"
MAX_STEPS="${MAX_STEPS:-3000}"
if [[ -n "${NS_TRAIN_BIN+x}" ]]; then
  NS_TRAIN="${NS_TRAIN_BIN}"
elif [[ -x "${ROOT_DIR}/.venv_3dgs/bin/ns-train" ]]; then
  NS_TRAIN="${ROOT_DIR}/.venv_3dgs/bin/ns-train"
else
  NS_TRAIN="$(command -v ns-train || true)"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
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

BG_FRAMES="${RUN_DIR}/video/bg_only/frames"
SPARSE_SRC="${RUN_DIR}/video/colmap/sparse/0"
DATASET_DIR="${RUN_DIR}/video/nerfstudio_bg"
IMAGES_DIR="${DATASET_DIR}/images"
SPARSE_DST="${DATASET_DIR}/sparse/0"
OUTPUT_DIR="${RUN_DIR}/video/3dgs"
STATUS_PATH="${RUN_DIR}/video/3dgs_status.json"

write_status() {
  local status="$1"
  local message="$2"
  python3 - "$STATUS_PATH" "$status" "$message" "$RUN_DIR" "$NS_TRAIN" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

path, status, message, run_dir, ns_train, max_steps = sys.argv[1:]
run_path = Path(run_dir)
latest_config = None
latest_checkpoint = None
latest_run_dir = None
if status == "completed":
    configs = sorted(
        (run_path / "video" / "3dgs").glob("**/config.yml"),
        key=lambda item: item.stat().st_mtime,
    )
    if configs:
        latest_config = configs[-1]
        latest_run_dir = latest_config.parent
        checkpoints = sorted(
            (latest_run_dir / "nerfstudio_models").glob("*.ckpt"),
            key=lambda item: item.stat().st_mtime,
        )
        if checkpoints:
            latest_checkpoint = checkpoints[-1]

def rel(path_obj):
    if path_obj is None:
        return None
    try:
        return str(path_obj.relative_to(run_path))
    except ValueError:
        return str(path_obj)

Path(path).parent.mkdir(parents=True, exist_ok=True)
Path(path).write_text(
    json.dumps(
        {
            "version": 1,
            "stage": "video_3dgs",
            "status": status,
            "message": message,
            "run_dir": run_dir,
            "runtime": {
                "ns_train": ns_train,
                "method": "splatfacto",
                "max_steps": int(max_steps),
            },
            "outputs": {
                "dataset_dir": "video/nerfstudio_bg",
                "images_dir": "video/nerfstudio_bg/images",
                "sparse_dir": "video/nerfstudio_bg/sparse/0",
                "output_dir": "video/3dgs",
                "latest_run_dir": rel(latest_run_dir),
                "latest_config": rel(latest_config),
                "latest_checkpoint": rel(latest_checkpoint),
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

if [[ ! -d "$BG_FRAMES" ]]; then
  echo "BG-only frame directory not found: $BG_FRAMES" >&2
  exit 2
fi
if [[ ! -d "$SPARSE_SRC" ]]; then
  echo "COLMAP sparse model not found: $SPARSE_SRC" >&2
  exit 2
fi

rm -rf "$DATASET_DIR"
mkdir -p "$IMAGES_DIR" "$SPARSE_DST" "$OUTPUT_DIR"
cp "$BG_FRAMES"/*.png "$IMAGES_DIR"/
cp "$SPARSE_SRC"/* "$SPARSE_DST"/

if [[ -z "$NS_TRAIN" || ! -x "$NS_TRAIN" ]]; then
  write_status "missing_3dgs_runtime" "Nerfstudio ns-train was not found. Create .venv_3dgs or set NS_TRAIN_BIN."
  exit_for_missing
fi

set +e
"$NS_TRAIN" splatfacto \
  --output-dir "$OUTPUT_DIR" \
  --max-num-iterations "$MAX_STEPS" \
  --vis tensorboard \
  colmap \
  --data "$DATASET_DIR" \
  --images-path images \
  --colmap-path sparse/0
code=$?
set -e
if [[ "$code" -ne 0 ]]; then
  write_status "failed" "3DGS training failed with exit code ${code}."
  cat "$STATUS_PATH" >&2
  exit "$code"
fi

write_status "completed" "3DGS training completed."
cat "$STATUS_PATH"
