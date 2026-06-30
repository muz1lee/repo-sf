#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RSF_PYTHON="${RSF_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_video_scene.sh --run-dir RUN_DIR [--target-label LABEL ...]
                             [--max-objects N] [--settle-steps N]
                             [--skip-genesis]

Builds the foreground object scene from the MoGe reference frame, attaches the
trained video 3DGS background to scene_manifest.json, exports the Genesis
launcher, and optionally runs no-viewer Genesis settle.
EOF
}

RUN_DIR=""
MAX_OBJECTS=""
SETTLE_STEPS="5"
RUN_GENESIS="1"
TARGET_LABEL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --target-label)
      TARGET_LABEL_ARGS+=(--target-label "$2")
      shift 2
      ;;
    --max-objects)
      MAX_OBJECTS="$2"
      shift 2
      ;;
    --settle-steps)
      SETTLE_STEPS="$2"
      shift 2
      ;;
    --skip-genesis)
      RUN_GENESIS="0"
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
if [[ ! -x "$RSF_PYTHON" ]]; then
  echo "Python not found or not executable: $RSF_PYTHON" >&2
  exit 2
fi

if [[ -f "${ROOT_DIR}/.qwen_env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.qwen_env.local"
  set +a
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

cmd=(
  "$RSF_PYTHON" -m real2sim_scene_foundry.cli video-scene
  --run-dir "$RUN_DIR"
  --settle-steps "$SETTLE_STEPS"
  "${TARGET_LABEL_ARGS[@]}"
)
if [[ -n "$MAX_OBJECTS" ]]; then
  cmd+=(--max-objects "$MAX_OBJECTS")
fi

"${cmd[@]}"

if [[ "$RUN_GENESIS" == "1" ]]; then
  "${ROOT_DIR}/scripts/rsf_genesis_settle.sh" \
    --run-dir "$RUN_DIR" \
    --settle-steps "$SETTLE_STEPS"
fi

cat <<EOF

Video scene run complete.
Run directory: ${RUN_DIR}
Scene manifest: ${RUN_DIR}/scene_manifest.json
QA report: ${RUN_DIR}/qa/qa_report.json
Interaction report: ${RUN_DIR}/qa/interaction_report.json
EOF
