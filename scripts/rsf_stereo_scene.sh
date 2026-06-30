#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RSF_PYTHON="${RSF_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DEFAULT_INPAINT_URL="http://localhost:5092/inpaint"

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_stereo_scene.sh --left LEFT.png --right RIGHT.png --calib CALIB.json
                              --out RUN_DIR [--label TEXT]
                              [--inpaint-url URL|none]
                              [--settle-steps N] [--skip-genesis]

Runs the current stereo real-to-sim path:
  Qwen proposals -> SAM3 masks -> S2M2 XYZ -> SAM3D -> alignment
  -> foreground removal -> manifest/USD -> Genesis no-viewer settle.

If .qwen_env.local exists in the project root it is sourced automatically.
EOF
}

LEFT=""
RIGHT=""
CALIB=""
OUT_DIR=""
LABEL="object"
INPAINT_URL="$DEFAULT_INPAINT_URL"
SETTLE_STEPS="5"
RUN_GENESIS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --left)
      LEFT="$2"
      shift 2
      ;;
    --right)
      RIGHT="$2"
      shift 2
      ;;
    --calib)
      CALIB="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --inpaint-url)
      INPAINT_URL="$2"
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

if [[ -z "$LEFT" || -z "$RIGHT" || -z "$CALIB" || -z "$OUT_DIR" ]]; then
  echo "--left, --right, --calib, and --out are required" >&2
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
  "$RSF_PYTHON" -m real2sim_scene_foundry.cli run
  --left "$LEFT"
  --right "$RIGHT"
  --calib "$CALIB"
  --label "$LABEL"
  --out "$OUT_DIR"
)
if [[ "$INPAINT_URL" != "none" ]]; then
  cmd+=(--inpaint-url "$INPAINT_URL")
fi

"${cmd[@]}"

if [[ "$RUN_GENESIS" == "1" ]]; then
  "${ROOT_DIR}/scripts/rsf_genesis_settle.sh" \
    --run-dir "$OUT_DIR" \
    --settle-steps "$SETTLE_STEPS"
fi

cat <<EOF

Stereo scene run complete.
Run directory: ${OUT_DIR}
Scene manifest: ${OUT_DIR}/scene_manifest.json
QA report: ${OUT_DIR}/qa/qa_report.json
EOF
