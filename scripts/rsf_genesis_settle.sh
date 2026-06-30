#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RSF_PYTHON="${RSF_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
KNOWIN_WORLD_PYTHON="${KNOWIN_WORLD_PYTHON:-/mnt/workspace/wenqian/knowin-world/.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_genesis_settle.sh --run-dir RUN_DIR [--settle-steps N] [--viewer]

Generates the Genesis launcher for a reconstructed scene and runs it with the
knowin-world virtual environment. Default mode is no-viewer for server smoke.
EOF
}

RUN_DIR=""
SETTLE_STEPS="5"
VIEWER="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --settle-steps)
      SETTLE_STEPS="$2"
      shift 2
      ;;
    --viewer)
      VIEWER="1"
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
if [[ ! -x "$KNOWIN_WORLD_PYTHON" ]]; then
  echo "knowin-world Python not found or not executable: $KNOWIN_WORLD_PYTHON" >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
"$RSF_PYTHON" -m real2sim_scene_foundry.cli interactive \
  --run-dir "$RUN_DIR" \
  --settle-steps "$SETTLE_STEPS"

cmd=(
  "$KNOWIN_WORLD_PYTHON"
  "${RUN_DIR}/exports/run_interactive_scene.py"
  --run-dir "$RUN_DIR"
  --settle-steps "$SETTLE_STEPS"
)
if [[ "$VIEWER" != "1" ]]; then
  cmd+=(--no-viewer)
fi

"${cmd[@]}"

cat <<EOF

Genesis settle complete.
Report: ${RUN_DIR}/qa/genesis_settle_report.json
QA: ${RUN_DIR}/qa/qa_report.json
EOF
