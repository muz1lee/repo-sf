#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_video_bg_only.sh --run-dir RUN_DIR [--target-label LABEL ...]
                              [--max-frames N] [--allow-missing]

Generates BG-only sampled video frames:
  video/frames -> Qwen bbox proposals -> SAM3 masks -> inpainted video/bg_only/frames
EOF
}

RUN_DIR=""
ALLOW_MISSING="0"
MAX_FRAMES=""
DILATION_PX="${DILATION_PX:-10}"
SAM3_URL="${SAM3_URL:-http://101.132.143.105:5081/segment}"
INPAINT_URL="${INPAINT_URL:-http://localhost:5092/inpaint}"
RSF_PYTHON="${RSF_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
QWEN_ENV_FILE="${QWEN_ENV_FILE:-${ROOT_DIR}/.qwen_env.local}"
TARGET_LABELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --target-label)
      TARGET_LABELS+=("$2")
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --sam3-url)
      SAM3_URL="$2"
      shift 2
      ;;
    --inpaint-url)
      INPAINT_URL="$2"
      shift 2
      ;;
    --dilation-px)
      DILATION_PX="$2"
      shift 2
      ;;
    --qwen-env-file)
      QWEN_ENV_FILE="$2"
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

STATUS_PATH="${RUN_DIR}/video/bg_only_status.json"

write_status() {
  local status="$1"
  local message="$2"
  python3 - "$STATUS_PATH" "$status" "$message" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

path, status, message, run_dir = sys.argv[1:]
Path(path).parent.mkdir(parents=True, exist_ok=True)
Path(path).write_text(
    json.dumps(
        {
            "version": 1,
            "stage": "video_bg_only",
            "status": status,
            "message": message,
            "run_dir": run_dir,
            "outputs": {
                "frames_dir": "video/bg_only/frames",
                "masks_dir": "video/bg_only/masks",
                "proposals_dir": "video/bg_only/proposals",
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

if [[ ! -x "$RSF_PYTHON" ]]; then
  write_status "missing_python_runtime" "Project Python runtime was not found or is not executable."
  exit_for_missing
fi

cmd=(
  "$RSF_PYTHON"
  - "$RUN_DIR"
  "$SAM3_URL"
  "$INPAINT_URL"
  "$DILATION_PX"
  "${MAX_FRAMES:-}"
  "${TARGET_LABELS[@]}"
)

if ! REAL2SIM_QWEN_ENV_FILE="$QWEN_ENV_FILE" "${cmd[@]}" <<'PY'
import json
import sys
from pathlib import Path

from real2sim_scene_foundry.background import HTTPInpaintClient, OpenCVInpaintClient
from real2sim_scene_foundry.clients import SAM3Client
from real2sim_scene_foundry.proposals import QwenProposalClient
from real2sim_scene_foundry.video_bg_only import create_video_bg_only_artifacts

run_dir, sam3_url, inpaint_url, dilation_px, max_frames, *target_labels = sys.argv[1:]
status_path = Path(run_dir) / "video" / "bg_only_status.json"
try:
    inpaint_client = HTTPInpaintClient(inpaint_url) if inpaint_url else OpenCVInpaintClient()
    result = create_video_bg_only_artifacts(
        run_dir=run_dir,
        proposal_client=QwenProposalClient(),
        mask_client=SAM3Client(sam3_url),
        inpaint_client=inpaint_client,
        target_labels=target_labels,
        max_frames=int(max_frames) if max_frames else None,
        dilation_px=int(dilation_px),
    )
    print(result.status_path.read_text(encoding="utf-8"))
except Exception as exc:  # noqa: BLE001 - shell wrapper converts service/runtime failures to status.
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "version": 1,
                "stage": "video_bg_only",
                "status": "failed",
                "message": str(exc),
                "run_dir": run_dir,
                "outputs": {
                    "frames_dir": "video/bg_only/frames",
                    "masks_dir": "video/bg_only/masks",
                    "proposals_dir": "video/bg_only/proposals",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(status_path.read_text(encoding="utf-8"), file=sys.stderr)
    raise SystemExit(4)
PY
then
  if [[ "$ALLOW_MISSING" == "1" ]]; then
    cat "$STATUS_PATH"
    exit 0
  fi
  exit 4
fi
