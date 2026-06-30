#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_video_view_3dgs.sh --run-dir RUN_DIR [--config CONFIG_YML] [--websocket-port PORT] [--host HOST]

Starts Nerfstudio's interactive viewer for the latest trained background 3DGS.
When --config is omitted, the script reads video/3dgs_status.json and falls
back to the newest config.yml under video/3dgs.
EOF
}

RUN_DIR=""
CONFIG_PATH=""
WEBSOCKET_PORT="${WEBSOCKET_PORT:-7007}"
WEBSOCKET_HOST="${WEBSOCKET_HOST:-0.0.0.0}"
if [[ -n "${NS_VIEWER_BIN+x}" ]]; then
  NS_VIEWER="${NS_VIEWER_BIN}"
elif [[ -x "${ROOT_DIR}/.venv_3dgs/bin/ns-viewer" ]]; then
  NS_VIEWER="${ROOT_DIR}/.venv_3dgs/bin/ns-viewer"
else
  NS_VIEWER="$(command -v ns-viewer || true)"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --websocket-port)
      WEBSOCKET_PORT="$2"
      shift 2
      ;;
    --host)
      WEBSOCKET_HOST="$2"
      shift 2
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
if [[ -z "$NS_VIEWER" || ! -x "$NS_VIEWER" ]]; then
  echo "Nerfstudio ns-viewer was not found. Create .venv_3dgs or set NS_VIEWER_BIN." >&2
  exit 3
fi

if [[ -z "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$(python3 - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
status_path = run_dir / "video" / "3dgs_status.json"
if status_path.is_file():
    status = json.loads(status_path.read_text(encoding="utf-8"))
    latest = status.get("outputs", {}).get("latest_config")
    if latest:
        candidate = run_dir / latest
        if candidate.is_file():
            print(candidate)
            raise SystemExit(0)

configs = sorted(
    (run_dir / "video" / "3dgs").glob("**/config.yml"),
    key=lambda item: item.stat().st_mtime,
)
if configs:
    print(configs[-1])
PY
)"
fi

if [[ -z "$CONFIG_PATH" || ! -f "$CONFIG_PATH" ]]; then
  echo "No Nerfstudio config.yml found for run: $RUN_DIR" >&2
  exit 4
fi

echo "Starting Nerfstudio viewer from: $CONFIG_PATH"
echo "Websocket host: $WEBSOCKET_HOST"
echo "Websocket port: $WEBSOCKET_PORT"
"$NS_VIEWER" \
  --load-config "$CONFIG_PATH" \
  --viewer.websocket-host "$WEBSOCKET_HOST" \
  --viewer.websocket-port "$WEBSOCKET_PORT"
