#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_video_m7.sh --video PATH --out RUN_DIR [--frame-stride N]
                          [--max-frames N] [--reference-frame-index N]

Runs the current M7 video front end:
  phone RGB video -> sampled frames/reference -> COLMAP status.

If COLMAP is not installed, this still writes video/colmap_status.json with
status=missing_colmap so the missing dependency is explicit.
EOF
}

VIDEO=""
OUT_DIR=""
FRAME_STRIDE="10"
MAX_FRAMES=""
REFERENCE_FRAME_INDEX="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      VIDEO="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --frame-stride)
      FRAME_STRIDE="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --reference-frame-index)
      REFERENCE_FRAME_INDEX="$2"
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

if [[ -z "$VIDEO" || -z "$OUT_DIR" ]]; then
  echo "--video and --out are required" >&2
  usage >&2
  exit 2
fi

prep_cmd=(
  "${ROOT_DIR}/scripts/rsf_video_prep.sh"
  --video "$VIDEO"
  --out "$OUT_DIR"
  --frame-stride "$FRAME_STRIDE"
  --reference-frame-index "$REFERENCE_FRAME_INDEX"
)
if [[ -n "$MAX_FRAMES" ]]; then
  prep_cmd+=(--max-frames "$MAX_FRAMES")
fi

"${prep_cmd[@]}"
"${ROOT_DIR}/scripts/rsf_video_colmap.sh" --run-dir "$OUT_DIR" --allow-missing

cat <<EOF

M7 video front end complete.
Run directory: ${OUT_DIR}
Video manifest: ${OUT_DIR}/video/video_manifest.json
COLMAP status: ${OUT_DIR}/video/colmap_status.json

Remaining M7 steps after COLMAP is available:
  1. Use camera poses to build BG-only frame set.
  2. Fit per-scene 3DGS.
  3. Align 3DGS background with object digital twins.
EOF
