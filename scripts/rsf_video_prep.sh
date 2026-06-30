#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RSF_PYTHON="${RSF_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  scripts/rsf_video_prep.sh --video PATH [--out RUN_DIR] [--run-id NAME]
                            [--frame-stride N] [--max-frames N]
                            [--reference-frame-index N]

Prepares a phone RGB video for the SimFoundry background path. This only
extracts sampled frames, a reference frame, and video_manifest.json; COLMAP/SLAM
camera poses and 3DGS fitting are later stages.
EOF
}

VIDEO=""
OUT_DIR=""
RUN_ID=""
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
    --run-id)
      RUN_ID="$2"
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

if [[ -z "$VIDEO" ]]; then
  echo "--video is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -x "$RSF_PYTHON" ]]; then
  echo "Python not found or not executable: $RSF_PYTHON" >&2
  echo "Set RSF_PYTHON or create the project .venv first." >&2
  exit 2
fi

if [[ -z "$OUT_DIR" ]]; then
  if [[ -z "$RUN_ID" ]]; then
    base="$(basename "$VIDEO")"
    base="${base%.*}"
    RUN_ID="video_${base}_$(date +%Y%m%d_%H%M%S)"
  fi
  OUT_DIR="${ROOT_DIR}/runs/${RUN_ID}"
fi

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

cmd=(
  "$RSF_PYTHON" -m real2sim_scene_foundry.cli video-prep
  --video "$VIDEO"
  --out "$OUT_DIR"
  --frame-stride "$FRAME_STRIDE"
  --reference-frame-index "$REFERENCE_FRAME_INDEX"
)
if [[ -n "$MAX_FRAMES" ]]; then
  cmd+=(--max-frames "$MAX_FRAMES")
fi

"${cmd[@]}"

cat <<EOF

Video preparation complete.
Run directory: ${OUT_DIR}
Manifest: ${OUT_DIR}/video/video_manifest.json
Reference frame: ${OUT_DIR}/video/reference.png

Next M7 steps:
  1. Estimate camera poses from ${OUT_DIR}/video/frames.
  2. Segment foreground objects per frame.
  3. Inpaint masks into BG-only frames.
  4. Fit per-scene 3DGS from BG-only frames + camera poses.
EOF
