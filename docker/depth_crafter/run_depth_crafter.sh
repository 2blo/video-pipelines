#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: run_depth_crafter.sh <input_video> <output_dir> <max_res> <process_length> <target_fps>"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
MAX_RES="$3"
PROCESS_LENGTH="$4"
TARGET_FPS="$5"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

if [[ ! "$MAX_RES" =~ ^[0-9]+$ ]] || [[ "$MAX_RES" -le 0 ]]; then
  echo "max_res must be a positive integer."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd /opt/DepthCrafter
/opt/DepthCrafter/.venv/bin/python run.py \
  --video-path "$INPUT_VIDEO" \
  --save-folder "$OUTPUT_DIR" \
  --max-res "$MAX_RES" \
  --process-length "$PROCESS_LENGTH" \
  --target-fps "$TARGET_FPS" \
  --save-npz=True

INPUT_BASENAME="$(basename "$INPUT_VIDEO")"
INPUT_STEM="${INPUT_BASENAME%.*}"
EXPECTED_NPZ="$OUTPUT_DIR/$INPUT_STEM.npz"

if [[ ! -f "$EXPECTED_NPZ" ]]; then
  echo "DepthCrafter output not found: $EXPECTED_NPZ"
  echo "Available output files:"
  ls -la "$OUTPUT_DIR"
  exit 1
fi
