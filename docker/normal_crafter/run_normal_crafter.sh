#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "Usage: run_normal_crafter.sh <input_video> <output_dir> <max_res> <process_length> <target_fps> <window_size> <time_step_size> <decode_chunk_size>"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
MAX_RES="$3"
PROCESS_LENGTH="$4"
TARGET_FPS="$5"
WINDOW_SIZE="$6"
TIME_STEP_SIZE="$7"
DECODE_CHUNK_SIZE="$8"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

if [[ ! "$MAX_RES" =~ ^[0-9]+$ ]] || [[ "$MAX_RES" -le 0 ]]; then
  echo "max_res must be a positive integer."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd /opt/NormalCrafter
ARGS=(
  --video-path "$INPUT_VIDEO"
  --save-folder "$OUTPUT_DIR"
  --max-res "$MAX_RES"
  --process-length "$PROCESS_LENGTH"
  --target-fps "$TARGET_FPS"
  --window-size "$WINDOW_SIZE"
  --time-step-size "$TIME_STEP_SIZE"
  --save-npz=True
)

if python3 run.py --help 2>&1 | grep -q -- "--decode-chunk-size"; then
  ARGS+=(--decode-chunk-size "$DECODE_CHUNK_SIZE")
fi

python3 run.py "${ARGS[@]}"

INPUT_BASENAME="$(basename "$INPUT_VIDEO")"
INPUT_STEM="${INPUT_BASENAME%.*}"
EXPECTED_NPZ="$OUTPUT_DIR/$INPUT_STEM.npz"

if [[ ! -f "$EXPECTED_NPZ" ]]; then
  echo "NormalCrafter output not found: $EXPECTED_NPZ"
  echo "Available output files:"
  ls -la "$OUTPUT_DIR"
  exit 1
fi
