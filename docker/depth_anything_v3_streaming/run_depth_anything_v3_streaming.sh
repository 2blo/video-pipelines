#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 11 ]]; then
  echo "Usage: run_depth_anything_v3_streaming.sh <input_video> <output_dir> <max_res> <fps> <device:auto|cuda> <chunk_size> <overlap> <loop_enable:0|1> <save_depth_conf_result:0|1> <delete_temp_files:0|1> <align_lib>"
  echo "Example: run_depth_anything_v3_streaming.sh /io/in/input.mp4 /io/out/depth_da3_stream 640 5 auto 120 60 1 1 1 triton"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
MAX_RES="$3"
FPS="$4"
DEVICE="$5"
CHUNK_SIZE="$6"
OVERLAP="$7"
LOOP_ENABLE="$8"
SAVE_DEPTH_CONF_RESULT="$9"
DELETE_TEMP_FILES="${10}"
ALIGN_LIB="${11}"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 /opt/depth-anything-3/infer_depth_video.py \
  --input-video "$INPUT_VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --model-dir "/models" \
  --max-res "$MAX_RES" \
  --fps "$FPS" \
  --device "$DEVICE" \
  --chunk-size "$CHUNK_SIZE" \
  --overlap "$OVERLAP" \
  --loop-enable "$LOOP_ENABLE" \
  --save-depth-conf-result "$SAVE_DEPTH_CONF_RESULT" \
  --delete-temp-files "$DELETE_TEMP_FILES" \
  --align-lib "$ALIGN_LIB"
