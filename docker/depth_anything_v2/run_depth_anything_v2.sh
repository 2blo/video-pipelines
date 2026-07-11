#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: run_depth_anything_v2.sh <input_video> <output_dir> <encoder> <input_size>"
  echo "Example: run_depth_anything_v2.sh /io/in/input.mp4 /io/out/depth_v2 vitg 518"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
ENCODER="$3"
INPUT_SIZE="$4"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

if [[ ! "$INPUT_SIZE" =~ ^[0-9]+$ ]] || [[ "$INPUT_SIZE" -le 0 ]]; then
  echo "input_size must be a positive integer."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 /opt/depth-anything-v2/infer_depth_video.py \
  --input-video "$INPUT_VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --encoder "$ENCODER" \
  --input-size "$INPUT_SIZE"
