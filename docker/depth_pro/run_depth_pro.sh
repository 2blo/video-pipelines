#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: run_depth_pro.sh <input_video> <output_dir> <precision>"
  echo "Example: run_depth_pro.sh /io/in/input.mp4 /io/out/depth_pro fp16"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
PRECISION="$3"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

if [[ "$PRECISION" != "fp16" && "$PRECISION" != "fp32" ]]; then
  echo "precision must be one of: fp16, fp32"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 /opt/ml-depth-pro/infer_depth_video.py \
  --input-video "$INPUT_VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --precision "$PRECISION"
