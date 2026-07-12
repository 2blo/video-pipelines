#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: run_depth_anything_v3.sh <input_video> <output_dir> <model> <max_res>"
  echo "Example: run_depth_anything_v3.sh /io/in/input.mp4 /io/out/depth_v3 small 1280"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
MODEL="$3"
MAX_RES="$4"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

if [[ "$MODEL" != "small" ]]; then
  echo "Unsupported model: $MODEL. Supported values: small"
  exit 1
fi

if [[ ! "$MAX_RES" =~ ^[0-9]+$ ]] || [[ "$MAX_RES" -le 0 ]]; then
  echo "max_res must be a positive integer."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 /opt/depth-anything-3/infer_depth_video.py \
  --input-video "$INPUT_VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --max-res "$MAX_RES"
