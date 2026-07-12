#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: run_depth_anything_v2.sh <input_video> <output_dir> <encoder> <input_size> [precision] [fast_resize_height] [temporal_smoothing_alpha]"
  echo "Example: run_depth_anything_v2.sh /io/in/input.mp4 /io/out/depth_v2 vitl 3840 fp16 518 0.25"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
ENCODER="$3"
INPUT_SIZE="$4"
PRECISION="${5:-fp32}"
FAST_RESIZE_HEIGHT="${6:-0}"
TEMPORAL_SMOOTHING_ALPHA="${7:-0.0}"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

if [[ ! "$INPUT_SIZE" =~ ^[0-9]+$ ]] || [[ "$INPUT_SIZE" -le 0 ]]; then
  echo "input_size must be a positive integer."
  exit 1
fi

if [[ "$PRECISION" != "fp32" && "$PRECISION" != "fp16" ]]; then
  echo "precision must be one of: fp32, fp16"
  exit 1
fi

if [[ ! "$FAST_RESIZE_HEIGHT" =~ ^[0-9]+$ ]] || [[ "$FAST_RESIZE_HEIGHT" -lt 0 ]]; then
  echo "fast_resize_height must be a non-negative integer (0 disables fast resize)."
  exit 1
fi

python3 - << 'PY' "$TEMPORAL_SMOOTHING_ALPHA"
import sys

try:
  alpha = float(sys.argv[1])
except ValueError:
  raise SystemExit("temporal_smoothing_alpha must be a float in [0.0, 1.0].")

if alpha < 0.0 or alpha > 1.0:
  raise SystemExit("temporal_smoothing_alpha must be in [0.0, 1.0].")
PY

mkdir -p "$OUTPUT_DIR"

python3 /opt/depth-anything-v2/infer_depth_video.py \
  --input-video "$INPUT_VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --encoder "$ENCODER" \
  --input-size "$INPUT_SIZE" \
  --precision "$PRECISION" \
  --fast-resize-height "$FAST_RESIZE_HEIGHT" \
  --temporal-smoothing-alpha "$TEMPORAL_SMOOTHING_ALPHA"
