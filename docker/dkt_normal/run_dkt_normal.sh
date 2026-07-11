#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "Usage: run_dkt_normal.sh <input_video> <output_dir> <model_id> <height> <width> <num_inference_steps> <window_size> <overlap>"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
MODEL_ID="$3"
HEIGHT="$4"
WIDTH="$5"
NUM_INFERENCE_STEPS="$6"
WINDOW_SIZE="$7"
OVERLAP="$8"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

for V in "$HEIGHT" "$WIDTH" "$NUM_INFERENCE_STEPS" "$WINDOW_SIZE" "$OVERLAP"; do
  if [[ ! "$V" =~ ^[0-9]+$ ]]; then
    echo "Numeric parameters must be non-negative integers."
    exit 1
  fi
done

if [[ "$HEIGHT" -le 0 || "$WIDTH" -le 0 || "$NUM_INFERENCE_STEPS" -le 0 || "$WINDOW_SIZE" -le 0 ]]; then
  echo "height/width/num_inference_steps/window_size must be positive integers."
  exit 1
fi

if [[ "$OVERLAP" -ge "$WINDOW_SIZE" ]]; then
  echo "overlap must be less than window_size."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd /opt/DKT
python3 /opt/DKT/docker/infer_dkt_normal.py \
  --input-video "$INPUT_VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --model-id "$MODEL_ID" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --window-size "$WINDOW_SIZE" \
  --overlap "$OVERLAP"

INPUT_BASENAME="$(basename "$INPUT_VIDEO")"
INPUT_STEM="${INPUT_BASENAME%.*}"
EXPECTED_NPZ="$OUTPUT_DIR/$INPUT_STEM.npz"

if [[ ! -f "$EXPECTED_NPZ" ]]; then
  echo "DKT normal output not found: $EXPECTED_NPZ"
  echo "Available output files:"
  ls -la "$OUTPUT_DIR"
  exit 1
fi
