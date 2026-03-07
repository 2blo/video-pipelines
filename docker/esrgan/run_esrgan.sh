#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: run_esrgan.sh <input_video> <target_width> <output_video>"
  echo "Example: run_esrgan.sh /work/input.mp4 2160 /work/output.mp4"
  exit 1
fi

INPUT_VIDEO="$1"
TARGET_WIDTH="$2"
OUTPUT_VIDEO="$3"
ESRGAN_BATCH_SIZE="${ESRGAN_BATCH_SIZE:-0}"
FFMPEG_THREAD_QUEUE_SIZE="${FFMPEG_THREAD_QUEUE_SIZE:-1024}"
FFMPEG_X264_PRESET="${FFMPEG_X264_PRESET:-medium}"
FFMPEG_X264_CRF="${FFMPEG_X264_CRF:-17}"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

if [[ ! "$TARGET_WIDTH" =~ ^[0-9]+$ ]] || [[ "$TARGET_WIDTH" -le 0 ]]; then
  echo "TARGET_WIDTH must be a positive integer."
  exit 1
fi

mkdir -p /opt/esrgan/models
MODEL_PATH="/opt/esrgan/models/RealESRGAN_x4plus.pth"
MODEL_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

download_model() {
  local tmp_model
  tmp_model="$(mktemp /tmp/esrgan_model_XXXXXX.pth)"
  echo "Downloading ESRGAN model..."
  wget -O "$tmp_model" "$MODEL_URL"
  mv -f "$tmp_model" "$MODEL_PATH"
}

validate_model() {
  python3 - "$MODEL_PATH" <<'PY'
import sys
import torch

model_path = sys.argv[1]
try:
    obj = torch.load(model_path, map_location="cpu", weights_only=False)
except TypeError:
    obj = torch.load(model_path, map_location="cpu")

if not isinstance(obj, dict):
    raise SystemExit(1)
PY
}

if [[ ! -f "$MODEL_PATH" ]]; then
  download_model
fi

if ! validate_model; then
  echo "Cached ESRGAN model appears corrupted. Re-downloading..."
  rm -f "$MODEL_PATH"
  download_model
  if ! validate_model; then
    echo "Model validation failed after re-download."
    exit 1
  fi
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model setup failed: $MODEL_PATH not found"
  exit 1
fi

WORK_DIR="$(mktemp -d /tmp/esrgan_video_XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

FRAMES_IN="$WORK_DIR/frames_in"
FRAMES_UP="$WORK_DIR/frames_up"
mkdir -p "$FRAMES_IN" "$FRAMES_UP" "$(dirname "$OUTPUT_VIDEO")"

FPS_RAW="$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "$INPUT_VIDEO")"
if [[ -z "$FPS_RAW" ]]; then
  echo "Could not read input FPS"
  exit 1
fi

echo "Extracting frames..."
ffmpeg -hide_banner -loglevel warning -y -i "$INPUT_VIDEO" "$FRAMES_IN/%08d.png"

echo "Upscaling frames with ESRGAN..."
PYTHONPATH="/opt/ESRGAN" python3 /opt/esrgan/upscale_frames.py \
  --input-dir "$FRAMES_IN" \
  --output-dir "$FRAMES_UP" \
  --model-path "$MODEL_PATH" \
  --batch-size "$ESRGAN_BATCH_SIZE"

OUTPUT_EXT="${OUTPUT_VIDEO##*.}"
TMP_VIDEO="$WORK_DIR/video_tmp.${OUTPUT_EXT}"

echo "Assembling output video..."

set +e
ffmpeg -hide_banner -loglevel warning -y \
  -thread_queue_size "$FFMPEG_THREAD_QUEUE_SIZE" -framerate "$FPS_RAW" -i "$FRAMES_UP/%08d.png" \
  -thread_queue_size "$FFMPEG_THREAD_QUEUE_SIZE" -i "$INPUT_VIDEO" \
  -map 0:v:0 -map 1:a? -map 1:s? \
  -vf "scale=${TARGET_WIDTH}:-2:flags=lanczos" \
  -c:v libx264 -preset "$FFMPEG_X264_PRESET" -crf "$FFMPEG_X264_CRF" \
  -c:a copy -c:s copy \
  "$TMP_VIDEO"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  echo "Subtitle copy/mux failed. Retrying without subtitles..."
  ffmpeg -hide_banner -loglevel warning -y \
    -thread_queue_size "$FFMPEG_THREAD_QUEUE_SIZE" -framerate "$FPS_RAW" -i "$FRAMES_UP/%08d.png" \
    -thread_queue_size "$FFMPEG_THREAD_QUEUE_SIZE" -i "$INPUT_VIDEO" \
    -map 0:v:0 -map 1:a? \
    -vf "scale=${TARGET_WIDTH}:-2:flags=lanczos" \
    -c:v libx264 -preset "$FFMPEG_X264_PRESET" -crf "$FFMPEG_X264_CRF" \
    -c:a copy \
    "$TMP_VIDEO"
fi

mv -f "$TMP_VIDEO" "$OUTPUT_VIDEO"

echo "Done: $OUTPUT_VIDEO"
