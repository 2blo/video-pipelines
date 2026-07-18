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
ESRGAN_OUTPUT_MODE="${ESRGAN_OUTPUT_MODE:-video}"
ESRGAN_VIDEO_CODEC="${ESRGAN_VIDEO_CODEC:-h264}"
ESRGAN_PRORES_PROFILE="${ESRGAN_PRORES_PROFILE:-hq}"
ESRGAN_X264_PRESET="${ESRGAN_X264_PRESET:-medium}"
ESRGAN_X264_CRF="${ESRGAN_X264_CRF:-17}"
ESRGAN_EXR_TYPE="${ESRGAN_EXR_TYPE:-float}"
ESRGAN_EXR_COMPRESSION="${ESRGAN_EXR_COMPRESSION:-zip}"

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

INPUT_DIMENSIONS="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$INPUT_VIDEO")"
if [[ -z "$INPUT_DIMENSIONS" ]] || [[ "$INPUT_DIMENSIONS" != *x* ]]; then
  echo "Could not read input dimensions"
  exit 1
fi
INPUT_WIDTH="${INPUT_DIMENSIONS%x*}"
INPUT_HEIGHT="${INPUT_DIMENSIONS#*x}"
if [[ ! "$INPUT_WIDTH" =~ ^[0-9]+$ ]] || [[ ! "$INPUT_HEIGHT" =~ ^[0-9]+$ ]] || [[ "$INPUT_WIDTH" -le 0 ]] || [[ "$INPUT_HEIGHT" -le 0 ]]; then
  echo "Invalid input dimensions from ffprobe: $INPUT_DIMENSIONS"
  exit 1
fi

if [[ -n "${ESRGAN_TARGET_HEIGHT:-}" ]]; then
  TARGET_HEIGHT="${ESRGAN_TARGET_HEIGHT}"
else
  TARGET_HEIGHT="$(( (INPUT_HEIGHT * TARGET_WIDTH + INPUT_WIDTH / 2) / INPUT_WIDTH ))"
fi

if [[ ! "$TARGET_HEIGHT" =~ ^[0-9]+$ ]]; then
  echo "TARGET_HEIGHT must be a positive integer when provided. Got: $TARGET_HEIGHT"
  exit 1
fi
if [[ "$TARGET_HEIGHT" -le 0 ]]; then
  echo "Computed invalid target height: $TARGET_HEIGHT"
  exit 1
fi

echo "Extracting frames..."
ffmpeg -hide_banner -loglevel warning -y -i "$INPUT_VIDEO" -pix_fmt rgb48le "$FRAMES_IN/%08d.png"

echo "Upscaling frames with ESRGAN..."
PYTHONPATH="/opt/ESRGAN" python3 /opt/esrgan/upscale_frames.py \
  --input-dir "$FRAMES_IN" \
  --output-dir "$FRAMES_UP" \
  --model-path "$MODEL_PATH" \
  --batch-size "$ESRGAN_BATCH_SIZE" \
  --target-width "$TARGET_WIDTH" \
  --target-height "$TARGET_HEIGHT" \
  --output-format "$([[ "$ESRGAN_OUTPUT_MODE" == "exr_sequence" ]] && echo exr || echo png)" \
  --exr-type "$ESRGAN_EXR_TYPE" \
  --exr-compression "$ESRGAN_EXR_COMPRESSION"

if [[ "$ESRGAN_OUTPUT_MODE" == "exr_sequence" ]]; then
  mkdir -p "$OUTPUT_VIDEO"
  shopt -s nullglob
  EXR_FRAMES=("$FRAMES_UP"/*.exr)
  if [[ ${#EXR_FRAMES[@]} -gt 0 ]]; then
    mv -f "$FRAMES_UP"/*.exr "$OUTPUT_VIDEO"/
  else
    PNG_FRAMES=("$FRAMES_UP"/*.png)
    if [[ ${#PNG_FRAMES[@]} -eq 0 ]]; then
      echo "No EXR or PNG frames were produced by ESRGAN."
      exit 1
    fi
    echo "OpenCV EXR writer unavailable; exporting PNG sequence instead."
    mv -f "$FRAMES_UP"/*.png "$OUTPUT_VIDEO"/
  fi
  shopt -u nullglob
  echo "Done: $OUTPUT_VIDEO"
  exit 0
fi

OUTPUT_EXT="${OUTPUT_VIDEO##*.}"
TMP_VIDEO="$WORK_DIR/video_tmp.${OUTPUT_EXT}"

echo "Assembling output video..."
if [[ "$ESRGAN_VIDEO_CODEC" == "h264" ]]; then
  ffmpeg -hide_banner -loglevel warning -y \
    -thread_queue_size "$FFMPEG_THREAD_QUEUE_SIZE" -framerate "$FPS_RAW" -i "$FRAMES_UP/%08d.png" \
    -thread_queue_size "$FFMPEG_THREAD_QUEUE_SIZE" -i "$INPUT_VIDEO" \
    -map 0:v:0 -map 1:a? -map 1:s? \
    -vf "scale=${TARGET_WIDTH}:-2:flags=lanczos" \
    -c:v libx264 -preset "$ESRGAN_X264_PRESET" -crf "$ESRGAN_X264_CRF" \
    -c:a copy -c:s copy \
    "$TMP_VIDEO"
elif [[ "$ESRGAN_VIDEO_CODEC" == "prores" ]]; then
  case "$ESRGAN_PRORES_PROFILE" in
    proxy) PROFILE_ID=0; PIX_FMT="yuv422p10le" ;;
    lt) PROFILE_ID=1; PIX_FMT="yuv422p10le" ;;
    422) PROFILE_ID=2; PIX_FMT="yuv422p10le" ;;
    hq) PROFILE_ID=3; PIX_FMT="yuv422p10le" ;;
    4444) PROFILE_ID=4; PIX_FMT="yuv444p10le" ;;
    4444xq) PROFILE_ID=5; PIX_FMT="yuv444p10le" ;;
    *)
      echo "Unsupported ESRGAN_PRORES_PROFILE: $ESRGAN_PRORES_PROFILE"
      exit 1
      ;;
  esac

  ffmpeg -hide_banner -loglevel warning -y \
    -thread_queue_size "$FFMPEG_THREAD_QUEUE_SIZE" -framerate "$FPS_RAW" -i "$FRAMES_UP/%08d.png" \
    -thread_queue_size "$FFMPEG_THREAD_QUEUE_SIZE" -i "$INPUT_VIDEO" \
    -map 0:v:0 -map 1:a? -map 1:s? \
    -vf "scale=${TARGET_WIDTH}:-2:flags=lanczos" \
    -c:v prores_ks -profile:v "$PROFILE_ID" -pix_fmt "$PIX_FMT" \
    -c:a copy -c:s copy \
    "$TMP_VIDEO"
else
  echo "Unsupported ESRGAN_VIDEO_CODEC: $ESRGAN_VIDEO_CODEC"
  exit 1
fi

mv -f "$TMP_VIDEO" "$OUTPUT_VIDEO"

echo "Done: $OUTPUT_VIDEO"
