#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: run_rife.sh <input_video> <temporal_scale_factor> <output_video>"
  echo "Example: run_rife.sh /work/input.mp4 2 /work/output.mp4"
  exit 1
fi

INPUT_VIDEO="$1"
SCALE_FACTOR="$2"
OUTPUT_VIDEO="$3"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO"
  exit 1
fi

if [[ "$SCALE_FACTOR" =~ ^[0-9]+$ ]] && [[ "$SCALE_FACTOR" -ge 2 ]]; then
  factor="$SCALE_FACTOR"
  exp=0
  while [[ "$factor" -gt 1 ]]; do
    if (( factor % 2 != 0 )); then
      echo "SCALE must be a power of two (2, 4, 8, ...)."
      exit 1
    fi
    factor=$((factor / 2))
    exp=$((exp + 1))
  done
else
  echo "SCALE must be an integer >= 2 and a power of two (2, 4, 8, ...)."
  exit 1
fi

FPS_RAW="$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "$INPUT_VIDEO")"
if [[ -z "$FPS_RAW" ]]; then
  echo "Could not read input FPS"
  exit 1
fi

TARGET_FPS="$(python3 - <<PY
fps_raw = "${FPS_RAW}"
exp = int("${exp}")
if "/" in fps_raw:
    num, den = fps_raw.split("/", 1)
    base_fps = float(num) / float(den)
else:
    base_fps = float(fps_raw)

if base_fps <= 0:
    raise SystemExit(1)

target_fps = int(round(base_fps * (2 ** exp)))
if target_fps <= 0:
    raise SystemExit(1)

print(target_fps)
PY
)"

if [[ -z "$TARGET_FPS" ]] || [[ ! "$TARGET_FPS" =~ ^[0-9]+$ ]] || [[ "$TARGET_FPS" -le 0 ]]; then
  echo "Failed to compute target fps from input fps '$FPS_RAW' and scale '$SCALE_FACTOR'"
  exit 1
fi

mkdir -p /opt/rife/train_log
if [[ ! -f /opt/rife/train_log/flownet.pkl ]]; then
  echo "Downloading RIFE HD model..."
  python3 -m gdown "1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_" -O /tmp/rife_hd.zip
  unzip -o /tmp/rife_hd.zip -d /opt/rife/train_log

  if [[ -f /opt/rife/train_log/train_log/flownet.pkl ]]; then
    mv /opt/rife/train_log/train_log/* /opt/rife/train_log/
    rm -rf /opt/rife/train_log/train_log
  fi

  rm -rf /opt/rife/train_log/__MACOSX /opt/rife/train_log/.DS_Store
fi

if [[ ! -f /opt/rife/train_log/flownet.pkl ]]; then
  echo "Model setup failed: /opt/rife/train_log/flownet.pkl not found"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_VIDEO")"
TMP_OUT="$(dirname "$OUTPUT_VIDEO")/.rife_tmp_$(basename "$OUTPUT_VIDEO")"
TMP_LOG="/tmp/rife_run.log"

cd /opt/rife
python3 inference_video.py --video="$INPUT_VIDEO" --exp="$exp" --fps="$TARGET_FPS" --output="$TMP_OUT" 2>&1 | tee "$TMP_LOG"

mv -f "$TMP_OUT" "$OUTPUT_VIDEO"

echo "Done: $OUTPUT_VIDEO"
