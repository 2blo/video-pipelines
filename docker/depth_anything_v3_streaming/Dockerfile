FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV"

RUN python -m pip install --upgrade pip
ARG TORCH_CHANNELS="cu128 cu129 cu124 cu121 nightly/cu128 nightly/cu129"
RUN bash -eu -o pipefail <<'SH'
channels="${TORCH_CHANNELS}"
found=0

for channel in $channels; do
    echo "Trying torch wheel channel: $channel"
    if python -m pip install --upgrade --force-reinstall --pre --index-url "https://download.pytorch.org/whl/$channel" torch torchvision; then
        if python - <<'PY'
import torch
import torchvision
from torchvision.ops import nms

arch_list = torch.cuda.get_arch_list()
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_arch_list={arch_list}")
print(f"torchvision={torchvision.__version__}")

_ = nms(
    torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
    torch.tensor([0.5], dtype=torch.float32),
    0.5,
)

if "sm_120" in arch_list:
    raise SystemExit(0)

raise SystemExit(1)
PY
        then
            found=1
            echo "Found sm_120 support using channel: $channel"
            break
        fi
    fi
done

if [ "$found" -ne 1 ]; then
    echo "Installed torch wheels do not include sm_120 support for RTX 5090 across channels: $channels"
    echo "Refusing to build an incompatible DA3 streaming image."
    exit 1
fi
SH

WORKDIR /opt
RUN git clone --depth 1 --recursive https://github.com/ByteDance-Seed/depth-anything-3.git depth-anything-3

WORKDIR /opt/depth-anything-3
RUN grep -Ev '^(torch>=2|torchvision|xformers)$' requirements.txt > /tmp/requirements_notorch.txt \
    && python -m pip install -r /tmp/requirements_notorch.txt
RUN grep -Ev '^(faiss-gpu|torch([<>=!].*)?|torchvision([<>=!].*)?|xformers([<>=!].*)?)$' da3_streaming/requirements.txt > /tmp/da3_streaming_requirements_nogpu.txt \
    && python -m pip install -r /tmp/da3_streaming_requirements_nogpu.txt \
    && python -m pip install faiss-cpu
RUN python -m pip install -e . opencv-python-headless pyyaml

COPY docker/depth_anything_v3_streaming/infer_depth_video.py /opt/depth-anything-3/infer_depth_video.py
COPY docker/depth_anything_v3_streaming/run_depth_anything_v3_streaming.sh /usr/local/bin/run_depth_anything_v3_streaming.sh
RUN chmod +x /usr/local/bin/run_depth_anything_v3_streaming.sh

ENTRYPOINT ["/usr/local/bin/run_depth_anything_v3_streaming.sh"]
