FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    unzip \
    ca-certificates \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
ARG TORCH_CHANNELS="cu128 cu129 cu124 cu121 nightly/cu128 nightly/cu129"
RUN bash -eu -o pipefail <<'SH'
channels="${TORCH_CHANNELS}"
found=0
selected_channel=""

for channel in $channels; do
    echo "Trying torch wheel channel: $channel"
    if python3 -m pip install --upgrade --force-reinstall --pre --index-url "https://download.pytorch.org/whl/$channel" torch torchvision; then
        if python3 - <<'PY'
import torch

arch_list = torch.cuda.get_arch_list()
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_arch_list={arch_list}")

if "sm_120" in arch_list:
    raise SystemExit(0)

raise SystemExit(1)
PY
        then
            found=1
            selected_channel="$channel"
            echo "Found sm_120 support using channel: $channel"
            break
        fi
    fi
done

if [ "$found" -ne 1 ]; then
    echo "Installed torch wheels do not include sm_120 support for RTX 5090 across channels: $channels"
    echo "Refusing to build an incompatible RIFE image."
    exit 1
fi

echo "Verifying torchvision import from selected channel: $selected_channel"
python3 - <<'PY'
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
PY
SH

WORKDIR /opt
RUN git clone --depth 1 https://github.com/hzwer/ECCV2022-RIFE.git rife

WORKDIR /opt/rife
RUN grep -Ev '^(torch([<>=!].*)?|torchvision([<>=!].*)?)$' requirements.txt > /tmp/rife_requirements_notorch.txt \
    && python3 -m pip install -r /tmp/rife_requirements_notorch.txt gdown

COPY docker/rife/run_rife.sh /usr/local/bin/run_rife.sh
RUN chmod +x /usr/local/bin/run_rife.sh

ENTRYPOINT ["/usr/local/bin/run_rife.sh"]
