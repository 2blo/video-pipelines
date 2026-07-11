FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    ca-certificates \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

WORKDIR /opt
RUN git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git depth-anything-v2

WORKDIR /opt/depth-anything-v2
RUN python3 -m pip install -r requirements.txt huggingface_hub

COPY docker/depth_anything_v2/infer_depth_video.py /opt/depth-anything-v2/infer_depth_video.py
COPY docker/depth_anything_v2/run_depth_anything_v2.sh /usr/local/bin/run_depth_anything_v2.sh
RUN chmod +x /usr/local/bin/run_depth_anything_v2.sh

ENTRYPOINT ["/usr/local/bin/run_depth_anything_v2.sh"]
