FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

WORKDIR /opt
RUN git clone --depth 1 https://github.com/apple/ml-depth-pro.git ml-depth-pro

WORKDIR /opt/ml-depth-pro
RUN python3 -m pip install -e . opencv-python-headless

RUN mkdir -p /opt/ml-depth-pro/checkpoints && \
    wget -O /opt/ml-depth-pro/checkpoints/depth_pro.pt \
      https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt

COPY docker/depth_pro/infer_depth_video.py /opt/ml-depth-pro/infer_depth_video.py
COPY docker/depth_pro/run_depth_pro.sh /usr/local/bin/run_depth_pro.sh
RUN chmod +x /usr/local/bin/run_depth_pro.sh

ENTRYPOINT ["/usr/local/bin/run_depth_pro.sh"]
