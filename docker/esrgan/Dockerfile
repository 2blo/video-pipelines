FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    wget \
    ca-certificates \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

RUN python3 -m pip install numpy opencv-python tqdm

WORKDIR /opt
RUN git clone --depth 1 https://github.com/xinntao/ESRGAN.git ESRGAN

RUN mkdir -p /opt/esrgan/models
COPY docker/esrgan/upscale_frames.py /opt/esrgan/upscale_frames.py
COPY docker/esrgan/run_esrgan.sh /usr/local/bin/run_esrgan.sh
RUN chmod +x /usr/local/bin/run_esrgan.sh

ENTRYPOINT ["/usr/local/bin/run_esrgan.sh"]
