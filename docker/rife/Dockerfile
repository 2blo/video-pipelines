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
RUN python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

WORKDIR /opt
RUN git clone --depth 1 https://github.com/hzwer/ECCV2022-RIFE.git rife

WORKDIR /opt/rife
RUN python3 -m pip install -r requirements.txt gdown

COPY docker/rife/run_rife.sh /usr/local/bin/run_rife.sh
RUN chmod +x /usr/local/bin/run_rife.sh

ENTRYPOINT ["/usr/local/bin/run_rife.sh"]
