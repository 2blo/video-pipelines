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
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu130 torch torchvision torchaudio \
    || python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

WORKDIR /opt
RUN git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git seedvr2

WORKDIR /opt/seedvr2
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install "transformers<5"

RUN mkdir -p /opt/seedvr2/models/SEEDVR2

ENTRYPOINT ["python3", "/opt/seedvr2/inference_cli.py", "--help"]
