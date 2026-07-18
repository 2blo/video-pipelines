FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_IO_ENABLE_OPENEXR=1

ARG TORCH_CHANNELS="cu128 cu129 cu124 cu121 nightly/cu128 nightly/cu129"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-opencv \
    git \
    ffmpeg \
    wget \
    ca-certificates \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN set -eux; \
    chosen_channel=""; \
    for channel in ${TORCH_CHANNELS}; do \
        python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true; \
        if [ "${channel#nightly/}" != "$channel" ]; then \
            stable_channel="${channel#nightly/}"; \
            index_url="https://download.pytorch.org/whl/nightly/${stable_channel}"; \
            python3 -m pip install --pre --index-url "$index_url" torch || continue; \
        else \
            index_url="https://download.pytorch.org/whl/${channel}"; \
            python3 -m pip install --index-url "$index_url" torch || continue; \
        fi; \
        if python3 -c 'import torch, sys; arch = torch.cuda.get_arch_list(); print("torch_cuda_arch_list=", arch); sys.exit(0 if "sm_120" in arch else 1)'; then \
            chosen_channel="$channel"; \
            break; \
        fi; \
    done; \
    if [ -z "$chosen_channel" ]; then \
        echo "Failed to install a torch wheel with sm_120 support. Tried: ${TORCH_CHANNELS}"; \
        exit 1; \
    fi; \
    echo "Using torch channel: $chosen_channel"

RUN python3 -m pip install numpy tqdm

RUN python3 -c 'import cv2, os, sys; os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"; ok = cv2.haveImageWriter(".exr"); print("opencv", cv2.__version__, "have_exr", ok); sys.exit(0 if ok else 1)'

WORKDIR /opt
RUN git clone --depth 1 https://github.com/xinntao/ESRGAN.git ESRGAN

RUN mkdir -p /opt/esrgan/models
COPY docker/esrgan/upscale_frames.py /opt/esrgan/upscale_frames.py
COPY docker/esrgan/run_esrgan.sh /usr/local/bin/run_esrgan.sh
RUN chmod +x /usr/local/bin/run_esrgan.sh

ENTRYPOINT ["/usr/local/bin/run_esrgan.sh"]
