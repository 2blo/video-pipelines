FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH}"

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
RUN python3 -m pip install uv
RUN python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision

WORKDIR /opt
RUN git clone --depth 1 https://github.com/Binyr/NormalCrafter.git NormalCrafter

WORKDIR /opt/NormalCrafter
RUN python3 -m pip install -r requirements.txt fire
RUN python3 -m pip uninstall -y torch torchvision xformers || true
RUN python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision
RUN python3 -m pip uninstall -y \
    torchaudio \
    nvidia-cublas-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 \
    nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-nccl-cu11 \
    || true
RUN python3 -m pip install --no-deps --extra-index-url https://pypi.org/simple \
    nvidia-cublas-cu12==12.8.4.1 \
    nvidia-cuda-cupti-cu12==12.8.90 \
    nvidia-cuda-nvrtc-cu12==12.8.93 \
    nvidia-cuda-runtime-cu12==12.8.90 \
    nvidia-cudnn-cu12==9.20.0.48 \
    nvidia-cufft-cu12==11.3.3.83 \
    nvidia-curand-cu12==10.3.9.90 \
    nvidia-cusolver-cu12==11.7.3.90 \
    nvidia-cusparse-cu12==12.5.8.93 \
    nvidia-cusparselt-cu12==0.7.1 \
    nvidia-nccl-cu12==2.29.7 \
    nvidia-nvjitlink-cu12==12.8.93 \
    nvidia-nvtx-cu12==12.8.90
RUN uv pip uninstall --python /usr/bin/python3 nvidia-nccl-cu12 || true
RUN uv pip install --python /usr/bin/python3 --no-deps --index-strategy unsafe-best-match --extra-index-url https://pypi.org/simple nvidia-nccl-cu12==2.29.7
RUN python3 -m pip uninstall -y xformers || true
RUN python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda)"

COPY docker/normal_crafter/run_normal_crafter.sh /usr/local/bin/run_normal_crafter.sh
RUN chmod +x /usr/local/bin/run_normal_crafter.sh

ENTRYPOINT ["/usr/local/bin/run_normal_crafter.sh"]
