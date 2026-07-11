FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

ENV LD_LIBRARY_PATH="/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cufft/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cufile/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/curand/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cusolver/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cusparse/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/nccl/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/nvtx/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/nvidia/nvshmem/lib:/opt/DepthCrafter/.venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti:${LD_LIBRARY_PATH}"

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
RUN python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision \
    || python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

WORKDIR /opt
RUN git clone --depth 1 https://github.com/Tencent/DepthCrafter.git DepthCrafter
RUN sed -i 's/requires-python = ">=3\.13"/requires-python = ">=3.10"/' /opt/DepthCrafter/pyproject.toml

WORKDIR /opt/DepthCrafter
RUN python3 -m pip install uv
RUN uv sync --python /usr/bin/python3
RUN uv pip uninstall --python /opt/DepthCrafter/.venv/bin/python torch torchvision || true
RUN uv pip install --python /opt/DepthCrafter/.venv/bin/python \
    --no-deps \
    "https://download.pytorch.org/whl/nightly/cu128/torch-2.12.0.dev20260407%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl" \
    "https://download.pytorch.org/whl/nightly/cu128/torchvision-0.27.0.dev20260407%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
RUN uv pip uninstall --python /opt/DepthCrafter/.venv/bin/python \
    torchaudio \
    nvidia-cublas-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 \
    nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-nccl-cu11 \
    nvidia-nvshmem-cu12 \
    triton \
    || true
RUN uv pip install --python /opt/DepthCrafter/.venv/bin/python --no-deps --index-strategy unsafe-best-match --extra-index-url https://pypi.org/simple \
    nvidia-cublas-cu12==12.8.4.1 \
    nvidia-cuda-cupti-cu12==12.8.90 \
    nvidia-cuda-nvrtc-cu12==12.8.93 \
    nvidia-cuda-runtime-cu12==12.8.90 \
    nvidia-cudnn-cu12==9.20.0.48 \
    nvidia-cufft-cu12==11.3.3.83 \
    nvidia-cufile-cu12==1.13.1.3 \
    nvidia-curand-cu12==10.3.9.90 \
    nvidia-cusolver-cu12==11.7.3.90 \
    nvidia-cusparse-cu12==12.5.8.93 \
    nvidia-cusparselt-cu12==0.7.1 \
    nvidia-nccl-cu12==2.29.7 \
    nvidia-nvshmem-cu12==3.4.5 \
    nvidia-nvjitlink-cu12==12.8.93 \
    nvidia-nvtx-cu12==12.8.90
RUN uv pip install --python /opt/DepthCrafter/.venv/bin/python --no-deps --index-strategy unsafe-best-match --extra-index-url https://pypi.org/simple triton==3.3.1
RUN sed -i 's/except (ImportError, ModuleNotFoundError, AttributeError) as e:/except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as e:/' /opt/DepthCrafter/depthcrafter/inference.py

COPY docker/depth_crafter/run_depth_crafter.sh /usr/local/bin/run_depth_crafter.sh
RUN chmod +x /usr/local/bin/run_depth_crafter.sh

ENTRYPOINT ["/usr/local/bin/run_depth_crafter.sh"]
