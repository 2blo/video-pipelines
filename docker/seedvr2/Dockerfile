FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ca-certificates \
    ffmpeg \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 (required for newer GPUs like RTX 5090 / sm_120)
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch \
    torchvision

# Fetch SeedVR (upstream inference scripts)
WORKDIR /opt
RUN git clone --depth 1 https://github.com/bytedance-seed/SeedVR.git seedvr
WORKDIR /opt/seedvr

# Patch SeedVR to allow running without flash-attn (fallback to torch SDPA varlen)
RUN python3.10 - <<'PY'
from pathlib import Path

replacement = '''try:
    from flash_attn import flash_attn_varlen_func
except Exception:
    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        **kwargs,
    ):
        outs = []
        bsz = int(cu_seqlens_q.numel()) - 1
        for i in range(bsz):
            q_start = int(cu_seqlens_q[i].item())
            q_end = int(cu_seqlens_q[i + 1].item())
            k_start = int(cu_seqlens_k[i].item())
            k_end = int(cu_seqlens_k[i + 1].item())

            q_i = q[q_start:q_end].transpose(0, 1)
            k_i = k[k_start:k_end].transpose(0, 1)
            v_i = v[k_start:k_end].transpose(0, 1)

            try:
                out_i = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=causal,
                    scale=softmax_scale,
                )
            except TypeError:
                out_i = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=causal,
                )
            outs.append(out_i.transpose(0, 1))

        return torch.cat(outs, dim=0)
'''

for rel in ["models/dit/attention.py", "models/dit_v2/attention.py"]:
    path = Path("/opt/seedvr") / rel
    text = path.read_text()
    needle = "from flash_attn import flash_attn_varlen_func\n"
    if needle in text:
        path.write_text(text.replace(needle, replacement + "\n"))

for rel in ["models/dit/normalization.py", "models/dit_v2/normalization.py"]:
    path = Path("/opt/seedvr") / rel
    text = path.read_text()
    text = text.replace(
        """        if norm_type == \"fusedln\":\n            from apex.normalization import FusedLayerNorm\n\n            return FusedLayerNorm(\n                normalized_shape=dim,\n                elementwise_affine=elementwise_affine,\n                eps=eps,\n            )\n""",
        """        if norm_type == \"fusedln\":\n            try:\n                from apex.normalization import FusedLayerNorm\n            except Exception:\n                return nn.LayerNorm(\n                    normalized_shape=dim,\n                    eps=eps,\n                    elementwise_affine=elementwise_affine,\n                )\n\n            return FusedLayerNorm(\n                normalized_shape=dim,\n                elementwise_affine=elementwise_affine,\n                eps=eps,\n            )\n""",
    )
    text = text.replace(
        """        if norm_type == \"fusedrms\":\n            from apex.normalization import FusedRMSNorm\n\n            return FusedRMSNorm(\n                normalized_shape=dim,\n                elementwise_affine=elementwise_affine,\n                eps=eps,\n            )\n""",
        """        if norm_type == \"fusedrms\":\n            try:\n                from apex.normalization import FusedRMSNorm\n            except Exception:\n                return RMSNorm(\n                    dim=dim,\n                    eps=eps,\n                    elementwise_affine=elementwise_affine,\n                )\n\n            return FusedRMSNorm(\n                normalized_shape=dim,\n                elementwise_affine=elementwise_affine,\n                eps=eps,\n            )\n""",
    )
    path.write_text(text)

rope_path = Path("/opt/seedvr/models/dit_v2/rope.py")
rope_text = rope_path.read_text()
rope_text = rope_text.replace(
    """        vid_freqs = self.get_axial_freqs(1024, 128, 128)\n        txt_freqs = self.get_axial_freqs(1024)\n""",
    """        max_t = max(1, int((vid_shape[:, 0] + txt_shape[:, 0]).max().item()))\n        max_h = max(1, int(vid_shape[:, 1].max().item()))\n        max_w = max(1, int(vid_shape[:, 2].max().item()))\n        max_l = max(1, int(txt_shape[:, 0].max().item()))\n        vid_freqs = self.get_axial_freqs(max_t, max_h, max_w)\n        txt_freqs = self.get_axial_freqs(max_l)\n""",
)
rope_path.write_text(rope_text)
PY

# Install upstream deps except torch/torchvision (we already installed CUDA wheels above)
RUN grep -vE '^(torch|torchvision)=' requirements.txt > /tmp/requirements.notorch.txt && \
    pip install -r /tmp/requirements.notorch.txt

# Required by torchvision.io.read_video in SeedVR inference script
RUN pip install av

# SeedVR uses flash-attn (imported at module import time)
ENV CUDAARCHS=120
RUN pip install psutil packaging
RUN pip install --only-binary=:all: flash-attn || true

# Needed for downloading model checkpoints at runtime
RUN pip install huggingface_hub==0.25.2

COPY docker/seedvr2/seedvr2_upscale.py /usr/local/bin/seedvr2-upscale
RUN chmod +x /usr/local/bin/seedvr2-upscale

ENV PYTHONPATH=/opt/seedvr
WORKDIR /opt/seedvr

ENTRYPOINT ["seedvr2-upscale"]
