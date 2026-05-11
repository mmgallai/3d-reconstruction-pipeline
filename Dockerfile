# ─────────────────────────────────────────────────────────────────────────────
# nerfstudio-blackwell
# Nerfstudio + Gaussian Splatting for NVIDIA Blackwell GPUs (RTX 50xx / sm_120)
# Tested with: RTX 5070 Ti, CUDA 12.8, PyTorch nightly
#
# Build:
#   docker build -t nerfstudio-blackwell .
#
# Run (example):
#   docker run --rm --gpus all -v "C:/path/to/project:/workspace" nerfstudio-blackwell ns-train splatfacto --help
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        python3-pip \
        python3-dev \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        cmake \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ── PyTorch (nightly, CUDA 12.8 — required for Blackwell / sm_120) ───────────
RUN pip install --no-cache-dir --pre \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128

# ── Tell the compiler to target Blackwell (sm_120) ───────────────────────────
ENV TORCH_CUDA_ARCH_LIST="12.0"

# ── Nerfstudio core ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir nerfstudio

# ── gsplat — recompile from source for sm_120 ─────────────────────────────────
# This is the key step: the pip wheel is not compiled for Blackwell.
# Building from source with TORCH_CUDA_ARCH_LIST=12.0 produces correct kernels.
RUN pip install --no-cache-dir \
        git+https://github.com/nerfstudio-project/gsplat.git

# NOTE: DN-Splatter is installed at training-time from /workspace/dn-splatter
# (the local clone) by lib/nerfstudio_pipeline.py — this preserves any local
# modifications to the dn-splatter source. See `is_dn` block in train_nerfstudio_format.

# ── Cache torch hub models in /workspace/torch_cache so they persist ──────────
# Without this, Docker downloads alexnet (~233 MB) fresh on every --rm run.
ENV TORCH_HOME=/workspace/torch_cache

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKDIR /workspace
