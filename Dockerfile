FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0a"

# RunPod path defaults: network volume (persistent) + local NVMe (fast 50GB).
ENV QDP_NETWORK_DIR=/runpod-volume
ENV QDP_LOCAL_DIR=/workspace/qdp_data
ENV HF_HOME=/workspace/qdp_data/torch_cache
ENV HF_HUB_CACHE=/workspace/qdp_data/torch_cache/hub
ENV TORCH_HOME=/workspace/qdp_data/torch_cache

RUN for i in 1 2 3 4 5; do \
      apt-get clean && \
      apt-get update -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 --fix-missing && \
      apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv python3-pip \
      git wget curl build-essential ninja-build \
      ffmpeg rubberband-cli libsndfile1 sox libsox-fmt-all && break || sleep 15; \
    done && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip wheel setuptools packaging

COPY requirements.txt /app/requirements.txt

# Pin torch stack to CUDA 12.4 for RTX/A100 compatibility.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Core runtime deps.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Model runtimes.
RUN pip install --no-cache-dir qwen-tts==0.1.1 qwen-asr==0.0.6

# Optional speedups (do not fail image build if unavailable).
ARG INSTALL_XFORMERS=1
ARG INSTALL_FLASH_ATTN=0
RUN if [ "$INSTALL_XFORMERS" = "1" ]; then \
      pip install --no-cache-dir xformers==0.0.28.post3 || echo "WARN: xformers install skipped"; \
    fi
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
      pip install --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation || echo "WARN: flash-attn install skipped"; \
    else \
      echo "INFO: flash-attn disabled at build time"; \
    fi

COPY . /app

EXPOSE 7860
CMD ["python", "/app/main_ui.py"]
