FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0a;10.0;12.0"

# RunPod path defaults: network volume (persistent) + local NVMe (fast 50GB).
ENV QDP_NETWORK_DIR=/runpod-volume
ENV QDP_LOCAL_DIR=/workspace/qdp_data
ENV HF_HOME=/workspace/qdp_data/torch_cache
ENV HF_HUB_CACHE=/workspace/qdp_data/torch_cache/hub
ENV TORCH_HOME=/workspace/qdp_data/torch_cache

# ========== GPU TUNING ENVIRONMENT VARIABLES ==========
# Enable aggressive batching and concurrency for faster dubbing on RunPod
# NLLB-200 (A100-80GB profile): batch=512 lines per pass
ENV PYVIDEOTRANS_NLLB_BATCH_LINES=512
# Optional operator knob: desired VRAM budget for NLLB (GB).
# Example override in RunPod: PYVIDEOTRANS_NLLB_TARGET_VRAM_GB=60
ENV PYVIDEOTRANS_NLLB_TARGET_VRAM_GB=60
# Qwen3-TTS: dynamic VRAM-first batching by default.
# Keep this at 0 so runtime auto-sizing can use most available VRAM.
ENV PYVIDEOTRANS_QWEN_TTS_BATCH_LINES=0
# Optional operator knob: desired VRAM budget for Qwen TTS (GB).
# 0 means auto (use ~92% of free VRAM detected at runtime).
ENV PYVIDEOTRANS_QWEN_TTS_TARGET_VRAM_GB=0
# Qwen3-TTS: PyTorch thread count for audio post-processing (CPU)
ENV PYVIDEOTRANS_QWEN_TTS_TORCH_THREADS=12
# Force CUDA for Qwen-TTS even if upstream detection fails
ENV PYVIDEOTRANS_FORCE_QWEN_TTS_CUDA=1
# Enable auto-detection of GPU process pool size
ENV PYVIDEOTRANS_AUTO_PROCESS_MAX_GPU=1
# Override default process_max_gpu setting: allows 2 concurrent GPU tasks
# (instead of default 1) for better throughput on high-VRAM GPU systems
ENV PYVIDEOTRANS_PROCESS_MAX_GPU=2

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

# Pin torch stack to >=2.6 to satisfy transformers safe-loading requirement.
# Use CUDA 12.8 wheels to support Blackwell GPUs (sm_120).
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Core runtime deps.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Model runtimes.
# qwen-tts==0.1.1 pins transformers 4.57.3 while qwen-asr==0.0.6 pins 4.57.6.
# We keep a single transformers version (4.57.6) and install qwen-tts without deps
# because the required runtime deps are already installed from requirements.
RUN pip install --no-cache-dir transformers==4.57.6 accelerate==1.12.0 einops sox \
  && pip install --no-cache-dir qwen-asr==0.0.6 \
  && pip install --no-cache-dir --no-deps qwen-tts==0.1.1

# Optional speedups (do not fail image build if unavailable).
ARG INSTALL_XFORMERS=0
ARG INSTALL_FLASH_ATTN=0
RUN if [ "$INSTALL_XFORMERS" = "1" ]; then \
      pip install --no-cache-dir --no-deps xformers==0.0.28.post3 || echo "WARN: xformers install skipped"; \
    fi
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
      pip install --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation || echo "WARN: flash-attn install skipped"; \
    else \
      echo "INFO: flash-attn disabled at build time"; \
    fi

# Hard-fail image build if torch gets downgraded by transitive deps.
RUN python - <<'PY'
import torch
from packaging.version import Version
v = torch.__version__.split('+')[0]
print(f"Torch resolved in image: {torch.__version__}")
if Version(v) < Version("2.6.0"):
    raise SystemExit(f"ERROR: torch>=2.6.0 required, got {torch.__version__}")
PY

COPY . /app

# ========== OPTIMIZE cfg.json FOR GPU PERFORMANCE ==========
# Ensure aggressive batching and concurrency settings are baked into the container
RUN python3 - <<'PY'
import json
cfg_path = "/app/pyvideotrans/videotrans/cfg.json"
with open(cfg_path, "r") as f:
    cfg = json.load(f)
# GPU tuning for aggressive performance
cfg["trans_thread"] = 10          # translation threads (balanced with GPU memory)
cfg["aitrans_thread"] = 50        # AI translation threads
cfg["process_max_gpu"] = 2        # enable 2 concurrent GPU tasks
cfg["multi_gpus"] = False         # use first GPU with high concurrency, not multi-GPU
cfg["dubbing_thread"] = 1         # dubbing runs serialized but with batching
cfg["crf"] = 23                   # video quality (default)
cfg["backaudio_volume"] = 0.8     # background audio volume
with open(cfg_path, "w") as f:
    json.dump(cfg, f)
print(f"[DOCKERFILE] Updated {cfg_path} with GPU tuning: process_max_gpu=2, trans_thread=10, aitrans_thread=50")
PY

EXPOSE 7860
CMD ["python", "/app/main_ui.py"]
