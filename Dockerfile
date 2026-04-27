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
ENV PYVIDEOTRANS_OFFLINE_MODE=1

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
# qwen-tts==0.1.1 requires transformers==4.57.3.
# Keep one validated transformers version and install Qwen packages without
# dependency resolution here to avoid accidental upgrades/downgrades.
RUN pip install --no-cache-dir transformers==4.57.3 accelerate==1.12.0 einops sox \
    && pip install --no-cache-dir --no-deps qwen-asr==0.0.6 \
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

# ========== PRE-BAKE DIARIZATION ONNX MODELS ==========
# Download the 3 sherpa_onnx diarization models at build time so they are
# available immediately at runtime (no cold-start download on the GPU pod).
RUN python3 - <<'PY'
import urllib.request, os
from pathlib import Path

onnx_dir = Path("/app/pyvideotrans/models/onnx")
onnx_dir.mkdir(parents=True, exist_ok=True)

models = [
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/seg_model.onnx",
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/nemo_en_titanet_small.onnx",
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
]

for url in models:
    fname = url.split("/")[-1]
    dest = onnx_dir / fname
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"[ONNX] Already exists: {fname} ({dest.stat().st_size // 1024}KB)")
        continue
    print(f"[ONNX] Downloading {fname}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"[ONNX] OK: {fname} ({dest.stat().st_size // 1024}KB)")
    except Exception as e:
        print(f"[ONNX] WARN: failed to pre-download {fname}: {e} — will retry at runtime")
print("[ONNX] Diarization model pre-bake done.")
PY

# ========== PRE-BAKE MODELSCOPE + FASTER-WHISPER MODELS ==========
# Keep Phase 1 self-contained: no runtime download for ali_CAM diarization
# and default ASR model (large-v3-turbo).
RUN python3 - <<'PY'
from pathlib import Path
import os

os.environ.setdefault("MODELSCOPE_CACHE", "/app/pyvideotrans/models")
os.environ.setdefault("HF_HOME", "/app/pyvideotrans/models")
os.environ.setdefault("HF_HUB_CACHE", "/app/pyvideotrans/models")

print("[PREBAKE] Downloading ModelScope CAMP++ diarization model...")
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
ms_snapshot_download(
    model_id="iic/speech_campplus_speaker-diarization_common",
    local_dir="/app/pyvideotrans/models/models/iic/speech_campplus_speaker-diarization_common",
    max_workers=1,
)

# ali_CAM pipeline resolves this speaker verification dependency dynamically.
# Pre-bake it so runtime stays offline and deterministic.
print("[PREBAKE] Downloading ModelScope CAMP++ speaker verification submodel...")
ms_snapshot_download(
    model_id="damo/speech_campplus_sv_zh-cn_16k-common",
    local_dir="/app/pyvideotrans/models/models/damo/speech_campplus_sv_zh-cn_16k-common",
    max_workers=1,
)

# ali_CAM also resolves the FunASR VAD model dynamically before clustering.
print("[PREBAKE] Downloading ModelScope VAD submodel...")
ms_snapshot_download(
    model_id="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    local_dir="/app/pyvideotrans/models/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    max_workers=1,
)

# ali_CAM lazily resolves this clustering backend only when diarization reaches
# the long-form clustering stage. Pre-bake it explicitly so 1h videos remain
# offline and deterministic too.
print("[PREBAKE] Downloading ModelScope speaker clustering submodel...")
ms_snapshot_download(
    model_id="damo/speech_campplus-transformer_scl_zh-cn_16k-common",
    local_dir="/app/pyvideotrans/models/models/damo/speech_campplus-transformer_scl_zh-cn_16k-common",
    max_workers=1,
)

print("[PREBAKE] Downloading Faster-Whisper large-v3-turbo model...")
from huggingface_hub import snapshot_download as hf_snapshot_download
hf_snapshot_download(
    repo_id="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    local_dir="/app/pyvideotrans/models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
    max_workers=1,
    local_files_only=False,
    ignore_patterns=["*.msgpack", "*.h5", ".git*", "*.md"],
)

print("[PREBAKE] Downloading Qwen3-TTS base model...")
hf_snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    local_dir="/app/pyvideotrans/models/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
    max_workers=1,
    local_files_only=False,
)

print("[PREBAKE] Downloading Qwen3-TTS custom voice model...")
hf_snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    local_dir="/app/pyvideotrans/models/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice",
    max_workers=1,
    local_files_only=False,
)

print("[PREBAKE] Model pre-bake done.")
PY

# ========== BUILD-TIME PHASE 1 AUDIT ==========
RUN python3 - <<'PY'
from pathlib import Path
import importlib

required_modules = [
    "modelscope",
    "funasr",
    "huggingface_hub",
    "faster_whisper",
    "simplejson",
    "sortedcontainers",
    "sherpa_onnx",
    "addict",
    "requests",
    "pydub",
]
for mod in required_modules:
    importlib.import_module(mod)
    print(f"[AUDIT] import ok: {mod}")

# Validate the exact ModelScope audio path used by ali_CAM diarization.
importlib.import_module("modelscope.models.audio.funasr.model")
print("[AUDIT] import ok: modelscope.models.audio.funasr.model")

required_files = [
    "/app/pyvideotrans/models/onnx/seg_model.onnx",
    "/app/pyvideotrans/models/onnx/nemo_en_titanet_small.onnx",
    "/app/pyvideotrans/models/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
    "/app/pyvideotrans/models/models/damo/speech_campplus-transformer_scl_zh-cn_16k-common/transformer_backend.pt",
    "/app/pyvideotrans/models/models/damo/speech_campplus-transformer_scl_zh-cn_16k-common/campplus_cn_encoder.pt",
    "/app/pyvideotrans/models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/model.bin",
    "/app/pyvideotrans/models/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/config.json",
    "/app/pyvideotrans/models/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/config.json",
]
for p in required_files:
    path = Path(p)
    if not path.exists() or path.stat().st_size <= 1024:
        raise SystemExit(f"[AUDIT] missing or invalid file: {p}")
    print(f"[AUDIT] file ok: {p}")

print("[AUDIT] Phase 1 Docker audit passed.")
PY

# ========== BUILD-TIME ALI_CAM RUNTIME SMOKE ==========
# Load the speaker-diarization pipeline and run VAD on a synthetic WAV.
# Goal: prove funasr + modelscope + all transitive deps load at build time.
# A sine tone has no speech so VAD returns 0 effective seconds; we treat the
# specific "audio duration is too short" AssertionError as a PASS (it means
# the full dep chain loaded and VAD ran). Only real import/runtime errors fail.
RUN python3 - <<'PY'
import math
import os
import struct
import traceback
import wave

os.environ.setdefault("MODELSCOPE_CACHE", "/app/pyvideotrans/models")
os.environ.setdefault("HF_HOME", "/app/pyvideotrans/models")
os.environ.setdefault("HF_HUB_CACHE", "/app/pyvideotrans/models")

wav_path = "/tmp/ali_cam_smoke.wav"
sample_rate = 16000
duration_sec = 2.0
freq = 440.0
n = int(sample_rate * duration_sec)

with wave.open(wav_path, "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    for i in range(n):
        v = int(0.08 * 32767 * math.sin(2.0 * math.pi * freq * i / sample_rate))
        w.writeframes(struct.pack("<h", v))

try:
    from modelscope.pipelines import pipeline

    diar = pipeline(
        task="speaker-diarization",
        model="iic/speech_campplus_speaker-diarization_common",
    )
    out = diar(wav_path)
    print(f"[AUDIT] ali_CAM runtime smoke ok: type={type(out)}")
except AssertionError as exc:
    # "The effective audio duration is too short" — VAD ran, all deps present.
    if "too short" in str(exc) or "audio duration" in str(exc):
        print(f"[AUDIT] ali_CAM runtime smoke ok (VAD ran, no speech in tone): {exc}")
    else:
        print(f"[AUDIT] ali_CAM runtime smoke failed (unexpected assertion): {exc}")
        traceback.print_exc()
        raise SystemExit(1)
except Exception as exc:
    print(f"[AUDIT] ali_CAM runtime smoke FAILED: {exc}")
    traceback.print_exc()
    raise SystemExit(1)
PY

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
# Use GPU-accelerated speaker diarization (ali_CAM = CAMP++ via ModelScope)
# This replaces the CPU-only 'built' sherpa_onnx backend.
cfg["speaker_type"] = "ali_CAM"
with open(cfg_path, "w") as f:
    json.dump(cfg, f)
print(f"[DOCKERFILE] Updated {cfg_path} with GPU tuning: process_max_gpu=2, trans_thread=10, aitrans_thread=50, speaker_type=ali_CAM")
PY

EXPOSE 7860
CMD ["python", "/app/main_ui.py"]
