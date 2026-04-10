# ─────────────────────────────────────────────────────────────────────────────
# EN->ES Voice Dubbing Pipeline
# Base: pytorch/pytorch 2.5.1 + CUDA 12.1 + cuDNN 9 (Ubuntu 22.04)
# Models are NOT baked in — they download to /runpod-volume at first start.
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        unzip \
        libsndfile1 \
        sox \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Deno — runtime JS recomendado por yt-dlp para resolver el n-challenge de YouTube
# (nodejs del apt de Ubuntu 22.04 = v12, no soportado; Deno es el default de yt-dlp EJS)
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="/root/.deno/bin:$PATH"

WORKDIR /app

# Python dependencies (torch is already in the base image — do NOT reinstall)
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Application code
COPY pipeline.py .
COPY app.py .
COPY shorts_pipeline.py .
COPY subtitle_pipeline.py .

# Voice reference files — baked into the image, never change
COPY voice_reference/ ./voice_reference/

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ── Create persistent output directories ─────────────────────────────────────
# These get mounted as volumes in production (RunPod network volume or bind mounts)
RUN mkdir -p \
    /app/cache \
    /app/enhance_output \
    /app/subtitle_output \
    /app/youtube_video \
    /app/archives \
    /app/shorts/short_video \
    /app/shorts/short_audio_doblado \
    /app/shorts/audio_original_en \
    /app/shorts/shorts_videos_lists \
    /app/shorts/background_music \
    /app/shorts/_cache

# ── Model cache points to the RunPod Network Volume ──────────────────────────
# All three vars ensure compatibility across huggingface-hub versions
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub

# App config (overridable in RunPod template)
ENV APP_DIR=/app
ENV WHISPER_MODEL=base
ENV TTS_LOCAL_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV TTS_BATCH_SIZE=8
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Declare volumes for persistent data
VOLUME ["/runpod-volume", "/app/cache", "/app/enhance_output", \
        "/app/subtitle_output", "/app/youtube_video", \
        "/app/archives", "/app/shorts"]

CMD ["/start.sh"]
