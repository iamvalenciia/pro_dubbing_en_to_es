# ─────────────────────────────────────────────────────────────────────────────
# EN->ES Voice Dubbing Pipeline
# Base: pytorch/pytorch 2.5.1 + CUDA 12.1 + cuDNN 9 (Ubuntu 22.04)
# Models are NOT baked in — they download to /runpod-volume at first start.
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# System dependencies (Añadido wget, libgl1 y libglib para OpenCV y Retalking)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        wget \
        unzip \
        libsndfile1 \
        sox \
        zip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Deno — runtime JS recomendado por yt-dlp
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="/root/.deno/bin:$PATH"

WORKDIR /app

# Python dependencies de tu app principal
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# ── INSTALACIÓN DE VIDEO-RETALKING (LIP-SYNC) ────────────────────────────────
# Clonamos, instalamos dependencias y descargamos los pesos de GFPGAN en el build
RUN git clone https://github.com/OpenTalker/video-retalking.git /app/video-retalking
WORKDIR /app/video-retalking
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gfpgan

RUN mkdir -p checkpoints && \
    wget -q https://github.com/xinntao/facexlib/releases/download/v0.2.2/alignment_WFLW_4modules.pth -O checkpoints/alignment_WFLW_4modules.pth && \
    wget -q https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -O checkpoints/detection_Resnet50_Final.pth && \
    wget -q https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -O checkpoints/GFPGANv1.4.pth && \
    wget -q https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -O checkpoints/parsing_parsenet.pth

# Regresamos al directorio de la app
WORKDIR /app
# ─────────────────────────────────────────────────────────────────────────────

# Application code
COPY pipeline.py .
COPY app.py .
COPY shorts_pipeline.py .
COPY subtitle_pipeline.py .

# Voice reference files
COPY voice_reference/ ./voice_reference/

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ── Create persistent output directories ─────────────────────────────────────
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

# ── Model cache points ───────────────────────────────────────────────────────
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub

# App config
ENV APP_DIR=/app
ENV WHISPER_MODEL=base
ENV TTS_LOCAL_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV TTS_BATCH_SIZE=8
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

VOLUME ["/runpod-volume", "/app/cache", "/app/enhance_output", \
        "/app/subtitle_output", "/app/youtube_video", \
        "/app/archives", "/app/shorts"]

CMD ["/start.sh"]