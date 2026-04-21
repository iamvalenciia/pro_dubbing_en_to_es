FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Variables para evadir bloqueos interactivos y compilar C++ suavemente
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Forzar optimizaciones en compilaciones de CUDA para A100 (8.0), RTX Ada (8.9) y Hopper (9.0a)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0a" 

# 1. Dependencias maestras de SO (FFmpeg, Rubberband-cli, Compiladores Ninja y C++)
# 1. Dependencias maestras de SO con un Loop Resiliente Anti-Caídas de DNS en WSL
# Ejecuta hasta 5 intentos si el servidor de Ubuntu rechaza o pierde paquetes goteando
RUN for i in 1 2 3 4 5; do \
        apt-get clean && \
        apt-get update -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 --fix-missing && \
        apt-get install -y --no-install-recommends \
        python3.10 python3.10-distutils python3.10-venv python3.10-dev \
        python3-pip \
        git wget curl build-essential ninja-build \
        ffmpeg rubberband-cli libsndfile1 sox libsox-fmt-all \
        && break || sleep 15; \
    done && rm -rf /var/lib/apt/lists/*

# Configurar alias robusto
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Upgrade de utilidades pilar de construcción
RUN pip install --no-cache-dir --upgrade pip wheel setuptools packaging

# Copia de requerimientos anclados
COPY requirements.txt .

# 2. MATCH PERFECTO: PyTorch 2.5.1 envuelto en CUDA 12.4 (Garantía Ada, Hopper, Blackwell)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. COMPILACION / INSTALACION ACELERADA Y COMPATIBLE
# xformers compatible con PyTorch 2.5.1+cu124
RUN pip install --no-cache-dir xformers==0.0.28.post3

# flash-attn: Con Ninja y Build-Essential presentes, el flag --no-build-isolation garantiza 
# que use el torch 2.5 ya instalado en vez de bajar uno conflictivo al crear el wheel.
RUN pip install --no-cache-dir psutil && \
    pip install --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation
# 4. Instalación de librerías del Pipeline Pinneadas
RUN pip install --no-cache-dir -r requirements.txt

# 4.5 LatentSync (Build-time setup controlado)
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git /app/LatentSync && \
    egrep -v "^(torch|torchvision|torchaudio|transformers|accelerate|xformers|gradio|numpy)" /app/LatentSync/requirements.txt > /app/LatentSync/req_filtered.txt && \
    pip install --no-cache-dir -r /app/LatentSync/req_filtered.txt

# 5. Qwen3-TTS primero, luego Qwen3-ASR.
# Ambos clavan transformers exacto (4.57.3 vs 4.57.6) — pip no resuelve juntos.
# Al instalar en 2 pasos, qwen-asr gana y deja transformers==4.57.6,
# que sigue siendo compatible con qwen-tts (solo patch bump).
RUN pip install --no-cache-dir qwen-tts==0.1.1
RUN pip install --no-cache-dir qwen-asr==0.0.6

# Copia del código fuente
COPY . .

# Defaults de data dir:
# - APP_DATA_DIR y TORCH_HOME al volumen (archivos grandes persisten)
# - HF_HOME al disco local del container: pyannote pesa poco (~200MB) pero
#   el fuse mount del volumen tiene IOPS muy bajos — el load inicial demora
#   5+ min desde fuse vs <1 min descargando desde HF CDN.
ENV APP_DATA_DIR=/runpod-volume
ENV TORCH_HOME=/runpod-volume/torch_cache
ENV HF_HOME=/root/.cache/huggingface

# Pre-descarga pyannote 3.1 al HF_HOME del container para eliminar los 5+ min
# de descarga en el primer request. Requiere --build-arg HF_TOKEN=xxx al construir.
# Sin token, la imagen sigue funcionando pero el primer run pagara el costo.
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
        echo "Baking pyannote/speaker-diarization-3.1 into image..." && \
        HF_TOKEN="$HF_TOKEN" python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='$HF_TOKEN')" && \
        echo "Pyannote baked OK en $HF_HOME"; \
    else \
        echo "WARN: HF_TOKEN build-arg vacio, pyannote NO horneado. Primer arranque del container descargara ~200MB (5+ min en fuse)."; \
    fi

# Exposición y Ejecución
EXPOSE 7860
CMD ["python", "/app/main_ui.py"]
