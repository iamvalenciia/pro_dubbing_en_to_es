#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — RunPod entrypoint
# 1. Descarga modelos al Network Volume si no están cacheados
# 2. Lanza la app Gradio
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo ""
echo "============================================================"
echo "  EN->ES Voice Dubbing  |  RunPod A100 SXM"
echo "============================================================"
echo "  TTS model  : ${TTS_LOCAL_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
echo "  Whisper    : ${WHISPER_MODEL:-base}"
echo "  Batch size : ${TTS_BATCH_SIZE:-8}"
echo "  HF cache   : ${HF_HOME:-/runpod-volume/huggingface}"
echo "============================================================"
echo ""

# Asegurar que el directorio de cache existe (RunPod lo crea, pero por si acaso)
mkdir -p "${HF_HOME:-/runpod-volume/huggingface}/hub"

# Asegurar que los directorios de salida existen (por si el volumen se montó vacío)
mkdir -p \
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

# ── 1. Whisper base (~145 MB) ─────────────────────────────────────────────────
WHISPER_CACHE="${HF_HOME:-/runpod-volume/huggingface}/hub/models--Systran--faster-whisper-base"
if [ ! -d "$WHISPER_CACHE" ]; then
    echo "[start.sh] Descargando faster-whisper 'base' (~145 MB)..."
    python3 -c "
from faster_whisper import WhisperModel
print('  Downloading Whisper base...')
WhisperModel('base', device='cpu', compute_type='int8')
print('  OK Whisper base cached.')
"
else
    echo "[start.sh] Whisper base: ya en cache (OK)"
fi

# ── 2. Qwen3-TTS 1.7B (~4.5 GB) ─────────────────────────────────────────────
# Bypass the complex HuggingFace cache Symlink network that fails on RunPod NFS.
# We download directly as physical files to a standard directory.
TTS_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TTS_DIR="/runpod-volume/Qwen3-TTS-Flat"
export TTS_LOCAL_MODEL="$TTS_DIR"

echo "[start.sh] Validando integridad de ${TTS_MODEL_ID} en ${TTS_DIR}..."
python3 -c "
import os, sys, shutil
from huggingface_hub import snapshot_download

model_id = '${TTS_MODEL_ID}'
local_dir = '${TTS_DIR}'

print(f'  Downloading/Verifying {model_id} to {local_dir}...')
try:
    snap_path = snapshot_download(model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    # Verificar que el feature_extractor fisico si está dentro
    chk = os.path.join(snap_path, 'speech_tokenizer', 'preprocessor_config.json')
    if not os.path.exists(chk):
        print(f'  WARN: Directorio {chk} no existe. Corrupcion física detectada. Borrando...')
        shutil.rmtree(snap_path, ignore_errors=True)
        sys.exit(1)
except Exception as e:
    print(f'  Error detectado en caché plano: {e}')
    sys.exit(1)
print('  OK Qwen3-TTS descargado de forma plana y validado.')
"
if [ $? -ne 0 ]; then
    echo "[start.sh] Fallo detectado. Purgando TODO el directorio plano..."
    rm -rf "$TTS_DIR"
    echo "[start.sh] Reintentando descarga pesada en modo plano (~3-8 min)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${TTS_MODEL_ID}', local_dir='${TTS_DIR}', local_dir_use_symlinks=False)"
fi

echo ""
echo "[start.sh] Todos los modelos listos."
echo "[start.sh] Iniciando Gradio en 0.0.0.0:7860 ..."
echo ""

# exec reemplaza el proceso bash -> SIGTERM llega directo a Python (shutdown limpio)
exec python3 /app/app.py
