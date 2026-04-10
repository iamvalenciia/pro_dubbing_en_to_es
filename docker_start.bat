@echo off
REM ─────────────────────────────────────────────────────────────────
REM  Inicia Qwen3-TTS como contenedor Docker local (requiere GPU)
REM  Después de que arranque, pipeline.py usará http://localhost:7860
REM ─────────────────────────────────────────────────────────────────
REM  REQUISITOS:
REM    - Docker Desktop con backend WSL2
REM    - nvidia-container-toolkit (nvidia-docker2)
REM    - HF_TOKEN con acceso al modelo (o dejarlo vacío si es público)
REM ─────────────────────────────────────────────────────────────────

SET HF_TOKEN=YOUR_HF_TOKEN_HERE

echo Iniciando Qwen3-TTS en Docker (GPU)...
echo Accesible en: http://localhost:7860
echo Presiona Ctrl+C para detener.
echo.

docker run -it --rm ^
  -p 7860:7860 ^
  --platform linux/amd64 ^
  --gpus all ^
  -e HF_TOKEN="%HF_TOKEN%" ^
  registry.hf.space/qwen-qwen3-tts:latest ^
  python app.py

pause
