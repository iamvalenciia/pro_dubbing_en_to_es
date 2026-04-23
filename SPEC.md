# Quantum Dubbing Pipeline - Production SPEC (RunPod First)

## 1. Objetivo del Sistema
Aplicacion web Gradio para:
- Doblaje EN -> ES con pyvideotrans como motor principal.
- Ajustes visuales globales del video final (brillo/contraste/color/nitidez).
- Subtitulado (SRT/JSON) desde video o audio.
- Generacion de reels virales 9:16 con Gemini + render por seleccion (checkboxes).

Este documento define el flujo oficial del proyecto y reemplaza flujos antiguos.

## 2. Arquitectura de Ejecucion
Soporte dual:
- Local (Windows/Linux): mismo flujo funcional que produccion.
- Produccion (RunPod Docker): despliegue headless con GPU.

Entrypoint de app:
- `main_ui.py`

Motor pesado:
- `pyvideotrans/` (ASR, traduccion, TTS, sincronizacion temporal)

Modulos de soporte:
- `subtitle_generator.py`
- `subtitle_renderer.py`
- `reel_generator.py`
- `frame_editor.py`

## 3. Storage y Rutas (Contrato Estricto)
Rutas centrales se resuelven en `src/paths.py`.

### 3.1 Volumen de Red Persistente
`QDP_NETWORK_DIR=/runpod-volume`

Subcarpetas:
- `user_input/`: videos originales subidos por usuario.
- `output_final/`: artefactos finales descargables.
- `models/`: modelos persistentes.

### 3.2 Disco Local NVMe Rapido (50GB)
`QDP_LOCAL_DIR=/workspace/qdp_data`

Subcarpetas:
- `temp_processing/`: ffmpeg intermedio, chunks y trabajo temporal.
- `logs/`: logs operativos.
- `torch_cache/`: cache de torch/huggingface.

### 3.3 Regla Operativa
- Input de usuario entra por red (`/runpod-volume/user_input`).
- Procesamiento intensivo ocurre en local NVMe (`/workspace/qdp_data`).
- Salida final vuelve a red (`/runpod-volume/output_final`).

## 4. UI Oficial (3 Pestañas)

### 4.1 Pestaña 1 - Doblaje Master
Objetivo:
- Generar video doblado final sin subtitulos quemados.

Entradas:
- Video original (upload o dropdown desde `user_input`).
- Flags:
  - Modo prueba (30s)
  - Speaker-aware clone
- Ajustes visuales:
  - Brillo
  - Contraste
  - Color
  - Nitidez
  - Checkbox: aplicar ajustes al video final

Proceso:
1. pyvideotrans ejecuta pipeline EN->ES.
2. Se copia MP4 final a `output_final`.
3. Si checkbox activo, se aplica postproceso visual al MP4 final completo.
4. Se extrae WAV final.
5. Se genera `timestamps.json` desde SRT adaptado.

Salidas:
- Video doblado final (`*_dubbed.mp4`)
- Audio WAV (`*_dubbed.wav`)
- SRT (`*_subtitles.srt`)
- JSON timestamps (`*_timestamps.json`)

### 4.2 Pestaña 2 - Subtitulos
Objetivo:
- Generar subtitulos desde video o audio.
- Renderizar subtitulos sobre video cuando la fuente es video.

Entradas:
- Upload o seleccion de media (video/audio).
- Idioma de transcripcion.

Reglas:
- Si entrada es audio (.wav/.mp3/...):
  - SI: generar SRT/JSON.
  - NO: renderizar video subtitulado (requiere video).

Salidas:
- SRT
- JSON
- Video subtitulado (solo si entrada base es video)

### 4.3 Pestaña 3 - Viral Shorts
Objetivo:
- Crear reels verticales 9:16 desde video doblado + JSON.

Entradas:
- Video doblado final:
  - upload o dropdown desde `output_final`.
- JSON:
  - JSON de timestamps para analizar con Gemini, o
  - JSON de reels ya editado por humano (re-upload).

Flujo:
1. Analizar/cargar JSON de reels.
2. Mostrar lista de reels en checkbox group.
3. Usuario marca reels a generar.
4. Renderizar solo seleccionados.
5. Descargar JSON de reels y archivos MP4 generados.

Subtitulos en reels:
- Dinamicos.
- Posicion centrada (medio del frame).

Salidas:
- Preview de primer reel generado.
- Descarga multiarchivo de reels (`gr.Files`).
- JSON de reels descargable/editable/reutilizable.

## 5. Contrato de JSON

### 5.1 timestamps.json (Pestaña 1 -> 3)
Campos esperados por segmento:
- `start` o `start_ms`
- `end` o `end_ms`
- `text_es`

### 5.2 reels.json (Gemini o edicion humana)
Estructura:
- `reels`: lista de objetos con:
  - `reel_num`
  - `start_ms`
  - `end_ms`
  - `title`
  - `caption_style`
  - `priority` (opcional)
- `metadata` (opcional)

Regla:
- Si el JSON subido ya contiene `reels`, se usa directo (sin reanalizar).

## 6. Docker y RunPod

### 6.1 Build
Dockerfile actual esta preparado para:
- CUDA 12.4 + cuDNN.
- Torch 2.5.1 cu124.
- qwen-tts + qwen-asr.
- ffmpeg/sox/libsndfile.
- variables de ruta para RunPod.

Variables clave en imagen:
- `QDP_NETWORK_DIR=/runpod-volume`
- `QDP_LOCAL_DIR=/workspace/qdp_data`
- `HF_HOME=/workspace/qdp_data/torch_cache`
- `TORCH_HOME=/workspace/qdp_data/torch_cache`

### 6.2 Comando de arranque
- `python /app/main_ui.py`

### 6.3 Flash Attention
- Es opcional por build arg (`INSTALL_FLASH_ATTN=1`).
- Si no se instala, la app funciona igual; solo reduce velocidad de inferencia.

## 7. Limpieza y No-Objetivos
No forman parte del workflow oficial:
- Flujos antiguos de reels (selector unico, tab duplicada).
- Artefactos visuales/caches locales (capturas y `__pycache__`).

## 8. Checklist de Release
Antes de desplegar:
1. `python -m py_compile main_ui.py reel_generator.py subtitle_generator.py subtitle_ui_wrapper.py`
2. Verificar dropdowns de red en `user_input` y `output_final`.
3. Verificar escritura de temporales en `/workspace/qdp_data/temp_processing`.
4. Validar que reels se renderizan por seleccion (checkbox) y se descargan en lote.
