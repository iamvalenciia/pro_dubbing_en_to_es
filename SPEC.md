# Quantum Dubbing Pipeline - SPEC Operativo Actual

## 1. Objetivo del sistema
Aplicación web Gradio para producción de doblaje EN -> ES con foco en RunPod GPU y paridad funcional local.

Capacidades oficiales:
- Doblaje en 2 fases con asignación manual de voz por speaker.
- Diarización obligatoria en Fase 1 (si falla, se aborta el run).
- Render final con ajustes de video y mezcla de audio configurable.
- Generación y render de subtítulos (SRT/JSON y video subtitulado).
- Generación de reels verticales por selección desde JSON.

Este documento reemplaza versiones anteriores y describe el estado real actual del proyecto.

## 2. Política de entorno Python (obligatoria)

### 2.1 Regla principal
No usar entornos virtuales locales (`.venv`) en este repositorio.

### 2.2 Ejecución permitida
- Usar siempre Python global del sistema en local.
- En Docker/RunPod, usar el Python global de la imagen (`/usr/bin/python` o `python`).

### 2.3 Operación diaria
- Instalar dependencias con `pip` global (`pip install -r requirements.txt`).
- Ejecutar la app con `python main_ui.py`.
- No activar ni crear `.venv` para correr el flujo de trabajo.

## 3. Arquitectura de ejecución

### 3.1 Entrypoint
- `main_ui.py`

### 3.2 Motor principal
- `pyvideotrans/` (ASR, diarización, traducción, TTS, alineación y ensamblado)

### 3.3 Módulos de soporte
- `subtitle_generator.py`
- `subtitle_renderer.py`
- `reel_generator.py`
- `frame_editor.py`
- `src/paths.py` (resolución de rutas y contrato de storage)

## 4. Storage y rutas (contrato vigente)

Rutas base resueltas por `src/paths.py`.

### 4.1 RunPod / Linux (modo volumen de red + NVMe)
- `QDP_NETWORK_DIR=/runpod-volume`
- `QDP_LOCAL_DIR=/workspace/qdp_data`

Subrutas efectivas:
- Red: `/runpod-volume/user_input`, `/runpod-volume/output_final`, `/runpod-volume/models`
- Local: `/workspace/qdp_data/temp_processing`, `/workspace/qdp_data/logs`, `/workspace/qdp_data/torch_cache`

### 4.2 Windows local (fallback actual)
Si no se define `QDP_NETWORK_DIR`:
- Input de red lógico: carpeta `input/` del repo
- Output de red lógico: carpeta `output/` del repo
- Models de red lógico: carpeta `models/` del repo
- Local temporal: `temp_workspace/qdp_data/`

### 4.3 Regla operativa
- Inputs entran por carpeta de red lógica.
- Proceso pesado ocurre en almacenamiento local rápido.
- Artefactos finales se publican en carpeta de salida de red lógica.

## 5. UI oficial (3 pestañas)

## 5.1 Pestaña 1 - Doblaje Master (flujo en 2 fases)

### Artifact-first AssemblyAI (nuevo flujo oficial)
El sistema prioriza artifacts AssemblyAI reutilizables para evitar reruns costosos.

Prioridad canónica de bootstrap:
1. `timestamps.json`
2. `sentences.json`
3. `paragraphs.json`
4. `transcript.srt`
5. `transcript.vtt`

Reglas:
- `transcript.txt` se persiste, pero no habilita fases por sí solo.
- El rescate manual siempre va ligado al video seleccionado en Tab 1.
- Si existe `speaker_segments.json` válido para ese video, Fase 1 se reutiliza sin relanzar analyze.
- Si existe `translated_segments.json` válido, Fase 2 se reutiliza y Fase 3 queda habilitada.
- Si faltan artifacts opcionales (sentences/paragraphs), el sistema degrada con fallback automático.

### Fase 1 - Analizar video
Objetivo:
- Detectar speakers reales y preparar artefactos de asignación.

Secuencia real:
1. `prepare`
2. `recogn`
3. `diariz`
4. Validación estricta de `speaker.json` (obligatoria)
5. `trans`
6. `run_speaker_analysis`
7. Emisión de `ANALYZE_DONE`

Regla crítica:
- Si diarización no produce speakers válidos, Fase 1 falla y no debe continuar.
- No se debe avanzar a traducción útil de negocio cuando no hay diarización válida.

Salida funcional de Fase 1:
- Lista dinámica de speakers en UI (máximo configurable por `PYVIDEOTRANS_MAX_SPEAKERS`, default 50, rango efectivo 12-100).
- Segmentos auditables por speaker.
- Preparación de mapeo speaker -> voz de referencia.
- Persistencia en `phase_state/<video>/` de exports AssemblyAI reutilizables.

### Fase 2 - Doblar con voces asignadas
Objetivo:
- Ejecutar pipeline completo de doblaje con mapeo manual por speaker.

Operación:
- Escribe `input/voice_refs/speaker_voice_map.json` con las asignaciones.
- Ejecuta tarea `vtv` de `pyvideotrans` con diarización activada y separación de audio (`--is_separate`).

Audio mix:
- Calidad (recomendado): usa pista instrumental para minimizar ghost voice.
- Overlay rápido (solo pruebas): superposición sobre original completo.

Salidas esperadas en red:
- Video final doblado (`*_dubbed.mp4`).
- Audio final (`*_dubbed.wav`).
- Subtítulos (`*_subtitles.srt`).
- Contrato temporal (`*_timestamps.json`).
- Artefactos de speaker (`*_speaker_profile.json`, `*_speaker_identity.json`) cuando aplica.

## 5.2 Pestaña 2 - Agregar subtítulos
Objetivo:
- Generar subtítulos desde video o audio.
- Renderizar subtítulos sobre video cuando la fuente base es video.

Reglas vigentes:
- Audio permite generar SRT/JSON.
- Audio no permite render visual de video subtitulado.
- Incluye preview de estilo con extracción de frame y presets.
- Permite import manual de subtítulos para render: `SRT`, `VTT` y `JSON` AssemblyAI normalizado.

## 5.3 Pestaña 3 - Viral Shorts
Objetivo:
- Generar reels verticales desde video final + JSON.

Flujo vigente:
1. Cargar/analizar JSON de reels.
2. Mostrar selección múltiple (checkboxes).
3. Renderizar solo reels seleccionados.
4. Exponer preview, JSON y descarga múltiple de MP4.

Compatibilidad de entrada para análisis/reels:
- `timestamps.json`
- `sentences.json`
- `paragraphs.json`
- JSON de reels ya curado (`reels`)

## 6. Contrato JSON

## 6.1 `timestamps.json`
Campos aceptados por segmento:
- `start` o `start_ms`
- `end` o `end_ms`
- `text_es`
- `speaker` o `speaker_id` opcional

Nota de origen:
- El backend de AssemblyAI persiste automáticamente: `timestamps.json`, `sentences.json`, `paragraphs.json`, `transcript.srt`, `transcript.vtt`, `transcript.txt`.
- `timestamps.json` se mantiene como fuente granular de verdad para reels/captions.

## 6.2 `reels.json`
Estructura:
- `reels`: lista de objetos con `reel_num`, `start_ms`, `end_ms`, `title`, `caption_style`, `priority` opcional
- `metadata` opcional

Regla:
- Si el JSON de entrada ya contiene `reels`, se usa directo (sin reanálisis).

## 7. Stack tecnológico vigente

## 7.1 Docker base (producción)
- Imagen base: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
- Python global del contenedor: 3.10
- PyTorch: `2.8.0` con CUDA 12.8 (`cu128`)

## 7.2 ML/IA principal
- `transformers==4.57.3`
- `qwen-asr==0.0.6`
- `qwen-tts==0.1.1`
- `faster-whisper==1.1.1`
- `modelscope>=1.34.0`
- `funasr>=1.3.1`
- `sherpa-onnx>=1.12.15`

## 7.3 Diarización y clustering
- Dependencias obligatorias en runtime/build: `hdbscan`, `umap-learn`, `datasets`, `simplejson`, `sortedcontainers`, `addict`.
- Modelos ONNX y submodelos de ModelScope predescargados en build de Docker para reducir cold-start.

## 7.4 App y API
- `gradio>=5`
- `fastapi>=0.115`
- `uvicorn>=0.32`

## 8. Variables de entorno operativas relevantes
- `QDP_NETWORK_DIR`
- `QDP_LOCAL_DIR`
- `QDP_TRANSLATE_MODEL` (default funcional: `nllb_local`)
- `QDP_ASR_MODEL` (default funcional: `large-v3-turbo`)
- `PYVIDEOTRANS_MAX_SPEAKERS`
- `API_GOOGLE_STUDIO`
- `GEMINI_MODEL`

## 9. Checklist de release (actual)
1. Validar arranque con Python global: `python main_ui.py`.
2. Verificar Fase 1: detectar speakers y poblar UI de asignación.
3. Verificar fail-fast de diarización: sin `speaker.json` válido debe fallar Fase 1.
4. Verificar Fase 2: salida MP4 + WAV + SRT + `timestamps.json` en carpeta de salida.
5. Verificar Pestaña 2 con video y audio (render solo para video).
6. Verificar Pestaña 3 con selección múltiple de reels y descarga en lote.

## 10. No objetivos
- Mantener flujos legacy duplicados de reels o variantes fuera del flujo en 3 pestañas.
- Reintroducir `.venv` como entorno de ejecución principal.
