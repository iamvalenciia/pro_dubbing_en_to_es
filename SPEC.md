# Quantum Dubbing Pipeline - SPEC Operativo Actual

## 1. Objetivo del sistema
Aplicación web Gradio para producción de doblaje EN → ES con foco en RunPod GPU y paridad funcional local.

Capacidades oficiales:
- Doblaje en 3 fases con asignación manual de voz por speaker.
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
## 5.1 Pestaña 1 - Doblaje Master (flujo en 3 fases)

### Visión general del flujo de doblaje

El pipeline de doblaje opera en **3 fases secuenciales** ejecutadas desde la UI.
Cada fase produce artefactos que la siguiente reutiliza para evitar trabajo duplicado.
El motor de síntesis y ensamble es `pyvideotrans` (ejecutado como subproceso via `cli.py`).

```
Fase 1 (ASR + Diarización con AssemblyAI)
		→ phase1_transcript_diarization.json  (word-level, tiempos en ms, speaker A/B/…)
		→ en.srt                              (en pyvt_target_dir)
		→ speaker.json                        (lista plana speaker_id por segmento)
		→ speaker_profile.json / speaker_identity.json

Fase 2 (Traducción EN→ES con Gemini)
		→ en.srt / es.srt                     (en pyvt_target_dir, preparado para vtv)
		→ translated_segments.json            (en output/, artefacto de recovery)
		→ timestamps_translate.json           (en output/, contrato canónico de recovery)

Fase 3 (TTS Qwen3 clone + mezcla final)
		→ {video}_dubbed.mp4                  (video final con doblaje + cama de audio)
		→ {video}_dubbed.wav                  (PCM 16kHz mono)
		→ {video}_subtitles.srt
		→ timestamps_translate.json actualizado (sobrescrito desde SRT final)
```

---

### Fase 1 — ASR + Diarización (AssemblyAI)

**Disparador:** botón "Ejecutar P1 · ASR + Diarización" en la UI.

**Modo test:** genera un video recortado a 30 s con nombre **estable** (sin timestamp):
```
temp_workspace/qdp_data/temp_processing/test30s_{safe_name}{ext}
```
El nombre es siempre el mismo para el mismo video de entrada → el `pyvt_target_dir` es siempre el mismo.

**Motor:** pyvideotrans `vtv` con modo `analyze`.
Internamente ejecuta: `prepare → recogn → diariz → trans → run_speaker_analysis → ANALYZE_DONE`.

**Diarización:** AssemblyAI (obligatoria). Si falla o no produce speakers válidos,
Fase 1 falla y se aborta. El resultado se escribe en `speaker.json` dentro del
`cache_folder` y `target_dir` de pyvideotrans.

**Directorio de artefactos de Fase 1 (`pyvt_target_dir`):**
```
pyvideotrans/output/{safe_video_name}-mp4/
├── en.srt                              ← transcript inglés en formato SRT
├── speaker.json                        ← lista plana de speaker_id por segmento ["A","A","B"…]
├── speaker_profile.json                ← perfil por speaker con time_ranges y ejemplos
├── speaker_identity.json               ← clasificación Gemini: role/gender/confidence
└── phase1_transcript_diarization.json  ← (también copiado a output/ para recovery)
```

**Formato canónico de `phase1_transcript_diarization.json`:**
Segmentos word-level con tiempos en **milisegundos**:
```json
{
	"segments": [
		{"text": "There", "start": 1230, "end": 1580, "speaker": "A", "confidence": 0.99},
		{"text": "has",   "start": 1600, "end": 1820, "speaker": "A", "confidence": 0.98}
	]
}
```
⚠ `start`/`end` son **ms**, no segundos. No cambiar este formato; todo el pipeline
depende de esta estructura para la resolución de speaker_id en Fases 2 y 3.

**Salida visible en UI:**
- Lista dinámica de speakers (A, B, C…) con preview de audio por speaker.
- Selector de voz de referencia para clonado (mapping speaker → `voice_ref_id`).
- Botón de descarga de `phase1_transcript_diarization.json` para recovery manual.

**Artefacto de recovery:**
`phase1_transcript_diarization.json` se copia a `output/` para que Fases 2 y 3
lo encuentren vía fallback aunque el `target_dir` de pyvideotrans haya rotado.

---

### Fase 2 — Traducción EN→ES (Gemini)

**Disparador:** botón "Ejecutar P2 · Traducción EN→ES" en la UI.

**Precondición:** Fase 1 debe haber completado. Se lee `phase1_transcript_diarization.json`
(en `pyvt_target_dir/` o en `output/` como fallback).

**Modo test:** usa el mismo nombre estable `test30s_{safe_name}{ext}` para derivar
`pyvt_target_dir` → los artefactos se escriben en el **mismo directorio** que Fase 1.

**Flujo interno (en `run_phase2()` de `main_ui.py`):**

1. Lee `phase1_transcript_diarization.json` → reconstruye `source_subs`
	 (lista con `startraw`/`endraw`/`text` por segmento).
2. Llama a Gemini (`vt_run_translate`) para traducir cada segmento EN→ES.
3. Para cada segmento traducido, resuelve `speaker_id` usando la
	 **cadena de fallback de speaker** (§5.4).
4. Escribe en `pyvt_target_dir` (preparando Fase 3 para saltar ASR/traducción):
	 - `en.srt` — transcript inglés en formato SRT
	 - `es.srt` — traducción española en formato SRT
5. Escribe en `output/` (artefactos de recovery):
	 - `translated_segments.json` — segmentos con `speaker_id` resuelto, tiempos en segundos
	 - `timestamps_translate.json` — contrato de recovery canónico para Fase 3

**Variables de entorno inyectadas al proceso hijo:**
- `API_GOOGLE_STUDIO` — clave Gemini (obligatoria)
- `GEMINI_MODEL` — modelo Gemini (default `gemini-3.1-flash-lite-preview`)

**Log de diagnóstico:**
```
[P2] referencia de speaker desde F1 cargada: N segmentos
[P2] speaker fallback F1 aplicado en X/Y segmentos
```
Si este log muestra `X > 0`, significa que los speaker_id de `timestamps.json` eran
`"default"` y fueron resueltos desde Phase 1.

---

### Fase 3 — TTS Qwen3 Clone + Ensamble Final

**Disparador:** botón "Ejecutar P3 · Clonación + Ensamble" en la UI.

**Precondiciones:** Fases 1 y 2 completadas. Se leen:
- `phase1_transcript_diarization.json` (para speaker fallback y reconstruir `en.srt`)
- `timestamps_translate.json` (para reconstruir `es.srt` y `speaker.json`)
- `speaker_voice_map.json` (mapping speaker → archivo WAV de referencia de voz)

#### Modo test: limitación de nombre no estable

En modo test, Fase 3 genera el video recortado con `int(time.time())` en el nombre:
```python
# main_ui.py línea ~2840
test_video_path = f"test30s_{safe_name}_{int(time.time())}.mp4"  # ← NUEVO en cada run
```
Esto hace que `pyvt_target_dir` sea diferente en cada presión del botón:
```
# Run 1 → pyvideotrans/output/test30s_full_video_en_1777709027-mp4/
# Run 2 → pyvideotrans/output/test30s_full_video_en_1777709424-mp4/  ← directorio nuevo
```
Como consecuencia, los WAVs sintetizados por Qwen TTS (que viven en `cache_folder` temporal,
nunca en `target_dir`) **no se reutilizan** entre runs. TTS se re-ejecuta completo en cada
presión del botón de Fase 3 (~5 min para 5 segmentos en GPU).

**Contraste:** Fases 1 y 2 usan nombre estable → mismo `pyvt_target_dir` → acumulan correctamente.

#### Pre-materialización de artefactos antes de lanzar `vtv`

Antes de ejecutar pyvideotrans, `main_ui.py` escribe en el nuevo `pyvt_target_dir`
tres archivos que permiten a `vtv` saltar etapas costosas:

| Archivo materializado | Función en `main_ui.py` | Fuente de datos | Efecto en `vtv` |
|---|---|---|---|
| `en.srt` | `_materialize_phase3_subtitles()` | `phase1_transcript_diarization.json` | Salta ASR/reconocimiento |
| `es.srt` | `_materialize_phase3_subtitles()` | `timestamps_translate.json` | Salta traducción |
| `speaker.json` | `_materialize_phase3_speaker_marks()` | `timestamps_translate.json` + fallback phase1 | Salta diarización |

Esta pre-materialización es la razón por la que los logs de Fase 3 muestran:
```
[P3] artefactos reutilizados para evitar re-ASR/re-traducción: en.srt=… es.srt=… segmentos=5
[P3] speaker.json reutilizado para evitar re-diarización: … (5 marcas)
[DIARIZ-DEBUG] ENTRY: enable_diariz=True … aligned_with_subs=True
[DIARIZ-DEBUG] SKIP: reason=already_valid
```

#### Comando `vtv` lanzado

```
python pyvideotrans/cli.py --task vtv
	--name {video_recortado}
	--source_language_code en  --target_language_code es
	--recogn_type 0  --model_name large-v3-turbo
	--translate_type 5  --tts_type 1
	--cuda  --no-clear-cache  --subtitle_type 0
	--video_autorate  --voice_role clone
	# --enable_diariz/--nums_diariz OMITIDOS cuando speaker.json está disponible
```

La bandera `--enable_diariz` se omite cuando `p3_speaker_count > 0`.
Si `speaker.json` no se pudo materializar, se agrega `--enable_diariz --nums_diariz 0`
y pyvideotrans hará diarización desde cero.

#### TTS (Qwen3-TTS-12Hz-1.7B)

- Modo: `clone` (clonado de voz por speaker).
- Agrupa segmentos por `speaker_id` del `speaker.json` materializado.
- Para cada speaker, usa el WAV de referencia en `input/voice_refs/normalized/`
	mapeado desde `speaker_voice_map.json`.
- Los WAVs generados se escriben en `cache_folder` (temporal, no persistente entre runs).

#### Variables de entorno inyectadas al subproceso `vtv`

| Variable | Valor | Efecto |
|---|---|---|
| `QDP_BACKAUDIO_VOLUME` | float del slider UI | Volumen de cama de audio (0.0–1.0) |
| `QDP_BACKAUDIO_SOURCE` | `"original"` | Extrae cama desde el audio del video original |
| `QDP_ORIGINAL_SOURCE_VIDEO` | ruta al video fuente completo | Cama se extrae del full video, no del recortado |
| `PYVIDEOTRANS_DUCKING_PURE` | `"1"` | No ejecuta separación vocal/instrumental |
| `PYVIDEOTRANS_FAST_NOVOICE_COPY` | `"1"` | Copia stream video sin re-encode (h264/yuv420p) |
| `PYVIDEOTRANS_REMOVE_DUBB_SILENCE` | `"0"` | No elimina silencios del doblaje |
| `PYVIDEOTRANS_DUBBING_THREAD` | `"4"` | Threads paralelos para TTS |
| `PYVIDEOTRANS_FORCE_QWEN_TTS_CUDA` | `"1"` | Fuerza CUDA para Qwen TTS |
| `API_GOOGLE_STUDIO` / `GEMINI_MODEL` | heredados del env | Usados por `_classify_speakers_with_gemini()` |

`QDP_DUBBED_AUDIO_VOLUME` **no se inyecta** → siempre usa el default de `trans_create.py` (1.05).

#### Salidas publicadas en `output/`

| Archivo | Descripción |
|---|---|
| `{video}_dubbed.mp4` | Video final con doblaje + cama de audio mezclados |
| `{video}_dubbed.wav` | Audio doblado PCM 16kHz mono |
| `{video}_subtitles.srt` | Subtítulos español del video final |
| `{video}_timestamps.json` | JSON de timestamps del SRT final |
| `timestamps_translate.json` | Sobrescrito con timestamps del SRT final (recovery) |

---

### §5.4 Cadena de resolución de speaker_id

Tanto Fase 2 como la pre-materialización de Fase 3 usan la misma cadena de niveles
(`_resolve_speaker_from_phase1()` en `main_ui.py`) para asignar `speaker_id`
cuando el valor actual es nulo o "default".

**`_is_default_speaker_id(value)`** detecta todos los tokens que significan "sin speaker":
`""`, `"default"`, `"unknown"`, `"unk"`, `"none"`, `"null"`, `"n/a"`, `"na"`

**`_load_phase1_speaker_reference(phase1_json_path, *, test_mode)`:**
- Lee `phase1_transcript_diarization.json`.
- Devuelve lista de `{idx, start_ms, end_ms, speaker_id}`.
- En modo test, omite entradas con `start_ms >= 30000`.

**`_resolve_speaker_from_phase1(idx, start_ms, end_ms, phase1_refs, current_speaker)`:**

1. Si `current_speaker` ya es no-default → devuelve sin cambio.
2. **Index-aligned:** busca `phase1_refs[idx]` → usa su `speaker_id` si es no-default.
3. **Time-overlap:** recorre todos los refs y busca el de mayor solapamiento temporal
	 con `[start_ms, end_ms]` del segmento actual.
4. **Primer no-default:** usa el primer `speaker_id` no-default encontrado en `phase1_refs`.
5. Si todo falla → devuelve `"default"`.

Esta cadena garantiza que incluso si `timestamps_translate.json` tiene todos los
`speaker_id` como `"default"`, los segmentos de Fase 3 recibirán los speaker reales
(A, B, C…) de Phase 1 y la clonación de voz funcionará correctamente.

---

### §5.5 Mezcla de audio (trans_create.py)

Implementada en `_back_music()` y `_backandvocal()` de
`pyvideotrans/videotrans/task/trans_create.py`.

**Resolvedores de volumen:**
```python
def _resolve_backaudio_volume(default_value=0.28):
		# Lee QDP_BACKAUDIO_VOLUME; clamp 0.0–1.0
		# Este valor se inyecta desde el slider UI de Phase 3

def _resolve_dubbed_audio_volume(default_value=1.05):
		# Lee QDP_DUBBED_AUDIO_VOLUME; clamp 0.0–3.0
		# NO expuesto en UI; para cambiar: setear env var antes de lanzar main_ui.py
		# o cambiar default_value en trans_create.py
```

**Filtro FFmpeg de mezcla final:**
```
[0:a]volume={dubbed_vol}[dub];[dub][1:a]amix=inputs=2:duration=first:dropout_transition=2
```
Donde:
- `[0:a]` = audio doblado generado por Qwen TTS
- `[1:a]` = cama de audio (del video original, ya reducida a `backaudio_vol`)
- `dropout_transition=2` = fade de 2s al final de la mezcla

**Proceso completo de `_back_music()` cuando `QDP_BACKAUDIO_SOURCE=original`:**
1. Lee `QDP_ORIGINAL_SOURCE_VIDEO` para obtener la ruta al video fuente completo.
2. Extrae audio completo del video fuente a WAV 44100Hz stereo (`original_full_bed.wav`).
3. Aplica `volume={backaudio_vol}` a la cama → `bgm_file_extend_volume.wav`.
4. Mezcla: `[doblaje × dubbed_vol] amix [cama reducida]` → `lastend.wav`.
5. `PYVIDEOTRANS_DUCKING_PURE=1` suprime la separación vocal/instrumental previa.

---

### §5.6 Artefactos de recovery (contrato canónico)

Para re-ejecutar Fase 3 sin repetir Fases 1 y 2, los siguientes archivos deben
existir en `output/` antes de lanzar Fase 3:

| Archivo en `output/` | Requerido para | Generado por |
|---|---|---|
| `phase1_transcript_diarization.json` | Speaker fallback, reconstruir `en.srt` | Fase 1 (copiado a output/) |
| `timestamps_translate.json` | Reconstruir `es.srt` y `speaker.json` | Fase 2 / sobrescrito por Fase 3 |

Si estos archivos existen, la UI los carga vía el diálogo de recovery (`btn_recovery`)
y Fase 3 puede ejecutarse de forma standalone (sin re-ejecutar Fases 1 y 2).

---

### §5.7 Limitación conocida: re-síntesis TTS en cada run de Fase 3 (test mode)

**Causa:** En test mode, Fase 3 genera un video temporal con `int(time.time())` en el nombre.
Esto produce un nuevo `pyvt_target_dir` y un nuevo `cache_folder` vacío en cada run.
Los WAVs de Qwen TTS solo existen en `cache_folder` (efímero), nunca en `target_dir` (persistente).

**Resultado:** TTS re-sintetiza todos los segmentos en cada run (~5 min para 5 segs en GPU),
incluso si solo se cambió el volumen del fondo de audio.

**Contraste con Fases 1 y 2:** usan nombre estable → mismo `pyvt_target_dir` → SRTs y
`speaker.json` se reutilizan correctamente entre runs.

**Workaround actual:** ninguno automático. Si solo se cambia `QDP_BACKAUDIO_VOLUME`,
hay que esperar la re-síntesis TTS completa.

**Corrección pendiente (no implementada):** usar nombre estable en Fase 3 test mode,
O detectar si ya existe el MP4 final en `pyvt_target_dir` y saltar directamente al ensamble final.

---

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

### 8.1 Storage y rutas
- `QDP_NETWORK_DIR` — ruta base de volumen de red (RunPod: `/runpod-volume`)
- `QDP_LOCAL_DIR` — ruta base de almacenamiento local rápido (RunPod: `/workspace/qdp_data`)

### 8.2 Modelos y backends
- `QDP_TRANSLATE_MODEL` — backend de traducción (default `gemini`)
- `QDP_ASR_MODEL` — modelo Whisper/ASR (default `large-v3-turbo`)
- `PYVIDEOTRANS_MAX_SPEAKERS` — máximo de speakers en UI (default 50, rango 12–100)
- `GEMINI_MODEL` — modelo Gemini para traducción e identidad (default `gemini-3.1-flash-lite-preview`)
- `API_GOOGLE_STUDIO` — API key de Google AI Studio (obligatoria para Fases 2 y 3)
- `ASSEMBLYAI_API_KEY` — API key de AssemblyAI (obligatoria para Fase 1)

### 8.3 Audio mix (Phase 3 / trans_create.py)
- `QDP_BACKAUDIO_VOLUME` — volumen de cama de audio (0.0–1.0, default 0.28). Controlado desde slider UI.
- `QDP_DUBBED_AUDIO_VOLUME` — volumen del doblaje antes del mix (0.0–3.0, default 1.05). **No expuesto en UI.**
- `QDP_BACKAUDIO_SOURCE` — fuente de la cama: `"original"` (extrae del video fuente) o `"file"`. Default `"original"`.
- `QDP_ORIGINAL_SOURCE_VIDEO` — ruta al video original completo para extraer cama en test mode.
- `QDP_MIN_DUB_AUDIO_RATIO` — ratio mínimo duración audio/video antes de fallar con error (default 0.20).

### 8.4 Pipeline pyvideotrans (Phase 3)
- `PYVIDEOTRANS_DUCKING_PURE` — si `"1"`, omite separación vocal/instrumental (inyectado como `"1"` por UI).
- `PYVIDEOTRANS_FAST_NOVOICE_COPY` — si `"1"`, copia stream de video sin re-encode para h264/yuv420p (inyectado como `"1"`).
- `PYVIDEOTRANS_REMOVE_DUBB_SILENCE` — si `"1"`, elimina silencios del doblaje (inyectado como `"0"`).
- `PYVIDEOTRANS_DUBBING_THREAD` — threads paralelos para TTS (inyectado como `"4"`).
- `PYVIDEOTRANS_FORCE_QWEN_TTS_CUDA` — fuerza GPU para Qwen TTS (inyectado como `"1"`).
- `PYVIDEOTRANS_DIARIZ_BACKEND` — backend de diarización (solo `assemblyai` soportado).
- `PYVIDEOTRANS_DIARIZ_STRICT_GPU` — falla en vez de caer a CPU para diarización (default `True`).
- `PYVIDEOTRANS_PYTHON` — ruta al Python a usar para el subproceso `vtv` (default `sys.executable`).

## 9. Checklist de release (actual)
1. Validar arranque con Python global: `python main_ui.py`.
2. Verificar Fase 1 (test mode): detectar speakers, poblar UI de asignación,
	 verificar `phase1_transcript_diarization.json` en `output/` con `speaker` = "A"/"B"/etc.
3. Verificar fail-fast de diarización: sin `speaker.json` válido debe fallar Fase 1 (no avanzar a Fase 2).
4. Verificar Fase 2 (test mode): `timestamps_translate.json` en `output/` con
	 `speaker_id` no-default. Log debe mostrar `[P2] speaker fallback F1 aplicado en X/Y segmentos`.
5. Verificar Fase 3 (test mode): logs deben mostrar:
	 - `[DIARIZ-DEBUG] SKIP: reason=already_valid`
	 - `[P3] artefactos reutilizados para evitar re-ASR/re-traducción`
	 - Video final en `output/`.
6. Verificar mezcla de audio: fondo audible con volumen reducido, doblaje claro sobre fondo.
	 Ajustar slider de volumen fondo y confirmar que `QDP_BACKAUDIO_VOLUME` se inyecta correctamente.
7. Verificar Pestaña 2 con video y audio (render solo para video).
8. Verificar Pestaña 3 con selección múltiple de reels y descarga en lote.

## 10. Reglas de implementación (no romper)

- **No cambiar el nombre estable del video test en Fases 1 y 2.**
	Fases 1 y 2 usan `test30s_{safe_name}{ext}` (sin timestamp) para que el `pyvt_target_dir`
	sea siempre el mismo y los artefactos se acumulen correctamente entre re-runs.

- **No cambiar la estructura de `phase1_transcript_diarization.json`.**
	Tiempos en ms, campo `speaker` = "A"/"B"/etc., nivel word. Es la fuente canónica de
	verdad para toda la cadena de resolución de speaker_id en Fases 2 y 3.

- **No eliminar `_resolve_speaker_from_phase1()` ni `_load_phase1_speaker_reference()`.**
	Son las funciones que propagan los speaker_id reales de Fase 1 a Fases 2 y 3.
	Sin ellas, `speaker.json` tendrá solo `"default"` y la clonación de voz
	seleccionará referencias incorrectas o fallará silenciosamente.

- **No modificar `_materialize_phase3_subtitles()` ni `_materialize_phase3_speaker_marks()`**
	sin entender que son el "puente" que permite a pyvideotrans saltarse ASR/traducción/diarización.
	Cualquier cambio que elimine o rompa la escritura de `en.srt`/`es.srt`/`speaker.json`
	antes de llamar a `vtv` provocará que pyvideotrans re-ejecute todo desde cero.

- **No agregar `--enable_diariz` cuando `speaker.json` está disponible.**
	La bandera se omite del comando `vtv` precisamente cuando `p3_speaker_count > 0`.
	Agregar `--enable_diariz` siempre causaría llamadas innecesarias a AssemblyAI en cada run de Fase 3.

- **`QDP_DUBBED_AUDIO_VOLUME` no está expuesto en UI.**
	Para cambiar el boost del doblaje: setear la variable de entorno antes de lanzar
	`main_ui.py` (`$env:QDP_DUBBED_AUDIO_VOLUME = "1.0"`) o cambiar `default_value`
	en `_resolve_dubbed_audio_volume()` en `trans_create.py`.

- **No reintroducir `.venv` como entorno de ejecución principal.**

## 11. No objetivos
- Mantener flujos legacy duplicados de reels o variantes fuera del flujo en 3 pestañas.
- Reintroducir `.venv` como entorno de ejecución principal.
- Soportar backends de diarización distintos a AssemblyAI.
