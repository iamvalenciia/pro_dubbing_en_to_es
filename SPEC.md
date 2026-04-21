# SPECIFICATION: Quantum Dubbing Pipeline (QDP)

## 1. Visión General de la Arquitectura
El QDP es un pipeline secuencial para doblar videos de inglés a español de forma automatizada para YouTube. 
- **Flujo Principal:** Video Original (EN) -> ASR (Faster-Whisper / WhisperX) -> Traducción (ES) -> TTS Cloning (ES) -> LatentSync (Lip-sync video a audio ES).
- **Audio Final:** El audio resultante será ÚNICAMENTE la voz doblada en español. **NO** se requiere separar el ambiente ni mezclarlo de vuelta. La voz original en inglés se descarta por completo en el mux final.
- **Locutores:** El pipeline asume ESTRICTAMENTE un solo locutor (`SPEAKER_00`). No se debe implementar ni mantener código de diarización múltiple.
- **Hardware Base:** GPU mínima garantizada es **RTX 6000 Ada (48GB VRAM)**. El código puede aprovechar batches grandes sin miedo a GPUs pequeñas.

## 2. Layout de Almacenamiento (Prevención de Cuellos de Botella y OOM)
El sistema opera en un contenedor RunPod que ha sido actualizado a **50GB de NVMe Container Disk** y un **Network Volume** persistente. El código NO debe usar rutas relativas. Debe definir y usar rutas absolutas para separar estrictamente el almacenamiento en red del procesamiento local (NVMe):

- **ENTORNO DE RED (Lento, Persistente): `NETWORK_DIR = "/runpod-volume"`**
  - `NETWORK_DIR/models/`: Contiene los pesos (Whisper, Marian, Qwen-TTS, LatentSync). El código lee los modelos directamente de aquí.
  - `NETWORK_DIR/user_input/`: Donde el usuario sube los originales. Fase 0 debe COPIAR estos archivos al entorno local, no usar symlinks.
  - `NETWORK_DIR/output/`: Solo para persistencia final. El archivo final se copia aquí tras terminar el proceso.

- **ENTORNO LOCAL NVMe (Ultra Rápido, Efímero, 50GB): `LOCAL_DIR = "/workspace/qdp_data"` (o la ruta raíz del contenedor equivalente)**
  - `LOCAL_DIR/input/`: Donde se copia el video/audio original antes de procesar.
  - `LOCAL_DIR/temp_workspace/`: TODOS los chunks de ASR, WAVs de TTS, timelines JSON y renders intermedios de FFmpeg DEBEN escribirse aquí.
  - `LOCAL_DIR/output/`: Donde se hace el MUX final del video y audio. La UI de Gradio servirá las descargas desde aquí para evitar timeouts de red.
  - `LOCAL_DIR/logs/` y `LOCAL_DIR/torch_cache/`: Los logs y cachés temporales deben operar localmente.

## 3. Contrato de Datos (El Timeline JSON)
El objeto JSON es la fuente de la verdad. Nace en la Fase 2 gracias a **Whisper**, el cual debe agrupar los segmentos con lógica gramatical (comas, puntos) para evitar fragmentos cortados a la mitad. Su propósito es estructurar el doblaje en español.
- **Estructura inmutable esperada por las fases finales:**
  ```json
  [
    {
      "segment_id": 0,
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 4.5,
      "original_duration": 4.5,
      "text_en": "Hello world", 
      "text_es": "Hola mundo", 
      "cloned_audio_path": "/app/data/temp_workspace/p4/cloned_00000.wav",
      "dubbed_duration": 1.2
    }
  ]
  ```

## 4. Restricción Dura: Sincronización y Tiempos (Isochrony)

- **Regla de Oro:** El audio doblado en español (`dubbed_duration`) NUNCA DEBE SUPERAR la duración del segmento original en inglés (`original_duration`).

- **Prohibición de Time-Stretching:** Queda ESTRICTAMENTE PROHIBIDO usar algoritmos de "aceleración sutil" (ej. `librosa.effects.time_stretch`). Esto provoca que el audio se sienta robotizado.

- **Tolerancia y Hard-Cap:** Es aceptable que el audio en español dure menos. Sin embargo, si el TTS alucina y el tensor generado supera `original_duration` (especialmente si es >1.4x), se debe aplicar un recorte brutal (trim en seco) al límite exacto de `original_duration` usando `numpy`. Cero distorsión robótica.

## 5. Tolerancia a Fallos y Manejo de Errores

- **Fase 0 (Inputs):** Los inputs se leerán principalmente desde rutas absolutas del volumen de red (ej. `/runpod-volume/user_input/video.mp4`). No se requiere lógica de reintentos para descargas de red externas.

- **ASR & TTS Hallucinations (Zero Tolerance):** Exigimos máxima calidad. Si en la Fase 2 (ASR con Whisper) el modelo empieza a loopear, o el TTS genera ruido que requiere ser cortado masivamente por el Runaway Guard, el código debe lanzar un `RuntimeError`, abortar el pipeline y notificar a la UI. No se admiten fallbacks de baja calidad.

- **Liberación de Memoria:** Entre fases (especialmente entre Fase 2, 3 y 4), los modelos deben borrarse de memoria explícitamente (del `model` seguido de `torch.cuda.empty_cache()`) para garantizar que el siguiente modelo tenga los 48GB enteros disponibles.

## . Formatos y Resoluciones

- **Video/Audio:** El sistema debe aceptar cualquier resolución (4K, 1080p, 720p) y cualquier sample rate de entrada. El pipeline interno normalizará el audio a 16kHz internamente solo cuando los modelos lo requieran explícitamente, pero el video final mantendrá la resolución original.

## 7. User Interface (UI) & Interaction Flow ("Apple-Style Minimalism")

La interfaz se construirá con Gradio usando un diseño minimalista de tres pestañas (Tabs) principales.

- **Inputs (Explorador de Archivos en Red):** La UI usará menús desplegables (`gr.Dropdown`) que leerán los videos originales directamente de `NETWORK_DIR/user_input/`.

- **UI Hints (AWS S3):** Se incluirá un "Acordeón" desplegable (`gr.Accordion`) llamado "¿Cómo subir archivos?" con los comandos de AWS S3 CLI para ingestar data al volumen de red.

- **Feedback Visual:** Un texto que muestre la Fase Activa actual y un emulador de Terminal (`gr.Textbox` con auto-scroll) mostrando los logs en tiempo real.

- **Salidas Modulares (Outputs):** Para evitar pérdida de datos ante fallos en fases críticas (Fase 6), el sistema DEBE exponer el **Audio Doblado (.wav)** para su descarga inmediata al concluir la Fase 5, independientemente de si el proceso de video falla o continúa. El video final se mostrará en un reproductor (`gr.Video`) al terminar.

## 8. Workflows, Ajustes y Modo de Prueba (Settings)

La UI se divide en 3 Pestañas (Tabs) correspondientes a 3 Workflows independientes, gobernados por ajustes globales.

### Ajustes Globales (Toggles en UI)
- **"Test Mode (30 Segundos Límite)":** Inyecta una regla en FFmpeg para truncar el procesamiento a los primeros 30 segundos del video original (aplica para todos los workflows).

- **"Renderizar en 4K (Original)":** La Fase 5b/6 debe respetar estrictamente la resolución original del input.

### Pestaña 1: Doblaje Master (End-to-End)

Produce el video horizontal completo, con audio doblado y LipSync.

- **Flujo:** Fase 0 hasta Fase 6 sin interrupciones.

- **Toggles Específicos:** "Habilitar LatentSync (Lip-sync)" (Activo por defecto). Si se apaga, salta la Fase 6.

- **Entregables:** Audio `.wav` (disponible tras Fase 5) y Video `.mp4` (al finalizar).

### Pestaña 2: Standalone Lip-Sync (Herramienta de Recuperación/Producción)

Ejecuta únicamente la Fase 6 para sincronizar un audio ya doblado con un video original.

- **Inputs:** 1. Video Original (Seleccionado vía `gr.Dropdown` desde la red).
  2. Audio Doblado en Español (Permite carga directa desde el navegador web vía `gr.Audio` o `gr.File`, ya que los audios pesan poco).

- **Flujo:** Extrae el video, normaliza el audio subido y ejecuta LatentSync directamente.

- **Entregable:** Video final sincronizado (`.mp4`).

### Pestaña 3: AI Viral Clips / Shorts (>3 min)

Genera recortes verticales a partir del Master final.

- **Paso 1 (AI Analista):** Consume el `master_timeline_with_audio.json`. Un LLM extrae segmentos de alto valor y retención, con una duración mínima estricta de 3 minutos por clip.

- **Paso 2 (Listado UI):** Genera un plan de recortes. La UI muestra este listado con checkboxes para que el usuario seleccione cuáles renderizar.

- **Paso 3 (Render):** Recorta el Master, aplica formato vertical (Crop 9:16) e inyecta subtítulos amarillos estilo viral en el centro de la pantalla.