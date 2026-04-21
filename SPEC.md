# SPECIFICATION: Quantum Dubbing Pipeline (QDP)

## 1. Visión General de la Arquitectura
El QDP es un pipeline secuencial para doblar videos de inglés a español de forma automatizada para YouTube. 
- **Flujo Principal:** Video Original (EN) -> ASR (EN) -> Traducción (ES) -> TTS Cloning (ES) -> LatentSync (Lip-sync video a audio ES).
- **Audio Final:** El audio resultante será ÚNICAMENTE la voz doblada en español. **NO** se requiere separar el ambiente ni mezclarlo de vuelta. La voz original en inglés se descarta por completo en el mux final.
- **Locutores:** El pipeline asume ESTRICTAMENTE un solo locutor (`SPEAKER_00`). No se debe implementar ni mantener código de diarización múltiple.
- **Hardware Base:** GPU mínima garantizada es **RTX 6000 Ada (48GB VRAM)**. El código puede aprovechar batches grandes sin miedo a GPUs pequeñas.

## 2. Layout de Almacenamiento (Prevención de Cuellos de Botella y OOM)
El sistema opera en un contenedor RunPod que ha sido actualizado a **50GB de NVMe Container Disk** y un **Network Volume** persistente. El código NO debe usar rutas relativas. Debe definir y usar rutas absolutas para separar estrictamente el almacenamiento en red del procesamiento local (NVMe):

- **ENTORNO DE RED (Lento, Persistente): `NETWORK_DIR = "/runpod-volume"`**
  - `NETWORK_DIR/models/`: Contiene los pesos (Qwen, Marian, LatentSync). El código lee los modelos directamente de aquí.
  - `NETWORK_DIR/user_input/`: Donde el usuario sube los originales. Fase 0 debe COPIAR estos archivos al entorno local, no usar symlinks.
  - `NETWORK_DIR/output/`: Solo para persistencia final. El archivo final se copia aquí tras terminar el proceso.

- **ENTORNO LOCAL NVMe (Ultra Rápido, Efímero, 50GB): `LOCAL_DIR = "/workspace/qdp_data"` (o la ruta raíz del contenedor equivalente)**
  - `LOCAL_DIR/input/`: Donde se copia el video/audio original antes de procesar.
  - `LOCAL_DIR/temp_workspace/`: TODOS los chunks de ASR, WAVs de TTS, timelines JSON y renders intermedios de FFmpeg DEBEN escribirse aquí.
  - `LOCAL_DIR/output/`: Donde se hace el MUX final del video y audio. La UI de Gradio servirá las descargas desde aquí para evitar timeouts de red.
  - `LOCAL_DIR/logs/` y `LOCAL_DIR/torch_cache/`: Los logs y cachés temporales deben operar localmente.

## 3. Contrato de Datos (El Timeline JSON)
El objeto JSON es la fuente de la verdad. Aunque nace en la Fase 2 (ASR), su propósito es estructurar el doblaje en español.
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
- **Tolerancia:** Es 100% aceptable que el audio en español dure MENOS que el original. Las pausas son manejables, pero un audio más largo provocará pantallas negras en YouTube.
- **Implementación requerida en Fase 4 (TTS):** El Runaway Guard no solo debe revisar los caracteres por segundo, sino aplicar un Hard-Cap: Si el tensor generado supera `original_duration`, se debe truncar (trim) brutalmente al límite de `original_duration` usando `numpy`.

## 5. Tolerancia a Fallos y Manejo de Errores
- **Fase 0 (Inputs):** Los inputs se leerán principalmente desde rutas absolutas del volumen de red (ej. `/runpod-volume/user_input/video.mp4`). No se requiere lógica compleja de reintentos para descargas de red externas.
- **ASR & TTS Hallucinations (Zero Tolerance):** Exigimos máxima calidad. Si en la Fase 2 (ASR) el modelo empieza a loopear (ej. repite la misma palabra más de 3 veces) o el TTS genera ruido que requiere ser cortado repetidamente por el Runaway Guard, el código debe lanzar un `RuntimeError`, abortar el pipeline y notificar a la UI. No se admiten fallbacks de baja calidad.
- **Liberación de Memoria:** Entre fases (especialmente entre Fase 2, 3 y 4), los modelos deben borrarse de memoria explícitamente (del `model` seguido de `torch.cuda.empty_cache()`) para garantizar que el siguiente modelo tenga los 48GB enteros disponibles.

## 6. Formatos y Resoluciones
- **Video/Audio:** El sistema debe aceptar cualquier resolución (4K, 1080p, 720p) y cualquier sample rate de entrada. El pipeline interno normalizará el audio a 16kHz internamente solo cuando los modelos lo requieran explícitamente, pero el video final mantendrá la resolución original.


## 7. User Interface (UI) & Interaction Flow ("Apple-Style Minimalism")
La interfaz se construirá con Gradio bajo una filosofía de diseño minimalista. 
- **Flujo Principal (One-Click Magic):** Un gran botón de "Ejecutar Doblaje" que dispara el pipeline de Fase 0 a Fase 6 sin pausas, leyendo desde NVMe y entregando el MP4 final.
- **Inputs (Explorador de Archivos):** No hay subida directa por navegador web. La UI usará menús desplegables (`gr.Dropdown`) que leerán los archivos directamente de `NETWORK_DIR/user_input/`.
- **UI Hints (AWS S3):** Cerca de los inputs, debe haber un componente tipo "Acordeón" desplegable (`gr.Accordion`) llamado "¿Cómo subir archivos?" que muestre textualmente estos comandos de AWS para que el usuario sepa cómo ingestar data al volumen de red:
  ```bash
  aws s3 ls --region us-ks-2 --endpoint-url [https://s3api-us-ks-2.runpod.io](https://s3api-us-ks-2.runpod.io) s3://x73d6lzlpq/
  aws s3 cp tu_video.mp4 s3://x73d6lzlpq/user_input/ --region us-ks-2 --endpoint-url [https://s3api-us-ks-2.runpod.io](https://s3api-us-ks-2.runpod.io)
      ```

- **Feedback Visual:** 
  1. Un texto o componente grande que muestre la Fase Activa actual (ej. "Fase 4: Clonación de Voz (45%)").
  2. Un emulador de Terminal (`gr.Textbox` o componente de log en tiempo real) que muestre la salida del logger del sistema.
- **Salidas (Outputs):** Un reproductor de video (`gr.Video`) para previsualizar el MP4 final y permitir su descarga. No se expondrán botones para audios sueltos ni JSONs.

## 8. Workflows, Ajustes y Modo de Prueba (Settings)

La UI tendrá una pequeña sección de ajustes (Toggles) antes de iniciar, para orquestar 3 Workflows distintos.

### Ajustes Globales (Toggles en UI)
- **"Habilitar LatentSync (Lip-sync)" (Activo por defecto):** Si se apaga, el pipeline omite la Fase 6 y entrega el video solo con el audio doblado superpuesto.
- **"Test Mode (30 Segundos Límite)":** Si está activo, el sistema debe inyectar una regla dura en la Fase 0/2 que TRUNQUE el procesamiento a los primeros 30 segundos del video original. Aplica para todos los workflows (video completo, shorts y subtítulos).
- **"Renderizar en 4K (Original)":** La Fase 5b/6 debe respetar estrictamente la resolución original del input, escalando los procesos si es necesario.

### The 3 Workflows
- **Workflow 1: Master Clean Video (Principal):** Produce el video horizontal completo, con audio doblado y LipSync. Estrictamente limpio (sin subtítulos quemados).
- **Workflow 1.1: Master con Subtítulos (Opcional):** Un Toggle en la UI que, tras generar el Clean Video, dispara un proceso de FFMPEG para quemar subtítulos de una sola línea en formato horizontal.
- **Workflow 2: AI Viral Clips / Shorts (>3 min):**
  - **Paso 1 (AI Analista):** Se toma el `master_timeline_with_audio.json`. Una IA (ej. Gemini o Claude) lo analiza para extraer ideas centrales y ganchos fuertes (hooks). La IA está configurada para extraer segmentos que duren mínimo 3 minutos.
  - **Paso 2 (Listado UI):** Genera un `shorts_plan.json` (título, start, end). La UI muestra este listado con checkboxes.
  - **Paso 3 (Render):** El sistema recorta el video Master Clean en las marcas de tiempo seleccionadas, aplica un "Crop" a formato vertical (9:16), e inyecta subtítulos (captions) de color amarillo centrados en la pantalla.