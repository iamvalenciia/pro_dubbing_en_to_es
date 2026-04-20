# SPECIFICATION: Quantum Dubbing Pipeline (QDP)

## 1. Visión General de la Arquitectura
El QDP es un pipeline secuencial para doblar videos de inglés a español de forma automatizada para YouTube. 
- **Flujo Principal:** Video Original (EN) -> ASR (EN) -> Traducción (ES) -> TTS Cloning (ES) -> LatentSync (Lip-sync video a audio ES).
- **Audio Final:** El audio resultante será ÚNICAMENTE la voz doblada en español. **NO** se requiere separar el ambiente ni mezclarlo de vuelta. La voz original en inglés se descarta por completo en el mux final.
- **Locutores:** El pipeline asume ESTRICTAMENTE un solo locutor (`SPEAKER_00`). No se debe implementar ni mantener código de diarización múltiple.
- **Hardware Base:** GPU mínima garantizada es **RTX 6000 Ada (48GB VRAM)**. El código puede aprovechar batches grandes sin miedo a GPUs pequeñas.

## 2. Layout de Almacenamiento (Prevención de Cuellos de Botella y OOM)
El sistema opera en un contenedor RunPod con dos discos. El código debe respetar esta división estrictamente para no saturar el disco ni la red:
- **`/runpod-volume/...` (Network Volume - Lento, 40GB+):**
  - `/runpod-volume/models/`: Contiene los pesos pesados (Qwen, Marian, LatentSync).
  - `/runpod-volume/user_input/`: Donde el usuario sube los videos/audios originales vía S3 (AWS CLI).
  - `/runpod-volume/output/`: Donde se copian los resultados finales (el `.mp4` y `.wav` final) para persistencia.
- **`/app/data/...` (Container NVMe Disk - Ultra Rápido, 15GB Límite):**
  - `/app/data/temp_workspace/`: Todos los chunks, wavs clonados, timelines intermedios y recortes de `ffmpeg` DEBEN vivir aquí para máximo IOPS.
- **Garbage Collection:** Al finalizar la Fase 5b o Fase 6 exitosamente, el pipeline DEBE purgar `/app/data/temp_workspace/` para evitar llenar los 15GB del contenedor.

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