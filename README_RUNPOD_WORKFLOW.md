# RunPod Workflow Definitivo (Pod + Network Volume + Subidas desde PC)

Esta guia esta pensada para tu flujo real:
- Encender Pod
- Procesar doblaje
- Descargar resultados
- Apagar Pod
- Conservar insumos/modelos en Network Volume

Asi no necesitas reconstruir imagen cada vez ni perder archivos al terminar la instancia.

## 1) Idea clave

En RunPod hay dos zonas de almacenamiento:

1. Network Volume (persistente)
- Sobrevive cuando apagas o terminas Pods.
- Lo accedes desde tu PC por endpoint S3 compatible.
- Ideal para: user_input, output_final, modelos cacheados.

2. Disco local del Pod (efimero)
- Rapido, pero se pierde al terminar el Pod.
- Ideal para temporales de procesamiento.

Recomendacion practica:
- Subir insumos y guardar finales en Network Volume.
- Usar disco local solo para temporales.

## 2) Variables sugeridas en tu PC (PowerShell)

Ajusta una vez por sesion:

$env:RUNPOD_S3_ENDPOINT = "https://s3api-us-ks-2.runpod.io"
$env:RUNPOD_REGION = "us-ks-2"
$env:RUNPOD_BUCKET = "l9dt5rqorw"

Opcional (si quieres simplificar comandos):

$env:AWS_DEFAULT_REGION = $env:RUNPOD_REGION

## 3) Verificar contenido actual del Network Volume

aws s3 ls s3://$env:RUNPOD_BUCKET/ --recursive --human-readable --summarize --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

En tu caso, ya viste que existe:
- user_input/full_video_en.mp4

Eso significa que el video esta listo para ser referenciado en el flujo de doblaje.

## 4) Subir archivos desde tu computadora al Network Volume

### 4.1 Subir un archivo puntual de entrada

aws s3 cp "C:\ruta\a\tu\video.mp4" "s3://$env:RUNPOD_BUCKET/user_input/full_video_en.mp4" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

### 4.2 Subir carpeta completa de entradas

aws s3 sync "C:\ruta\a\inputs" "s3://$env:RUNPOD_BUCKET/user_input/" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

### 4.3 Subir modelos/cache predescargados

aws s3 sync "C:\ruta\a\models" "s3://$env:RUNPOD_BUCKET/models/" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

## 5) Descargar resultados del Network Volume a tu PC

### 5.1 Descargar salida final completa

aws s3 sync "s3://$env:RUNPOD_BUCKET/output_final/" "C:\ruta\local\output_final" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

### 5.2 Descargar un archivo especifico

aws s3 cp "s3://$env:RUNPOD_BUCKET/output_final/mi_video_doblado.mp4" "C:\ruta\local\mi_video_doblado.mp4" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

## 6) Comandos base para manejar Pods (runpodctl)

### 6.1 Ver Pods

runpodctl pod list
runpodctl pod list --all

### 6.2 Ver detalle de un Pod

runpodctl pod get POD_ID

### 6.3 Crear Pod con tu imagen y Network Volume

runpodctl pod create --name "qdp-dubbing" --image "TU_USUARIO/TU_IMAGEN:latest" --gpu-id "NVIDIA GeForce RTX 4090" --container-disk-in-gb 50 --network-volume-id "TU_NETWORK_VOLUME_ID" --volume-mount-path "/runpod-volume" --ports "7860/http,22/tcp" --env '{"QDP_NETWORK_DIR":"/runpod-volume","QDP_LOCAL_DIR":"/workspace/qdp_data","HF_HOME":"/workspace/qdp_data/torch_cache","TORCH_HOME":"/workspace/qdp_data/torch_cache"}'

Nota:
- Si no quieres montar volumen ahora, puedes omitir network-volume-id y volume-mount-path.
- Pero para flujo definitivo, si conviene montarlo.

### 6.4 Encender, apagar, reiniciar, borrar

runpodctl pod start POD_ID
runpodctl pod stop POD_ID
runpodctl pod restart POD_ID
runpodctl pod delete POD_ID

## 7) Flujo diario recomendado (simple y estable)

1. Subir video a user_input desde tu PC.
2. Encender Pod existente (o crear uno si no existe).
3. Ejecutar doblaje en la app.
4. Verificar que salida quede en output_final.
5. Descargar resultados con aws s3 sync/cp.
6. Apagar Pod para no pagar compute innecesario.

Con esto ya no necesitas reconstruir imagen en cada corrida.

## 8) Buenas practicas para evitar reprocesos

- Mantener nombres estables:
  - user_input/full_video_en.mp4
  - output_final/...
- Guardar modelos en models/ dentro del bucket.
- Usar disco local del Pod solo para temporales.
- Antes de apagar, verificar que output_final ya este en Network Volume.

## 9) Troubleshooting rapido

1. Error de credenciales AWS CLI
- Ejecuta aws configure o define AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY para tu S3 de RunPod.

2. No aparece archivo al listar
- Revisa bucket, region y endpoint.
- Ejecuta de nuevo el ls con --summarize.

3. El Pod inicia pero no encuentras archivos
- Confirma que el Pod fue creado con network-volume-id y volume-mount-path correctos.
- Confirma variables QDP_NETWORK_DIR y QDP_LOCAL_DIR.

4. Quieres conservar trabajo entre sesiones
- Guarda insumos/modelos/salidas en Network Volume.
- No dependas del disco local del Pod para persistencia.

## 10) Comandos listos para tu caso actual

Listar todo:

aws s3 ls s3://l9dt5rqorw/ --recursive --human-readable --summarize --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io

Subir un video nuevo:

aws s3 cp "C:\ruta\local\full_video_en.mp4" "s3://l9dt5rqorw/user_input/full_video_en.mp4" --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io

Descargar todos los resultados:

aws s3 sync "s3://l9dt5rqorw/output_final/" "C:\ruta\local\output_final" --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io

---

## 11) PERFORMANCE & GPU TUNING EN RUNPOD (Optimizado Abril 2026)

### 11.1 Por qué puede estar lento en RunPod

Si sientes que todo va lento **al iniciar a doblar audio**, las causas más comunes son:

1. **Doble descarga de modelos Qwen3-TTS** (~30 segundos)
   - Ya fue corregida en commit 51b7549 (modelo cache singleton)

2. **Configuración conservadora de GPU** (heredada de cfg.json defaults)
   - process_max_gpu=1 (solo 1 tarea de GPU concurrente)
   - trans_thread=10 (traducción sin batching agresivo)
   - aitrans_thread=50 (threads para traducción IA)

3. **Sin aceleración de batching**
   - NLLB-200: default batch=10, debería ser 256
   - Qwen3-TTS: default batch=2, debería ser 12

### 11.2 Nueva imagen Docker (v25+) con GPU tuning baked-in

**A partir del commit ee72e0f, la imagen Docker incluye:**

```dockerfile
# ENV variables para batching agresivo
ENV PYVIDEOTRANS_NLLB_BATCH_LINES=256
ENV PYVIDEOTRANS_QWEN_TTS_BATCH_LINES=12
ENV PYVIDEOTRANS_PROCESS_MAX_GPU=2

# cfg.json actualizado automáticamente en build:
# process_max_gpu=2    (2 tareas de GPU concurrentes)
# trans_thread=10      (traducción con threading)
# aitrans_thread=50    (hilos para traducción IA)
```

**Impacto esperado:**
- Primera ejecución: ~30s más rápido (no double-download)
- Cada batch de traducción: 2-3x más rápido
- Audio dubbing: 1.5-2x más rápido en GPUs con 24GB+ VRAM

### 11.3 Cómo usar la imagen optimizada en RunPod

**Opción A: Usar imagen más reciente (recomendada)**

1. Construir localmente:
```bash
docker build -t iamvalenciia/dubbing-app:v25 .
docker push iamvalenciia/dubbing-app:v25
```

2. En RunPod, crear Pod con:
```bash
runpodctl pod create \
  --name "qdp-dubbing" \
  --image "iamvalenciia/dubbing-app:v25" \
  --gpu-id "NVIDIA A100-80GB" \
  --container-disk-in-gb 50 \
  --network-volume-id "TU_VOLUME_ID" \
  --volume-mount-path "/runpod-volume"
```

**Opción B: Override con ENV variables en RunPod**

Si no quieres reconstruir imagen, puedes setear en RunPod UI:

```
PYVIDEOTRANS_NLLB_BATCH_LINES=256
PYVIDEOTRANS_QWEN_TTS_BATCH_LINES=12
PYVIDEOTRANS_PROCESS_MAX_GPU=2
PYVIDEOTRANS_FORCE_QWEN_TTS_CUDA=1
```

### 11.4 Validar que GPU tuning está activo

En la consola de RunPod, busca estos logs cuando inicie doblaje:

```
[nllb200] ... alloc_gb=10+ reserved_gb=15+ batch_lines=256  ← Batching activo
[qwen3tts] batch_start ... size=12 mode=clone              ← Batch 12 activo
Concurrent task_nums=2                                      ← 2 tareas GPU en paralelo
```

Sin tuning (lento), verías:
```
[nllb200] ... batch_lines=10                ← Solo batch=10 (conservador)
[qwen3tts] batch_start ... size=2           ← Solo batch=2
Concurrent task_nums=1                      ← 1 tarea GPU (serializado)
```

### 11.5 Recomendaciones por tipo de GPU

| GPU Model | VRAM | Recomendado | Notas |
|-----------|------|-------------|-------|
| RTX 4090 | 24GB | NLLB batch=256, Qwen TTS batch=12, process_max_gpu=2 | ✅ Óptimo |
| A100-80GB | 80GB | NLLB batch=512, Qwen TTS batch=24, process_max_gpu=4 | Puedes ser más agresivo |
| A100-40GB | 40GB | NLLB batch=256, Qwen TTS batch=12, process_max_gpu=2 | Estándar |
| H100 | 80GB | NLLB batch=512, Qwen TTS batch=24, process_max_gpu=4 | Máximo rendimiento |
| L40 | 48GB | NLLB batch=256, Qwen TTS batch=12, process_max_gpu=2 | Bueno |

### 11.6 Si aún está lento después de tuning

1. **Verificar que cfg.json tiene valores correctos:**
   ```bash
   cat /app/pyvideotrans/videotrans/cfg.json | jq '.process_max_gpu, .trans_thread, .aitrans_thread'
   ```

2. **Verificar GPU memory:**
   ```bash
   nvidia-smi
   ```

3. **Revisar logs detallados:**
   - En app Gradio: visor de logs real-time
   - Buscar `[nllb200]` y `[qwen3tts]` para ver memory usage actual

4. **Aumentar batch size si hay VRAM disponible:**
   - Si `reserved_gb < total_vram * 0.6`, puedes aumentar batch
   - Ej: PYVIDEOTRANS_NLLB_BATCH_LINES=512 para A100-80GB

### 11.7 Commits relacionados con performance

- **51b7549**: Qwen3-TTS model singleton cache (elimina double-download)
- **66eb9a3**: GPU tuning env override support
- **ee72e0f**: Dockerfile con GPU tuning baked-in + cfg.json auto-update

### 11.8 Troubleshooting performance específico

**Síntoma: "Fetching 11 files" aparece dos veces (lento al inicio)**
- ✅ Corregido en 51b7549 (actualiza a latest main)

**Síntoma: Traducción muy lenta (<20 lin/seg)**
- Verificar: PYVIDEOTRANS_NLLB_BATCH_LINES env var
- Verificar: trans_thread en cfg.json sea >= 10
- Verificar: GPU memory available (nvidia-smi)

**Síntoma: Qwen3-TTS audio generation slow**
- Verificar: PYVIDEOTRANS_QWEN_TTS_BATCH_LINES=12
- Verificar: PYVIDEOTRANS_FORCE_QWEN_TTS_CUDA=1
- Nota: Primera línea es siempre lenta (model warmup); líneas 2-N son rápidas

**Síntoma: GPU utilization baja (<50%)**
- Aumentar process_max_gpu si tienes múltiples GPUs
- Aumentar batch_lines si hay VRAM libre
- Verificar: aitrans_thread >= 50 en cfg.json
