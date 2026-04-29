# Optimización de Extracción de Embeddings - Guía Completa

## 🔍 Diagnóstico Rápido

Tu video de 2.38 GB está atorado en **"Extracting embeddings..."** después de ~30 minutos.

### ¿Es normal?
- **VAD (Voice Activity Detection)**: 33.65s ✅ (correcto)
- **Segmentation**: <1s ✅ (correcto)
- **Embedding Extraction**: 30+ min ⏳ (puede serlo, pero depende de)

La diarización ali_CAM (CAMP++) tiene 3 fases:
1. **VAD** — Detectar dónde hay audio (33s)
2. **Segmentation** — Dividir audio en ventanas de 1.5s (~4000 segmentos)
3. **Embedding Extraction** — Extraer speaker embeddings de cada segmento
4. **Clustering** — Agrupar embeddings en speakers (HDBSCAN)

La fase 3+4 es donde está el cuello de botella.

---

## 🚀 Optimizaciones Ya Aplicadas

### 1. **Auto-detección de Batch Size** (ACTIVADA)
El código ahora detecta automáticamente tu GPU VRAM y ajusta:
- GPU 12GB+ → batch_size=64
- GPU 10-12GB → batch_size=48
- GPU 8-10GB → batch_size=32
- GPU 6-8GB → batch_size=24
- GPU <6GB → batch_size=16

**Qué hace**: En lugar de procesar 4000 embeddings uno por uno, los agrupa en lotes de 16-64.
- Sin batching: 4000 llamadas GPU = lento
- Con batching (batch_size=32): 125 llamadas GPU = mucho más rápido

### 2. **Timeout Inteligente** (ACTIVADA)
- Máximo 60 minutos por Phase 1
- Si tarda más, se cancela automáticamente
- Te aconseja parámetros optimizados

---

## 📊 Cómo Diagnosticar

### Opción A: Monitor de GPU en Tiempo Real
Abre otra terminal (PowerShell) y corre:
```powershell
nvidia-smi -l 1
```
Mientras Phase 1 está corriendo, observa:
- **Memory**: Uso VRAM (si está >90% = OOM risk, reduce batch_size)
- **GPU%**: Utilización (si <20% = CPU-bound, aumenta batch_size; si >80% = normal)

### Opción B: Diagnosticar con script
```powershell
.\diagnose_diarization_stall.ps1
```
Te mostrará:
- Procesos Python activos y su memoria
- GPU status
- Logs recientes
- Recomendaciones

### Opción C: Monitorear logs
```powershell
Get-Content "$env:HOMEDRIVE$env:HOMEPATH\OneDrive\Escritorio\qwen-en-to-es\temp_workspace\qdp_data\logs\pipeline.log" -Wait
```
Busca líneas con `[DIAR-BATCH]` para ver progreso de batching.

---

## 🎛️ Ajustes de Optimización

### Situación 1: "GPU utilization <20%, GPU memory <50%"
**Diagnóstico**: CPU-bound, batch_size es muy pequeño

**Solución**:
```powershell
$env:PYVIDEOTRANS_DIARIZATION_BATCH_SIZE="64"
python main_ui.py
```

Reinicia Phase 1. Deberías ver:
- GPU utilization sube a 50-80%
- Tiempo total baja a 10-20 min

---

### Situación 2: "GPU memory >95%, proceso lento"
**Diagnóstico**: Out-of-Memory, batch_size es demasiado grande

**Solución**:
```powershell
$env:PYVIDEOTRANS_DIARIZATION_BATCH_SIZE="8"
python main_ui.py
```

El proceso será más lento pero no hará OOM.

---

### Situación 3: "GPU memory 80-90%, GPU util 70-80%, todo normal"
**Diagnóstico**: Funcionando correctamente, embedding extraction simplemente es lento

**Solución**: Espera. Para un video de 2.38GB con 4000+ segmentos, 30-60 minutos es realista con ali_CAM. Puedes intentar:

```powershell
$env:PYVIDEOTRANS_DIARIZATION_BATCH_SIZE="32"  # O más alto
python main_ui.py
```

Si aún es lento, considera usar un backend más rápido (Pyannote, Reverb, o built-in).

---

## 🔧 Parámetros Avanzados

### Variable de Entorno: `PYVIDEOTRANS_DIARIZATION_BATCH_SIZE`
- **Default**: Auto-detectado según GPU
- **Override**: `$env:PYVIDEOTRANS_DIARIZATION_BATCH_SIZE="32"`
- **Rango recomendado**: 8-64
- **Trade-off**: 
  - Batch más grande = más rápido pero más memoria
  - Batch más pequeño = más lento pero menos memoria

### Variable de Entorno: `QDP_PHASE1_MAX_TIMEOUT`
- **Default**: 3600 segundos (60 minutos)
- **Override**: `$env:QDP_PHASE1_MAX_TIMEOUT="7200"` (120 minutos)
- **Nota**: Si se alcanza, Phase 1 se cancela y te aconseja optimización

### Variable de Entorno: `PYVIDEOTRANS_DIARIZATION_DEBUG`
- **Enable**: `$env:PYVIDEOTRANS_DIARIZATION_DEBUG="1"`
- **Efecto**: Más logs detallados sobre batching
- **Logs**: Verás `[DIAR-BATCH]` en los logs con detalles de cada batch

---

## 📈 Rendimiento Esperado

### Video 2.38 GB típico (~3000 seg, 1 hora)

| Backend | Batch Size | GPU | VAD+Seg | Embeddings | Clustering | Total |
|---------|-----------|-----|---------|-----------|-----------|-------|
| ali_CAM | 16 | 8GB | 1min | 45min | 10min | **56min** |
| ali_CAM | 32 | 8GB | 1min | 25min | 8min | **34min** |
| ali_CAM | 64 | 12GB+ | 1min | 15min | 5min | **21min** |
| Pyannote | N/A | 8GB+ | 2min | 8min | 2min | **12min** ⚡ |

**Conclusión**: ali_CAM con batch_size=32-64 es reasonably fast. Para uso en producción, considera Pyannote (más rápido pero requiere HF token).

---

## 🛑 Si Aún Está Lento

### Plan A: Cambiar Backend
Edita `.env` o establece variables:
```powershell
# Cambiar a Pyannote (más rápido)
$env:PYVIDEOTRANS_DIARIZ_BACKEND="pyannote"
$env:HF_TOKEN="hf_xxxxxxxxxxxx"  # Necesitas token de huggingface.co

python main_ui.py
```

### Plan B: Cambiar a Backend Built-in (CPU)
```powershell
$env:PYVIDEOTRANS_DIARIZ_BACKEND="built"
python main_ui.py
```
Más lento pero no requiere GPU. Útil si ali_CAM está fallando.

### Plan C: Aumentar Timeout (para videos muy grandes)
```powershell
$env:QDP_PHASE1_MAX_TIMEOUT="7200"  # 120 minutos
python main_ui.py
```

---

## 🧪 Test de Optimización Rápido

Si solo quieres probar sin esperar 30+ min:

1. Recorta el video a 5 minutos (280 segundos ~370 segmentos):
   ```powershell
   ffmpeg -i full_video_en_upload_1777394770.mp4 -t 300 -c copy test_5min.mp4
   ```

2. Ejecuta Phase 1 en el video de prueba:
   ```powershell
   $env:PYVIDEOTRANS_DIARIZATION_BATCH_SIZE="32"
   python main_ui.py
   ```
   
3. Observa GPUutilización y tiempo total
4. Extrapola: si 5min tarda X segundos, entonces 120min (2.38GB) tardará ~24X segundos

---

## 📊 Monitoreo en Tiempo Real Recomendado

Abre 3 terminales:

**Terminal 1**: Monitoreo GPU
```powershell
nvidia-smi -l 1
```

**Terminal 2**: Monitoreo logs en vivo
```powershell
Get-Content temp_workspace\qdp_data\logs\pipeline.log -Wait | Select-String -Pattern "DIAR|Extracting|batch"
```

**Terminal 3**: Ejecuta Phase 1
```powershell
python main_ui.py
```

Esto te permite ver en tiempo real:
- GPU memory/util
- Progreso de diarización
- Si está realmente procesando o atorado

---

## ✅ Resumen de Acciones

1. **Inmediato**: Tu código ya tiene auto-detección de batch_size. Ejecuta Phase 1 nuevamente.
2. **Si aún lento (30+ min)**: Establece `PYVIDEOTRANS_DIARIZATION_BATCH_SIZE=64` (o más bajo si OOM)
3. **Si extremadamente lento (60+ min)**: Considera cambiar a Pyannote o backend built-in
4. **Para monitoreo**: Usa `nvidia-smi -l 1` en terminal aparte

---

## 🆘 Si Nothing Funciona

1. Ejecuta diagnóstico:
   ```powershell
   .\diagnose_diarization_stall.ps1
   ```

2. Mira los logs:
   ```powershell
   Get-Content temp_workspace\qdp_data\logs\pipeline.log | Select-String -Pattern "ERROR|WARN|Failed" | Select-Object -Last 20
   ```

3. Busca:
   - GPU OOM (Out of Memory)
   - CUDA provider no disponible
   - Modelo no encontrado

4. Contacta con los logs de error
