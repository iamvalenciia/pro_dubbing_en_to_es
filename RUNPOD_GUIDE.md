# 🚀 Guía RunPod — EN→ES Voice Dubbing

## Configuración inicial (una sola vez)

### 1. Construir y publicar la imagen Docker

```bash
# En tu máquina local (necesitas Docker Desktop instalado)
docker build -t tuusuario/dubbing-app:latest .
docker login
docker push tuusuario/dubbing-app:latest
```

> Reemplaza `tuusuario` con tu nombre de usuario en Docker Hub (hub.docker.com).

---

### 2. Crear un Network Volume en RunPod

1. Ve a [RunPod.io](https://www.runpod.io) → **Storage** → **+ New Volume**
2. Nombre: `dubbing-volume`
3. Tamaño: **20 GB** (modelos ~5 GB + espacio para tus archivos)
4. Región: elige la misma donde crearás tu pod

> El Network Volume persiste aunque pares o termines el pod. Cuesta ~$0.07/GB/mes.

---

### 3. Crear el Pod Template

1. Ve a **Templates** → **+ New Template**
2. Rellena los campos:
   - **Container Image:** `tuusuario/dubbing-app:latest`
   - **Container Disk:** 20 GB
   - **Volume Mount Path:** `/runpod-volume` → selecciona `dubbing-volume`
   - **Expose HTTP Ports:** `7860`
3. Añade las variables de entorno:
   ```
   WHISPER_MODEL=base
   TTS_LOCAL_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
   TTS_BATCH_SIZE=8
   PYTHONUNBUFFERED=1
   ```
4. Guarda el template.

---

## ▶️ Turn ON — Encender el servidor

1. **Pods → + Deploy** → selecciona tu template
2. Elige la GPU (ver recomendaciones abajo)
3. Haz clic en **Deploy**
4. Espera ~2 minutos:
   - Primera vez: descarga modelos (~5 GB al Network Volume) → puede tardar 5-10 min
   - Siguientes veces: los modelos ya están en caché → arranque en ~1 min
5. Haz clic en **Connect → HTTP Service [Port 7860]**

✅ Ya tienes acceso a la app en el navegador.

---

## ⏹️ Turn OFF — Apagar para ahorrar dinero

1. Ve a **Pods** → tu pod → **Stop**
2. El pod se detiene → **costo GPU = $0/hora**
3. Los archivos en `/runpod-volume` (modelos HuggingFace) **se conservan siempre**
4. Los archivos en el disco del pod (cache, youtube_video, etc.) **también persisten al hacer Stop**

> ⚠️ **Terminate ≠ Stop.** Si haces **Terminate** (borrar el pod), pierdes el disco local del contenedor pero NO el Network Volume.

---

## 💾 Qué se guarda y qué se pierde

| Directorio | Tipo de storage | Persiste al Stop | Persiste al Terminate |
|---|---|:---:|:---:|
| `/runpod-volume/huggingface/` | Network Volume | ✅ Sí | ✅ Sí |
| `/app/cache/` | Disco del pod | ✅ Sí | ❌ No |
| `/app/youtube_video/` | Disco del pod | ✅ Sí | ❌ No |
| `/app/shorts/` | Disco del pod | ✅ Sí | ❌ No |
| `/app/archives/` | Disco del pod | ✅ Sí | ❌ No |
| `/app/enhance_output/` | Disco del pod | ✅ Sí | ❌ No |
| `/app/subtitle_output/` | Disco del pod | ✅ Sí | ❌ No |

**Tip:** Usa el botón **"🗄 Archivar Proyecto"** en la app antes de apagar para guardar todo en un ZIP descargable.

---

## 🖥 GPUs recomendadas

| GPU | VRAM | Precio aprox. | Notas |
|---|---|---|---|
| RTX 3090 | 24 GB | ~$0.69/hr | ✅ Óptima — corre el modelo TTS sin problemas |
| RTX 4090 | 24 GB | ~$1.99/hr | Más rápida pero más cara |
| A40 | 48 GB | ~$0.79/hr | Buena relación precio/VRAM |
| RTX 3080 | 10 GB | ~$0.34/hr | ⚠️ Puede quedarse corta con TTS 1.7B |

> El modelo Qwen3-TTS-12Hz-1.7B-Base necesita ~8-10 GB de VRAM. Usa **Community Cloud** para las GPUs más baratas.

---

## 💡 Tips para reducir costos

1. **Para siempre** el pod cuando no lo uses (un día olvidado = $16+ en RTX 3090)
2. Usa pods **Spot/Interruptible** para ~40% de descuento (el pod puede reiniciarse sin aviso)
3. El **Network Volume** cuesta $0.07/GB/mes → 20 GB = $1.40/mes aunque el pod esté apagado
4. Usa la opción **"Test 5 minutos"** o **"Test 20 segundos"** para verificar antes de procesar completo

---

## 🔧 Prueba local con Docker Compose

Para probar sin RunPod (necesitas NVIDIA Container Toolkit en tu máquina):

```bash
docker-compose up --build
# Gradio disponible en http://localhost:7860
```

Los archivos se montan desde tu directorio local, por lo que no pierdes datos al reiniciar.

---

## 📦 Actualizar la imagen después de cambiar código

```bash
docker build -t tuusuario/dubbing-app:latest .
docker push tuusuario/dubbing-app:latest
```

Luego en RunPod: **Pods → Stop → Edit → Update Image → Start**
