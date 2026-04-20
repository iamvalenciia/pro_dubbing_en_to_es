# LatentSync en RunPod — Auto-setup

**No hay setup manual.** LatentSync se instala solo la primera vez que
generas un video. Esta guía documenta lo que pasa por debajo para
entender el sistema y debuggear si algo falla.

---

## Cómo funciona (TL;DR)

1. El usuario hace click en **"Renderizar completo"** (Tab 2) por
   primera vez después de crear el pod.
2. El wrapper `src/latentsync_wrapper.py` detecta que falta el repo y/o
   los pesos en `/runpod-volume/LatentSync/`.
3. **Bootstrap automático** (~3–5 min la primera vez):
   - `git clone --depth 1 https://github.com/bytedance/LatentSync.git
     /runpod-volume/LatentSync` (~20 MB)
   - `huggingface_hub.snapshot_download("ByteDance/LatentSync-1.6")` ⇒
     `/runpod-volume/LatentSync/checkpoints/latentsync_unet.pt` (~1.8 GB)
     \+ `whisper/tiny.pt` (~75 MB)
   - `pip install -r /runpod-volume/LatentSync/requirements.txt`
4. Se escribe un marker `/runpod-volume/LatentSync/.deps_installed`
   para saltar el `pip install` en futuras llamadas.
5. Se ejecuta `python -m scripts.inference ...` en subprocess con
   `cwd=/runpod-volume/LatentSync`.
6. Siguientes renders: bootstrap es **noop** (solo stat de archivos) →
   instantáneo.

Si algo falla en el bootstrap → **RuntimeError**. No hay fallback a
muxeo placeholder — el pipeline se detiene con un error claro.

---

## Persistencia en el volumen

RunPod monta `/runpod-volume/` como volumen persistente. Todo esto
**sobrevive al restart del pod**:

```
/runpod-volume/LatentSync/
├── .git/                                # repo clonado
├── .deps_installed                      # marker (persiste entre pods)
├── scripts/inference.py                 # entrypoint del subprocess
├── configs/
│   └── unet/
│       ├── stage2.yaml                  # 256x256, 8 GB VRAM
│       └── stage2_512.yaml              # 512x512, 18 GB VRAM (default)
├── requirements.txt
└── checkpoints/
    ├── latentsync_unet.pt               # ~1.8 GB — UNet diffusion
    └── whisper/
        └── tiny.pt                      # ~75 MB — audio encoder
```

**Lo que NO persiste**: el entorno Python del pod (site-packages).
Por eso el `pip install` se ejecuta una vez por pod nuevo. El marker
`.deps_installed` vive en el volumen, pero tras recrear el pod el
entorno está vacío — la primera llamada detecta deps faltantes
implícitamente cuando el subprocess falla. Para ser robusto, podés
borrar el marker antes de relanzar el pod:

```bash
rm -f /runpod-volume/LatentSync/.deps_installed
```

o setear una env var para forzar reinstall:

```bash
# en tu Dockerfile o en el pod:
rm -f /runpod-volume/LatentSync/.deps_installed
```

---

## Verificación rápida

Desde Python:

```python
from src.latentsync_wrapper import status_report, ensure_latentsync_ready
import json

# Diagnostico (no dispara bootstrap)
print(json.dumps(status_report(), indent=2))

# Forzar bootstrap ahora (util para pre-calentar al boot del app)
ensure_latentsync_ready()
```

Output esperado después del primer bootstrap exitoso:

```json
{
  "repo_path": "/runpod-volume/LatentSync",
  "repo_present": true,
  "unet_ckpt": {
    "path": "/runpod-volume/LatentSync/checkpoints/latentsync_unet.pt",
    "present": true,
    "size_mb": 1823.4
  },
  "whisper_ckpt": {
    "path": "/runpod-volume/LatentSync/checkpoints/whisper/tiny.pt",
    "present": true,
    "size_mb": 74.7
  },
  "unet_config": "configs/unet/stage2_512.yaml",
  "deps_installed": true,
  "ready": true
}
```

---

## Integración en el pipeline

### Tab 2 (master longform) — lipsync MANDATORY

El master (`output/master_lipsynced_nosubs.mp4`) **siempre** pasa por
LatentSync. No hay checkbox para apagarlo. Los únicos controles en la
UI son:

- **Inference steps** (10–50, default 20): más steps = mejor calidad
  y más tiempo.
- **Guidance scale** (1.0–3.0, default 1.5): mayor = mejor sync pero
  más distorsión en frames sin cara clara.

### Tab 3 (shorts) — hereda el lipsync del master

Los shorts consumen el master como fuente. Como el master ya tiene
lipsync aplicado, Tab 3 **no re-ejecuta LatentSync**. Solo:

- Recorta el rango `[start, end]`
- Hace crop/pad a 9:16 (1080×1920)
- Quema captions amarillos word-by-word en ES
- Aplica color grading

Esto ahorra mucho compute (de otro modo, cada short añadiría
~3-5 min de inferencia adicional).

---

## Costos de compute

LatentSync corre a **~0.3-0.5× realtime** en GPU decente:

| GPU | Config (VRAM) | Resolución | Velocidad | 1h video |
|---|---|---|---|---|
| RTX 3090 24GB | stage2_512 (18 GB) | 512×512 | ~0.3× RT | ~3 h compute |
| A5000 24GB | stage2_512 (18 GB) | 512×512 | ~0.4× RT | ~2.5 h compute |
| RTX 4090 24GB | stage2_512 (18 GB) | 512×512 | ~0.5× RT | ~2 h compute |
| RTX 3060 12GB | stage2 (8 GB) | 256×256 | ~0.5× RT | ~2 h compute (peor calidad) |

Para GPUs con <16 GB VRAM, forzá el config 256×256:

```bash
export LATENTSYNC_UNET_CONFIG=configs/unet/stage2.yaml
```

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `LATENTSYNC_PATH` | `/runpod-volume/LatentSync` | Path donde se clona el repo y viven los checkpoints |
| `LATENTSYNC_UNET_CONFIG` | `configs/unet/stage2_512.yaml` | Config UNet. Relativo al repo path |
| `LATENTSYNC_KEEP_WORKDIR` | `0` | Si `1`, conserva tmpdirs de inference para debug |
| `HF_TOKEN` | — | Token HuggingFace (si el repo de pesos lo requiere) |

---

## Troubleshooting

**El primer render se queda "colgado" 3–5 minutos.**
Es normal — está descargando ~2 GB de pesos en background. Mirá el log
de la UI (accordion "Log del render"). Vas a ver:
```
[bootstrap 1/3] Cloning LatentSync from ...
[bootstrap 2/3] Downloading LatentSync weights from ByteDance/LatentSync-1.6 ...
[bootstrap 3/3] pip install -r requirements.txt ...
LatentSync listo (repo + pesos + deps).
LatentSync infer: ...
```
Si pasaron 10 min sin progreso, el HF puede estar rate-limited —
reintenta.

**`RuntimeError: Fallo clonando LatentSync desde ...`**
No hay conexión a GitHub desde el pod, o `/runpod-volume` no es
escribible. Checkeá `git clone --depth 1 https://github.com/bytedance/LatentSync.git /tmp/ls-test`
manualmente.

**`RuntimeError: Fallo descargando pesos LatentSync desde ByteDance/LatentSync-1.6: ...`**
- Si menciona `401/403`: el repo requiere `HF_TOKEN` — exportalo antes
  de lanzar la app.
- Si menciona `Part size mismatch`: problema transitorio de HF. El
  `resume_download=True` retoma desde el último shard OK, así que
  reintenta.

**`LatentSync fallo (code=1)` durante inferencia.**
El subprocess imprime las últimas ~600 chars de stderr en el error. Casos
típicos:
- `CUDA out of memory` → forzá config 256×256 o reducí `inference_steps`.
- `ModuleNotFoundError: No module named 'X'` → `pip install` no completó.
  Borrá el marker y reintenta:
  ```bash
  rm /runpod-volume/LatentSync/.deps_installed
  ```

**Quiero pre-descargar los pesos antes del primer render para que la
UI no se quede esperando.**
Agregá en tu Dockerfile o en un wrapper de arranque del pod:
```python
from src.latentsync_wrapper import ensure_latentsync_ready
ensure_latentsync_ready()
```
Esto dispara el bootstrap al boot del app, antes de que el usuario
haga click.

**Borre por error los checkpoints.**
Simplemente relanza un render — el bootstrap los re-descarga
automáticamente.

---

## Por qué LatentSync y no EchoMimicV3 u otros

- **Video-to-video nativo**: preserva el video original (gestos,
  escenas, cámara). EMV3 regenera desde imagen fija → pierde
  continuidad visual.
- **Liviano**: ~2 GB de pesos vs ~14 GB de EMV3.
- **Sin chunking manual**: maneja videos largos internamente.
- **VRAM razonable**: 8 GB (v1.5 @ 256×256) o 18 GB (v1.6 @ 512×512).
- **Arquitectura SD + audio-conditioned latent diffusion** — estado del
  arte al 2025 para lipsync de video real.
