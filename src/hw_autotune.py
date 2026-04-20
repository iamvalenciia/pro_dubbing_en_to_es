"""Hardware auto-tuning para batch sizes dinamicos.

Objetivo: que el pipeline corra bien en CUALQUIER pod (desde una RTX 3060 12GB
hasta una H100 94GB) sin tocar un solo env var. Cada fase declara su baseline
y que recurso escala (VRAM, RAM, CPU), este modulo detecta la capacidad real
del host y escala el baseline por la proporcion contra un "baseline ref".

Como usarlo desde un phase:

    from src.hw_autotune import autotune
    DEMUCS_BATCH_SIZE = autotune(
        "DEMUCS_BATCH_SIZE", baseline=8, scale_with="vram",
        baseline_ref=24, min_v=2, max_v=32,
    )

Semantica:
  - Si la env var esta seteada, la respetamos (override manual gana).
  - Si no: factor = capacidad_detectada / baseline_ref, luego
    resultado = clamp(round(baseline * factor), min_v, max_v).

Baseline refs recomendadas (calibradas para modelos medianos):
  - VRAM: 24 GB  (RTX 3090 / 4090 / A10, un "pod tipico")
  - RAM : 32 GB
  - CPU : 8 vCPU
"""

from __future__ import annotations

import os
from functools import lru_cache


# ---------------------------------------------------------------------------
# Deteccion (cacheada porque no cambia durante la ejecucion).
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def detect_gpu_vram_gb() -> float:
    """VRAM total del GPU 0 en GB. 0.0 si no hay GPU disponible."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def detect_gpu_name() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "no-gpu"
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def detect_system_ram_gb() -> float:
    """RAM total del sistema en GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    # Fallback: leer /proc/meminfo en Linux
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 0.0


@lru_cache(maxsize=1)
def detect_cpu_count() -> int:
    """Numero logico de CPUs (incluye hyperthreads)."""
    try:
        n = os.cpu_count()
        if n and n > 0:
            return int(n)
    except Exception:
        pass
    return 1


# ---------------------------------------------------------------------------
# Helper principal.
# ---------------------------------------------------------------------------
_SCALE_MAP = {
    "vram": detect_gpu_vram_gb,
    "ram": detect_system_ram_gb,
    "cpu": detect_cpu_count,
}


def autotune(
    env_var: str,
    *,
    baseline: int,
    scale_with: str,
    baseline_ref: float,
    min_v: int,
    max_v: int,
) -> int:
    """Devuelve un batch size autotuneado, respetando override por env.

    Args:
      env_var: nombre de la variable de entorno que puede forzar el valor.
      baseline: valor "nominal" pensado para un host con capacidad = baseline_ref.
      scale_with: "vram" | "ram" | "cpu" — que recurso escala este valor.
      baseline_ref: capacidad del host en la que `baseline` fue calibrado.
      min_v, max_v: clamp duro (no bajar/subir mas de esto).

    Fallback:
      Si no se puede detectar el recurso (p.ej. sin GPU y scale_with="vram"),
      devolvemos `min_v` para no crashear por asumir GPU.
    """
    # Override manual: siempre gana.
    raw = os.environ.get(env_var)
    if raw is not None and raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass  # env mal formateada, seguimos con autotune

    detector = _SCALE_MAP.get(scale_with)
    if detector is None:
        raise ValueError(f"scale_with desconocido: {scale_with!r}")

    capacity = float(detector())
    if capacity <= 0 or baseline_ref <= 0:
        # No pudimos detectar o baseline mal definido; devolver minimo seguro.
        return max(1, min_v)

    factor = capacity / float(baseline_ref)
    scaled = int(round(baseline * factor))
    return max(min_v, min(max_v, scaled))


# ---------------------------------------------------------------------------
# Resumen para logs (util al arrancar cada fase).
# ---------------------------------------------------------------------------
def hw_summary() -> str:
    vram = detect_gpu_vram_gb()
    ram = detect_system_ram_gb()
    cpu = detect_cpu_count()
    gpu = detect_gpu_name()
    return (
        f"GPU={gpu} VRAM={vram:.1f}GB | RAM={ram:.1f}GB | CPU={cpu} logical"
    )


if __name__ == "__main__":
    print(hw_summary())
    print(f"autotune example DEMUCS_BATCH_SIZE baseline=8 vram24 -> "
          f"{autotune('DEMUCS_BATCH_SIZE', baseline=8, scale_with='vram', baseline_ref=24, min_v=2, max_v=32)}")
    print(f"autotune example QWEN_TTS_BATCH_SIZE baseline=16 vram24 -> "
          f"{autotune('QWEN_TTS_BATCH_SIZE', baseline=16, scale_with='vram', baseline_ref=24, min_v=2, max_v=64)}")
