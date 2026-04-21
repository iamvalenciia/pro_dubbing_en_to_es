"""Gestion centralizada de rutas del pipeline QDP (SPEC §2).

Dos almacenamientos disjuntos:
- NETWORK_DIR (/runpod-volume): persistente y lento. Solo para modelos, inputs
  del usuario y respaldo de outputs finales.
- LOCAL_DIR (/workspace/qdp_data): NVMe efimero de 50GB. Aqui aterriza TODO el
  I/O intermedio (chunks ASR, WAVs TTS, JSONs, renders, logs, cache).

Los defaults se pueden sobrescribir con QDP_NETWORK_DIR / QDP_LOCAL_DIR para
desarrollo local (ej. Windows).
"""
import os
import shutil

from src.logger import get_logger, step_timer

log = get_logger("paths")


NETWORK_DIR = os.environ.get("QDP_NETWORK_DIR", "/runpod-volume")
LOCAL_DIR = os.environ.get("QDP_LOCAL_DIR", "/workspace/qdp_data")

NETWORK_MODELS = os.path.join(NETWORK_DIR, "models")
NETWORK_USER_INPUT = os.path.join(NETWORK_DIR, "user_input")
NETWORK_OUTPUT = os.path.join(NETWORK_DIR, "output")

LOCAL_INPUT = os.path.join(LOCAL_DIR, "input")
LOCAL_TEMP = os.path.join(LOCAL_DIR, "temp_workspace")
LOCAL_OUTPUT = os.path.join(LOCAL_DIR, "output")
LOCAL_LOGS = os.path.join(LOCAL_DIR, "logs")
LOCAL_CACHE = os.path.join(LOCAL_DIR, "torch_cache")

_LOCAL_SUBDIRS = (LOCAL_INPUT, LOCAL_TEMP, LOCAL_OUTPUT, LOCAL_LOGS, LOCAL_CACHE)


def ensure_local_dirs() -> None:
    """Crea LOCAL_DIR y subcarpetas. Jamas toca NETWORK_DIR.

    Tambien redirige caches de torch/HF al NVMe local para que los pesos
    intermedios no se escriban al volumen de red.
    """
    os.makedirs(LOCAL_DIR, exist_ok=True)
    for d in _LOCAL_SUBDIRS:
        os.makedirs(d, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", LOCAL_CACHE)
    os.environ.setdefault("HF_HOME", LOCAL_CACHE)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(LOCAL_CACHE, "hub"))


def _is_under(path: str, root: str) -> bool:
    try:
        p = os.path.abspath(path)
        r = os.path.abspath(root)
        return p == r or p.startswith(r.rstrip(os.sep) + os.sep)
    except (OSError, ValueError):
        return False


def stage_from_network(src: str, dest_name: str | None = None) -> str:
    """Copia fisica de un archivo al NVMe local.

    - Si `src` ya vive en LOCAL_DIR, no hace nada y devuelve tal cual.
    - Si `src` vive en NETWORK_DIR (o cualquier otra ruta externa), copia
      fisicamente a LOCAL_INPUT/<dest_name>. NUNCA usa symlinks (el SPEC §2
      lo prohibe: los symlinks mantienen los reads en el volumen de red).
    """
    if not src or not os.path.exists(src):
        raise FileNotFoundError(f"stage_from_network: no existe {src!r}")
    if _is_under(src, LOCAL_DIR):
        return src
    ensure_local_dirs()
    name = dest_name or os.path.basename(src)
    dest = os.path.join(LOCAL_INPUT, name)
    with step_timer(log, f"stage {src} -> {dest}"):
        if os.path.exists(dest):
            os.remove(dest)
        shutil.copyfile(src, dest)
    return dest


def backup_to_network(local_path: str, subdir: str = "") -> str:
    """Respalda un archivo final de LOCAL_OUTPUT a NETWORK_OUTPUT.

    Idempotente: sobrescribe si ya existe. Solo debe llamarse cuando el
    pipeline termino OK. Retorna la ruta final en el volumen de red.
    """
    if not local_path or not os.path.exists(local_path):
        raise FileNotFoundError(f"backup_to_network: no existe {local_path!r}")
    target_dir = os.path.join(NETWORK_OUTPUT, subdir) if subdir else NETWORK_OUTPUT
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, os.path.basename(local_path))
    with step_timer(log, f"backup {local_path} -> {dest}"):
        shutil.copyfile(local_path, dest)
    return dest
