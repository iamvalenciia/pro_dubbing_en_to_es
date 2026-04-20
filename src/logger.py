"""Sistema de logs centralizado para el pipeline.

Cada fase usa get_logger(name) para emitir logs estructurados.
Todos los logs se escriben a disco (pipeline.log) y a stdout del container.
El UI lee el tail del archivo para mostrar progreso en vivo.
"""
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional


_LOG_PATH: Optional[str] = None
_CONFIGURED = False


class _TeeStream:
    """Duplica escrituras a stdout original + archivo de log.
    Necesario para capturar prints de librerias que no usan logging (tqdm, hf, etc.)
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except (ValueError, OSError):
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except (ValueError, OSError):
                pass

    def isatty(self):
        return False


def setup_pipeline_logger(log_dir: str = "logs") -> str:
    """Configura logger root con handler de archivo + stdout y tee de stdout/stderr.
    Idempotente: solo configura una vez por proceso.
    IMPORTANTE: log_dir debe estar FUERA de temp_workspace/output/input (que son
    purgados por clear_workspace). Por defecto usa './logs/pipeline.log'.
    Retorna la ruta del archivo de log.
    """
    global _LOG_PATH, _CONFIGURED

    os.makedirs(log_dir, exist_ok=True)
    _LOG_PATH = os.path.join(log_dir, "pipeline.log")

    if _CONFIGURED:
        return _LOG_PATH

    root = logging.getLogger("qdp")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(levelname).1s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.__stdout__)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    root.addHandler(sh)

    log_file_raw = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, log_file_raw)
    sys.stderr = _TeeStream(sys.__stderr__, log_file_raw)

    _CONFIGURED = True

    root.info("=" * 80)
    root.info(f"Pipeline logger inicializado. Archivo: {_LOG_PATH}")
    root.info("=" * 80)
    return _LOG_PATH


def get_logger(name: str) -> logging.Logger:
    """Retorna logger namespaced bajo 'qdp.<name>'."""
    return logging.getLogger(f"qdp.{name}")


def read_log_tail(lines: int = 300) -> str:
    """Lee las ultimas N lineas del archivo de log.
    Usado por el UI para mostrar progreso en vivo.
    """
    if not _LOG_PATH or not os.path.exists(_LOG_PATH):
        return "(log vacio — pipeline no ha iniciado)"
    try:
        with open(_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except OSError as e:
        return f"(error leyendo log: {e})"


def clear_log():
    """Marca un reset visual en el log. No trunca el archivo para no invalidar
    el file handle del FileHandler (que escribiria en inode fantasma). El UI
    solo muestra el tail reciente, asi que el separador es suficiente.
    """
    root = logging.getLogger("qdp")
    if not root.handlers:
        return
    root.info("")
    root.info("#" * 80)
    root.info("# LOG RESET — NUEVA CORRIDA")
    root.info("#" * 80)
    root.info("")


def log_gpu_snapshot(log: logging.Logger, tag: str = ""):
    """Loguea estado de CUDA: device count, memoria usada/libre, nombre del device."""
    try:
        import torch
        if not torch.cuda.is_available():
            log.info(f"[GPU {tag}] CUDA NO disponible (torch.cuda.is_available=False)")
            return
        n = torch.cuda.device_count()
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        alloc = torch.cuda.memory_allocated(dev) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(dev) / 1024 ** 3
        total = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3
        log.info(
            f"[GPU {tag}] dev={dev}/{n} name='{name}' "
            f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB "
            f"cuda_version={torch.version.cuda}"
        )
    except (ImportError, RuntimeError) as e:
        log.warning(f"[GPU {tag}] no se pudo obtener snapshot: {e}")


def log_file_info(log: logging.Logger, path: str, label: str = ""):
    """Loguea tamano y existencia de un archivo."""
    if not os.path.exists(path):
        log.error(f"[FILE {label}] NO EXISTE: {path}")
        return
    size_mb = os.path.getsize(path) / 1024 ** 2
    log.info(f"[FILE {label}] {path} ({size_mb:.2f} MB)")


@contextmanager
def phase_timer(log: logging.Logger, phase_name: str):
    """Context manager que loguea inicio/fin de una fase con duracion."""
    log.info("#" * 70)
    log.info(f"### {phase_name} — INICIO")
    log.info("#" * 70)
    t0 = time.time()
    try:
        yield
    except Exception as e:
        dt = time.time() - t0
        log.error(f"### {phase_name} — FALLO tras {dt:.2f}s: {type(e).__name__}: {e}")
        raise
    else:
        dt = time.time() - t0
        log.info("#" * 70)
        log.info(f"### {phase_name} — OK en {dt:.2f}s")
        log.info("#" * 70)


@contextmanager
def step_timer(log: logging.Logger, step_name: str):
    """Context manager para sub-pasos dentro de una fase."""
    log.info(f"  -> {step_name} ...")
    t0 = time.time()
    try:
        yield
    except Exception as e:
        dt = time.time() - t0
        log.error(f"  <- {step_name} FALLO tras {dt:.2f}s: {type(e).__name__}: {e}")
        raise
    else:
        dt = time.time() - t0
        log.info(f"  <- {step_name} OK ({dt:.2f}s)")
