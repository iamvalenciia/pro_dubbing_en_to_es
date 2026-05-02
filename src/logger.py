"""Simple pipeline logger used by the Gradio wrapper."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque
from typing import Optional

_LOG_PATH: Optional[str] = None
_CONFIGURED = False
_TAIL_LOCK = threading.Lock()
_TAIL_LINES: deque[str] = deque(maxlen=3000)
_TAIL_POS = 0
_TAIL_RENDER_CACHE: dict[int, tuple[int, str]] = {}
_TAIL_VERSION = 0


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        for stream in self.streams:
            try:
                return bool(stream.isatty())
            except Exception:
                continue
        return False


def setup_pipeline_logger(log_dir: str | None = None) -> str:
    global _LOG_PATH, _CONFIGURED

    if log_dir is None:
        from src.paths import LOCAL_LOGS

        log_dir = LOCAL_LOGS

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

    file_handler = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.__stdout__)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)

    try:
        log_raw = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.__stdout__, log_raw)
        sys.stderr = _TeeStream(sys.__stderr__, log_raw)
    except Exception:
        pass

    _CONFIGURED = True
    root.info("Pipeline logger inicializado en %s", _LOG_PATH)
    return _LOG_PATH


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"qdp.{name}")


def read_log_tail(lines: int = 300) -> str:
    if not _LOG_PATH or not os.path.exists(_LOG_PATH):
        return "(log vacio — pipeline no ha iniciado)"
    try:
        if lines <= 0:
            lines = 1

        with _TAIL_LOCK:
            global _TAIL_POS, _TAIL_VERSION
            file_size = os.path.getsize(_LOG_PATH)

            # Handle truncation/rotation.
            if _TAIL_POS > file_size:
                _TAIL_POS = 0
                _TAIL_LINES.clear()
                _TAIL_RENDER_CACHE.clear()
                _TAIL_VERSION += 1

            with open(_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                if _TAIL_POS == 0 and not _TAIL_LINES:
                    # First read: load once, then incremental updates only.
                    for ln in f.readlines():
                        _TAIL_LINES.append(ln)
                    _TAIL_POS = f.tell()
                    _TAIL_VERSION += 1
                elif _TAIL_POS < file_size:
                    f.seek(_TAIL_POS)
                    chunk = f.read()
                    _TAIL_POS = f.tell()
                    if chunk:
                        for ln in chunk.splitlines(keepends=True):
                            _TAIL_LINES.append(ln)
                        _TAIL_VERSION += 1

            cached = _TAIL_RENDER_CACHE.get(lines)
            if cached and cached[0] == _TAIL_VERSION:
                return cached[1]

            output = "".join(list(_TAIL_LINES)[-lines:])
            _TAIL_RENDER_CACHE[lines] = (_TAIL_VERSION, output)
            return output
    except OSError as exc:
        return f"(error leyendo log: {exc})"


def clear_log() -> None:
    return
