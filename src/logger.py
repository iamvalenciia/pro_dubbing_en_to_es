"""Simple pipeline logger used by the Gradio wrapper."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_LOG_PATH: Optional[str] = None
_CONFIGURED = False


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
        with open(_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except OSError as exc:
        return f"(error leyendo log: {exc})"


def clear_log() -> None:
    logger = logging.getLogger("qdp")
    if logger.handlers:
        logger.info("")
        logger.info("#" * 72)
        logger.info("# LOG RESET — NUEVA CORRIDA")
        logger.info("#" * 72)
        logger.info("")
