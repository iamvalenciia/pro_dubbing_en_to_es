"""Fase 1: PASS-THROUGH (antes hacia separacion Demucs voces/ambiente).

La nueva filosofia del pipeline ya NO hace diarizacion ni separa voz/fondo:
  - El doblaje ES se genera con MarianMT + Qwen3-TTS sobre un hablante unico.
  - El audio doblado se monta directo sobre el video original.
  - LatentSync (Fase 6) re-sincroniza los labios al nuevo audio.
  - No hay "mezcla con ambiente" — el video original ya trae la mezcla final,
    simplemente le reemplazamos la pista de audio.

Esta fase queda como pass-through para no romper la firma que usaba main_ui.py.
Retorna `(audio_input_path, audio_input_path)` — el mismo archivo como "vocals"
(input de Fase 2 ASR) y como "ambient" (ignorado por el pipeline nuevo, pero
mantenido para compat con llamadas viejas).
"""

from __future__ import annotations

import os

from src.logger import get_logger, phase_timer, log_file_info

log = get_logger("phase1")


def run_phase1_audio_separation(audio_input_path: str, output_dir: str):
    """Pass-through: no separa nada. Retorna (audio, audio) para mantener la firma."""
    with phase_timer(log, "FASE 1 — Pass-through (Demucs deshabilitado)"):
        log.info(f"Input audio: {audio_input_path}")
        log.info(
            "Fase 1 esta en modo PASS-THROUGH. El pipeline nuevo no hace "
            "separacion voz/ambiente — Fase 2 corre el ASR sobre el audio original."
        )
        if not audio_input_path or not os.path.exists(audio_input_path):
            raise RuntimeError(f"Audio input no existe: {audio_input_path!r}")
        os.makedirs(output_dir, exist_ok=True)
        log_file_info(log, audio_input_path, "input_audio")

    return audio_input_path, audio_input_path
