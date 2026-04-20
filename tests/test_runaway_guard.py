"""Unit test del runaway guard de Fase 4.

El guard corta en seco los WAVs cuya duracion excede lo esperable para la
longitud del texto (proxy: len(text) / CHARS_PER_SEC). Este test simula la
rama relevante de `_batched_generate_serial_decode` con tensores fake, sin
necesidad de GPU ni del modelo Qwen3-TTS real.

El objetivo: garantizar que:
  1. Un WAV "normal" (duracion ~ expected) no se toca.
  2. Un WAV runaway (20s para un text de 10 chars) se trunca al limite.
  3. Un WAV borderline (dentro del RUNAWAY_FACTOR) no se toca.
  4. Segmentos con text vacio toleran el FLOOR.
"""
from __future__ import annotations

import sys
import os

# Dependencias minimas del modulo (no importamos torch ni librosa porque el
# test usa solo np). Forzamos stubs para que el import no falle en Windows
# donde torch CUDA no esta disponible.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


CHARS_PER_SEC = 14.0
RUNAWAY_FACTOR = 1.8
RUNAWAY_FLOOR_S = 2.0


def apply_runaway_guard(wav: np.ndarray, sr: int, text: str) -> tuple[np.ndarray, bool]:
    """Replica EXACTA de la logica del guard en phase4_tts_cloning.py.
    Devuelve (wav_posiblemente_truncado, fue_truncado)."""
    text_len = len(text)
    expected_s = max(0.1, text_len / max(CHARS_PER_SEC, 1e-3))
    limit_s = expected_s * RUNAWAY_FACTOR + RUNAWAY_FLOOR_S
    actual_s = wav.shape[0] / sr
    if actual_s > limit_s:
        limit_samples = int(limit_s * sr)
        return wav[:limit_samples], True
    return wav, False


def _make_wav(duration_s: float, sr: int = 24000) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def test_normal_segment_untouched():
    # Texto de 42 chars → expected ~3s → limit ~3*1.8 + 2 = 7.4s
    # WAV de 3.5s esta muy dentro del limite. No se debe tocar.
    wav = _make_wav(3.5)
    out, truncated = apply_runaway_guard(wav, 24000, "Hola mundo, esto es un texto de prueba ok")
    assert not truncated, "WAV normal de 3.5s para 42ch no deberia marcarse runaway"
    assert out.shape[0] == wav.shape[0]


def test_runaway_long_wav_truncated():
    # Texto de 20 chars → expected ~1.43s → limit ~1.43*1.8 + 2 = 4.57s
    # WAV de 25s (model hizo runaway). Debe truncarse a ~4.57s.
    wav = _make_wav(25.0)
    out, truncated = apply_runaway_guard(wav, 24000, "Buenos dias señor hi")  # 20 ch
    assert truncated, "WAV de 25s para 20ch deberia detectarse como runaway"
    actual_s = out.shape[0] / 24000
    assert abs(actual_s - 4.571) < 0.01, f"trunc a {actual_s}, esperado ~4.57s"


def test_borderline_just_under_limit_untouched():
    # Texto 20ch → limit 4.57s. WAV de 4.5s esta justo debajo del limite.
    wav = _make_wav(4.5)
    out, truncated = apply_runaway_guard(wav, 24000, "Buenos dias señor hi")
    assert not truncated
    assert out.shape[0] == wav.shape[0]


def test_empty_text_floor_protects():
    # Texto vacio: expected=0.1s, limit = 0.1*1.8 + 2 = 2.18s.
    # WAV de 1s para text vacio NO deberia marcarse runaway (floor absoluto).
    wav = _make_wav(1.0)
    out, truncated = apply_runaway_guard(wav, 24000, "")
    assert not truncated, "WAV de 1s con text vacio no deberia ser runaway"


def test_very_short_text_protected_by_floor():
    # Texto "Si." (3 chars) → expected=0.214s. Sin floor, limit seria 0.386s,
    # recortando cualquier "Si." real (que dura ~0.7s). CON floor de 2s,
    # limit = 0.386 + 2 = 2.386s. Un "Si." de 0.9s esta a salvo.
    wav = _make_wav(0.9)
    out, truncated = apply_runaway_guard(wav, 24000, "Si.")
    assert not truncated, "WAV corto legitimo no deberia ser runaway gracias al floor"


def test_very_short_text_runaway_still_detected():
    # Mismo "Si." (3 chars) limit=2.386s. Pero si el modelo runaway genera
    # 30s de silencio/ruido, DEBE ser truncado.
    wav = _make_wav(30.0)
    out, truncated = apply_runaway_guard(wav, 24000, "Si.")
    assert truncated
    actual_s = out.shape[0] / 24000
    assert actual_s < 3.0, f"'Si.' runaway de 30s deberia truncarse a ~2.4s, dio {actual_s}s"


if __name__ == "__main__":
    test_normal_segment_untouched()
    test_runaway_long_wav_truncated()
    test_borderline_just_under_limit_untouched()
    test_empty_text_floor_protects()
    test_very_short_text_protected_by_floor()
    test_very_short_text_runaway_still_detected()
    print("OK — todos los tests del runaway guard pasan")
