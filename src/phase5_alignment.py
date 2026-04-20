"""Fase 5: Concatenacion natural de WAVs doblados (sin alineamiento).

Reemplaza la version previa que hacia placement con timeline anchor + overlap
controlado + mezcla con ambiente. La nueva filosofia:
  - No hay ambiente (sin Demucs). El audio doblado reemplaza el original entero.
  - No hay que "cuadrar" con el timeline del video fuente — Fase 6 (LatentSync)
    re-sincroniza los labios al nuevo audio.
  - Solo concatenamos los WAVs clonados en orden de `segment_id`, metiendo
    pausas naturales entre segmentos (usando el gap del timeline original
    capeado a un rango razonable para que la voz fluya sin silencios eternos).

El resultado es un WAV unico que puede durar MENOS que el video original, y
eso es OK — al mux-earlo con `-shortest` en Fase 5b, el MP4 final queda igual
a la duracion del audio doblado.
"""

from __future__ import annotations

import os
import json
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

from src.logger import get_logger, phase_timer, step_timer, log_file_info

try:
    from src.hw_autotune import autotune
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("phase5")


if _HAS_AUTOTUNE:
    PHASE5_LOAD_WORKERS = autotune(
        "PHASE5_LOAD_WORKERS", baseline=16, scale_with="cpu",
        baseline_ref=16, min_v=2, max_v=32,
    )
else:
    PHASE5_LOAD_WORKERS = int(os.environ.get("PHASE5_LOAD_WORKERS", "8"))


# Pausa entre segmentos. Tomamos el gap original del timeline (end_prev -> start_curr),
# clampeado a [MIN, MAX]. Asi se preserva el ritmo natural del orador pero evitamos
# silencios artificiales si la fuente tenia pausas muy largas (p.ej. cortes de edicion).
PHASE5_MIN_GAP_S = float(os.environ.get("PHASE5_MIN_GAP_S", "0.08"))
PHASE5_MAX_GAP_S = float(os.environ.get("PHASE5_MAX_GAP_S", "0.60"))


def _load_wav_float(path: str):
    """Lee un WAV como np.float32 mono 1D. Si es estereo, promedia a mono."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def run_phase5_time_alignment(json_path: str, output_final_audio: str, _unused=None):
    """Fase 5: Concatena WAVs clonados en orden con pausas naturales.

    Firma compatible (best-effort) con versiones previas que pasaban `ambient_path`
    como segundo argumento: lo aceptamos pero lo ignoramos. main_ui.py nuevo
    llama con solo (json_path, output_final_audio).

    Args:
      json_path: timeline JSON con cloned_audio_path por segmento (de Fase 4).
      output_final_audio: WAV final concatenado.

    Returns:
      output_final_audio (mismo string).
    """
    # Compatibilidad: si se paso un 2do arg posicional (era `ambient_path`), swap.
    if _unused is not None and isinstance(output_final_audio, str) and not output_final_audio.endswith(".wav"):
        # Muy improbable, pero por si acaso.
        pass

    with phase_timer(log, "FASE 5 — Concatenacion natural de voces dobladas"):
        log.info(f"Input timeline: {json_path}")
        log.info(f"Output final audio: {output_final_audio}")
        log_file_info(log, json_path, "timeline_in")

        with open(json_path, "r", encoding="utf-8") as f:
            master_timeline = json.load(f)
        log.info(f"Segmentos a concatenar: {len(master_timeline)}")

        # Ordenar por segment_id (deberia venir en orden, pero nos aseguramos).
        master_timeline = sorted(master_timeline, key=lambda s: s.get("segment_id", 0))

        load_specs = []
        for seg in master_timeline:
            p = seg.get("cloned_audio_path")
            if p and os.path.exists(p):
                load_specs.append((seg["segment_id"], seg, p))
            else:
                log.warning(
                    f"  seg[{seg.get('segment_id')}] sin cloned_audio_path — skip"
                )

        if not load_specs:
            raise RuntimeError("Fase 5: no hay WAVs clonados para concatenar.")

        with step_timer(
            log,
            f"Cargando {len(load_specs)} voces en paralelo [workers={PHASE5_LOAD_WORKERS}]"
        ):
            def _load(item):
                sid, seg, path = item
                audio, sr = _load_wav_float(path)
                return sid, seg, audio, sr

            with ThreadPoolExecutor(max_workers=PHASE5_LOAD_WORKERS) as ex:
                loaded = list(ex.map(_load, load_specs))

        srs = set(sr for _, _, _, sr in loaded)
        if len(srs) > 1:
            raise RuntimeError(f"Sample rates inconsistentes entre voces: {srs}")
        sr = srs.pop()
        log.info(f"Sample rate unificado: {sr}Hz")

        # Construir pieces: [voice_0, gap_0, voice_1, gap_1, ...]
        pieces: list[np.ndarray] = []
        total_samples = 0
        prev_seg_end = None
        gap_stats: list[float] = []

        with step_timer(log, f"Encadenando {len(loaded)} segmentos con pausas naturales"):
            for i, (sid, seg, audio, _) in enumerate(loaded):
                if i > 0 and prev_seg_end is not None:
                    raw_gap = float(seg.get("start", 0.0)) - prev_seg_end
                    gap_s = max(PHASE5_MIN_GAP_S, min(PHASE5_MAX_GAP_S, raw_gap))
                    gap_stats.append(gap_s)
                    gap_samples = int(gap_s * sr)
                    if gap_samples > 0:
                        pieces.append(np.zeros(gap_samples, dtype=np.float32))
                        total_samples += gap_samples

                pieces.append(audio)
                total_samples += len(audio)

                prev_seg_end = float(seg.get("end", 0.0))

        # Concatenacion final en un solo buffer (evita muchos append ineficientes).
        final_mono = np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=np.float32)

        # Clip [-1, 1] por seguridad antes de exportar PCM_16.
        np.clip(final_mono, -1.0, 1.0, out=final_mono)

        if gap_stats:
            gs = np.asarray(gap_stats, dtype=np.float32)
            log.info(
                f"Pausas entre segmentos: n={len(gs)} "
                f"mean={gs.mean():.3f}s median={float(np.median(gs)):.3f}s "
                f"min={gs.min():.3f}s max={gs.max():.3f}s "
                f"(clamp [{PHASE5_MIN_GAP_S}, {PHASE5_MAX_GAP_S}]s)"
            )

        duration_s = len(final_mono) / sr
        log.info(f"Audio doblado final: {duration_s:.2f}s @ {sr}Hz ({len(final_mono)} samples)")

        os.makedirs(os.path.dirname(output_final_audio) or ".", exist_ok=True)
        with step_timer(log, f"Escribiendo WAV PCM_16 ({duration_s:.1f}s)"):
            sf.write(output_final_audio, final_mono, sr, subtype="PCM_16")

        log_file_info(log, output_final_audio, "final_audio")

    return output_final_audio
