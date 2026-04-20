import os
import json
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

from src.logger import get_logger, phase_timer, step_timer, log_file_info

try:
    from src.hw_autotune import autotune, hw_summary
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("phase5")


# Workers para leer cloned_*.wav en paralelo. soundfile.read suelta GIL en el
# decode, asi que threads saturan disco sin contention. Escalado por CPU count.
if _HAS_AUTOTUNE:
    PHASE5_LOAD_WORKERS = autotune(
        "PHASE5_LOAD_WORKERS", baseline=16, scale_with="cpu",
        baseline_ref=16, min_v=2, max_v=32,
    )
else:
    PHASE5_LOAD_WORKERS = int(os.environ.get("PHASE5_LOAD_WORKERS", "8"))


# ---------------------------------------------------------------------------
# Placement con timeline anchor + overlap controlado (Fix D del drift pipeline)
# ---------------------------------------------------------------------------
# Algoritmo anterior: `pos = cursor_s + natural_gap_s` -> cada segmento
# empujaba al siguiente, acumulando el exceso del TTS ES vs EN. En un video
# de 1.5h acumulaba +27min de drift (logs reales del usuario).
#
# Algoritmo nuevo: usamos `seg["start"]` como anchor y nos permitimos hasta
# MAX_OVERLAP_S de solape con la voz previa antes de empujar hacia adelante.
# Asi cada segmento re-sincroniza con el timeline original cuando puede, y
# solo se pierde sync cuando es inevitable (TTS previo desbordado).
#
# Cap a 0.3s porque overlaps > 300ms empiezan a ser perceptibles como "voces
# encimadas". 300ms cae tipicamente en la cola decayente de una palabra final
# contra el arranque consonantico de la siguiente — inaudible a volumen normal.
PHASE5_MAX_OVERLAP_S = float(os.environ.get("PHASE5_MAX_OVERLAP_S", "0.3"))


def _load_wav_float(path):
    """Lee un WAV como np.float32. Retorna (audio_1d_o_2d, sr). Si es estereo,
    promedia a mono para facilitar el overlay posterior."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # mix L+R -> mono
    return audio, sr


def _resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float32 si los sr no coinciden. Usa librosa (rapido, bien probado)."""
    if orig_sr == target_sr:
        return audio
    import librosa  # lazy import
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def run_phase5_time_alignment(json_path: str, ambient_path: str, output_final_audio: str):
    """Fase 5: Mezcla natural de voces TTS + ambiente, VECTORIZADA con numpy.

    Diferencias vs la version pydub previa:
      - Reemplaza `combined_voices.overlay()` O(N²) por canvas numpy + suma in-place O(N·L).
      - Carga todos los WAV en paralelo con ThreadPoolExecutor (16 workers) en vez de serial.
      - En videos con cientos de segmentos es 50-100x mas rapido.

    Filosofia (ACTUALIZADA — Fix D del drift pipeline):
      - NO hay time-stretching agresivo en esta fase (eso es Fase 4, cap 1.15x).
      - Placement con TIMELINE ANCHOR: cada segmento intenta colocarse en su
        `seg["start"]` original; solo se empuja si la voz previa invadio mas
        alla de MAX_OVERLAP_S (default 0.3s).
      - El drift NO se acumula — cada seg re-sincroniza con el timeline cuando
        puede. Antes (cursor_s + natural_gap_s) era linealmente acumulativo.
      - Si el total excede la duracion del ambiente, se extiende con silencio.
    """
    with phase_timer(log, "FASE 5 — Mezcla natural vectorizada (numpy)"):
        log.info(f"Input timeline: {json_path}")
        log.info(f"Input ambiente: {ambient_path}")
        log.info(f"Output final audio: {output_final_audio}")
        log_file_info(log, json_path, "timeline_in")
        log_file_info(log, ambient_path, "ambient_in")

        with open(json_path, 'r', encoding='utf-8') as f:
            master_timeline = json.load(f)
        log.info(f"Segmentos a colocar: {len(master_timeline)}")

        # === Cargar ambient === (float32 stereo, si es mono duplicar canales)
        ambient_audio, ambient_sr = sf.read(ambient_path, dtype="float32", always_2d=True)
        if ambient_audio.shape[1] == 1:
            ambient_audio = np.repeat(ambient_audio, 2, axis=1)
        log.info(
            f"Ambient: {len(ambient_audio)/ambient_sr:.2f}s "
            f"@ {ambient_sr}Hz canales={ambient_audio.shape[1]}"
        )

        # === Cargar voces en paralelo ===
        load_specs = []
        skip_count = 0
        for seg in master_timeline:
            p = seg.get("cloned_audio_path")
            if p and os.path.exists(p):
                load_specs.append((seg["segment_id"], p))
            else:
                log.warning(
                    f"  seg[{seg.get('segment_id')}] sin cloned_audio_path o no existe: skip"
                )
                skip_count += 1

        with step_timer(
            log,
            f"Cargando {len(load_specs)} voces en paralelo [workers={PHASE5_LOAD_WORKERS}]"
        ):
            def _load(item):
                sid, path = item
                audio, sr = _load_wav_float(path)
                return sid, audio, sr

            with ThreadPoolExecutor(max_workers=PHASE5_LOAD_WORKERS) as ex:
                loaded = list(ex.map(_load, load_specs))

        # Verificar sr consistente entre voces
        voice_srs = set(sr for _, _, sr in loaded)
        if len(voice_srs) > 1:
            raise RuntimeError(f"Voces con sample rates inconsistentes: {voice_srs}")
        voice_sr = voice_srs.pop() if voice_srs else ambient_sr
        log.info(f"Voice sr={voice_sr}Hz | ambient sr={ambient_sr}Hz")

        # === Resamplear voces si hace falta (TTS emite a 24kHz, ambient a 44.1kHz tipico) ===
        if voice_sr != ambient_sr:
            with step_timer(log, f"Resample voces {voice_sr}Hz -> {ambient_sr}Hz"):
                loaded = [
                    (sid, _resample_if_needed(audio, voice_sr, ambient_sr), ambient_sr)
                    for sid, audio, _ in loaded
                ]

        sr = ambient_sr
        voices_by_id = {sid: audio for sid, audio, _ in loaded}

        # === Calcular posiciones: timeline anchor + overlap controlado ===
        # Para cada seg intentamos colocarlo en su `seg["start"]` original (anchor).
        # Si la voz previa se extendio mas alla del anchor, permitimos hasta
        # MAX_OVERLAP_S de solape; si hace falta mas, empujamos adelante lo minimo.
        # Resultado: el drift NO se acumula — cada segmento re-sincroniza cuando puede.
        #
        # NOTA: removimos el time-stretch pyrubberband 1.45x de esta fase.
        # 1.45x distorsiona la voz audiblemente (suena robotica/apurada).
        # Ahora el stretch se hace en Fase 4 capped a 1.15x (imperceptible),
        # y el residuo sub-15% lo absorbe este placement con overlaps.
        placements = []  # (position_samples, voice_array_1d_mono)
        prev_cursor_s = None   # end del TTS anterior ya colocado
        ok_count = 0
        overlap_count = 0
        max_overlap_seen = 0.0
        max_drift_seen = 0.0
        resync_count = 0       # segmentos que pudieron usar su anchor original
        pushed_count = 0       # segmentos empujados adelante (anchor no alcanzable)

        with step_timer(
            log,
            f"Posicionando {len(master_timeline)} segmentos "
            f"[MAX_OVERLAP_S={PHASE5_MAX_OVERLAP_S}]"
        ):
            for seg in master_timeline:
                sid = seg.get("segment_id")
                if sid not in voices_by_id:
                    continue
                voice_arr = voices_by_id[sid]
                voice_dur_s = len(voice_arr) / sr

                # Anchor: donde deberia ir segun el timeline original
                target_pos_s = float(seg["start"])

                if prev_cursor_s is None:
                    # Primer segmento: usa posicion original sin ajustes
                    pos_s = target_pos_s
                    overlap_s = 0.0
                    resync_count += 1
                else:
                    # Earliest permitido: permitimos overlap con voz previa
                    # hasta MAX_OVERLAP_S (imperceptible a volumen normal).
                    earliest_pos_s = prev_cursor_s - PHASE5_MAX_OVERLAP_S
                    if target_pos_s >= earliest_pos_s:
                        # Cabe en anchor -> re-sincronizamos con timeline original.
                        pos_s = target_pos_s
                        resync_count += 1
                    else:
                        # No cabe: empujamos al earliest permitido (drift minimo).
                        pos_s = earliest_pos_s
                        pushed_count += 1

                    # Overlap real con voz previa (para logging/diagnostico)
                    overlap_s = max(0.0, prev_cursor_s - pos_s)
                    if overlap_s > 0.001:
                        overlap_count += 1
                        if overlap_s > max_overlap_seen:
                            max_overlap_seen = overlap_s

                drift_s = pos_s - target_pos_s  # >= 0 por construccion
                if drift_s > max_drift_seen:
                    max_drift_seen = drift_s

                pos_samples = int(pos_s * sr)
                placements.append((pos_samples, voice_arr))

                log.info(
                    f"  seg[{sid}] {seg['speaker']} "
                    f"target={target_pos_s:.2f}s pos={pos_s:.2f}s "
                    f"drift=+{drift_s:.2f}s overlap={overlap_s:.3f}s "
                    f"dur={voice_dur_s:.2f}s "
                    f"(orig={float(seg['start']):.2f}-{float(seg['end']):.2f}s)"
                )

                prev_cursor_s = pos_s + voice_dur_s
                ok_count += 1

        log.info(
            f"Colocacion: ok={ok_count} skipped={skip_count} | "
            f"resync={resync_count} pushed={pushed_count} | "
            f"overlaps={overlap_count} (max={max_overlap_seen:.3f}s) | "
            f"drift_max_local={max_drift_seen:.2f}s "
            f"(no acumulativo — re-sincronia en cada seg posible)"
        )

        # === Duracion final ===
        voices_end_samples = max((p + len(a) for p, a in placements), default=0)
        final_samples = max(voices_end_samples, len(ambient_audio))
        n_channels = ambient_audio.shape[1]
        log.info(
            f"Voices end: {voices_end_samples/sr:.2f}s | "
            f"Ambient: {len(ambient_audio)/sr:.2f}s | "
            f"Final: {final_samples/sr:.2f}s | channels={n_channels}"
        )

        # Extender ambient con silencio si voces exceden
        if final_samples > len(ambient_audio):
            pad = final_samples - len(ambient_audio)
            log.info(f"Extendiendo ambiente con {pad/sr:.2f}s de silencio")
            ambient_audio = np.concatenate([
                ambient_audio,
                np.zeros((pad, n_channels), dtype=np.float32),
            ], axis=0)

        # === Overlay-add vectorizado: voces en canvas + ambient ===
        # +3 dB gain = factor 10^(3/20) ≈ 1.4125 (mismo gain que la version pydub: `combined_voices + 3`)
        voice_gain = float(10 ** (3 / 20))
        with step_timer(log, f"Overlay-add vectorizado {len(placements)} voces (+3dB)"):
            voices_canvas = np.zeros((final_samples, n_channels), dtype=np.float32)
            for pos, voice_arr in placements:
                end = pos + len(voice_arr)
                # voice_arr es 1D mono → broadcasting a los n_channels
                voices_canvas[pos:end, :] += voice_arr[:, np.newaxis] * voice_gain

        with step_timer(log, "Mezcla final: ambient + voces"):
            mixed = ambient_audio + voices_canvas
            # Clip a [-1, 1] para no distorsionar al exportar a PCM_16
            np.clip(mixed, -1.0, 1.0, out=mixed)

        os.makedirs(os.path.dirname(output_final_audio) or ".", exist_ok=True)
        with step_timer(log, f"Escribiendo audio final ({final_samples/sr:.1f}s WAV PCM_16)"):
            sf.write(output_final_audio, mixed, sr, subtype="PCM_16")

        log_file_info(log, output_final_audio, "final_audio")

    return output_final_audio
