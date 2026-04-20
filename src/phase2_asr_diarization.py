import os
import json
import torch
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor

from src.logger import get_logger, phase_timer, step_timer, log_gpu_snapshot, log_file_info

try:
    from src.hw_autotune import autotune, hw_summary
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("phase2")

# Parametros del chunking / segmentacion
CHUNK_DURATION_S = 25.0      # tamano de cada chunk que entra al ASR
CHUNK_OVERLAP_S = 1.0        # overlap entre chunks para no cortar palabras
PAUSE_SPLIT_S = 0.5          # gap entre palabras que dispara nuevo segmento
MAX_SEG_DURATION_S = 12.0    # duracion maxima de un segmento logico
MIN_SEG_DURATION_S = 0.5     # duracion minima (descartar ruido)
SINGLE_SPEAKER_ID = "SPEAKER_00"

# Batch size de Qwen3-ASR. Calibrado: 96 corre bien en H100 94GB VRAM con bf16 +
# flash_attention_2. Escalado lineal: 24GB pod -> ~24, 48GB -> ~49, etc.
# Min 2 para no crashear en GPUs muy chicas; max 128 (sweet-spot del modelo).
if _HAS_AUTOTUNE:
    QWEN_ASR_BATCH_SIZE = autotune(
        "QWEN_ASR_BATCH_SIZE", baseline=96, scale_with="vram",
        baseline_ref=94, min_v=2, max_v=128,
    )
else:
    QWEN_ASR_BATCH_SIZE = int(os.environ.get("QWEN_ASR_BATCH_SIZE", "24"))

# Workers para el slicing CPU-serial de chunks (librosa decode + sf.write).
# Escalado por CPU count: 16 vCPU -> 16 workers, 8 vCPU -> 8 workers, etc.
if _HAS_AUTOTUNE:
    ASR_SLICE_WORKERS = autotune(
        "ASR_SLICE_WORKERS", baseline=16, scale_with="cpu",
        baseline_ref=16, min_v=2, max_v=32,
    )
else:
    ASR_SLICE_WORKERS = int(os.environ.get("ASR_SLICE_WORKERS", "8"))


def _slice_chunk(vocals_path: str, start_s: float, duration_s: float, target_sr: int, out_path: str):
    """Recorta [start, start+duration] del audio a wav mono 16k para Qwen3-ASR."""
    y, _ = librosa.load(vocals_path, sr=target_sr, mono=True, offset=start_s, duration=duration_s)
    sf.write(out_path, y, target_sr, subtype="PCM_16")
    return out_path


def _audio_duration_s(path: str) -> float:
    try:
        return float(librosa.get_duration(path=path))
    except TypeError:
        return float(librosa.get_duration(filename=path))


def _extract_words_from_result(r, chunk_offset_s: float) -> list[dict]:
    """Extrae word-level timestamps de un ASR result. chunk_offset_s se suma al tiempo local."""
    words = []
    try:
        raw_ts = r.time_stamps if r.time_stamps else []
        if raw_ts and not hasattr(raw_ts[0], "text") and hasattr(raw_ts[0], "__iter__"):
            items = raw_ts[0]
        else:
            items = raw_ts
        for ws in items:
            words.append({
                "text": ws.text,
                "start": chunk_offset_s + float(ws.start_time),
                "end": chunk_offset_s + float(ws.end_time),
            })
    except (AttributeError, IndexError, TypeError) as e:
        log.warning(f"  extract_words: {type(e).__name__}: {e}")
    return words


def _group_words_into_segments(words: list[dict]) -> list[dict]:
    """Agrupa palabras en segmentos logicos por:
    - Pausa entre palabras > PAUSE_SPLIT_S
    - Duracion acumulada > MAX_SEG_DURATION_S
    """
    if not words:
        return []

    segments = []
    current = {"words": [words[0]], "start": words[0]["start"], "end": words[0]["end"]}

    for w in words[1:]:
        gap = w["start"] - current["end"]
        seg_len = w["end"] - current["start"]

        if gap > PAUSE_SPLIT_S or seg_len > MAX_SEG_DURATION_S:
            segments.append(current)
            current = {"words": [w], "start": w["start"], "end": w["end"]}
        else:
            current["words"].append(w)
            current["end"] = w["end"]

    segments.append(current)

    segments = [s for s in segments if (s["end"] - s["start"]) >= MIN_SEG_DURATION_S]
    return segments


def _dedupe_overlapping_words(all_words: list[dict]) -> list[dict]:
    """Elimina palabras duplicadas producidas por el overlap entre chunks.
    Dos palabras se consideran duplicadas si su rango temporal se solapa significativamente
    y su texto coincide (case-insensitive, trimmed).
    """
    if not all_words:
        return []

    all_words.sort(key=lambda w: w["start"])
    deduped = [all_words[0]]
    for w in all_words[1:]:
        prev = deduped[-1]
        overlaps = w["start"] < prev["end"] - 0.05
        same_text = w["text"].strip().lower() == prev["text"].strip().lower()
        if overlaps and same_text:
            continue
        if overlaps and not same_text:
            w = {**w, "start": max(w["start"], prev["end"])}
        deduped.append(w)
    return deduped


def run_phase2_diarization_and_asr(
    vocals_path: str,
    output_json: str,
    temp_workspace: str = None,
):
    """Fase 2 (sin diarizacion): Qwen3-ASR directo con forced aligner.
    Agrupa word-timestamps en segmentos logicos por pausas naturales.
    Todos los segmentos se asignan a SPEAKER_00 (single-speaker assumption).
    """
    with phase_timer(log, "FASE 2 — ASR + Segmentacion por Pausas"):
        log.info(f"Input vocals: {vocals_path}")
        log_file_info(log, vocals_path, "vocals_input")

        log_gpu_snapshot(log, "pre-asr")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA no disponible dentro del contenedor. "
                "Verifica que el pod tenga GPU asignada (nvidia-smi) y runtime NVIDIA activo."
            )
        torch.cuda.empty_cache()

        if temp_workspace is None:
            temp_workspace = os.path.dirname(vocals_path)
        chunk_dir = os.path.join(temp_workspace, "asr_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        log.info(f"Chunks dir: {chunk_dir}")

        total_duration = _audio_duration_s(vocals_path)
        log.info(f"Duracion total del audio: {total_duration:.2f}s")

        # Generar chunks con overlap
        chunk_specs = []
        t = 0.0
        step = max(1.0, CHUNK_DURATION_S - CHUNK_OVERLAP_S)
        idx = 0
        while t < total_duration:
            dur = min(CHUNK_DURATION_S, total_duration - t)
            if dur < 0.5:
                break
            chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.wav")
            chunk_specs.append({"idx": idx, "start": t, "duration": dur, "path": chunk_path})
            t += step
            idx += 1
        log.info(f"Generando {len(chunk_specs)} chunks de ~{CHUNK_DURATION_S}s "
                 f"con {CHUNK_OVERLAP_S}s overlap")

        target_sr = 16000
        with step_timer(
            log,
            f"Recortando {len(chunk_specs)} chunks a 16kHz mono "
            f"[workers={ASR_SLICE_WORKERS}]"
        ):
            # Paralelizamos con threads — librosa suelta el GIL en la parte C (soundfile
            # decode), asi que threads saturan los 16 vCPU del pod sin penalty.
            def _do_slice(spec):
                _slice_chunk(vocals_path, spec["start"], spec["duration"], target_sr, spec["path"])
                return spec["idx"]

            with ThreadPoolExecutor(max_workers=ASR_SLICE_WORKERS) as ex:
                # list() fuerza a que todos terminen antes de salir del context manager.
                list(ex.map(_do_slice, chunk_specs))

        # Cargar Qwen3-ASR
        from qwen_asr import Qwen3ASRModel

        asr_model_path = os.environ.get("QWEN_ASR_PATH", "/app/models/Qwen/Qwen3-ASR-1.7B")
        aligner_path = os.environ.get("QWEN_ALIGNER_PATH", "/app/models/Qwen/Qwen3-ForcedAligner-0.6B")
        log.info(f"QWEN_ASR_PATH = {asr_model_path}")
        log.info(f"QWEN_ALIGNER_PATH = {aligner_path}")

        with step_timer(
            log,
            f"Qwen3-ASR + Aligner: from_pretrained (bf16, flash_attention_2, "
            f"batch={QWEN_ASR_BATCH_SIZE})"
        ):
            asr_model = Qwen3ASRModel.from_pretrained(
                asr_model_path,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                attn_implementation="flash_attention_2",
                max_inference_batch_size=QWEN_ASR_BATCH_SIZE,
                max_new_tokens=256,
                forced_aligner=aligner_path,
                forced_aligner_kwargs=dict(
                    dtype=torch.bfloat16,
                    device_map="cuda:0",
                    attn_implementation="flash_attention_2",
                ),
            )
        log_gpu_snapshot(log, "post-asr-load")

        chunk_paths = [spec["path"] for spec in chunk_specs]
        log.info(f"Transcribiendo {len(chunk_paths)} chunks con Qwen3-ASR (batch)...")
        with step_timer(log, "Qwen3-ASR.transcribe(batch) con forced_aligner"):
            results = asr_model.transcribe(
                audio=chunk_paths,
                language="English",
                return_time_stamps=True,
            )
        log.info(f"ASR retorno {len(results)} resultados")

        # Agregar words de todos los chunks con offsets globales
        all_words = []
        for spec, r in zip(chunk_specs, results):
            chunk_text = (r.text or "").strip()
            words = _extract_words_from_result(r, chunk_offset_s=spec["start"])
            log.info(f"  chunk[{spec['idx']}] [{spec['start']:.1f}s-"
                     f"{spec['start']+spec['duration']:.1f}s] "
                     f"words={len(words)} text='{chunk_text[:80]}'")
            all_words.extend(words)

        log.info(f"Total words antes de dedupe: {len(all_words)}")
        all_words = _dedupe_overlapping_words(all_words)
        log.info(f"Total words despues de dedupe: {len(all_words)}")

        with step_timer(log, "Agrupando words en segmentos logicos por pausas"):
            raw_segments = _group_words_into_segments(all_words)
        log.info(f"Segmentos logicos generados: {len(raw_segments)} "
                 f"(pause>{PAUSE_SPLIT_S}s, max_dur={MAX_SEG_DURATION_S}s)")

        master_timeline = []
        for i, seg in enumerate(raw_segments):
            text = " ".join(w["text"] for w in seg["words"]).strip()
            start = float(seg["start"])
            end = float(seg["end"])
            master_timeline.append({
                "segment_id": i,
                "speaker": SINGLE_SPEAKER_ID,
                "start": start,
                "end": end,
                "duration": end - start,
                "text_en": text,
                "language": "English",
                "words": seg["words"],
            })
            preview = text[:80] + ("..." if len(text) > 80 else "")
            log.info(f"  seg[{i}] [{start:.2f}s-{end:.2f}s] "
                     f"dur={end-start:.2f}s words={len(seg['words'])} text='{preview}'")

        del asr_model
        torch.cuda.empty_cache()
        log_gpu_snapshot(log, "post-asr-free")

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(master_timeline, f, indent=2, ensure_ascii=False)
        log_file_info(log, output_json, "timeline_json")

        log.info(f"Total segmentos en timeline: {len(master_timeline)}")

    return output_json
