import os
import json
import time
import signal
import traceback
import numpy as np
import torch
import librosa
import soundfile as sf
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

from src.logger import get_logger, phase_timer, step_timer, log_gpu_snapshot, log_file_info

try:
    from src.hw_autotune import autotune, hw_summary
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("phase4")


REF_DURATION_S = 30.0  # Ventana de referencia para Zero-Shot Cloning

# Chunked mini-batches: balance entre paralelismo y padding waste.
# Calibrado: 64 corre bien en H100 94GB (VRAM ~20-30GB). Escalado lineal por VRAM:
#   94GB -> 64 | 48GB -> ~33 | 24GB -> ~16 | 12GB -> ~8
# Min 2 para no crashear en GPUs muy chicas; max 96 (mas no acelera, padding waste).
# Batch size: re-calibrado agresivo para aprovechar A100 80GB.
# baseline=176 con baseline_ref=80 -> 80GB recibe 176 directo.
#   80GB -> 176 | 94GB -> ~207 (cap a 256) | 48GB -> ~105 | 24GB -> ~52
# El post-load reserva ~4GB (solo pesos); el pico real durante generate()
# con batch=176 y max_new_tokens=1024 se mueve en ~30-45GB en 80GB, muy
# lejos del limite. Si OOM, bajar por env: QWEN_TTS_BATCH_SIZE=128.
if _HAS_AUTOTUNE:
    QWEN_TTS_BATCH_SIZE = autotune(
        "QWEN_TTS_BATCH_SIZE", baseline=176, scale_with="vram",
        baseline_ref=80, min_v=2, max_v=256,
    )
else:
    QWEN_TTS_BATCH_SIZE = int(os.environ.get("QWEN_TTS_BATCH_SIZE", "16"))

# Habilitar torch.compile (mode=reduce-overhead) para +20-30% en generate().
# Set a "0" para desactivar (util para debug).
QWEN_TTS_USE_COMPILE = os.environ.get("QWEN_TTS_USE_COMPILE", "1") not in ("0", "false", "False")

# dynamic=True puede disparar tormentas de recompile al cambiar de shape-bucket,
# y en algunos casos se cuelga con 0 progreso por varios minutos. dynamic=False
# recompila por cada bucket de shape pero el primer paso por bucket es determinista
# y corto (~10-20s). Con bucketing por len(text) los buckets son estables.
# Nuevo default: dynamic=True para evitar recompiles por bucket al subir batch size.
# En A100 80GB con batch=176, el primer chunk paga warmup una vez y ya; sin dynamic
# cada bucket de shape nuevo pagaba 10-20s. Si ves cuelgues >1 min con 0 progreso,
# bajar a "0" por env (pero cuesta recompile por bucket).
QWEN_TTS_COMPILE_DYNAMIC = os.environ.get("QWEN_TTS_COMPILE_DYNAMIC", "1") not in ("0", "false", "False")

# Cap duro de tokens por chunk. Default del modelo es 2048. Con la constraint
# de longitud de Fase 3 (max_chars propocional a dur_s, ~15ch/s), ningun segmento
# necesita generar mas de ~1024 tokens. Bajarlo protege contra runaway generation
# en algun segmento patologico (texto con repeticion infinita, caracteres raros).
QWEN_TTS_MAX_NEW_TOKENS = int(os.environ.get("QWEN_TTS_MAX_NEW_TOKENS", "1024"))

# Watchdog por chunk. Si un chunk no termina en N segundos, mata el proceso con
# log claro en vez de colgarse 10+ minutos en silencio (lo que vimos en prod).
# 0 = desactivado. Default 300s (5 min) es amplio para batch=128 incluso con compile
# warmup. Solo funciona en POSIX (linux/mac). En Windows es no-op.
QWEN_TTS_CHUNK_TIMEOUT_S = int(os.environ.get("QWEN_TTS_CHUNK_TIMEOUT_S", "300"))

# Cada cuantos chunks forzamos torch.cuda.empty_cache. El empty_cache fuerza sync
# CPU<->GPU (barrera dura), asi que llamarlo en CADA chunk frena el pipeline.
# Default subido de 10 -> 50: en A100 80GB tenemos headroom de sobra, y menos
# empty_cache = menos barreras de sync = mas throughput. Bajar solo si ves OOM.
QWEN_TTS_EMPTY_CACHE_EVERY = int(os.environ.get("QWEN_TTS_EMPTY_CACHE_EVERY", "50"))

# Workers para escribir los WAV files al final del pipeline con sf.write.
# sf.write suelta GIL en el encode, asi que escala lineal por CPU count.
if _HAS_AUTOTUNE:
    QWEN_TTS_WRITE_WORKERS = autotune(
        "QWEN_TTS_WRITE_WORKERS", baseline=16, scale_with="cpu",
        baseline_ref=16, min_v=2, max_v=32,
    )
else:
    QWEN_TTS_WRITE_WORKERS = int(os.environ.get("QWEN_TTS_WRITE_WORKERS", "8"))


# ---------------------------------------------------------------------------
# Sin stretch post-TTS — voz natural, lipsync arregla el timing (Fase 6).
# ---------------------------------------------------------------------------
# La version previa medía duracion WAV vs slot original y aceleraba con
# librosa.effects.time_stretch hasta 1.15x para compensar drift. Con la nueva
# filosofia "LatentSync re-sincroniza labios al audio doblado natural", el
# stretch es contraproducente: introduce artefactos y no hace falta — el video
# final se mueve a la velocidad del audio doblado, no al reves.


def get_speaker_reference(voices_path: str, start_sec: float, ref_out_path: str, max_duration_sec: float = REF_DURATION_S):
    """Extrae hasta `max_duration_sec` del locutor original para Zero-Shot Cloning."""
    audio = AudioSegment.from_file(voices_path)
    start_ms = int(start_sec * 1000)
    end_ms = start_ms + int(max_duration_sec * 1000)
    ref_audio = audio[start_ms:end_ms]
    ref_audio.export(ref_out_path, format="wav")
    return ref_out_path


def _build_ref_text_for_window(timeline: list, speaker: str, ref_start: float, ref_end: float) -> str:
    """Concatena text_en de los segmentos del speaker que caen (total o parcialmente) en [ref_start, ref_end].
    Ese texto debe corresponder al audio de referencia para un ICL correcto.
    """
    parts = []
    for s in timeline:
        if s.get("speaker") != speaker:
            continue
        s_start = float(s.get("start", 0.0))
        s_end = float(s.get("end", 0.0))
        if s_end <= ref_start or s_start >= ref_end:
            continue
        t = (s.get("text_en") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


# =============================================================================
# CLONING MODE — CRITICO PARA CALIDAD DEL DOBLAJE
# =============================================================================
# True  = x_vector_only: usa SOLO speaker embedding (timbre). El modelo pronuncia
#         espanol nativo SIN contaminacion del ref EN. Voz mas plana pero:
#           • se entienden las palabras al 100%
#           • NO mezcla palabras en ingles
#           • NO copia prosodia/ritmo del ingles (ideal para TTS doblado)
# False = ICL: condiciona sobre (ref_audio, ref_text) juntos. En teoria copia
#         prosodia + emocion del ref. En la practica con Qwen3-TTS:
#           • en ~20-40% de segmentos, despues de terminar el ES el modelo
#             eco-copia tokens del ref EN → palabras en ingles mezcladas
#           • los sampling defaults agresivos (temp=0.9, top_p=1.0) amplifican
#             el problema en segmentos largos
# Default TRUE porque la calidad "faithful Spanish" >>> "prosody preserved".
USE_X_VECTOR_ONLY = os.environ.get("QWEN_TTS_X_VECTOR_ONLY", "1") not in ("0", "false", "False")

# =============================================================================
# SAMPLING PARAMS — overrides al _merge_generate_kwargs del modelo
# =============================================================================
# Los hard_defaults del modelo (do_sample=True, temp=0.9, top_p=1.0,
# rep_penalty=1.05) estan calibrados para generacion creativa, NO para dubbing
# faithful. Para TTS fidelitario queremos:
#   • temperature bajo → mas deterministico, menos creatividad
#   • top_p moderado → evita tokens de baja probabilidad que meten ruido
#   • repetition_penalty alto → CRITICO contra loops del codec ("disco rayado")
QWEN_TTS_TEMPERATURE = float(os.environ.get("QWEN_TTS_TEMPERATURE", "0.9"))
QWEN_TTS_TOP_P = float(os.environ.get("QWEN_TTS_TOP_P", "0.85"))
QWEN_TTS_TOP_K = int(os.environ.get("QWEN_TTS_TOP_K", "30"))
QWEN_TTS_REPETITION_PENALTY = float(os.environ.get("QWEN_TTS_REPETITION_PENALTY", "1.2"))

# =============================================================================
# RUNAWAY DETECTION — post-TTS duration sanity check
# =============================================================================
# Velocidad de habla en espanol natural: ~14-16 caracteres/segundo.
# Si la duracion del WAV generado excede `len(text_es) / CHARS_PER_SEC *
# RUNAWAY_FACTOR`, el modelo hizo runaway (no disparo EOS, loopeo, o eco del ref).
# Cortamos al limite esperado — mejor un seg truncado que uno contaminado.
QWEN_TTS_CHARS_PER_SEC = float(os.environ.get("QWEN_TTS_CHARS_PER_SEC", "14.0"))
QWEN_TTS_RUNAWAY_FACTOR = float(os.environ.get("QWEN_TTS_RUNAWAY_FACTOR", "1.8"))
# Margen absoluto adicional (segundos) que se tolera ANTES de marcar runaway.
# Protege segmentos cortos (p.ej. "Si." de 3 chars → limite 0.21s sin margen).
QWEN_TTS_RUNAWAY_FLOOR_S = float(os.environ.get("QWEN_TTS_RUNAWAY_FLOOR_S", "2.0"))


def _batched_generate_serial_decode(
    tts_model,
    batch_texts: list,
    batch_refs: list,
    batch_ref_texts: list,
    language: str = "Spanish",
    x_vector_only_mode: bool = USE_X_VECTOR_ONLY,
):
    """Reemplazo de tts_model.generate_voice_clone() que separa las dos etapas:

      1) BATCH en model.generate(): flash_attention_2, sin padding-mask bug → usa VRAM a tope.
      2) SERIAL en speech_tokenizer.decode(): evita el bug de create_sliding_window_causal_mask
         que ocurre con tensores padded bajo SDPA (torch 2.4 + transformers X).

    Modo de clonacion:
      - x_vector_only_mode=True: usa solo el speaker embedding (timbre). El modelo
        pronuncia espanol nativo SIN copiar el acento del audio ingles de referencia.
      - x_vector_only_mode=False (ICL): condiciona sobre (ref_audio, ref_text) juntos →
        copia prosodia y acento del ref (suena agringado al generar espanol).

    El costo de generate() domina (autoregresivo, lento); decode() es vocoder barato.
    Decodificar serial anade overhead despreciable y libra el pipeline del crash.
    """
    model = tts_model.model

    # 1) Construir voice-clone prompts (batched, rapido)
    # En x_vector_only_mode, ref_text es ignorado por el modelo.
    prompt_items = tts_model.create_voice_clone_prompt(
        ref_audio=batch_refs,
        ref_text=batch_ref_texts if not x_vector_only_mode else None,
        x_vector_only_mode=x_vector_only_mode,
    )
    voice_clone_prompt_dict = tts_model._prompt_items_to_voice_clone_prompt(prompt_items)
    ref_texts_for_ids = [it.ref_text for it in prompt_items]

    # 2) Tokenizar textos
    input_texts = [tts_model._build_assistant_text(t) for t in batch_texts]
    input_ids = tts_model._tokenize_texts(input_texts)

    ref_ids = []
    for rt in ref_texts_for_ids:
        if rt is None or rt == "":
            ref_ids.append(None)
        else:
            ref_tok = tts_model._tokenize_texts([tts_model._build_ref_text(rt)])[0]
            ref_ids.append(ref_tok)

    # 3) GENERATE BATCHED (flash_attn_2 — este es el paso pesado, VRAM sube aqui)
    languages = [language] * len(batch_texts)

    # max_new_tokens adaptativo por batch: el codec de Qwen3-TTS corre a 12Hz,
    # pero genera *varios* codes por "step" (codebooks); en la practica medida
    # empiricamente ~2.5 tokens de generate() por caracter de texto generado.
    # Cap del batch = max(len(texts)) * 2.8 + 80 tokens de margen, floor 128,
    # ceiling QWEN_TTS_MAX_NEW_TOKENS. Esto evita que el batch entero pague por
    # un solo seg que no dispara EOS (antes: todos los segs quedaban esperando
    # 1024 tokens, ahora se limita a lo que el seg mas largo legitimamente necesita).
    longest_text = max((len(t) for t in batch_texts), default=0)
    adaptive_max_new = max(128, min(QWEN_TTS_MAX_NEW_TOKENS, int(longest_text * 2.8) + 80))

    # Sampling params tight — defaults del modelo (temp=0.9, top_p=1.0,
    # rep_penalty=1.05) causan loops + contaminacion de lenguaje en dubbing.
    # Pasamos explicitamente valores conservadores para faithful TTS.
    gen_kwargs = tts_model._merge_generate_kwargs(
        max_new_tokens=adaptive_max_new,
        temperature=QWEN_TTS_TEMPERATURE,
        top_p=QWEN_TTS_TOP_P,
        top_k=QWEN_TTS_TOP_K,
        repetition_penalty=QWEN_TTS_REPETITION_PENALTY,
        # sub-talker replica los mismos params (genera los codebooks 2..N
        # condicionados en el primero; mismos defaults sueltos aplican).
        subtalker_temperature=QWEN_TTS_TEMPERATURE,
        subtalker_top_p=QWEN_TTS_TOP_P,
        subtalker_top_k=QWEN_TTS_TOP_K,
    )
    talker_codes_list, _ = model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=languages,
        non_streaming_mode=False,
        **gen_kwargs,
    )

    # 4) Concat ref_code + generated codes por sample (igual que generate_voice_clone)
    ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
    codes_for_decode = []
    for i, codes in enumerate(talker_codes_list):
        if ref_code_list is not None and ref_code_list[i] is not None:
            codes_for_decode.append(
                torch.cat([ref_code_list[i].to(codes.device), codes], dim=0)
            )
        else:
            codes_for_decode.append(codes)

    # 5) DECODE SERIAL (batch=1 → sin padding → sin bug SDPA)
    wavs_out = []
    sr = None
    runaway_stats = []
    for i, codes in enumerate(codes_for_decode):
        wav_list, sr = model.speech_tokenizer.decode([{"audio_codes": codes}])
        wav = wav_list[0]
        # Recortar el prefijo de ref_audio si corresponde (solo en ICL mode)
        if ref_code_list is not None and ref_code_list[i] is not None:
            ref_len = int(ref_code_list[i].shape[0])
            total_len = int(codes.shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]

        # RUNAWAY GUARD: si la duracion excede lo esperado por el texto,
        # el modelo no disparo EOS y esta generando garbage (loops, eco del
        # ref, o silence padding). Truncar en seco al limite esperado evita
        # contaminar la concat final con minutos de basura.
        if sr is not None and sr > 0:
            text_len = len(batch_texts[i])
            expected_s = max(
                0.1,
                text_len / max(QWEN_TTS_CHARS_PER_SEC, 1e-3)
            )
            limit_s = expected_s * QWEN_TTS_RUNAWAY_FACTOR + QWEN_TTS_RUNAWAY_FLOOR_S
            actual_s = wav.shape[0] / sr
            if actual_s > limit_s:
                limit_samples = int(limit_s * sr)
                wav = wav[:limit_samples]
                runaway_stats.append({
                    "idx": i,
                    "text_len": text_len,
                    "expected_s": expected_s,
                    "actual_s": actual_s,
                    "truncated_to_s": limit_s,
                    "preview": batch_texts[i][:50],
                })

        wavs_out.append(wav)

    # Report de runaways del chunk — si hay muchos, hay un problema sistemico
    # (texto con caracteres raros, ref_audio dañado, params aun muy sueltos).
    if runaway_stats:
        log.warning(
            f"RUNAWAY GUARD activo en {len(runaway_stats)}/{len(codes_for_decode)} "
            f"segs del batch (texto excedio {QWEN_TTS_RUNAWAY_FACTOR}x + "
            f"{QWEN_TTS_RUNAWAY_FLOOR_S}s). Truncados en seco."
        )
        for r in runaway_stats[:5]:
            log.warning(
                f"  runaway idx={r['idx']} text={r['text_len']}ch "
                f"expected={r['expected_s']:.1f}s actual={r['actual_s']:.1f}s "
                f"truncado={r['truncated_to_s']:.1f}s  '{r['preview']}'"
            )

    return wavs_out, sr


def run_phase4_tts_cloning(json_path: str, voices_path: str, temp_workspace: str, output_dir: str):
    """Fase 4: Zero-Shot Voice Cloning con Qwen3-TTS."""
    with phase_timer(log, "FASE 4 — Zero-Shot Voice Cloning (Qwen3-TTS)"):
        log.info(f"Input timeline JSON: {json_path}")
        log.info(f"Input voices: {voices_path}")
        log.info(f"Temp workspace: {temp_workspace}")
        log.info(f"Output dir: {output_dir}")
        log_file_info(log, json_path, "timeline_json")
        log_file_info(log, voices_path, "voices_wav")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_workspace, exist_ok=True)

        log_gpu_snapshot(log, "pre-tts-load")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA no disponible para Qwen3-TTS. Fase 4 es GPU-only por diseno — "
                "no hay fallback a CPU. Verifica que el pod tenga GPU asignada."
            )
        torch.cuda.empty_cache()

        # A100 / Ampere: TF32 matmul + cudnn benchmark aceleran operaciones
        # auxiliares (projections, convs). BF16 ya esta activo via dtype=bfloat16.
        # Tambien habilita allow_fp16_reduced_precision para matmuls mixtos.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        dev_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        log.info(
            f"GPU forzada dev={dev_idx}: {props.name} "
            f"sm={props.major}.{props.minor} "
            f"vram={props.total_memory/1e9:.2f}GB "
            f"| tf32=ON cudnn.benchmark=ON compile={'ON' if QWEN_TTS_USE_COMPILE else 'OFF'} "
            f"chunk_size={QWEN_TTS_BATCH_SIZE}"
        )
        # Log explicito de params de sampling y clone-mode. Critico para debugear
        # regresiones de calidad: si cambian los defaults del modelo o alguien
        # pisa un env var, esto lo deja en el pipeline.log del run.
        log.info(
            f"TTS sampling: temp={QWEN_TTS_TEMPERATURE} top_p={QWEN_TTS_TOP_P} "
            f"top_k={QWEN_TTS_TOP_K} rep_penalty={QWEN_TTS_REPETITION_PENALTY} "
            f"max_new_tokens_cap={QWEN_TTS_MAX_NEW_TOKENS} (adaptativo por batch)"
        )
        log.info(
            f"TTS clone-mode: x_vector_only={USE_X_VECTOR_ONLY} "
            f"({'timbre-only, Spanish puro' if USE_X_VECTOR_ONLY else 'ICL con ref_text (riesgo de mezcla EN/ES)'})"
        )
        log.info(
            f"TTS runaway guard: expected=len(text)/{QWEN_TTS_CHARS_PER_SEC}ch/s "
            f"limit=expected*{QWEN_TTS_RUNAWAY_FACTOR}+{QWEN_TTS_RUNAWAY_FLOOR_S}s"
        )

        from qwen_tts import Qwen3TTSModel

        tts_model_path = os.environ.get("QWEN_TTS_PATH", "/app/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        log.info(f"QWEN_TTS_PATH = {tts_model_path}")

        with step_timer(log, "Qwen3-TTS: from_pretrained (bf16, flash_attention_2)"):
            tts_model = Qwen3TTSModel.from_pretrained(
                tts_model_path,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        log_gpu_snapshot(log, "post-tts-load")

        # torch.compile del transformer: reduce kernel launch overhead y fusiona ops.
        # mode=reduce-overhead + dynamic=True porque los chunks pueden variar de tamano
        # (el ultimo chunk suele ser mas chico). Sin dynamic se recompila por shape.
        # Si compile falla internamente, el modelo sigue corriendo en GPU (eager) —
        # NO hay fallback a CPU. torch.compile no mueve tensores entre devices.
        if QWEN_TTS_USE_COMPILE:
            try:
                with step_timer(
                    log,
                    f"torch.compile(mode=reduce-overhead, dynamic={QWEN_TTS_COMPILE_DYNAMIC})"
                ):
                    tts_model.model = torch.compile(
                        tts_model.model,
                        mode="reduce-overhead",
                        fullgraph=False,
                        dynamic=QWEN_TTS_COMPILE_DYNAMIC,
                    )
                log.info(
                    f"torch.compile OK dynamic={QWEN_TTS_COMPILE_DYNAMIC} "
                    f"(primer chunk por bucket incluye warmup ~10-30s)"
                )
            except Exception as e:
                log.warning(
                    f"torch.compile fallo — sigue en eager mode (GPU, sin fallback CPU): {e}"
                )

        with open(json_path, "r", encoding="utf-8") as f:
            master_timeline = json.load(f)
        log.info(f"Segmentos a sintetizar: {len(master_timeline)}")

        # Cachea una referencia por speaker: audio + transcripcion correspondiente
        speaker_refs = {}        # spk -> ref_audio_path
        speaker_ref_texts = {}   # spk -> ref_text (transcripcion de la ventana de referencia)
        with step_timer(log, f"Extrayendo referencias de {REF_DURATION_S:.0f}s por speaker"):
            for seg in master_timeline:
                spk = seg["speaker"]
                if spk in speaker_refs:
                    continue
                ref_start = float(seg["start"])
                ref_end = ref_start + REF_DURATION_S
                ref_path = os.path.join(temp_workspace, f"ref_{spk}.wav")
                get_speaker_reference(voices_path, ref_start, ref_path, max_duration_sec=REF_DURATION_S)
                speaker_refs[spk] = ref_path

                ref_text_concat = _build_ref_text_for_window(
                    master_timeline, spk, ref_start, ref_end
                )
                if not ref_text_concat:
                    # Fallback: usar text_en del propio segmento (no deberia pasar con SPEAKER_00)
                    ref_text_concat = (seg.get("text_en") or "").strip()
                speaker_ref_texts[spk] = ref_text_concat

                log_file_info(log, ref_path, f"ref_{spk}")
                preview = ref_text_concat[:120] + ("..." if len(ref_text_concat) > 120 else "")
                log.info(
                    f"  {spk} ref window=[{ref_start:.1f}s-{ref_end:.1f}s] "
                    f"ref_text={len(ref_text_concat)}ch '{preview}'"
                )

        fail_count = 0
        success_count = 0

        # Construir batch (skip segmentos sin text_es)
        batch_indices: list[int] = []
        batch_texts: list[str] = []
        batch_refs: list[str] = []
        batch_text_en: list[str] = []

        for i, seg in enumerate(master_timeline):
            text_es = (seg.get("text_es") or "").strip()
            if not text_es:
                log.warning(f"  seg[{i}] sin text_es, skip TTS")
                seg["cloned_audio_path"] = None
                continue
            spk = seg["speaker"]
            batch_indices.append(i)
            batch_texts.append(text_es)
            batch_refs.append(speaker_refs[spk])
            # IMPORTANTE: ref_text debe corresponder al ref_audio (misma ventana),
            # no al text_en del segmento actual. ICL condiciona sobre (ref_audio, ref_text) juntos.
            batch_text_en.append(speaker_ref_texts[spk])

        if not batch_texts:
            log.warning("No hay segmentos con text_es — nada que sintetizar")
        else:
            clone_mode = "x_vector_only (timbre sin acento gringo)" if USE_X_VECTOR_ONLY else "ICL (copia prosodia+acento)"
            N = len(batch_texts)

            # BUCKETING: ordenar por len(text_es) para que cada mini-batch tenga
            # secuencias de longitud similar → minimiza padding waste y maximiza
            # utilizacion efectiva de la GPU en cada generate() batched.
            # Mantenemos sorted_indices para remapear wavs al orden original del timeline.
            order = sorted(range(N), key=lambda i: len(batch_texts[i]))
            sorted_texts = [batch_texts[i] for i in order]
            sorted_refs = [batch_refs[i] for i in order]
            sorted_ref_texts = [batch_text_en[i] for i in order]
            sorted_indices = [batch_indices[i] for i in order]

            num_chunks = (N + QWEN_TTS_BATCH_SIZE - 1) // QWEN_TTS_BATCH_SIZE
            log.info(
                f"Batch TTS: {N} segmentos | bucketed by text_len | "
                f"chunk_size={QWEN_TTS_BATCH_SIZE} | chunks={num_chunks} | "
                f"max_new_tokens={QWEN_TTS_MAX_NEW_TOKENS} | "
                f"chunk_timeout={QWEN_TTS_CHUNK_TIMEOUT_S}s | "
                f"clone_mode={clone_mode}"
            )
            log_gpu_snapshot(log, "pre-tts-batch")

            # Watchdog por chunk: SIGALRM mata el proceso si un chunk excede
            # timeout. Solo POSIX — Windows no tiene SIGALRM. En WIN es no-op.
            # signal.signal solo puede registrarse en el main thread; bajo
            # Gradio el worker corre en un thread secundario, asi que ahi
            # desactivamos el watchdog.
            import threading as _threading
            _has_alarm = (
                hasattr(signal, "SIGALRM")
                and _threading.current_thread() is _threading.main_thread()
            )
            def _chunk_timeout_handler(signum, frame):
                raise RuntimeError(
                    f"QWEN_TTS_CHUNK_TIMEOUT_S={QWEN_TTS_CHUNK_TIMEOUT_S}s excedido. "
                    f"Chunk colgado (posible runaway generate o recompile stall). "
                    f"Sugerencia: bajar QWEN_TTS_BATCH_SIZE, o QWEN_TTS_USE_COMPILE=0, "
                    f"o subir QWEN_TTS_CHUNK_TIMEOUT_S si el hardware es lento."
                )
            if _has_alarm and QWEN_TTS_CHUNK_TIMEOUT_S > 0:
                try:
                    signal.signal(signal.SIGALRM, _chunk_timeout_handler)
                except ValueError:
                    _has_alarm = False

            all_wavs: list = []
            sr_set: set = set()
            # Tracking de pico global de VRAM durante generate(). max_memory_allocated
            # persiste el pico desde el ultimo reset; lo reseteamos al entrar al loop
            # para que el pico que reportamos sea solo de este batch, no del load.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                _vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            else:
                _vram_total_gb = 0.0
            try:
                t0_total = time.time()
                for chunk_id, chunk_start in enumerate(range(0, N, QWEN_TTS_BATCH_SIZE)):
                    chunk_end = min(chunk_start + QWEN_TTS_BATCH_SIZE, N)
                    ct = sorted_texts[chunk_start:chunk_end]
                    cr = sorted_refs[chunk_start:chunk_end]
                    crt = sorted_ref_texts[chunk_start:chunk_end]

                    # Log explicito ANTES del step_timer. Si se cuelga adentro,
                    # al menos sabemos que fue en este chunk exacto.
                    max_len_ch = max(len(t) for t in ct) if ct else 0
                    log.info(
                        f"-> chunk[{chunk_id+1}/{num_chunks}] INICIO "
                        f"n={len(ct)} max_text={max_len_ch}ch"
                    )
                    if _has_alarm and QWEN_TTS_CHUNK_TIMEOUT_S > 0:
                        signal.alarm(QWEN_TTS_CHUNK_TIMEOUT_S)
                    try:
                        with step_timer(log, f"chunk[{chunk_id+1}/{num_chunks}] generate+decode (n={len(ct)})"):
                            t0 = time.time()
                            wavs, sr = _batched_generate_serial_decode(
                                tts_model,
                                batch_texts=ct,
                                batch_refs=cr,
                                batch_ref_texts=crt,
                                language="Spanish",
                            )
                            dt = time.time() - t0
                    finally:
                        if _has_alarm and QWEN_TTS_CHUNK_TIMEOUT_S > 0:
                            signal.alarm(0)  # desarmar
                    all_wavs.extend(wavs)
                    sr_set.add(sr)
                    # Pico de VRAM real durante este chunk (allocated + reserved).
                    # Esto revela si el batch size actual esta aprovechando la GPU o
                    # si hay headroom para subirlo mas. Reset del peak por chunk para
                    # que cada log sea independiente del anterior.
                    if torch.cuda.is_available():
                        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                        peak_reserv_gb = torch.cuda.max_memory_reserved() / 1024**3
                        util_pct = (peak_reserv_gb / _vram_total_gb * 100) if _vram_total_gb > 0 else 0.0
                        log.info(
                            f"  chunk[{chunk_id+1}/{num_chunks}] {len(ct)} segs en {dt:.2f}s "
                            f"({dt/len(ct):.2f}s/seg) | "
                            f"VRAM peak: alloc={peak_alloc_gb:.1f}GB "
                            f"reserved={peak_reserv_gb:.1f}GB / {_vram_total_gb:.0f}GB "
                            f"({util_pct:.0f}%)"
                        )
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        log.info(
                            f"  chunk[{chunk_id+1}/{num_chunks}] {len(ct)} segs en {dt:.2f}s "
                            f"({dt/len(ct):.2f}s/seg)"
                        )
                    # Libera activaciones cada N chunks en vez de cada uno: empty_cache
                    # fuerza sync CPU<->GPU (barrera dura); llamarlo cada chunk corta
                    # throughput. El modelo (~4GB) queda persistente en VRAM siempre.
                    if (chunk_id + 1) % QWEN_TTS_EMPTY_CACHE_EVERY == 0:
                        torch.cuda.empty_cache()

                dt_total = time.time() - t0_total
                log_gpu_snapshot(log, "post-tts-batch")

                if len(sr_set) != 1:
                    raise RuntimeError(
                        f"Chunks retornaron sample rates inconsistentes: {sr_set}"
                    )
                sr = sr_set.pop()
                log.info(
                    f"Batch completo: dt={dt_total:.2f}s "
                    f"({dt_total/N:.2f}s/seg promedio) "
                    f"sr={sr} wavs={len(all_wavs)} chunks={num_chunks}"
                )

                if len(all_wavs) != N:
                    raise RuntimeError(
                        f"TTS retorno {len(all_wavs)} wavs pero esperabamos {N}"
                    )

                # ---- Convert tensor -> numpy float32 (sin stretch) -----------
                # Qwen3-TTS devuelve tensor torch; pasamos a numpy para sf.write.
                # Ya NO aplicamos time_stretch: el audio sale natural, el video
                # se re-sincroniza con LatentSync en Fase 6.
                for wav_i, wav in enumerate(all_wavs):
                    if hasattr(wav, "cpu"):
                        wav_np = wav.cpu().numpy()
                    else:
                        wav_np = np.asarray(wav)
                    if wav_np.dtype != np.float32:
                        wav_np = wav_np.astype(np.float32, copy=False)
                    all_wavs[wav_i] = wav_np
                # -------------------------------------------------------------

                # Re-mapear wavs (en orden bucketed) a segmentos (en orden timeline)
                # y escribirlos en paralelo con ThreadPoolExecutor. sf.write suelta GIL
                # en el encode, asi que 16 workers saturan disco + vCPU sin penalty.
                write_jobs = []
                for wav, idx in zip(all_wavs, sorted_indices):
                    seg = master_timeline[idx]
                    cloned_out_path = os.path.join(
                        output_dir, f"cloned_{seg['segment_id']:05d}.wav"
                    )
                    seg["cloned_audio_path"] = cloned_out_path
                    write_jobs.append((idx, wav, cloned_out_path))

                def _do_write(job):
                    idx, wav, path = job
                    sf.write(path, wav, sr)
                    return idx, path, os.path.getsize(path)

                with step_timer(
                    log,
                    f"Guardando {len(write_jobs)} WAVs [workers={QWEN_TTS_WRITE_WORKERS}]"
                ):
                    with ThreadPoolExecutor(max_workers=QWEN_TTS_WRITE_WORKERS) as ex:
                        for idx, path, size_bytes in ex.map(_do_write, write_jobs):
                            seg = master_timeline[idx]
                            size_kb = size_bytes / 1024
                            success_count += 1
                            log.info(
                                f"  seg[{idx}] OK {seg['speaker']} "
                                f"ES={len(seg['text_es'])}ch "
                                f"file={os.path.basename(path)} ({size_kb:.1f}KB)"
                            )

            except torch.cuda.OutOfMemoryError as e:
                fail_count = N
                log.error("=" * 70)
                log.error(f"[TTS OOM] chunk_size={QWEN_TTS_BATCH_SIZE} fue demasiado grande.")
                log.error(f"[TTS OOM] Exporta QWEN_TTS_BATCH_SIZE=16 (o 8) y reintenta.")
                log.error(f"[TTS OOM] Excepcion: {e}")
                log.error("=" * 70)
                raise RuntimeError(
                    f"Qwen3-TTS OOM con QWEN_TTS_BATCH_SIZE={QWEN_TTS_BATCH_SIZE}. "
                    f"Baja el env var (ej: 16 o 8) y reintenta. "
                    f"NO hay fallback a CPU por diseno."
                ) from e

            except Exception as e:
                fail_count = N
                log.error("=" * 70)
                log.error(f"[TTS FAIL] batch de {N} segmentos EXPLOTO")
                log.error(f"[TTS FAIL] sorted_indices: {sorted_indices}")
                log.error(f"[TTS FAIL] Excepcion: {type(e).__name__}: {e}")
                log.error(f"[TTS FAIL] Traceback:\n{traceback.format_exc()}")
                log.error("=" * 70)
                raise RuntimeError(
                    f"Qwen3-TTS fallo en batch de {N} segmentos. "
                    f"Sin fallback silencioso: revisa log arriba."
                ) from e

        log.info(f"Resumen: {success_count} exitos, {fail_count} fallos")

        del tts_model
        torch.cuda.empty_cache()
        log_gpu_snapshot(log, "post-tts-free")

        updated_json = os.path.join(temp_workspace, "master_timeline_with_audio.json")
        with open(updated_json, "w", encoding="utf-8") as f:
            json.dump(master_timeline, f, indent=2, ensure_ascii=False)
        log_file_info(log, updated_json, "timeline_with_audio")

    return updated_json
