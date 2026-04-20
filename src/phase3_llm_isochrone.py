"""Fase 3: Traduccion EN->ES con MarianMT (Helsinki-NLP/opus-mt-en-es).

Reemplaza la implementacion previa con Gemini (google-generativeai). Motivo:
  - Gemini colgaba indefinidamente sin timeout en pods remotos.
  - Requeria API key externa y cuota.
  - Con la nueva filosofia "lipsync arregla el timing", no necesitamos la
    constraint isocronica de max_chars/duracion. MarianMT traduce natural y
    directo; si el audio doblado sale mas corto o mas largo que el original,
    LatentSync en Fase 6 lo absorbe.

Modelo:
  Helsinki-NLP/opus-mt-en-es (300MB, BLEU ~40). Muy rapido en GPU (2-5ms/seg
  en A100 bf16). Calidad suficiente para dubbing conversacional. No hay
  control de longitud — simplemente traduce el texto entero tal cual.

Batching: simple torch.batch forward — sin split/retry (no puede fallar
silenciosamente como la API remota).
"""

from __future__ import annotations

import os
import json

import torch

from src.logger import get_logger, phase_timer, step_timer, log_file_info

try:
    from src.hw_autotune import autotune, hw_summary
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("phase3")


# Batch size para MarianMT.generate(). El modelo es chico (~300MB); en A100
# 80GB podemos tirar batch=64 sin problema. Escalado por VRAM.
if _HAS_AUTOTUNE:
    MARIAN_BATCH_SIZE = autotune(
        "MARIAN_BATCH_SIZE", baseline=64, scale_with="vram",
        baseline_ref=80, min_v=4, max_v=128,
    )
else:
    MARIAN_BATCH_SIZE = int(os.environ.get("MARIAN_BATCH_SIZE", "16"))

MARIAN_MODEL_ID = os.environ.get("MARIAN_MODEL_ID", "Helsinki-NLP/opus-mt-en-es")
MARIAN_MAX_NEW_TOKENS = int(os.environ.get("MARIAN_MAX_NEW_TOKENS", "512"))


def _resolve_local_model_path() -> str:
    """Prefer local snapshot en /runpod-volume/models si existe; si no, usa hub id.

    ensure_models.py baja el modelo al volumen para que el primer boot no
    dependa de la red de HuggingFace.
    """
    override = os.environ.get("MARIAN_MODEL_PATH", "").strip()
    if override:
        return override
    local = "/runpod-volume/models/Helsinki-NLP/opus-mt-en-es"
    if os.path.isdir(local) and os.path.isfile(os.path.join(local, "config.json")):
        return local
    return MARIAN_MODEL_ID


def _load_marian(log_):
    """Carga MarianMT + tokenizer. bf16 en GPU si hay CUDA, fp32 en CPU como fallback."""
    from transformers import MarianMTModel, MarianTokenizer

    model_path = _resolve_local_model_path()
    log_.info(f"MarianMT path: {model_path}")

    tokenizer = MarianTokenizer.from_pretrained(model_path)

    # use_safetensors=True fuerza a transformers a usar model.safetensors
    # (convertido post-download en ensure_models.py) en vez de pytorch_model.bin,
    # dodgeando el bloqueo por CVE-2025-32434 en torch < 2.6.
    if torch.cuda.is_available():
        model = MarianMTModel.from_pretrained(
            model_path, torch_dtype=torch.float16, use_safetensors=True,
        )
        model = model.to("cuda:0").eval()
        log_.info("MarianMT cargado en CUDA fp16")
    else:
        model = MarianMTModel.from_pretrained(model_path, use_safetensors=True).eval()
        log_.warning("MarianMT cargado en CPU (fp32) — sin CUDA disponible")
    return model, tokenizer


def _translate_batch(model, tokenizer, texts: list[str]) -> list[str]:
    """Traduce una lista de textos EN -> ES con un solo forward batched."""
    if not texts:
        return []
    device = next(model.parameters()).device
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
    ).to(device)
    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=MARIAN_MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def run_phase3_llm_translation(json_path: str, output_path: str):
    """Fase 3: Traduccion EN->ES con MarianMT.

    Firma compatible con la version anterior (Gemini) para no romper main_ui.py.
    Lee timeline JSON de Fase 2, inyecta `text_es` en cada segmento, escribe
    output_path. No aplica ninguna constraint de longitud — Fase 6 (LatentSync)
    se encarga del sincronizado labial sobre el audio doblado natural.
    """
    with phase_timer(log, "FASE 3 — Traduccion EN->ES (MarianMT)"):
        if _HAS_AUTOTUNE:
            log.info(f"Hardware detectado: {hw_summary()}")
        log.info(f"Input JSON: {json_path}")
        log_file_info(log, json_path, "input_timeline")

        with open(json_path, "r", encoding="utf-8") as f:
            master_timeline = json.load(f)
        log.info(f"Segmentos en timeline: {len(master_timeline)}")

        # Pre-marca segmentos vacios como "" para no pegarle al modelo.
        to_translate_idx: list[int] = []
        empty_count = 0
        for i, seg in enumerate(master_timeline):
            text = (seg.get("text_en") or "").strip()
            if text:
                to_translate_idx.append(i)
            else:
                seg["text_es"] = ""
                empty_count += 1
        if empty_count:
            log.info(f"Segmentos vacios skip: {empty_count}")

        if not to_translate_idx:
            log.warning("Timeline sin segmentos traducibles.")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(master_timeline, f, indent=4, ensure_ascii=False)
            return output_path

        with step_timer(log, f"Cargando MarianMT {MARIAN_MODEL_ID}"):
            model, tokenizer = _load_marian(log)

        n = len(to_translate_idx)
        num_batches = (n + MARIAN_BATCH_SIZE - 1) // MARIAN_BATCH_SIZE
        log.info(
            f"Traduciendo {n} segs | batch_size={MARIAN_BATCH_SIZE} "
            f"batches={num_batches} max_new_tokens={MARIAN_MAX_NEW_TOKENS}"
        )

        with step_timer(log, f"MarianMT.generate() x{num_batches} batches"):
            for b_idx, start in enumerate(range(0, n, MARIAN_BATCH_SIZE)):
                end = min(start + MARIAN_BATCH_SIZE, n)
                chunk_idx = to_translate_idx[start:end]
                chunk_texts = [master_timeline[i]["text_en"].strip() for i in chunk_idx]

                translations = _translate_batch(model, tokenizer, chunk_texts)
                if len(translations) != len(chunk_texts):
                    raise RuntimeError(
                        f"MarianMT devolvio {len(translations)} traducciones "
                        f"pero esperabamos {len(chunk_texts)} en batch {b_idx+1}"
                    )
                for gi, es in zip(chunk_idx, translations):
                    master_timeline[gi]["text_es"] = es

                sample_en = chunk_texts[0][:60]
                sample_es = translations[0][:60]
                log.info(
                    f"  batch {b_idx+1}/{num_batches} OK ({len(translations)} segs) "
                    f"ej: '{sample_en}' -> '{sample_es}'"
                )

        # Liberar GPU antes de pasar a Fase 4 (TTS necesita la VRAM).
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validacion: ningun segmento con text_en debe quedar sin text_es.
        missing = [
            seg.get("segment_id", idx)
            for idx, seg in enumerate(master_timeline)
            if seg.get("text_en", "").strip() and not seg.get("text_es", "").strip()
        ]
        if missing:
            raise RuntimeError(
                f"Traduccion incompleta: {len(missing)} segmentos sin text_es. "
                f"IDs: {missing[:20]}{'...' if len(missing) > 20 else ''}"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(master_timeline, f, indent=4, ensure_ascii=False)
        log_file_info(log, output_path, "translated_timeline")

    return output_path
