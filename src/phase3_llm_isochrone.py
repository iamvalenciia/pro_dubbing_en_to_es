import os
import json
import asyncio
import google.generativeai as genai

from src.logger import get_logger, phase_timer, step_timer, log_file_info

try:
    # Autotune opcional: si el modulo existe, escala defaults a la VRAM/CPU reales.
    from src.hw_autotune import autotune, hw_summary
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("phase3")


# ---------------------------------------------------------------------------
# Defaults autotuneados
# ---------------------------------------------------------------------------
# IMPORTANTE: se bajo el default de 32 -> 12 porque con 32 segmentos largos
# Gemini a veces devuelve JSON + tokens extras (hit max_tokens mid-array), lo
# que rompia json.loads. Con 12 el prompt es mas corto y cabe siempre.
# autotune() lo escala por numero de CPUs (Gemini es red + parsing JSON, no
# GPU-bound), pero con tope bajo para no saturar ni rate-limitar la API.
if _HAS_AUTOTUNE:
    GEMINI_BATCH_SIZE = autotune(
        "GEMINI_BATCH_SIZE", baseline=12, scale_with="cpu",
        baseline_ref=8, min_v=6, max_v=20,
    )
    GEMINI_CONCURRENCY = autotune(
        "GEMINI_CONCURRENCY", baseline=12, scale_with="cpu",
        baseline_ref=8, min_v=4, max_v=24,
    )
else:
    GEMINI_BATCH_SIZE = int(os.environ.get("GEMINI_BATCH_SIZE", "12"))
    GEMINI_CONCURRENCY = int(os.environ.get("GEMINI_CONCURRENCY", "12"))

# Retries por batch con backoff exponencial (1s, 2s, 4s, 8s).
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "4"))

# Minimo tamaño al que permitimos dividir un batch fallido antes de rendirnos.
# Si un batch de size=1 sigue fallando, es un error real (no de tamaño).
GEMINI_MIN_SPLIT_SIZE = int(os.environ.get("GEMINI_MIN_SPLIT_SIZE", "1"))


# ---------------------------------------------------------------------------
# Isocronia por presupuesto de caracteres
# ---------------------------------------------------------------------------
# El castellano hablado nativo ronda ~14-16 caracteres/segundo en conversacion
# natural. Usamos 15 como target: si una traduccion supera `duration * 15` chars,
# casi seguro el TTS generara audio mas largo que el slot original y veremos drift
# acumulado en Fase 5. Gemini recibe este presupuesto explicito en el prompt.
#
# No es un corte duro: Fase 4 aplica time-stretch (hasta 1.15x) para compensar
# traducciones que excedan marginalmente, y Fase 5 permite micro-solapes. Pero
# si Gemini respeta este presupuesto, la cadena aguas abajo no tiene que trabajar.
PHASE3_CHARS_PER_SEC = float(os.environ.get("PHASE3_CHARS_PER_SEC", "15"))
# Piso minimo para segmentos muy cortos (<~1.3s): si no, quedan sin margen para
# nada util ("sí", "no", "ok"). 20 chars alcanza para cualquier interjeccion.
PHASE3_MIN_MAX_CHARS = int(os.environ.get("PHASE3_MIN_MAX_CHARS", "20"))


def _max_chars_for_duration(dur_s: float) -> int:
    """Target de caracteres para que el ES hablado ~15ch/s entre en dur_s."""
    return max(PHASE3_MIN_MAX_CHARS, int(dur_s * PHASE3_CHARS_PER_SEC))


_BATCH_PROMPT_HEADER = (
    'You translate English to natural, fluent Spanish for TTS dubbing.\n'
    'Each segment is paired with a SOURCE DURATION and a MAX_CHARS budget.\n'
    'The Spanish translation MUST fit that window when spoken at ~15 chars/second,\n'
    'otherwise the dubbing drifts out of sync with the video.\n'
    '\n'
    'LENGTH CONSTRAINT (CRITICAL — dubbing loses lip-sync otherwise):\n'
    '- Each segment has a `max_chars` budget. STAY AT OR BELOW that budget.\n'
    '- Spanish is ~20-25% longer than English by default; you MUST compress.\n'
    '- Drop fillers aggressively: "you know", "I mean", "kind of", "like",\n'
    '  "just", "really", "actually", "basically", "literally", "sort of".\n'
    '- Prefer short synonyms: "pero" > "sin embargo"; "hay que" > "es necesario";\n'
    '  "para" > "con el fin de"; "aunque" > "a pesar de que"; "si" > "en caso de".\n'
    '- Use contractions: "del", "al" (never "de el", "a el").\n'
    '- Cut redundancy: "start to run" -> "corre"; "begin to see" -> "ve";\n'
    '  "try to figure out" -> "averiguar"; "make sure" -> "asegurarse".\n'
    '- Prefer verbs over noun phrases: "hacer una llamada" -> "llamar".\n'
    '- Losing minor detail is OK. Exceeding max_chars is NOT OK.\n'
    '\n'
    'MEANING & STYLE:\n'
    '- Preserve core meaning, tone and emotional register faithfully.\n'
    '- Use idiomatic Spanish phrasing, not literal word-for-word.\n'
    '\n'
    'PROSODY / EMOTION MARKERS (critical — the TTS uses these to intonate):\n'
    '- Use Spanish opening marks: "¿...?" for questions and "¡...!" for exclamations.\n'
    '  If the EN is a question or exclamation, the ES MUST have both opening and closing marks.\n'
    '- Preserve or add emphasis punctuation when it fits the emotion:\n'
    '  ellipsis (…) for hesitation/trailing off, em-dash (—) for sharp pauses,\n'
    '  commas for natural breathing, "¡¿...?!" only if the original is very emphatic.\n'
    '- Keep interjections and discourse markers ("oh"/"wow"/"hey" -> "oh"/"guau"/"oye", etc.).\n'
    '- Do NOT flatten exclamations into statements. If EN has energy, ES must too.\n'
    '\n'
    'OUTPUT FORMAT (strict):\n'
    '- Return ONLY a JSON array of strings, one translation per input segment, in the SAME ORDER.\n'
    '- The array length MUST match exactly the number of input segments.\n'
    '- No markdown, no explanation, no keys, no numbering — just the JSON array: ["trad1", "trad2", ...]\n'
    '\n'
    'SEGMENTS TO TRANSLATE (format: "N. [dur=Xs, max_chars=M] English text"):\n'
)


def _build_batch_prompt(texts, durations):
    """Lista numerada con header + constraint de longitud por segmento.

    Cada item: "N. [dur=Xs, max_chars=M] English text" — Gemini usa el max_chars
    como presupuesto duro. Si una traduccion lo excede, el wrapper aguas abajo
    (Fase 4 speed-up, Fase 5 overlap) la absorbe, pero es mejor que Gemini la
    compense en origen.
    """
    lines = []
    for i, (t, d) in enumerate(zip(texts, durations)):
        mx = _max_chars_for_duration(d)
        lines.append(f"{i+1}. [dur={d:.2f}s, max_chars={mx}] {t}")
    return _BATCH_PROMPT_HEADER + "\n".join(lines)


def _strip_code_fences(txt: str) -> str:
    """Remueve fences ``` o ```json si los hay."""
    if txt.startswith("```"):
        lines = txt.split("\n")
        # drop first fence line
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        # drop trailing fence(s)
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    return txt


def _extract_first_json_value(txt: str):
    """Extrae la PRIMERA estructura JSON valida del texto, tolerando basura al final.

    Gemini a veces devuelve:
      '[\\n  "trad1",\\n  "trad2"\\n]\\n\\nAlguna cola extra'
    json.loads() revienta con "Extra data"; raw_decode() solo consume hasta
    cerrar la estructura y nos permite ignorar el resto.
    """
    decoder = json.JSONDecoder()
    # Buscamos el primer '[' o '{' para arrancar raw_decode alli.
    for i, ch in enumerate(txt):
        if ch in "[{":
            try:
                value, _end = decoder.raw_decode(txt[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("No se encontro JSON valido en la respuesta.", txt, 0)


def _parse_batch_response(raw_text: str, expected_n: int):
    """Parsea respuesta JSON tolerando markdown fences y cola extra de tokens."""
    txt = _strip_code_fences((raw_text or "").strip())

    try:
        parsed = _extract_first_json_value(txt)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini no devolvio JSON valido: {e}. Respuesta cruda: {txt[:500]!r}"
        ) from e

    if not isinstance(parsed, list):
        raise RuntimeError(
            f"Gemini devolvio {type(parsed).__name__}, esperaba list. "
            f"Respuesta cruda: {txt[:500]!r}"
        )
    if len(parsed) != expected_n:
        raise RuntimeError(
            f"Gemini devolvio {len(parsed)} traducciones, esperaba {expected_n}. "
            f"Respuesta: {parsed!r}"
        )
    return [str(x).strip() for x in parsed]


async def _call_gemini_once(model, texts, durations, sem, retries, log_, label):
    """Una sola llamada a Gemini con retry exponencial. Retorna list[str] de tamaño len(texts).

    `durations` es paralelo a `texts` — se usa para calcular max_chars por segmento
    y meterlo en el prompt como constraint duro.

    Falla si despues de `retries` intentos sigue sin devolver JSON valido/completo.
    """
    prompt = _build_batch_prompt(texts, durations)
    expected_n = len(texts)

    async with sem:
        last_err = None
        for attempt in range(retries):
            try:
                # google-generativeai es sync; lo corremos en threadpool para no bloquear el loop.
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                )
                raw = response.text or ""
                translations = _parse_batch_response(raw, expected_n)
                # Validacion dura: ninguna traduccion puede quedar vacia.
                for k, t in enumerate(translations):
                    if not t:
                        raise RuntimeError(
                            f"Traduccion vacia para item {k} en {label}."
                        )
                return translations
            except Exception as e:
                last_err = e
                backoff = min(2 ** attempt, 16)
                log_.warning(
                    f"  [{label}] intento {attempt+1}/{retries} fallo: "
                    f"{type(e).__name__}: {e}. Reintentando en {backoff}s..."
                )
                await asyncio.sleep(backoff)
        raise RuntimeError(
            f"{label} fallo definitivamente despues de {retries} intentos. "
            f"Ultimo error: {last_err}"
        )


async def _translate_with_split_fallback(model, batch_idx, batch_texts, batch_durs, sem, retries, log_, depth=0):
    """Traduce un batch; si falla despues de retries, lo PARTE EN DOS y reintenta recursivamente.

    Asi un pod mal dimensionado nunca tumba el pipeline: incluso batches de 1 seg
    tienen su chance. Solo si falla con size=GEMINI_MIN_SPLIT_SIZE explotamos.

    `batch_durs` es paralelo a `batch_texts` y se splitea igual que ellos al dividir.
    """
    label = f"batch {batch_idx}" + (f" (depth={depth})" if depth else "")
    try:
        translations = await _call_gemini_once(model, batch_texts, batch_durs, sem, retries, log_, label)
        return batch_idx, translations
    except Exception as e:
        n = len(batch_texts)
        if n <= GEMINI_MIN_SPLIT_SIZE:
            # Llegamos a size minimo y sigue sin servir -> error real del prompt/API.
            raise RuntimeError(
                f"{label} (size={n}) fallo incluso al tamaño minimo: {e}"
            ) from e

        half = n // 2
        left_t, right_t = batch_texts[:half], batch_texts[half:]
        left_d, right_d = batch_durs[:half], batch_durs[half:]
        log_.warning(
            f"  [{label}] size={n} persiste fallo. Partiendo en {len(left_t)}+{len(right_t)} "
            f"y reintentando recursivamente (depth={depth+1}). Causa: {type(e).__name__}: {e}"
        )
        # Las dos mitades corren en paralelo (comparten el mismo semaforo global).
        (_, left_tr), (_, right_tr) = await asyncio.gather(
            _translate_with_split_fallback(model, batch_idx, left_t, left_d, sem, retries, log_, depth=depth+1),
            _translate_with_split_fallback(model, batch_idx, right_t, right_d, sem, retries, log_, depth=depth+1),
        )
        return batch_idx, left_tr + right_tr


async def _translate_all(model, timeline, batch_size, concurrency, retries, log_):
    """Divide timeline en batches y los despacha en paralelo con sem=concurrency."""
    # Solo segmentos con texto real; los vacios se marcan "" directo sin pegarle al LLM.
    indexed = [(i, seg) for i, seg in enumerate(timeline)
               if seg.get("text_en", "").strip()]

    if not indexed:
        log_.warning("Timeline sin segmentos traducibles (todos text_en vacios).")
        return

    # Chunk en batches fijos de batch_size.
    batches = []
    for start in range(0, len(indexed), batch_size):
        batches.append(indexed[start:start + batch_size])

    log_.info(
        f"Traduccion batcheada: total_segs={len(indexed)} "
        f"batches={len(batches)} batch_size={batch_size} concurrency={concurrency}"
    )

    sem = asyncio.Semaphore(concurrency)

    def _seg_duration(seg):
        # `duration` es el campo canonico; si falta por data vieja, derivamos de end-start.
        d = seg.get("duration")
        if d is None:
            d = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        return max(0.1, float(d))  # piso a 0.1s — evita max_chars=0 en timestamps raros

    tasks = [
        _translate_with_split_fallback(
            model, batch_idx,
            [seg["text_en"].strip() for _, seg in batch_items],
            [_seg_duration(seg) for _, seg in batch_items],
            sem, retries, log_,
        )
        for batch_idx, batch_items in enumerate(batches)
    ]

    results = await asyncio.gather(*tasks)

    # Reinyecta respetando el mapping global_i -> traduccion.
    for batch_idx, translations in results:
        batch_items = batches[batch_idx]
        for (global_i, _seg), text_es in zip(batch_items, translations):
            timeline[global_i]["text_es"] = text_es

        # Log de ejemplo por batch para no spamear.
        sample_en = batch_items[0][1]["text_en"][:50]
        sample_es = translations[0][:50]
        log_.info(
            f"  batch {batch_idx+1}/{len(batches)} OK ({len(translations)} segs) "
            f"ej: '{sample_en}' -> '{sample_es}'"
        )


def run_phase3_llm_translation(json_path: str, output_path: str):
    """Fase 3: Traduccion isocronica EN->ES via Gemini, BATCHED + CONCURRENT + SELF-HEALING.

    Diferencias vs la version previa (segmento-por-segmento, serial):
      - Multi-segmento por prompt (GEMINI_BATCH_SIZE): un request traduce N segs.
      - Paralelismo via asyncio.gather + Semaphore(GEMINI_CONCURRENCY): los batches
        corren simultaneos.
      - response_mime_type=application/json + raw_decode: Gemini devuelve array JSON,
        parseo tolerante a tokens extra (cola de texto despues del `]`).
      - Retry con backoff exponencial + SPLIT-AND-RETRY: si un batch falla los N
        reintentos, se parte en dos y cada mitad reintenta recursivamente hasta
        llegar a size=1. Asi un fallo en 1 segmento no tumba 32.

    Env vars:
      GEMINI_BATCH_SIZE (default 12, auto-escalado por CPU): segmentos por prompt.
      GEMINI_CONCURRENCY (default 12, auto-escalado por CPU): batches en paralelo.
      GEMINI_MAX_RETRIES (default 4): reintentos por batch antes de split.
      GEMINI_MIN_SPLIT_SIZE (default 1): tamaño minimo al que NO seguimos dividiendo.
      GEMINI_MODEL (default gemini-3.1-flash-lite-preview).
      API_GOOGLE_STUDIO: API key (requerido).
      PHASE3_CHARS_PER_SEC (default 15): target de chars/s del ES hablado — determina
        max_chars por segmento. Bajar (ej 13) = traducciones mas cortas = menos drift
        pero mas paraphrasing agresivo.
      PHASE3_MIN_MAX_CHARS (default 20): piso minimo del max_chars para segmentos
        muy cortos (<1.3s), evita presupuesto=0.
    """
    with phase_timer(log, "FASE 3 — Traduccion Isocronica (Gemini) [BATCHED+CONCURRENT]"):
        if _HAS_AUTOTUNE:
            log.info(f"Hardware detectado: {hw_summary()}")
        log.info(f"Input JSON: {json_path}")
        log_file_info(log, json_path, "input_timeline")

        api_key = os.environ.get("API_GOOGLE_STUDIO")
        if not api_key:
            raise ValueError("API_GOOGLE_STUDIO env variable is missing for Gemini.")
        log.info(f"API_GOOGLE_STUDIO presente (len={len(api_key)})")

        genai.configure(api_key=api_key)
        default_model = "gemini-3.1-flash-lite-preview"
        model_name = os.environ.get("GEMINI_MODEL", default_model)
        log.info(
            f"Gemini: model={model_name} | "
            f"batch_size={GEMINI_BATCH_SIZE} concurrency={GEMINI_CONCURRENCY} "
            f"retries={GEMINI_MAX_RETRIES}"
        )
        model = genai.GenerativeModel(model_name)

        with open(json_path, 'r', encoding='utf-8') as f:
            master_timeline = json.load(f)
        log.info(f"Segmentos en timeline: {len(master_timeline)}")

        # Pre-marca segmentos vacios como "" para no pegarle al LLM.
        empty_count = 0
        for seg in master_timeline:
            if not seg.get("text_en", "").strip():
                seg["text_es"] = ""
                empty_count += 1
        if empty_count:
            log.info(f"Segmentos vacios pre-marcados (skip LLM): {empty_count}")

        with step_timer(log, f"Traduciendo EN->ES [batch={GEMINI_BATCH_SIZE} conc={GEMINI_CONCURRENCY}]"):
            asyncio.run(_translate_all(
                model, master_timeline,
                batch_size=GEMINI_BATCH_SIZE,
                concurrency=GEMINI_CONCURRENCY,
                retries=GEMINI_MAX_RETRIES,
                log_=log,
            ))

        # Validacion post-traduccion: ningun segmento con text_en puede quedar sin text_es.
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

        # Reporte de presupuesto de caracteres: cuantos segmentos exceden el
        # target_chars y en cuanto. No es error fatal — Fase 4 aplica speed-up
        # (cap 1.15x) y Fase 5 permite micro-solapes. Pero mientras mas se
        # respete el presupuesto aqui, menos trabajo aguas abajo.
        over_budget = []
        translated = 0
        for seg in master_timeline:
            text_es = (seg.get("text_es") or "").strip()
            if not text_es:
                continue
            translated += 1
            d = seg.get("duration")
            if d is None:
                d = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
            d = max(0.1, float(d))
            target_max = _max_chars_for_duration(d)
            if len(text_es) > target_max:
                over_budget.append((
                    seg.get("segment_id"), d, len(text_es), target_max,
                    len(text_es) - target_max,
                ))

        if over_budget:
            over_budget.sort(key=lambda x: x[4], reverse=True)  # peores primero
            total_excess = sum(x[4] for x in over_budget)
            log.warning(
                f"Presupuesto de chars excedido: {len(over_budget)}/{translated} segmentos "
                f"(total +{total_excess} chars sobre target @ {PHASE3_CHARS_PER_SEC}ch/s). "
                f"Fase 4 aplicara speed-up (cap 1.15x) y Fase 5 micro-solapes."
            )
            for sid, dur, got, target, excess in over_budget[:5]:
                log.warning(
                    f"  worst seg[{sid}] dur={dur:.2f}s got={got}ch "
                    f"target={target}ch excess=+{excess}ch"
                )
        else:
            log.info(
                f"Presupuesto de chars OK: {translated}/{translated} segmentos "
                f"dentro del target ({PHASE3_CHARS_PER_SEC}ch/s)."
            )

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(master_timeline, f, indent=4, ensure_ascii=False)
        log_file_info(log, output_path, "translated_timeline")

    return output_path
