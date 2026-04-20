import os
import json
import re
import math
import tempfile
import shutil

import google.generativeai as genai
import ffmpeg


# =========================================================================
# 1. Gemini analysis (Spanish transcript)
# =========================================================================

def analyze_timeline_for_shorts(
    json_path: str,
    min_duration_s: float = 180.0,   # 3 min
    max_duration_s: float = 900.0,   # 15 min
) -> list:
    """
    Lee el JSON de timeline traducido y pide a Gemini una lista de shorts
    autocontenidos (con duración configurable) basados en el texto EN ESPAÑOL.

    ¿Por qué en español? Porque ese es el audio final que va a escuchar el
    espectador. Analizar en inglés producía cortes donde la emoción vivía
    en el EN pero se perdía en el ES tras la traducción.

    Devuelve [{title, hook, start, end, description}, ...].
    """
    api_key = os.environ.get("API_GOOGLE_STUDIO")
    if not api_key:
        raise ValueError("API_GOOGLE_STUDIO env variable is missing.")
    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    model = genai.GenerativeModel(model_name)

    with open(json_path, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    lines = []
    for seg in timeline:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        # Preferimos text_es; si por alguna razón no existe (edge case), caemos a text_en
        text = (seg.get("text_es") or seg.get("text_en") or "").strip()
        if text:
            lines.append(f"[{start:.1f}-{end:.1f}] {text}")
    transcript = "\n".join(lines)

    min_min = min_duration_s / 60.0
    max_min = max_duration_s / 60.0

    prompt = f"""Eres un editor experto en contenido corto viral en español.

Se te da una transcripción larga con timestamps en segundos (el audio está en español).

REQUISITOS ESTRICTOS por cada short:
- Duración entre {min_duration_s:.0f} y {max_duration_s:.0f} segundos ({min_min:.1f}-{max_min:.1f} min)
- Gancho fuerte en los primeros 3 segundos (la primera frase debe atrapar)
- Comunica UNA sola idea o concepto clara, de principio a fin
- Cierre satisfactorio que aterrice la idea central
- Solo cortar en los bordes de segmentos presentes en la transcripción (no cortes a mitad de frase)
- Prefiere shorts sin overlap; maximiza cobertura de momentos de alto valor
- Máximo 10 shorts propuestos (prioriza calidad sobre cantidad)

OUTPUT: un array JSON puro, sin preámbulo, sin bloques de markdown. Cada item:
{{
  "title": "título corto y atractivo en español",
  "hook": "la primera frase que atrapa (en español)",
  "start": <número, segundos>,
  "end": <número, segundos>,
  "description": "una frase que resume la idea central (en español)"
}}

TRANSCRIPCIÓN:
{transcript}
"""

    response = model.generate_content(prompt)
    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        proposals = json.loads(raw)
    except json.JSONDecodeError:
        # Gemini a veces devuelve objetos sueltos o texto — intenta extraer un array
        match = re.search(r"\[\s*{.*?}\s*]", raw, re.DOTALL)
        if not match:
            raise
        proposals = json.loads(match.group(0))

    cleaned = []
    for p in proposals:
        try:
            s = float(p["start"])
            e = float(p["end"])
            dur = e - s
            # Rechaza los que quedan fuera del rango solicitado (con tolerancia)
            if dur < max(30.0, min_duration_s * 0.75):
                continue
            if dur > max_duration_s * 1.25:
                continue
            cleaned.append({
                "title": str(p.get("title", "Sin título"))[:120],
                "hook": str(p.get("hook", ""))[:240],
                "start": s,
                "end": e,
                "description": str(p.get("description", ""))[:240],
            })
        except (KeyError, ValueError, TypeError):
            continue
    return cleaned


# =========================================================================
# 2. ASS helpers (shared)
# =========================================================================

def _fmt_ass_time(s: float) -> str:
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{int(sec):02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _ass_filter_path(ass_path: str) -> str:
    return ass_path.replace("\\", "/").replace(":", r"\:")


# =========================================================================
# 3. Word-by-word Spanish captions for shorts (vertical, yellow)
# =========================================================================

def _estimate_spanish_word_timings(seg: dict) -> list[dict]:
    """
    Estima timestamps word-level en español distribuyendo proporcionalmente
    por longitud de caracter entre seg["start"] y seg["end"].

    Si el segmento tiene seg["words"] en inglés (lo genera phase2), NO los
    usamos para el español porque la traducción puede tener distinta cantidad
    de palabras. En su lugar interpolamos sobre el texto ES real.

    Devuelve [{"text": str, "start": float, "end": float}, ...].
    """
    text_es = (seg.get("text_es") or "").strip()
    if not text_es:
        return []
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    duration = max(0.01, end - start)

    # Tokeniza por whitespace preservando puntuación dentro de cada token
    words = text_es.split()
    if not words:
        return []

    # Peso por longitud en caracteres (más estable que peso uniforme para frases
    # con palabras cortas como artículos/preposiciones)
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []

    timings = []
    cursor = start
    for w in words:
        share = len(w) / total_chars
        w_dur = duration * share
        w_start = cursor
        w_end = min(end, cursor + w_dur)
        timings.append({"text": w, "start": w_start, "end": w_end})
        cursor = w_end
    return timings


def _build_ass_yellow_wordbyword(
    timeline: list,
    start_s: float,
    end_s: float,
    output_ass: str,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_size: int = 72,
    chunk_words: int = 3,
):
    """
    Genera un .ass con captions estilo "shorts virales":
    - Amarillo saturado sobre caja negra semitransparente
    - Word-by-word: se agrupan hasta `chunk_words` palabras por evento,
      cada grupo dura el tiempo combinado de esas palabras.
    - Centrado en el tercio inferior (pero visible desde centro-abajo).

    start_s / end_s acotan al rango del short (los tiempos se normalizan a
    coordenadas locales 0..duration).
    """
    # Amarillo brillante (&HBBGGRR) con outline negro
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Yellow, Arial Black, {font_size}, &H0000FFFF, &H000000FF, &H00000000, &HA0000000, 1, 0, 0, 0, 100, 100, 0, 0, 3, 5, 2, 2, 80, 80, 320, 1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    out_lines = [header]

    # Recolecta todas las palabras del rango y agrúpalas en chunks
    all_words = []
    for seg in timeline:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        if e < start_s or s > end_s:
            continue
        word_timings = _estimate_spanish_word_timings(seg)
        for wt in word_timings:
            if wt["end"] < start_s or wt["start"] > end_s:
                continue
            all_words.append(wt)

    # Agrupa en chunks de N palabras para que se lean cómodo
    for i in range(0, len(all_words), chunk_words):
        chunk = all_words[i:i + chunk_words]
        if not chunk:
            continue
        c_start = max(0.0, chunk[0]["start"] - start_s)
        c_end = min(end_s - start_s, chunk[-1]["end"] - start_s)
        if c_end <= c_start:
            continue
        text = " ".join(w["text"] for w in chunk).strip().upper()
        if not text:
            continue
        text = _escape_ass_text(text)
        out_lines.append(
            f"Dialogue: 0,{_fmt_ass_time(c_start)},{_fmt_ass_time(c_end)},Yellow,,0,0,0,,{text}\n"
        )

    with open(output_ass, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    return output_ass


# =========================================================================
# 4. Thumbnail generator
# =========================================================================

def generate_short_thumbnail(
    master_video: str,
    start_s: float,
    output_jpg: str,
    offset_s: float = 1.5,
):
    """
    Extrae un frame del master en (start_s + offset_s), lo recorta/pone en
    canvas 1080x1920 (9:16) y lo guarda como JPG.

    El offset por defecto (1.5s) evita el primer frame negro/fade-in típico
    y da una imagen representativa del short.
    """
    os.makedirs(os.path.dirname(output_jpg) or ".", exist_ok=True)

    ts = max(0.0, start_s + offset_s)
    vf = (
        "scale=1080:-2:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
    )

    (
        ffmpeg
        .input(master_video, ss=ts)
        .output(output_jpg, vframes=1, vf=vf, q=2)
        .overwrite_output()
        .run(quiet=True)
    )
    return output_jpg


# =========================================================================
# 5. Short renderer
# =========================================================================

def render_short(
    source_video: str,
    json_path: str,
    start_s: float,
    end_s: float,
    output_path: str,
    sharpness: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    preview_seconds: float | None = None,
    captions_word_by_word: bool = True,
    chunk_words: int = 3,
):
    """
    Renderiza un short 1080x1920 (9:16) a partir del MASTER LIP-SYNC:
    - Vídeo recortado al rango [start_s, end_s] y pasado a 9:16 con letterbox
    - Color grading (nitidez, brillo, contraste) consistente con longform
    - Captions amarillos word-by-word en español, quemados

    `source_video` DEBE ser el master_lipsynced_nosubs (el que sale de
    `run_phase6_lipsync_render_nosubs`). Ese master ya tiene los labios
    re-sincronizados al audio ES via LatentSync — por eso aqui NO volvemos
    a correr lipsync: solo recortamos + 9:16 + captions + color grading.

    Si `preview_seconds` se pasa, recorta solo esos segundos iniciales del short.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    effective_end = end_s
    if preview_seconds is not None:
        effective_end = min(end_s, start_s + preview_seconds)

    ass_path = output_path.replace(".mp4", ".ass")
    if captions_word_by_word:
        _build_ass_yellow_wordbyword(
            timeline, start_s, effective_end, ass_path,
            chunk_words=chunk_words,
        )

    duration = effective_end - start_s

    vf_parts = [
        "scale=1080:-2:force_original_aspect_ratio=decrease",
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
        f"eq=brightness={brightness}:contrast={contrast}",
        f"unsharp=5:5:{sharpness}:5:5:0.0",
    ]
    if captions_word_by_word:
        vf_parts.append(f"ass={_ass_filter_path(ass_path)}")
    vf = ",".join(vf_parts)

    # NVENC por default en A100; env PHASE7AI_VCODEC=libx264 para fallback CPU.
    _ai_vcodec = os.environ.get("PHASE7AI_VCODEC", "h264_nvenc")
    if _ai_vcodec.endswith("_nvenc"):
        _ai_out_kwargs = {
            "vf": vf,
            "vcodec": _ai_vcodec,
            "acodec": "aac",
            "preset": os.environ.get("PHASE7AI_NVENC_PRESET", "p4"),
            "cq": int(os.environ.get("PHASE7AI_NVENC_CQ", "20")),
            "b:v": "0",
            "movflags": "+faststart",
        }
    else:
        _ai_out_kwargs = {
            "vf": vf,
            "vcodec": _ai_vcodec,
            "acodec": "aac",
            "preset": "medium",
            "crf": 20,
            "movflags": "+faststart",
        }

    (
        ffmpeg
        .input(source_video, ss=start_s, t=duration)
        .output(output_path, **_ai_out_kwargs)
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path
