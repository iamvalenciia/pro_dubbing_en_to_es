"""Minimal AI shorts helpers for timeline analysis and clip rendering."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import ffmpeg


def _load_timeline(json_path: str) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El timeline JSON debe ser una lista de segmentos.")
    return data


def _timeline_duration(segments: list[dict[str, Any]]) -> float:
    if not segments:
        return 0.0
    return float(max((seg.get("end", 0.0) for seg in segments), default=0.0))


def _heuristic_proposals(
    timeline: list[dict[str, Any]],
    min_duration_s: float,
    max_duration_s: float,
) -> list[dict[str, Any]]:
    total = _timeline_duration(timeline)
    if total <= 0:
        return []

    target = max(min_duration_s, min(max_duration_s, max(total / 3.0, min_duration_s)))
    proposals = []
    start = 0.0
    idx = 1

    while start < total and len(proposals) < 6:
        end = min(total, start + target)
        title = f"Clip {idx}"
        hook = "Momento clave del video"
        description = "Recorte sugerido por análisis local (fallback)."
        proposals.append(
            {
                "title": title,
                "hook": hook,
                "start": round(start, 2),
                "end": round(end, 2),
                "description": description,
            }
        )
        idx += 1
        start = end

    return proposals


def analyze_timeline_for_shorts(
    json_path: str,
    min_duration_s: float = 45.0,
    max_duration_s: float = 180.0,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Analyze timeline and return short clip proposals.

    Uses Gemini if API key exists; otherwise falls back to deterministic local heuristic.
    """
    timeline = _load_timeline(json_path)

    api_key = os.environ.get("API_GOOGLE_STUDIO")
    if not api_key:
        return _heuristic_proposals(timeline, min_duration_s, max_duration_s)

    try:
        import google.generativeai as genai
    except Exception:
        return _heuristic_proposals(timeline, min_duration_s, max_duration_s)

    lines = []
    for seg in timeline:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text_es") or seg.get("text") or "").strip()
        if text:
            lines.append(f"[{start:.2f}-{end:.2f}] {text}")

    transcript = "\n".join(lines)
    if not transcript:
        return _heuristic_proposals(timeline, min_duration_s, max_duration_s)

    genai.configure(api_key=api_key)
    effective_model = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    model = genai.GenerativeModel(effective_model)

    prompt = f"""Eres editor de reels en español. Devuelve únicamente un array JSON.
Cada item debe tener: title, hook, start, end, description.
Duración por clip entre {min_duration_s} y {max_duration_s} segundos.
Usa solo tiempos presentes en la transcripción.

TRANSCRIPCION:
{transcript}
"""

    try:
        response = model.generate_content(prompt)
        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?\\s*", "", raw)
        raw = re.sub(r"\\s*```$", "", raw)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return _heuristic_proposals(timeline, min_duration_s, max_duration_s)

        cleaned = []
        for item in parsed:
            try:
                s = float(item["start"])
                e = float(item["end"])
                if e <= s:
                    continue
                dur = e - s
                if dur < min_duration_s or dur > max_duration_s:
                    continue
                cleaned.append(
                    {
                        "title": str(item.get("title", "Clip"))[:120],
                        "hook": str(item.get("hook", ""))[:240],
                        "start": round(s, 2),
                        "end": round(e, 2),
                        "description": str(item.get("description", ""))[:240],
                    }
                )
            except Exception:
                continue
        if cleaned:
            return cleaned
    except Exception:
        pass

    return _heuristic_proposals(timeline, min_duration_s, max_duration_s)


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
) -> str:
    """Render a 9:16 short clip from source video."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _ = (json_path, sharpness, captions_word_by_word, chunk_words)  # Reserved for future expansion.

    effective_end = min(end_s, start_s + preview_seconds) if preview_seconds else end_s
    duration = max(0.1, effective_end - start_s)

    vf = (
        "scale=1080:-2:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        f"eq=brightness={brightness}:contrast={contrast}"
    )

    (
        ffmpeg.input(source_video, ss=start_s, t=duration)
        .output(
            output_path,
            vf=vf,
            vcodec="libx264",
            acodec="aac",
            preset="medium",
            crf=20,
            movflags="+faststart",
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path


def generate_short_thumbnail(
    master_video: str,
    start_s: float,
    output_jpg: str,
    offset_s: float = 1.5,
) -> str:
    """Generate a 1080x1920 thumbnail JPG from source video."""
    os.makedirs(os.path.dirname(output_jpg) or ".", exist_ok=True)
    ts = max(0.0, start_s + offset_s)

    (
        ffmpeg.input(master_video, ss=ts)
        .output(
            output_jpg,
            vframes=1,
            vf="scale=1080:-2:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
            q=2,
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return output_jpg
