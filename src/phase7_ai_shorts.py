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


def analyze_timeline_for_shorts(
    json_path: str,
    min_duration_s: float = 45.0,
    max_duration_s: float = 180.0,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Analyze timeline and return short clip proposals.
    """
    timeline = _load_timeline(json_path)

    api_key = os.environ.get("API_GOOGLE_STUDIO")
    if not api_key:
        raise RuntimeError("API_GOOGLE_STUDIO is required for high-quality reel analysis.")

    try:
        import google.generativeai as genai
    except Exception as exc:
        raise RuntimeError("google-generativeai is required for reel analysis.") from exc

    lines = []
    for seg in timeline:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text_es") or seg.get("text") or "").strip()
        if text:
            lines.append(f"[{start:.2f}-{end:.2f}] {text}")

    transcript = "\n".join(lines)
    if not transcript:
        raise RuntimeError("No transcript content available for reel analysis.")

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

    response = model.generate_content(prompt)
    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?\\s*", "", raw)
    raw = re.sub(r"\\s*```$", "", raw)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise RuntimeError("Gemini response must be a JSON array of reel proposals.")

    cleaned = []
    for item in parsed:
        s = float(item["start"])
        e = float(item["end"])
        if e <= s:
            raise RuntimeError(f"Invalid reel timing from Gemini: start={s}, end={e}")
        dur = e - s
        if dur < min_duration_s or dur > max_duration_s:
            raise RuntimeError(
                f"Gemini proposed out-of-range duration ({dur:.2f}s). "
                f"Allowed range: {min_duration_s:.2f}s-{max_duration_s:.2f}s"
            )
        cleaned.append(
            {
                "title": str(item.get("title", "Clip"))[:120],
                "hook": str(item.get("hook", ""))[:240],
                "start": round(s, 2),
                "end": round(e, 2),
                "description": str(item.get("description", ""))[:240],
            }
        )

    if not cleaned:
        raise RuntimeError("Gemini returned no valid reel proposals.")
    return cleaned


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
