"""
Subtitle generator: Extract subtitles from video/audio using local faster-whisper.
Generates SRT and JSON with proper timestamps.
"""

import json
import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

from faster_whisper import WhisperModel


def get_audio_from_video(
    video_path: str,
    output_audio: Optional[str] = None,
    preview_duration_sec: Optional[float] = None,
) -> str:
    """Get 16k mono WAV from video or audio input using ffmpeg."""
    if output_audio is None:
        output_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    
    ext = Path(video_path).suffix.lower()
    audio_exts = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path]
    if preview_duration_sec:
        cmd.extend(["-t", str(float(preview_duration_sec))])

    if ext in audio_exts:
        # Normalize input audio to 16k mono WAV for ASR consistency.
        cmd.extend([
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_audio
        ])
    else:
        cmd.extend([
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_audio
        ])
    subprocess.run(cmd, check=True)
    return output_audio


def _resolve_whisper_runtime() -> tuple[str, str, str]:
    model_name = str(
        os.environ.get("QDP_SUBTITLE_ASR_MODEL")
        or os.environ.get("QDP_ASR_MODEL")
        or "large-v3"
    ).strip()

    requested_device = str(os.environ.get("QDP_SUBTITLE_ASR_DEVICE") or "auto").strip().lower()
    device = "cpu"
    if requested_device in {"cuda", "cpu"}:
        device = requested_device
    else:
        try:
            import torch  # type: ignore
            device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
        except Exception:
            device = "cpu"

    default_compute = "float16" if device == "cuda" else "int8"
    compute_type = str(os.environ.get("QDP_SUBTITLE_ASR_COMPUTE_TYPE") or default_compute).strip()
    return model_name, device, compute_type


def _split_segment_by_chars(start: float, end: float, text: str, max_chars_per_line: int) -> List[Dict]:
    token = str(text or "").strip()
    if not token:
        return []

    max_chars = max(int(max_chars_per_line or 35), 8)
    chunks = textwrap.wrap(
        token,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [token]

    duration = max(float(end) - float(start), 0.01)
    total_chars = sum(max(len(c), 1) for c in chunks)
    cur = float(start)
    rows: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        portion = max(len(chunk), 1) / max(total_chars, 1)
        seg_dur = duration * portion
        seg_start = cur
        seg_end = float(end) if idx == (len(chunks) - 1) else min(float(end), cur + seg_dur)
        cur = seg_end
        if seg_end <= seg_start:
            continue
        rows.append({
            "start": seg_start,
            "end": seg_end,
            "text": chunk.strip(),
        })
    return rows


def _parse_srt_to_segments(srt_text: str) -> List[Dict]:
    blocks = re.split(r"\r?\n\s*\r?\n", str(srt_text or "").strip())
    segments: List[Dict] = []
    ts_re = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")

    for block in blocks:
        lines = [ln.strip("\r") for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        ts_idx = -1
        for idx, line in enumerate(lines):
            if ts_re.search(line):
                ts_idx = idx
                break
        if ts_idx < 0:
            continue
        match = ts_re.search(lines[ts_idx])
        if not match:
            continue
        start = _srt_to_seconds(match.group(1))
        end = _srt_to_seconds(match.group(2))
        if end <= start:
            continue
        text = "\n".join(lines[ts_idx + 1:]).strip()
        segments.append({
            "start": start,
            "end": end,
            "text": text,
            "id": len(segments) + 1,
        })
    return segments


def _srt_to_seconds(value: str) -> float:
    hh, mm, ss_ms = str(value).strip().split(":")
    ss, ms = ss_ms.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + (int(ms) / 1000.0)


def transcribe_audio(audio_path: str, language: str = 'es', max_chars_per_line: int = 35) -> List[Dict]:
    """Transcribe audio using local faster-whisper and return normalized segments."""
    model_name, device, compute_type = _resolve_whisper_runtime()
    beam_size = int(os.environ.get("QDP_SUBTITLE_ASR_BEAM_SIZE", "5"))

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    whisper_segments, _info = model.transcribe(
        audio_path,
        language=(language or "").strip() or None,
        vad_filter=True,
        beam_size=max(1, beam_size),
    )

    rows: List[Dict] = []
    for seg in whisper_segments:
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", 0.0) or 0.0)
        text = str(getattr(seg, "text", "") or "").strip()
        if end <= start or not text:
            continue
        rows.extend(_split_segment_by_chars(start, end, text, max_chars_per_line=max_chars_per_line))

    normalized = []
    for idx, it in enumerate(rows, start=1):
        normalized.append({
            "start": float(it["start"]),
            "end": float(it["end"]),
            "text": str(it["text"]),
            "id": idx,
        })

    if not normalized:
        raise RuntimeError("faster-whisper no devolvió segmentos válidos de subtítulos.")
    return normalized


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: List[Dict]) -> str:
    """Convert segments to SRT format string."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = format_timestamp(seg['start'])
        end_ts = format_timestamp(seg['end'])
        text = seg['text']
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(text)
        srt_lines.append("")  # blank line between segments
    
    return "\n".join(srt_lines)


def generate_subtitles_from_video(
    video_path: str,
    output_srt: Optional[str] = None,
    language: str = 'es',
    output_json: Optional[str] = None,
    preview_duration_sec: Optional[float] = None,
    max_chars_per_line: int = 35,
) -> tuple[str, str]:
    """
    Complete pipeline: extract audio, transcribe, generate SRT and JSON.
    
    Args:
        video_path: Path to dubbed video
        output_srt: Output SRT file path (auto-generated if None)
        language: Language code for transcription (default 'es' for Spanish)
        output_json: Output JSON file path (auto-generated if None)
    
    Returns:
        (srt_file_path, json_file_path)
    """
    # Generate default paths if not provided
    base_name = Path(video_path).stem
    if output_srt is None:
        output_srt = f"{base_name}_subtitles.srt"
    if output_json is None:
        output_json = f"{base_name}_subtitles.json"
    
    # Extract audio
    print(f"[Subtitle] Extracting audio from video...")
    temp_audio = get_audio_from_video(video_path, preview_duration_sec=preview_duration_sec)
    
    try:
        # Transcribe
        print(f"[Subtitle] Transcribing locally with faster-whisper language={language}...")
        segments = transcribe_audio(
            temp_audio,
            language=language,
            max_chars_per_line=max_chars_per_line,
        )
        
        # Generate SRT
        print(f"[Subtitle] Writing SRT file...")
        srt_content = segments_to_srt(segments)
        Path(output_srt).write_text(srt_content, encoding='utf-8')
        
        # Generate JSON for easier manipulation
        print(f"[Subtitle] Writing JSON file...")
        json_content = json.dumps(segments, indent=2, ensure_ascii=False)
        Path(output_json).write_text(json_content, encoding='utf-8')
        
        print(f"[Subtitle] ✓ Generated {len(segments)} segments")
        print(f"[Subtitle] ✓ SRT: {output_srt}")
        print(f"[Subtitle] ✓ JSON: {output_json}")
        
        return output_srt, output_json
    
    finally:
        # Cleanup temp audio
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python subtitle_generator.py <video_path> [output_dir]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    srt, json_file = generate_subtitles_from_video(video_path)
    print(f"\nDone! SRT and JSON saved.")
