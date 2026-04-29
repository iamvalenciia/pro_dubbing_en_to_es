"""
Subtitle generator: Extract subtitles from video/audio using AssemblyAI.
Generates SRT and JSON with proper timestamps.
"""

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


def get_audio_from_video(
    video_path: str,
    output_audio: Optional[str] = None,
    preview_duration_sec: Optional[float] = None,
) -> str:
    """Get 16k mono WAV from video or audio input using ffmpeg."""
    from videotrans.util import tools
    
    if output_audio is None:
        output_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    
    ext = Path(video_path).suffix.lower()
    audio_exts = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}

    cmd = ['-y', '-i', video_path]
    if preview_duration_sec:
        cmd.extend(['-t', str(float(preview_duration_sec))])

    if ext in audio_exts:
        # Normalize input audio to 16k mono WAV for ASR consistency.
        tools.runffmpeg(cmd + [
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio
        ])
    else:
        tools.runffmpeg(cmd + [
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio
        ])
    return output_audio


def _resolve_assemblyai_key() -> str:
    key = str(os.environ.get("ASSEMBLY_AI_KEY") or os.environ.get("ASSEMBLYAI_API_KEY") or "").strip()
    if key:
        return key
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            env_key, env_value = line.split("=", 1)
            if env_key.strip() == "ASSEMBLY_AI_KEY":
                return env_value.strip().strip('"').strip("'")
    raise RuntimeError("ASSEMBLY_AI_KEY is required for subtitle generation.")


def _resolve_speech_models() -> List[str]:
    raw = str(
        os.environ.get("ASSEMBLYAI_SPEECH_MODELS")
        or os.environ.get("ASSEMBLYAI_SPEECH_MODEL")
        or "universal-3-pro"
    ).strip()
    models = []
    for item in raw.split(","):
        token = item.strip().lower()
        if token and token not in models:
            models.append(token)
    return models or ["universal-3-pro"]


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


def transcribe_audio(audio_path: str, language: str = 'es') -> List[Dict]:
    """
    Transcribe audio using AssemblyAI and return normalized segments.
    """
    api_key = _resolve_assemblyai_key()
    base_url = os.environ.get("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com/v2").rstrip("/")
    request_timeout = float(os.environ.get("ASSEMBLYAI_REQUEST_TIMEOUT", "60"))
    poll_seconds = float(os.environ.get("ASSEMBLYAI_POLL_SECONDS", "2.0"))
    max_seconds = int(os.environ.get("ASSEMBLYAI_TIMEOUT_SECONDS", "1800"))

    with open(audio_path, "rb") as handle:
        upload_resp = requests.post(
            f"{base_url}/upload",
            headers={"authorization": api_key},
            data=handle,
            timeout=request_timeout,
        )
    upload_resp.raise_for_status()
    upload_url = str((upload_resp.json() or {}).get("upload_url") or "").strip()
    if not upload_url:
        raise RuntimeError("AssemblyAI upload failed: missing upload_url.")

    payload = {
        "audio_url": upload_url,
        "language_detection": False,
        "language_code": language,
        "speech_model": _resolve_speech_models()[0],
    }
    create_resp = requests.post(
        f"{base_url}/transcript",
        headers={"authorization": api_key, "content-type": "application/json"},
        json=payload,
        timeout=request_timeout,
    )
    create_resp.raise_for_status()
    transcript_id = str((create_resp.json() or {}).get("id") or "").strip()
    if not transcript_id:
        raise RuntimeError("AssemblyAI transcript creation failed: missing transcript id.")

    started = time.time()
    while True:
        if (time.time() - started) > max_seconds:
            raise RuntimeError(f"AssemblyAI subtitle transcription timeout after {max_seconds}s.")

        status_resp = requests.get(
            f"{base_url}/transcript/{transcript_id}",
            headers={"authorization": api_key},
            timeout=request_timeout,
        )
        status_resp.raise_for_status()
        status_body = status_resp.json() or {}
        status = str(status_body.get("status") or "").strip().lower()
        if status == "completed":
            break
        if status == "error":
            raise RuntimeError(f"AssemblyAI subtitle transcription failed: {status_body.get('error')}")
        time.sleep(max(0.5, poll_seconds))

    srt_resp = requests.get(
        f"{base_url}/transcript/{transcript_id}/srt",
        headers={"authorization": api_key},
        timeout=request_timeout,
    )
    srt_resp.raise_for_status()
    segments = _parse_srt_to_segments(srt_resp.text)
    if not segments:
        raise RuntimeError("AssemblyAI returned no subtitle segments.")
    return segments


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
        print(f"[Subtitle] Transcribing with language={language}...")
        segments = transcribe_audio(temp_audio, language=language)
        
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
