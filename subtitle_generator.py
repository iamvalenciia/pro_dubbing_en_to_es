"""
Subtitle generator: Extract subtitles from dubbed audio using Whisper.
Generates SRT file from video/audio with proper timestamps.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


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


def transcribe_audio(audio_path: str, language: str = 'es') -> List[Dict]:
    """
    Transcribe audio using faster_whisper.
    Returns list of segments with timestamps.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster_whisper not installed. Install via: pip install faster-whisper")
    
    # Initialize model (tiny for speed, accurate for accuracy)
    model = WhisperModel("tiny", device="cuda", compute_type="int8_float16")
    
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False
    )
    
    result = []
    for seg in segments:
        result.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text.strip(),
            'id': seg.id if hasattr(seg, 'id') else len(result) + 1
        })
    
    return result


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
