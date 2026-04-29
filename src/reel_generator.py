"""
Reel Generator: Use Gemini AI to analyze timestamps and generate viral short specs.
Detects hooks, generates 3+ minute vertical reels with captions.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def _to_seconds(raw) -> float:
    try:
        val = float(raw)
    except Exception:
        return 0.0
    if val < 0:
        return 0.0
    if val > 86400:
        return val / 1000.0
    return val


def _load_segments_from_json(json_path: str) -> List[Dict]:
    payload = json.loads(Path(json_path).read_text(encoding='utf-8'))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get('segments') or payload.get('sentences') or payload.get('paragraphs') or []
    else:
        rows = []

    normalized = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        start = _to_seconds(item.get('start') if item.get('start') is not None else item.get('start_ms'))
        end = _to_seconds(item.get('end') if item.get('end') is not None else item.get('end_ms'))
        if end <= start:
            continue
        normalized.append(
            {
                'start_ms': int(round(start * 1000.0)),
                'end_ms': int(round(end * 1000.0)),
                'text_es': str(item.get('text_es') or item.get('text') or item.get('text_en') or '').strip(),
                'speaker_id': str(item.get('speaker_id') or item.get('speaker') or 'unknown').strip() or 'unknown',
            }
        )
    return normalized


def analyze_content_with_gemini(
    json_path: str,
    gemini_model: str = "gemini-2.0-flash-lite",
    api_key: Optional[str] = None
) -> Dict:
    """
    Use Gemini to analyze timestamps JSON and detect hooks for viral content.
    
    Args:
        json_path: Path to timestamps JSON file
        gemini_model: Gemini model name to use
        api_key: Google Gemini API key (uses env if None)
    
    Returns:
        Dict with generated reel specifications
    """
    import google.generativeai as genai
    
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("API_GOOGLE_STUDIO")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or API_GOOGLE_STUDIO not set in environment")
    
    genai.configure(api_key=api_key)
    
    # Read timestamps
    segments = _load_segments_from_json(json_path)
    
    if not segments:
        raise ValueError("No segments found in timestamps JSON")
    
    # Build content summary for Gemini
    content_summary = "## Video Segments Content Analysis\n\n"
    for seg in segments[:50]:  # Limit to first 50 for token efficiency
        start_sec = seg['start_ms'] // 1000
        end_sec = seg['end_ms'] // 1000
        content_summary += f"[{start_sec}s - {end_sec}s] {seg.get('text_es', '')}\n"
    
    prompt = f"""Analiza el siguiente contenido de video en ESPAÑOL y propón REELS VIRALES para redes sociales.

{content_summary}

INSTRUCCIONES:
1. Genera 3-5 propuestas de REELS VERTICALES (1080x1920)
2. Cada reel debe tener MÍNIMO 3 MINUTOS de duración
3. PRIORIZA segmentos con HOOKS fuertes (inicio impactante, giro sorprendente, pregunta interesante)
4. Para cada reel especifica:
   - start_ms: timestamp de inicio en milisegundos
   - end_ms: timestamp de fin en milisegundos
   - title: Título viral para redes sociales (máx 60 caracteres)
   - hook_description: Descripción del gancho inicial
   - caption_style: "documentary" o "energetic"

Responde SOLO en formato JSON con estructura:
{{
  "reels": [
    {{
      "reel_num": 1,
      "start_ms": 0,
      "end_ms": 180000,
      "duration_seconds": 180,
      "title": "...",
      "hook_description": "...",
      "caption_style": "documentary",
      "priority": "high"
    }}
  ],
  "analysis": "Resumen de por qué estos reels funcionan viralmente"
}}
"""
    
    model = genai.GenerativeModel(gemini_model)
    response = model.generate_content(prompt)
    
    # Parse response
    response_text = response.text
    
    # Extract JSON from response
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON found in Gemini response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Gemini JSON response: {e}\n{response_text}")
    
    return result


def generate_reel_specifications(
    json_path: str,
    output_dir: Optional[str] = None,
    min_duration_ms: int = 180000  # 3 minutes
) -> str:
    """
    Generate reel specifications JSON using Gemini analysis.
    
    Args:
        json_path: Path to timestamps JSON
        output_dir: Output directory for reel specs (default: same as input)
        min_duration_ms: Minimum reel duration in milliseconds
    
    Returns:
        Path to output reel specs JSON
    """
    if output_dir is None:
        output_dir = str(Path(json_path).parent)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Analyze with Gemini
    reel_specs = analyze_content_with_gemini(json_path)
    
    # Validate and process reels
    validated_reels = []
    for reel in reel_specs.get('reels', []):
        duration = reel['end_ms'] - reel['start_ms']
        
        # Only include reels meeting minimum duration
        if duration >= min_duration_ms:
            validated_reels.append({
                'reel_num': reel.get('reel_num', len(validated_reels) + 1),
                'start_ms': reel['start_ms'],
                'end_ms': reel['end_ms'],
                'duration_ms': duration,
                'duration_seconds': duration / 1000,
                'title': reel.get('title', f'Reel {len(validated_reels) + 1}'),
                'hook_description': reel.get('hook_description', ''),
                'caption_style': reel.get('caption_style', 'documentary'),
                'priority': reel.get('priority', 'medium'),
                'vertical_resolution': '1080x1920',
                'frame_rate': 30
            })
    
    output_dict = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'source_file': json_path,
            'total_reels': len(validated_reels),
            'min_duration_ms': min_duration_ms,
            'gemini_analysis': reel_specs.get('analysis', '')
        },
        'reels': validated_reels
    }
    
    output_path = os.path.join(output_dir, f"{Path(json_path).stem}_reels.json")
    Path(output_path).write_text(
        json.dumps(output_dict, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    
    return output_path


def extract_reel_segment(
    video_path: str,
    audio_path: str,
    start_ms: int,
    end_ms: int,
    output_dir: str,
    reel_num: int
) -> Tuple[str, str]:
    """
    Extract video segment for a reel (vertical 1080x1920).
    
    Returns: (video_path, audio_path) for the reel segment
    """
    import subprocess
    import ffmpeg
    
    duration_sec = (end_ms - start_ms) / 1000
    start_sec = start_ms / 1000
    
    # Extract video segment with vertical resolution
    reel_video = os.path.join(output_dir, f"reel_{reel_num:02d}_video.mp4")
    (
        ffmpeg
        .input(video_path, ss=start_sec, t=duration_sec)
        .output(reel_video, vf='scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2', 
                c='h264_nvenc', preset='fast')
        .overwrite_output()
        .run(quiet=True)
    )
    
    # Extract audio segment
    reel_audio = os.path.join(output_dir, f"reel_{reel_num:02d}_audio.wav")
    (
        ffmpeg
        .input(audio_path, ss=start_sec, t=duration_sec)
        .output(reel_audio, acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    
    return reel_video, reel_audio


def add_captions_to_reel(
    video_path: str,
    subtitle_path: str,
    start_ms: int,
    end_ms: int,
    caption_style: str,
    output_path: str,
    title: Optional[str] = None
) -> str:
    """
    Add captions to reel video with specified style.
    Captions positioned at screen midpoint (vertically centered).
    
    Args:
        video_path: Input video
        subtitle_path: SRT or JSON subtitle file
        start_ms: Reel segment start
        end_ms: Reel segment end
        caption_style: "documentary" or "energetic"
        output_path: Output video with captions
        title: Optional title to overlay at top
    
    Returns:
        Path to captioned video
    """
    import subprocess
    
    # Parse subtitles for this segment
    captions = _extract_captions_for_segment(subtitle_path, start_ms, end_ms)
    
    # Build FFmpeg filter for captions
    if caption_style == "energetic":
        # Bright center captions with stronger outline.
        fontsize = 72
        primary = "&H0000D7FF"  # yellow-ish in ASS BGR format
        outline = 3
        shadow = 0
    else:  # documentary
        # Clean center captions with moderate outline.
        fontsize = 60
        primary = "&H00FFFFFF"
        outline = 2
        shadow = 0
    
    # Create subtitle file for this segment (adjusted timestamps)
    temp_srt = f"{output_path}.temp.srt"
    _write_segment_srt(captions, start_ms, temp_srt)
    
    # Build FFmpeg command. Alignment=5 places captions at center-middle.
    srt_for_filter = temp_srt.replace('\\', '/').replace(':', '\\:')
    force_style = (
        f"Alignment=5,FontSize={fontsize},PrimaryColour={primary},"
        f"OutlineColour=&H00000000,BackColour=&H64000000,"
        f"BorderStyle=3,Outline={outline},Shadow={shadow},MarginV=0"
    )

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles='{srt_for_filter}':force_style='{force_style}'",
        "-c:a", "aac",
        "-c:v", "h264_nvenc",
        "-preset", "fast",
        output_path,
        "-y"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if os.path.exists(temp_srt):
            os.remove(temp_srt)
    
    return output_path


def _extract_captions_for_segment(subtitle_path: str, start_ms: int, end_ms: int) -> List[Dict]:
    """Extract subtitles that fall within segment time range."""
    captions = []
    
    if subtitle_path.endswith('.srt'):
        import re
        content = Path(subtitle_path).read_text(encoding='utf-8')
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                ts_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    lines[1]
                )
                if ts_match:
                    s_h, s_m, s_s, s_ms = map(int, ts_match.groups()[:4])
                    e_h, e_m, e_s, e_ms = map(int, ts_match.groups()[4:])
                    
                    seg_start = s_h * 3600000 + s_m * 60000 + s_s * 1000 + s_ms
                    seg_end = e_h * 3600000 + e_m * 60000 + e_s * 1000 + e_ms
                    
                    # Include if overlaps with segment
                    if seg_start < end_ms and seg_end > start_ms:
                        text = '\n'.join(lines[2:])
                        captions.append({
                            'start_ms': max(seg_start, start_ms),
                            'end_ms': min(seg_end, end_ms),
                            'text': text
                        })
            except Exception:
                continue
    elif subtitle_path.endswith('.json'):
        rows = _load_segments_from_json(subtitle_path)
        for item in rows:
            seg_start = int(item['start_ms'])
            seg_end = int(item['end_ms'])
            if seg_start < end_ms and seg_end > start_ms:
                captions.append({
                    'start_ms': max(seg_start, start_ms),
                    'end_ms': min(seg_end, end_ms),
                    'text': str(item.get('text_es') or '').strip(),
                })

    return captions


def _write_segment_srt(captions: List[Dict], start_ms: int, output_path: str):
    """Write captions to SRT file with adjusted timestamps (relative to segment start)."""
    srt_lines = []
    
    for i, cap in enumerate(captions, 1):
        # Adjust timestamps relative to segment start
        adj_start = cap['start_ms'] - start_ms
        adj_end = cap['end_ms'] - start_ms
        
        start_str = _ms_to_srt_time(adj_start)
        end_str = _ms_to_srt_time(adj_end)
        
        srt_lines.append(f"{i}\n{start_str} --> {end_str}\n{cap['text']}\n")
    
    Path(output_path).write_text('\n'.join(srt_lines), encoding='utf-8')


def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format HH:MM:SS,mmm"""
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


if __name__ == "__main__":
    print("Reel generator module loaded. Use with main_ui.py")
