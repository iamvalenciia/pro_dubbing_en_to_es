"""
Subtitle UI wrapper functions - handles integration with Gradio UI.
Manages subtitle generation, rendering, and file downloads.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import threading
import time


def _infer_subtitle_format(subtitle_path: str, declared: str = "srt") -> str:
    ext = Path(subtitle_path or "").suffix.lower()
    if ext == ".json":
        return "json"
    if ext == ".vtt":
        return "vtt"
    if ext == ".srt":
        return "srt"
    return (declared or "srt").lower()


def generate_subtitles_worker(
    video_path: str,
    language: str = 'es',
    progress_callback=None,
    preview_duration_sec: Optional[float] = None,
    max_chars_per_line: int = 35,
) -> Tuple[str, str, str]:
    """
    Background worker for subtitle generation.
    Returns: (srt_file, json_file, status_msg)
    """
    try:
        if not os.path.exists(video_path):
            return "", "", f"❌ Video no encontrado: {video_path}"
        
        # Import here to avoid issues if subtitle_generator not in path
        from src.subtitle_generator import generate_subtitles_from_video
        
        if progress_callback:
            progress_callback("🎬 Extrayendo audio del video...")
        
        base_name = Path(video_path).stem
        output_dir = Path(video_path).parent
        
        suffix = "_subtitles_preview" if preview_duration_sec else "_subtitles"
        srt_file = str(output_dir / f"{base_name}{suffix}.srt")
        json_file = str(output_dir / f"{base_name}{suffix}.json")
        
        if progress_callback:
            progress_callback(f"📝 Transcribiendo local (faster-whisper) con {language}...")
        
        srt_result, json_result = generate_subtitles_from_video(
            video_path,
            output_srt=srt_file,
            language=language,
            output_json=json_file,
            preview_duration_sec=preview_duration_sec,
            max_chars_per_line=max_chars_per_line,
        )
        
        if progress_callback:
            progress_callback(f"✅ Subtítulos generados correctamente")
        
        return srt_result, json_result, f"✅ Generados {len(json.loads(Path(json_result).read_text(encoding='utf-8')))} segmentos"
    
    except Exception as e:
        return "", "", f"❌ Error: {str(e)}"


def render_subtitles_worker(
    video_path: str,
    subtitle_path: str,
    subtitle_format: str = 'srt',
    progress_callback=None,
    style_config: Optional[dict] = None,
    preview_duration_sec: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Background worker for subtitle rendering.
    Returns: (output_video_path, status_msg)
    """
    try:
        if not os.path.exists(video_path):
            return "", f"❌ Video no encontrado: {video_path}"
        
        if not os.path.exists(subtitle_path):
            return "", f"❌ Archivo de subtítulos no encontrado: {subtitle_path}"
        
        from src.subtitle_renderer import add_subtitles_to_video_ffmpeg
        
        if progress_callback:
            progress_callback("🎨 Agregando subtítulos al video...")
        
        base_name = Path(video_path).stem
        output_dir = Path(video_path).parent
        output_video = str(output_dir / f"{base_name}_subtitled.mp4")
        
        suffix = "_subtitled_preview.mp4" if preview_duration_sec else "_subtitled.mp4"
        output_video = str(output_dir / f"{base_name}{suffix}")

        resolved_format = _infer_subtitle_format(subtitle_path, subtitle_format)

        result = add_subtitles_to_video_ffmpeg(
            video_path,
            subtitle_path,
            output_path=output_video,
            subtitle_format=resolved_format,
            fontsize=60,
            style_config=style_config,
            preview_duration_sec=preview_duration_sec,
        )
        
        if progress_callback:
            progress_callback(f"✅ Video con subtítulos completado")
        
        return result, f"✅ Video con subtítulos: {Path(result).name}"
    
    except Exception as e:
        return "", f"❌ Error: {str(e)}"


def ui_generate_subtitles(
    video_path: str,
    language: str = 'es',
    ui_log_update=None,
    preview_duration_sec: Optional[float] = None,
    max_chars_per_line: int = 35,
):
    """
    UI wrapper for subtitle generation - runs in background.
    """
    if not video_path:
        return "", "", "❌ Selecciona un video primero"
    
    def progress(msg):
        if ui_log_update:
            ui_log_update(f"[Subtítulos] {msg}")
    
    srt, json_file, msg = generate_subtitles_worker(
        video_path,
        language,
        progress,
        preview_duration_sec=preview_duration_sec,
        max_chars_per_line=max_chars_per_line,
    )
    
    # Return file objects for Gradio outputs
    if os.path.exists(srt) and os.path.exists(json_file):
        return srt, json_file, msg
    return "", "", msg


def ui_render_subtitles(
    video_path: str,
    subtitle_path: str,
    subtitle_format: str = 'srt',
    ui_log_update=None,
    style_config: Optional[dict] = None,
    preview_duration_sec: Optional[float] = None,
):
    """
    UI wrapper for subtitle rendering - runs in background.
    """
    if not video_path:
        return "", "❌ Selecciona un video primero"
    
    if not subtitle_path:
        return "", "❌ Carga un archivo de subtítulos primero"
    
    def progress(msg):
        if ui_log_update:
            ui_log_update(f"[Render] {msg}")
    
    output_video, msg = render_subtitles_worker(
        video_path,
        subtitle_path,
        subtitle_format,
        progress,
        style_config=style_config,
        preview_duration_sec=preview_duration_sec,
    )
    
    if os.path.exists(output_video):
        return output_video, msg
    return "", msg


if __name__ == "__main__":
    print("Subtitle UI module loaded. Use with main_ui.py")
