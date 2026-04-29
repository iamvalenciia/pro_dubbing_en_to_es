"""
Reel UI Wrapper: Integration of reel generation with Gradio UI.
Handles reel analysis, specification generation, and rendering.
"""

import os
import json
import threading
from pathlib import Path
from typing import Optional, Tuple, List
from src.reel_generator import generate_reel_specifications, add_captions_to_reel


def analyze_reels_worker(
    json_path: str,
    on_progress=None
) -> Tuple[str, str, List]:
    """
    Analyze video content and generate reel specifications.
    
    Returns:
        (status_msg, error_or_specs_json, reel_list_for_dropdown)
    """
    try:
        if not os.path.exists(json_path):
            return "Error", f"JSON file not found: {json_path}", []
        
        if on_progress:
            on_progress("Analizando contenido con Gemini AI...")
        
        specs_path = generate_reel_specifications(json_path)
        
        if on_progress:
            on_progress(f"✅ Reels analizados: {specs_path}")
        
        # Load specs for dropdown
        specs_data = json.loads(Path(specs_path).read_text(encoding='utf-8'))
        reel_list = [
            f"Reel {r['reel_num']}: {r['title']} ({r['duration_seconds']:.0f}s)"
            for r in specs_data.get('reels', [])
        ]
        
        return "Análisis completado", specs_path, reel_list
    
    except Exception as e:
        return "Error", f"Fallo en análisis de Gemini: {str(e)}", []


def render_reel_worker(
    specs_json_path: str,
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    reel_num: int,
    output_dir: str,
    on_progress=None
) -> Tuple[str, str]:
    """
    Render a single reel with captions.
    
    Returns:
        (status_msg, output_video_path)
    """
    try:
        if not all(os.path.exists(p) for p in [specs_json_path, video_path, audio_path]):
            return "Error", "Archivo faltante"
        
        specs_data = json.loads(Path(specs_json_path).read_text(encoding='utf-8'))
        reels = specs_data.get('reels', [])
        
        # Find reel by number
        target_reel = None
        for r in reels:
            if r['reel_num'] == reel_num:
                target_reel = r
                break
        
        if not target_reel:
            return "Error", f"Reel {reel_num} no encontrado"
        
        if on_progress:
            on_progress(f"Extrayendo segmento Reel {reel_num}...")
        
        # Extract segment
        from src.reel_generator import extract_reel_segment
        seg_video, seg_audio = extract_reel_segment(
            video_path,
            audio_path,
            target_reel['start_ms'],
            target_reel['end_ms'],
            output_dir,
            reel_num
        )
        
        if on_progress:
            on_progress(f"Agregando subtítulos con estilo '{target_reel['caption_style']}'...")
        
        # Add captions
        output_video = os.path.join(output_dir, f"reel_{reel_num:02d}_final.mp4")
        final_path = add_captions_to_reel(
            seg_video,
            subtitle_path,
            target_reel['start_ms'],
            target_reel['end_ms'],
            target_reel['caption_style'],
            output_video,
            title=target_reel['title']
        )
        
        if on_progress:
            on_progress(f"✅ Reel {reel_num} renderizado: {final_path}")
        
        return "Reel renderizado", final_path
    
    except Exception as e:
        return "Error", f"Fallo en renderizado: {str(e)}"


def ui_analyze_reels(json_file, progress_callback=None):
    """
    Gradio-compatible function to analyze reels.
    Returns: (status, specs_json_path, reel_choices)
    """
    status, specs_or_error, reel_list = analyze_reels_worker(json_file, progress_callback)
    
    if status == "Error":
        return status, specs_or_error, []
    
    return status, specs_or_error, reel_list


def ui_render_reel(
    specs_json_path: str,
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    selected_reel: str,
    output_dir: str,
    progress_callback=None
) -> Tuple[str, str]:
    """
    Gradio-compatible function to render a reel.
    """
    if not selected_reel:
        return "Error", "Selecciona un reel"
    
    # Parse reel number from dropdown string (format: "Reel N: Title (XXXs)")
    try:
        reel_num = int(selected_reel.split(':')[0].replace('Reel', '').strip())
    except:
        return "Error", "Formato de reel inválido"
    
    status, result = render_reel_worker(
        specs_json_path,
        video_path,
        audio_path,
        subtitle_path,
        reel_num,
        output_dir,
        progress_callback
    )
    
    return status, result


if __name__ == "__main__":
    print("Reel UI wrapper module loaded")
