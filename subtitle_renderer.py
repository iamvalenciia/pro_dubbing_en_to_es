"""
Subtitle renderer: Add styled subtitles to dubbed video.
Generates video with professional white-text-on-black-outline subtitles.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess


def parse_srt(srt_path: str) -> List[Dict]:
    """Parse SRT file into segments."""
    content = Path(srt_path).read_text(encoding='utf-8')
    segments = []
    
    # Split by double newlines
    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # lines[0] = segment number
        # lines[1] = timestamps (00:00:10,500 --> 00:00:13,000)
        # lines[2:] = text
        
        try:
            ts_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
            if ts_match:
                start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, ts_match.groups())
                start = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                end = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
                
                text = '\n'.join(lines[2:])  # Handle multi-line subtitles
                segments.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
        except Exception as e:
            print(f"[Subtitle] Warning: Failed to parse segment: {e}")
            continue
    
    return segments


def parse_json(json_path: str) -> List[Dict]:
    """Parse JSON subtitle file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_ffmpeg_drawtext_filter(
    segments: List[Dict],
    video_width: int = 1920,
    video_height: int = 1080,
    fontsize: int = 60,
    fontfile: str = "Arial.ttf"  # System default, may vary by OS
) -> str:
    """
    Generate FFmpeg drawtext filter string for subtitles.
    Style: White text with black outline, centered, professional documentary look.
    """
    
    # Escape special characters for FFmpeg
    def escape_text(text: str) -> str:
        # Replace newlines with escaped newlines for FFmpeg
        text = text.replace('\n', '\\n')
        # Escape special FFmpeg chars
        text = text.replace("'", "\\'")
        text = text.replace('\\', '\\\\')
        return text
    
    filters = []
    
    for seg in segments:
        start_time = seg['start']
        end_time = seg['end']
        text = escape_text(seg['text'])
        
        # drawtext filter with:
        # - text
        # - timing (enable from start_time to end_time)
        # - white color with black outline
        # - centered horizontally, positioned in lower third (for documentary style)
        # - font settings
        
        y_offset = int(video_height * 0.65)  # Lower third of screen
        
        # drawtext syntax:
        # drawtext=fontsize=60:fontcolor=white:borderw=3:bordercolor=black:x=(w-text_w)/2:y=y_offset:
        #         text='subtitle text':enable='between(t,start,end)'
        
        filter_str = (
            f"drawtext="
            f"fontsize={fontsize}:"
            f"fontcolor=white:"
            f"borderw=4:"
            f"bordercolor=black:"
            f"x='(w-text_w)/2':"  # Center horizontally
            f"y={y_offset}:"  # Position in lower third
            f"text='{text}':"
            f"enable='between(t,{start_time},{end_time})'"
        )
        
        filters.append(filter_str)
    
    # Chain all filters with comma
    return ','.join(filters)


def add_subtitles_to_video_ffmpeg(
    video_path: str,
    subtitle_path: str,
    output_path: Optional[str] = None,
    subtitle_format: str = 'srt',  # 'srt' or 'json'
    fontsize: int = 60,
    fontfile: Optional[str] = None
) -> str:
    """
    Add subtitles to video using FFmpeg drawtext filter.
    
    Args:
        video_path: Input video file
        subtitle_path: SRT or JSON subtitle file
        output_path: Output video file (auto-generated if None)
        subtitle_format: 'srt' or 'json'
        fontsize: Font size for subtitles
        fontfile: Path to font file (uses system default if None)
    
    Returns:
        Path to output video with subtitles
    """
    
    if output_path is None:
        base_name = Path(video_path).stem
        output_path = f"{base_name}_subtitled.mp4"
    
    # Parse subtitles
    print(f"[Subtitle] Parsing {subtitle_format} file...")
    if subtitle_format.lower() == 'json':
        segments = parse_json(subtitle_path)
    else:
        segments = parse_srt(subtitle_path)
    
    if not segments:
        raise ValueError(f"No subtitles found in {subtitle_path}")
    
    # Get video dimensions (needed for drawtext positioning)
    print(f"[Subtitle] Analyzing video dimensions...")
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        dims = result.stdout.strip().split(',')
        video_width = int(dims[0])
        video_height = int(dims[1])
        print(f"[Subtitle] Video: {video_width}x{video_height}")
    except Exception as e:
        print(f"[Subtitle] Warning: Could not determine video dimensions ({e}), using defaults")
        video_width, video_height = 1920, 1080
    
    # Generate drawtext filter
    print(f"[Subtitle] Generating subtitle filter ({len(segments)} segments)...")
    filter_str = generate_ffmpeg_drawtext_filter(
        segments,
        video_width=video_width,
        video_height=video_height,
        fontsize=fontsize,
        fontfile=fontfile
    )
    
    # Build FFmpeg command
    # Use hardware acceleration if available (nvenc for NVIDIA)
    print(f"[Subtitle] Encoding video with subtitles...")
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', video_path,
        '-vf', filter_str,
        '-c:v', 'h264_nvenc',  # NVIDIA GPU encoding
        '-c:a', 'aac',  # Copy audio codec
        '-b:v', '8M',  # Bitrate
        output_path
    ]
    
    # Fallback to libx264 if GPU encoding fails
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0 and 'h264_nvenc' in result.stderr:
            print(f"[Subtitle] GPU encoding not available, falling back to CPU...")
            ffmpeg_cmd[9] = 'libx264'
            ffmpeg_cmd[10] = '-preset'
            ffmpeg_cmd.insert(11, 'slow')  # Better quality, slower encoding
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg encoding timed out")
    except Exception as e:
        raise RuntimeError(f"FFmpeg error: {e}")
    
    if not os.path.exists(output_path):
        raise RuntimeError(f"Failed to create output video: {output_path}")
    
    print(f"[Subtitle] ✓ Video with subtitles: {output_path}")
    return output_path


def add_subtitles_to_video_simple(
    video_path: str,
    subtitle_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Simple wrapper: Add subtitles using moviepy (slower but simpler).
    Falls back to FFmpeg if moviepy not available.
    """
    try:
        import moviepy.editor as mpy
        from moviepy.video.VideoClip import TextClip
        
        print(f"[Subtitle] Using moviepy for subtitle rendering...")
        
        if output_path is None:
            base_name = Path(video_path).stem
            output_path = f"{base_name}_subtitled.mp4"
        
        # Parse subtitles
        if subtitle_path.endswith('.json'):
            segments = parse_json(subtitle_path)
        else:
            segments = parse_srt(subtitle_path)
        
        # Load video
        video = mpy.VideoFileClip(video_path)
        
        # Create text clips for each subtitle
        text_clips = []
        for seg in segments:
            txt_clip = TextClip(
                seg['text'],
                fontsize=60,
                color='white',
                method='caption',
                size=(video.w - 100, None),  # Leave margins
                font='Arial'
            ).set_position(('center', 0.75), relative=True).set_duration(seg['end'] - seg['start']).set_start(seg['start'])
            
            text_clips.append(txt_clip)
        
        # Composite video with subtitles
        final_video = mpy.CompositeVideoClip([video] + text_clips)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False)
        
        print(f"[Subtitle] ✓ Video with subtitles: {output_path}")
        return output_path
    
    except ImportError:
        print(f"[Subtitle] moviepy not available, using FFmpeg...")
        return add_subtitles_to_video_ffmpeg(video_path, subtitle_path, output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python subtitle_renderer.py <video_path> <subtitle_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    subtitle_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    output = add_subtitles_to_video_ffmpeg(video_path, subtitle_path, output_path)
    print(f"Success!")
