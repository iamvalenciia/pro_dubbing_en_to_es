"""
Subtitle renderer: add styled subtitles to video and generate still previews.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import textwrap
from typing import Dict, List, Optional, Tuple


@dataclass
class SubtitleStyleConfig:
    preset: str = "documental_limpio"
    fontsize: int = 60
    font_color: str = "#FFFFFF"
    border_color: str = "#000000"
    border_width: int = 4
    box_enabled: bool = False
    box_color: str = "#000000"
    box_opacity: float = 0.0
    max_chars_per_line: int = 34
    max_lines: int = 2
    line_spacing: int = 10
    alignment: str = "center"
    position_preset: str = "bottom"
    y_offset_ratio: float = 0.78
    bottom_margin: int = 72
    margin_x: int = 72
    fontfile: Optional[str] = None
    sample_text: str = "Este es un preview profesional de subtitulos en espanol."

    def to_dict(self) -> Dict:
        return asdict(self)


_STYLE_PRESETS: Dict[str, Dict] = {
    "documental_limpio": {
        "fontsize": 60,
        "font_color": "#FFFFFF",
        "border_color": "#101010",
        "border_width": 4,
        "box_enabled": False,
        "box_color": "#000000",
        "box_opacity": 0.0,
        "max_chars_per_line": 34,
        "max_lines": 2,
        "line_spacing": 10,
        "alignment": "center",
        "position_preset": "bottom",
        "y_offset_ratio": 0.78,
        "bottom_margin": 72,
        "margin_x": 72,
    },
    "youtube_moderno": {
        "fontsize": 70,
        "font_color": "#FFFFFF",
        "border_color": "#111111",
        "border_width": 5,
        "box_enabled": False,
        "box_color": "#000000",
        "box_opacity": 0.0,
        "max_chars_per_line": 26,
        "max_lines": 2,
        "line_spacing": 12,
        "alignment": "center",
        "position_preset": "bottom",
        "y_offset_ratio": 0.74,
        "bottom_margin": 88,
        "margin_x": 64,
    },
    "caja_negra_broadcast": {
        "fontsize": 56,
        "font_color": "#FFFFFF",
        "border_color": "#000000",
        "border_width": 2,
        "box_enabled": True,
        "box_color": "#000000",
        "box_opacity": 0.76,
        "max_chars_per_line": 36,
        "max_lines": 2,
        "line_spacing": 8,
        "alignment": "center",
        "position_preset": "bottom",
        "y_offset_ratio": 0.8,
        "bottom_margin": 54,
        "margin_x": 72,
    },
}


def available_style_presets() -> List[str]:
    return list(_STYLE_PRESETS.keys())


def build_style_config(style_config: Optional[Dict] = None, preset: Optional[str] = None) -> SubtitleStyleConfig:
    raw = dict(style_config or {})
    chosen_preset = str(preset or raw.get("preset") or "documental_limpio").strip().lower()
    base = dict(_STYLE_PRESETS.get(chosen_preset, _STYLE_PRESETS["documental_limpio"]))
    base.update({k: v for k, v in raw.items() if v is not None and k in SubtitleStyleConfig.__dataclass_fields__})
    base["preset"] = chosen_preset
    return SubtitleStyleConfig(**base)


def parse_srt(srt_path: str) -> List[Dict]:
    content = Path(srt_path).read_text(encoding="utf-8")
    segments = []
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            ts_match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})", lines[1])
            if ts_match:
                start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, ts_match.groups())
                start = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                end = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
                text = "\n".join(lines[2:])
                segments.append({"start": start, "end": end, "text": text})
        except Exception as exc:
            print(f"[Subtitle] Warning: failed to parse segment: {exc}")
    return segments


def parse_json(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _probe_dimensions(media_path: str) -> Tuple[int, int]:
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        media_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10, check=True)
    dims = result.stdout.strip().split(",")
    return int(dims[0]), int(dims[1])


def _normalize_color(color: str) -> str:
    raw = str(color or "#FFFFFF").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return "0xFFFFFF"
    return f"0x{raw.upper()}"


def _color_with_opacity(color: str, opacity: float) -> str:
    return f"{_normalize_color(color)}@{max(0.0, min(float(opacity), 1.0)):.2f}"


def _escape_text(text: str) -> str:
    escaped = str(text)
    escaped = escaped.replace("\\", r"\\")
    escaped = escaped.replace("\n", r"\\n")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(",", r"\,")
    escaped = escaped.replace("%", r"\%")
    escaped = escaped.replace("[", r"\[")
    escaped = escaped.replace("]", r"\]")
    return escaped


def _wrap_text(text: str, style: SubtitleStyleConfig) -> str:
    chunks: List[str] = []
    for raw_line in str(text or "").splitlines() or [""]:
        clean = re.sub(r"\s+", " ", raw_line).strip()
        if not clean:
            continue
        wrapped = textwrap.wrap(
            clean,
            width=max(8, int(style.max_chars_per_line)),
            break_long_words=False,
            break_on_hyphens=False,
        )
        chunks.extend(wrapped or [clean])

    if not chunks:
        return ""

    max_lines = max(1, int(style.max_lines))
    if len(chunks) <= max_lines:
        return "\n".join(chunks)

    visible = chunks[:max_lines]
    tail = " ".join(chunks[max_lines - 1 :]).strip()
    tail_width = max(8, int(style.max_chars_per_line) - 1)
    if len(tail) > tail_width:
        tail = tail[:tail_width].rstrip() + "…"
    visible[-1] = tail
    return "\n".join(visible)


def _position_expressions(style: SubtitleStyleConfig, video_height: int) -> Tuple[str, str]:
    alignment = str(style.alignment or "center").lower()
    if alignment == "left":
        x_expr = str(max(8, int(style.margin_x)))
    elif alignment == "right":
        x_expr = f"w-text_w-{max(8, int(style.margin_x))}"
    else:
        x_expr = "(w-text_w)/2"

    pos = str(style.position_preset or "bottom").lower()
    if pos == "top":
        y_expr = str(max(24, int(style.bottom_margin)))
    elif pos == "middle":
        y_expr = "(h-text_h)/2"
    else:
        computed = int(video_height * float(style.y_offset_ratio)) - int(style.bottom_margin)
        y_expr = str(max(24, computed))
    return x_expr, y_expr


def _prepare_segments_for_render(segments: List[Dict], style: SubtitleStyleConfig, preview_duration_sec: Optional[float] = None) -> List[Dict]:
    prepared: List[Dict] = []
    preview_limit = float(preview_duration_sec) if preview_duration_sec else None
    for seg in segments:
        start_time = float(seg.get("start", 0.0))
        end_time = float(seg.get("end", start_time))
        if preview_limit is not None and start_time >= preview_limit:
            continue
        if preview_limit is not None:
            end_time = min(end_time, preview_limit)
        wrapped_text = _wrap_text(seg.get("text", ""), style)
        if not wrapped_text or end_time <= start_time:
            continue
        prepared.append({"start": start_time, "end": end_time, "text": wrapped_text})
    return prepared


def generate_ffmpeg_drawtext_filter(
    segments: List[Dict],
    video_width: int = 1920,
    video_height: int = 1080,
    style_config: Optional[Dict] = None,
    preview_duration_sec: Optional[float] = None,
) -> str:
    style = build_style_config(style_config)
    prepared = _prepare_segments_for_render(segments, style, preview_duration_sec=preview_duration_sec)
    if not prepared:
        raise ValueError("No subtitle segments available after layout processing")

    x_expr, y_expr = _position_expressions(style, video_height)
    filters: List[str] = []
    for seg in prepared:
        text = _escape_text(seg["text"])
        parts = [
            "drawtext",
            f"fontsize={int(style.fontsize)}",
            f"fontcolor={_normalize_color(style.font_color)}",
            f"borderw={int(style.border_width)}",
            f"bordercolor={_normalize_color(style.border_color)}",
            f"line_spacing={int(style.line_spacing)}",
            f"x='{x_expr}'",
            f"y={y_expr}",
            f"text='{text}'",
            f"enable='between(t,{seg['start']:.3f},{seg['end']:.3f})'",
        ]
        if style.fontfile:
            parts.insert(1, f"fontfile={_escape_text(style.fontfile)}")
        if style.box_enabled:
            parts.extend([
                "box=1",
                f"boxcolor={_color_with_opacity(style.box_color, style.box_opacity)}",
                "boxborderw=18",
            ])
        filters.append(":".join(parts))
    return ",".join(filters)


def render_subtitle_preview_on_frame(
    frame_path: str,
    style_config: Optional[Dict] = None,
    sample_text: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    style = build_style_config(style_config)
    frame_width, frame_height = _probe_dimensions(frame_path)
    preview_text = sample_text or style.sample_text
    output = output_path or str(Path(frame_path).with_name("subtitle_style_preview.jpg"))
    filter_str = generate_ffmpeg_drawtext_filter(
        [{"start": 0.0, "end": 4.0, "text": preview_text}],
        video_width=frame_width,
        video_height=frame_height,
        style_config=style.to_dict(),
        preview_duration_sec=4.0,
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        frame_path,
        "-vf",
        filter_str,
        "-frames:v",
        "1",
        output,
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(output):
        raise RuntimeError(f"Failed to create subtitle preview image: {output}")
    return output


def _run_ffmpeg_with_codec_fallback(base_cmd: List[str], output_path: str) -> None:
    attempts = [
        [*base_cmd, "-c:v", "h264_nvenc", "-c:a", "aac", "-b:v", "8M", output_path],
        [*base_cmd, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", output_path],
    ]
    last_error = None
    for cmd in attempts:
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=True)
            return
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("FFmpeg encoding timed out") from exc
        except subprocess.CalledProcessError as exc:
            last_error = exc.stderr or exc.stdout or str(exc)
    raise RuntimeError(f"FFmpeg error: {last_error}")


def add_subtitles_to_video_ffmpeg(
    video_path: str,
    subtitle_path: str,
    output_path: Optional[str] = None,
    subtitle_format: str = "srt",
    fontsize: int = 60,
    fontfile: Optional[str] = None,
    style_config: Optional[Dict] = None,
    preview_duration_sec: Optional[float] = None,
) -> str:
    if output_path is None:
        output_path = f"{Path(video_path).stem}_subtitled.mp4"

    print(f"[Subtitle] Parsing {subtitle_format} file...")
    segments = parse_json(subtitle_path) if subtitle_format.lower() == "json" else parse_srt(subtitle_path)
    if not segments:
        raise ValueError(f"No subtitles found in {subtitle_path}")

    print("[Subtitle] Analyzing video dimensions...")
    try:
        video_width, video_height = _probe_dimensions(video_path)
    except Exception as exc:
        raise RuntimeError(f"Could not determine video dimensions for subtitle rendering: {exc}") from exc

    style = build_style_config(style_config, preset=(style_config or {}).get("preset") if style_config else None)
    if fontsize is not None:
        style.fontsize = int(fontsize)
    if fontfile:
        style.fontfile = fontfile

    print(f"[Subtitle] Generating subtitle filter ({len(segments)} segments)...")
    filter_str = generate_ffmpeg_drawtext_filter(
        segments,
        video_width=video_width,
        video_height=video_height,
        style_config=style.to_dict(),
        preview_duration_sec=preview_duration_sec,
    )

    print("[Subtitle] Encoding video with subtitles...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        filter_str,
    ]
    if preview_duration_sec:
        ffmpeg_cmd.extend(["-t", str(float(preview_duration_sec))])

    _run_ffmpeg_with_codec_fallback(ffmpeg_cmd, output_path)
    if not os.path.exists(output_path):
        raise RuntimeError(f"Failed to create output video: {output_path}")
    print(f"[Subtitle] ✓ Video with subtitles: {output_path}")
    return output_path


def add_subtitles_to_video_simple(
    video_path: str,
    subtitle_path: str,
    output_path: Optional[str] = None,
    style_config: Optional[Dict] = None,
) -> str:
    try:
        import moviepy.editor as mpy
        from moviepy.video.VideoClip import TextClip

        style = build_style_config(style_config)
        if output_path is None:
            output_path = f"{Path(video_path).stem}_subtitled.mp4"

        segments = parse_json(subtitle_path) if subtitle_path.endswith(".json") else parse_srt(subtitle_path)
        video = mpy.VideoFileClip(video_path)
        prepared = _prepare_segments_for_render(segments, style)
        text_clips = []
        rel_y = max(0.05, min(0.95, float(style.y_offset_ratio)))
        for seg in prepared:
            txt_clip = TextClip(
                seg["text"],
                fontsize=int(style.fontsize),
                color=style.font_color,
                method="caption",
                size=(video.w - (int(style.margin_x) * 2), None),
                font="Arial",
            ).set_position(("center", rel_y), relative=True).set_duration(seg["end"] - seg["start"]).set_start(seg["start"])
            text_clips.append(txt_clip)

        final_video = mpy.CompositeVideoClip([video] + text_clips)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False)
        return output_path
    except ImportError:
        print("[Subtitle] moviepy not available, using FFmpeg...")
        return add_subtitles_to_video_ffmpeg(video_path, subtitle_path, output_path, style_config=style_config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python subtitle_renderer.py <video_path> <subtitle_path> [output_path]")
        raise SystemExit(1)

    video_path = sys.argv[1]
    subtitle_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    add_subtitles_to_video_ffmpeg(video_path, subtitle_path, output_path)
    print("Success!")