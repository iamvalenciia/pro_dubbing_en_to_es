import os
import json
import torch
import ffmpeg

from src.latentsync_wrapper import lipsync_video_to_video


# Codec de video final. NVENC (h264_nvenc) es GPU-accelerated en Ampere (A100) —
# 5-10x mas rapido que libx264 CPU preset=medium. Para volver a CPU por compatibilidad
# (pod sin NVIDIA, testing local), exportar PHASE6_VCODEC=libx264.
PHASE6_VCODEC = os.environ.get("PHASE6_VCODEC", "h264_nvenc")

# Preset NVENC: p1 (fastest) .. p7 (slowest/best). p4 ≈ equivalente a libx264 medium.
# Si se vuelve a libx264 via env, este preset no aplica (se usa "medium" hardcoded).
PHASE6_NVENC_PRESET = os.environ.get("PHASE6_NVENC_PRESET", "p4")

# Constant Quality para NVENC (equivalente a CRF). 20 = calidad alta/publicable.
# Rango tipico 18-23 (mas bajo = mas calidad, archivo mas grande).
PHASE6_NVENC_CQ = int(os.environ.get("PHASE6_NVENC_CQ", "20"))


def _video_encode_kwargs(include_quality: bool = True) -> dict:
    """Devuelve los kwargs de codec/preset segun PHASE6_VCODEC. Encapsula la
    diferencia NVENC vs libx264 (crf vs cq, preset names distintos)."""
    if PHASE6_VCODEC == "h264_nvenc" or PHASE6_VCODEC == "hevc_nvenc":
        kw = {"vcodec": PHASE6_VCODEC, "preset": PHASE6_NVENC_PRESET}
        if include_quality:
            kw.update({"cq": PHASE6_NVENC_CQ, "b:v": "0"})  # constant-quality NVENC
        return kw
    else:
        # libx264 fallback
        kw = {"vcodec": PHASE6_VCODEC, "preset": "medium"}
        if include_quality:
            kw["crf"] = 20
        return kw


# =========================================================================
# Helpers
# =========================================================================

def _fmt_ass_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{int(sec):02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape text para ASS (evitar que caracteres rompan el parser)."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _ass_filter_path(ass_path: str) -> str:
    """Normaliza path de .ass para pasar al filtro ass= de ffmpeg en Windows."""
    return ass_path.replace("\\", "/").replace(":", r"\:")


# =========================================================================
# Subtitle generators
# =========================================================================

def generate_dynamic_subtitles_ass(json_path: str, output_ass: str):
    """
    Subtítulos multicolor por speaker (estilo "dialog-party"):
    colores saturados, uno por speaker. Útil para video de debug.

    Mantengo esta funcion por compatibilidad — no es lo que usamos para
    el master longform (ver generate_longform_pro_subtitles_ass).
    """
    colors = ["&HFFFFFF&", "&H00FFFF&", "&H00FF00&", "&HFFFF00&", "&HFF00FF&"]
    speaker_color_map = {}

    header = """[Script Info]
ScriptType: v4.00+
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default, Arial, 24, &H00FFFFFF, &H000000FF, &H00000000, &H00000000, 0, 0, 0, 0, 100, 100, 0, 0, 1, 2, 2, 2, 10, 10, 10, 1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for seg in data:
        speaker = seg.get("speaker", "SPEAKER_00")
        if speaker not in speaker_color_map:
            speaker_color_map[speaker] = colors[len(speaker_color_map) % len(colors)]
        color_code = speaker_color_map[speaker]
        start_t = _fmt_ass_time(seg["start"])
        end_t = _fmt_ass_time(seg["end"])
        text = _escape_ass_text(seg.get("text_es", ""))
        styled_text = f"{{\\c{color_code}}}{text}"
        lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{styled_text}\n")

    with open(output_ass, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return output_ass


def generate_longform_pro_subtitles_ass(
    json_path: str,
    output_ass: str,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font_size: int = 46,
):
    """
    Subtítulos profesionales estilo Netflix/YouTube para el master longform:
    - Blanco puro sobre caja negra semitransparente
    - 1 sola línea centrada en la parte inferior
    - Fuente legible, con outline sutil
    - Sin colores por speaker (diseño limpio y neutro)
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: LongformPro, Arial, {font_size}, &H00FFFFFF, &H000000FF, &H00000000, &HA0000000, 1, 0, 0, 0, 100, 100, 0, 0, 3, 2, 1, 2, 120, 120, 70, 1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = [header]
    for seg in data:
        start_t = _fmt_ass_time(float(seg.get("start", 0.0)))
        end_t = _fmt_ass_time(float(seg.get("end", 0.0)))
        text = (seg.get("text_es") or "").replace("\n", " ").strip()
        if not text:
            continue
        text = _escape_ass_text(text)
        lines.append(f"Dialogue: 0,{start_t},{end_t},LongformPro,,0,0,0,,{text}\n")

    with open(output_ass, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return output_ass


# =========================================================================
# Phase 6 entrypoints
# =========================================================================

def run_phase6_lipsync_render_nosubs(
    video_mute_path: str,
    final_audio_path: str,
    output_master_mp4: str,
    duration_s: float | None = None,
    lipsync_steps: int | None = None,
    lipsync_guidance: float | None = None,
):
    """
    Fase 6a: Master lip-sync SIN subtítulos (OBLIGATORIO).

    Produce el archivo base del que salen:
      - El master longform (con subs quemados) para publicación horizontal.
      - Los shorts verticales (recortados + con captions amarillos).

    LatentSync SIEMPRE se aplica. No hay modo "rapido"/"placeholder":
    el master ES el video con labios resincronizados al audio ES,
    preservando gestos, escenas y movimientos de camara del original.

    BOOTSTRAP AUTOMATICO: la primera vez que corre, si el repo o los pesos
    de LatentSync faltan, se descargan automaticamente al volumen
    persistente (ver src/latentsync_wrapper.py). Si la descarga falla,
    levanta RuntimeError — NO hay fallback.

    - lipsync_steps: 10-50 (20 default). +steps = +calidad, +tiempo.
    - lipsync_guidance: 1.0-3.0 (1.5 default). Mayor = mejor sync, mas distorsion.
    """
    print(f"--- Fase 6a: Master LatentSync (sin subs) [{video_mute_path}] ---")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(output_master_mp4) or ".", exist_ok=True)

    # Si el usuario pidió duración recortada, recortamos primero a un temp
    source_video = video_mute_path
    source_audio = final_audio_path
    tmp_cut_video = None
    tmp_cut_audio = None
    if duration_s:
        tmp_cut_video = output_master_mp4 + ".cut_v.mp4"
        tmp_cut_audio = output_master_mp4 + ".cut_a.wav"
        # Cut rapido (sin quality kw): preset=p1 (fastest NVENC) si nvenc, ultrafast si libx264.
        cut_preset = "p1" if PHASE6_VCODEC.endswith("_nvenc") else "ultrafast"
        (
            ffmpeg.input(video_mute_path, t=duration_s)
            .output(
                tmp_cut_video,
                vcodec=PHASE6_VCODEC, acodec="copy", preset=cut_preset,
            )
            .overwrite_output().run(quiet=True)
        )
        (
            ffmpeg.input(final_audio_path, t=duration_s)
            .output(tmp_cut_audio, ar=16000, ac=1, acodec="pcm_s16le")
            .overwrite_output().run(quiet=True)
        )
        source_video = tmp_cut_video
        source_audio = tmp_cut_audio

    try:
        print("LatentSync: video-to-video lipsync (preserva video original)...")
        lipsync_video_to_video(
            video_path=source_video,
            audio_path=source_audio,
            output_path=output_master_mp4,
            inference_steps=lipsync_steps,
            guidance_scale=lipsync_guidance,
        )
    finally:
        for tmp in (tmp_cut_video, tmp_cut_audio):
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    print(f"--- Fase 6a: Master sin subs listo: {output_master_mp4} ---")
    return output_master_mp4


def render_longform_with_subs(
    master_nosubs_path: str,
    json_path: str,
    output_final_mp4: str,
    duration_s: float | None = None,
):
    """
    Fase 6b: Quema subtítulos profesionales (blanco/caja negra 1-línea)
    sobre el master para producir el video longform final publicable.

    - Respeta la resolución original del master (16:9 horizontal típico).
    - Reencodifica video (libx264) porque el filtro 'ass' dibuja pixeles.
    - Copia audio sin reencodear para preservar calidad.
    """
    print(f"--- Fase 6b: Quemando subs longform pro [{master_nosubs_path}] ---")
    os.makedirs(os.path.dirname(output_final_mp4) or ".", exist_ok=True)

    ass_path = os.path.join(os.path.dirname(output_final_mp4) or ".", "longform_subs.ass")
    generate_longform_pro_subtitles_ass(json_path, ass_path)

    out_kwargs = {
        "vf": f"ass={_ass_filter_path(ass_path)}",
        "acodec": "copy",
        "movflags": "+faststart",
        **_video_encode_kwargs(include_quality=True),
    }
    if duration_s:
        out_kwargs["t"] = duration_s

    stream = ffmpeg.input(master_nosubs_path)
    stream = ffmpeg.output(stream, output_final_mp4, **out_kwargs)
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    print(f"--- Fase 6b: Longform final listo: {output_final_mp4} ---")
    return output_final_mp4


def run_phase6_lipsync_render(
    video_mute_path: str,
    final_audio_path: str,
    json_path: str,
    output_final_mp4: str,
    duration_s: float | None = None,
):
    """
    Entrypoint legacy: combina ambas fases (nosubs + subs multicolor legacy)
    para no romper callers antiguos. El nuevo flujo UI debe usar por separado
    run_phase6_lipsync_render_nosubs + render_longform_with_subs (pro subs).

    Aquí mantengo los subtítulos multicolor originales para retrocompatibilidad
    con tests; el flujo "publicable" va por render_longform_with_subs.
    """
    print(f"--- Fase 6 (legacy): Render Lip-Sync + Subtítulos [{video_mute_path}] ---")
    out_dir = os.path.dirname(output_final_mp4) or "."
    os.makedirs(out_dir, exist_ok=True)

    master_temp = os.path.join(out_dir, "master_lipsynced_nosubs.mp4")
    run_phase6_lipsync_render_nosubs(
        video_mute_path, final_audio_path, master_temp, duration_s=duration_s,
    )

    ass_path = os.path.join(out_dir, "subs.ass")
    generate_dynamic_subtitles_ass(json_path, ass_path)

    print("Quemando subtítulos dinámicos (legacy multicolor)...")
    out_kwargs = {
        "vf": f"ass={_ass_filter_path(ass_path)}",
        "acodec": "copy",
        **_video_encode_kwargs(include_quality=True),
    }
    if duration_s:
        out_kwargs["t"] = duration_s

    stream = ffmpeg.input(master_temp)
    stream = ffmpeg.output(stream, output_final_mp4, **out_kwargs)
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    print(f"--- Fase 6 (legacy): Finalizada. Video: {output_final_mp4} ---")
    return output_final_mp4
