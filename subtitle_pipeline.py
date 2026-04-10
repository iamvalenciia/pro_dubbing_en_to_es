"""
subtitle_pipeline.py — Genera videos con subtítulos quemados (burn-in) en pantalla.

Pipeline:
  1. [Opcional] Recorta video + audio a test_seconds.
  2. Detecta resolución del video fuente.
  3. Extrae WAV 16 kHz mono del audio doblado para Whisper.
  4. Transcribe con Whisper (es, word_timestamps) → subtítulos ES.
  5. Genera archivo .ass (PlayRes fijo 1920×1080; ffmpeg escala automáticamente
     al tamaño real del video, sea 1080p o 4K).
  6. Combina video + audio doblado (+ audio original opcional) con filtros
     de mejora (eq/unsharp) y subtítulos quemados vía ffmpeg.
     Codec: h264_nvenc si NVENC está disponible (RTX/GTX), libx264 en otro caso (A100/datacenter).

Exports:
  generate_subtitled_video(...)  → (out_video_path: str, status: str)
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path


# ─── Detección de NVENC (una sola vez al inicio) ──────────────────────────────

def _detect_nvenc() -> bool:
    """
    Detecta si h264_nvenc está operativo en este sistema.

    Usa 320x240 porque el RTX 3060 Laptop (GA106/Ampere) impone un mínimo
    de 145px de ancho a nivel de SDK (NV_ENC_CAPS_WIDTH_MIN). Resoluciones
    menores devuelven NV_ENC_ERR_INVALID_PARAM → AVERROR(EINVAL) = -22.
    320x240 es seguro para cualquier GPU NVIDIA desde Kepler en adelante.
    """
    null_out = "NUL" if sys.platform == "win32" else "/dev/null"
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner",
             "-f", "lavfi", "-i", "color=c=black:s=320x240:r=30",
             "-frames:v", "5",
             "-vf", "format=yuv420p",
             "-c:v", "h264_nvenc",
             "-f", "null", null_out],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False

NVENC_AVAILABLE: bool = _detect_nvenc()
print(f"  [startup] NVENC: {'disponible (GPU)' if NVENC_AVAILABLE else 'no disponible — usando CPU'}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list, description: str = "", cwd: str = None,
         duration_s: float = None, progress_cb=None,
         pct_start: float = 0.40, pct_end: float = 0.98) -> str:
    """
    Ejecuta un comando ffmpeg/ffprobe.

    Si duration_s y progress_cb se proporcionan, usa streaming con -progress pipe:1
    para reportar el progreso en tiempo real durante el encoding.
    """
    if description:
        print(f"  [{time.strftime('%H:%M:%S')}] -> {description}")

    # ── Modo streaming: progreso en tiempo real ───────────────────────────────
    if duration_s and progress_cb and duration_s > 0:
        stream_cmd = list(cmd[:-1]) + ["-progress", "pipe:1", "-nostats", cmd[-1]]
        process = subprocess.Popen(
            stream_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )

        # Drenar stderr en un thread independiente para evitar deadlock.
        # El buffer de la pipe (~4KB en Windows) se llena con el output inicial
        # de FFmpeg (banner + info de streams), bloqueando el proceso si nadie lo lee.
        stderr_buf = []
        def _drain_stderr():
            for line in process.stderr:
                stderr_buf.append(line)
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        # Videos de YouTube (y otros) tienen start_time != 0 en el container.
        # FFmpeg reporta out_time relativo al PTS del input, no desde 0.
        # Guardamos el primer valor y lo usamos como base para obtener
        # el tiempo relativo real (elapsed desde el inicio del encode).
        _time_base = [None]

        for line in process.stdout:
            if line.strip().startswith("out_time="):
                try:
                    timestr = line.strip().split("=", 1)[1].strip()
                    if timestr in ("N/A", ""):
                        continue
                    timestr = timestr.lstrip("-")
                    h, m, s = timestr.split(":")
                    raw = int(h) * 3600 + int(m) * 60 + float(s)
                    # Primer update → establece la base (offset del container)
                    if _time_base[0] is None:
                        _time_base[0] = raw
                    elapsed = max(0.0, raw - _time_base[0])
                    frac  = min(elapsed / duration_s, 1.0)
                    pct   = pct_start + frac * (pct_end - pct_start)
                    total = int(duration_s)
                    m_e, s_e = divmod(int(elapsed), 60)
                    m_t, s_t = divmod(total, 60)
                    progress_cb(
                        f"[4/4] Codificando con GPU... {m_e}m {s_e:02d}s / {m_t}m {s_t:02d}s",
                        pct,
                    )
                except (ValueError, AttributeError):
                    pass

        process.wait()
        stderr_thread.join()
        if process.returncode != 0:
            stderr_text = "".join(stderr_buf)
            raise RuntimeError(
                f"ffmpeg falló (código {process.returncode}).\n"
                f"stderr: {stderr_text[-1200:]}"
            )
        return ""

    # ── Modo estándar bloqueante ──────────────────────────────────────────────
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode == 0:
        return result.stdout

    raise RuntimeError(
        f"ffmpeg/ffprobe falló (código {result.returncode}).\n"
        f"Comando: {' '.join(str(c) for c in cmd)}\n"
        f"stderr: {result.stderr[-1200:]}"
    )


def _get_duration(path: str) -> float:
    """Retorna duración en segundos de un archivo de audio/video."""
    out = _run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", str(path),
    ])
    return float(json.loads(out)["format"]["duration"])


def _get_video_resolution(path: str) -> tuple:
    """
    Retorna (width, height) del primer stream de video.
    Retorna (0, 0) si no puede determinarlos.
    """
    try:
        out = _run([
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-print_format", "json",
            str(path),
        ])
        streams = json.loads(out).get("streams", [])
        if streams:
            return int(streams[0].get("width", 0)), int(streams[0].get("height", 0))
    except Exception:
        pass
    return (0, 0)


def _fmt_ts(seconds: float) -> str:
    """Convierte segundos a formato ASS: H:MM:SS.cs"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ─── ASS subtitle style ────────────────────────────────────────────────────────
#
# PlayRes 1920×1080 es el espacio de coordenadas para los estilos.
# ffmpeg escala automáticamente a la resolución real del video de salida:
#   • 1080p → sin escala (1x)
#   • 4K    → escala 2x  (el texto aparece al mismo tamaño relativo en pantalla)
#
# Esto garantiza que el mismo archivo .ass funciona correctamente
# tanto para 1080p como para 4K sin necesidad de cambiar PlayRes.

ASS_HEADER_WIDE = """\
[Script Info]
Title: Subtitulos ES
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1920
PlayResY: 1080
PlayDepth: 0
Timer: 100.0000
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,68,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,3,8,0,2,80,80,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
# Style: Default  —  estilo YouTube (fondo negro sólido, texto blanco)
#   FontSize 68pt  @ PlayRes 1920×1080
#   PrimaryColour  &H00FFFFFF = blanco opaco
#   OutlineColour  &H00000000 = negro (borde de la caja)
#   BackColour     &H00000000 = negro sólido (AA=00 → completamente opaco)
#   Bold           -1 = activo
#   BorderStyle    3  = caja opaca (opaque box, igual que YouTube)
#   Outline        8  = padding interior de la caja (px en coordenadas PlayRes)
#   Shadow         0  = sin sombra (no necesaria con caja)
#   Alignment      2  = bottom-center
#   MarginV        200 = 200px desde el borde inferior (en coordenadas PlayRes 1080p)
#                       → ffmpeg escala proporcionalmente para 4K


_MIN_SUB_DUR    = 0.20   # duración mínima de cada subtitle en pantalla (s)
_MAX_WORDS_LINE = 7      # palabras máximas por línea


def _build_subtitle_chunks(words: list) -> list:
    """
    Agrupa palabras en líneas de una sola línea (máx _MAX_WORDS_LINE palabras).
    Corte en silencios > 0.6 s o al alcanzar el límite de palabras.

    Retorna: [{"text": str, "start": float, "end": float}, ...]
    """
    if not words:
        return []

    chunks  = []
    current = [words[0]]

    for w in words[1:]:
        gap = w["start"] - current[-1]["end"]
        if len(current) >= _MAX_WORDS_LINE or gap > 0.6:
            chunks.append({
                "text":  " ".join(c["word"] for c in current),
                "start": current[0]["start"],
                "end":   max(current[-1]["end"], current[0]["start"] + _MIN_SUB_DUR),
            })
            current = [w]
        else:
            current.append(w)

    if current:
        chunks.append({
            "text":  " ".join(c["word"] for c in current),
            "start": current[0]["start"],
            "end":   max(current[-1]["end"], current[0]["start"] + _MIN_SUB_DUR),
        })

    return chunks


def _write_ass(chunks: list, out_path: str) -> None:
    """Escribe el archivo .ass con los subtítulos."""
    lines = [ASS_HEADER_WIDE]
    for chunk in chunks:
        start = chunk["start"]
        end   = max(chunk["end"], start + _MIN_SUB_DUR)
        text  = chunk["text"]
        lines.append(
            f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},Default,,0,0,0,,{text}"
        )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ─── Transcripción con Whisper ────────────────────────────────────────────────

def _transcribe_words(whisper_model, audio_path: str) -> list:
    """
    Transcribe el audio doblado (ES) con Whisper y retorna lista de palabras.
    Retorna: [{"word": str, "start": float, "end": float}, ...]
    """
    print(f"  [{time.strftime('%H:%M:%S')}] Transcribiendo subtítulos ES con Whisper...")
    segments_iter, _ = whisper_model.transcribe(
        audio_path,
        language="es",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 200},
    )

    words = []
    for seg in segments_iter:
        if seg.words:
            for w in seg.words:
                word_text = w.word.strip()
                if word_text:
                    words.append({
                        "word":  word_text,
                        "start": round(w.start, 3),
                        "end":   round(w.end, 3),
                    })

    print(f"  [{time.strftime('%H:%M:%S')}] {len(words)} palabras transcritas")
    return words


# ─── Elección de codec según resolución ──────────────────────────────────────

def _choose_codec(width: int, height: int) -> tuple:
    """
    Retorna (codec, cq, label) según la resolución y disponibilidad de NVENC.

    GPU con NVENC (RTX/GTX local):
      • ≤ 1080p → h264_nvenc CQ 18
      • 1440p   → h264_nvenc CQ 16
      • 4K+     → h264_nvenc CQ 14

    Sin NVENC (A100/datacenter RunPod):
      • ≤ 1080p → libx264 CRF 22
      • 1440p   → libx264 CRF 20
      • 4K+     → libx264 CRF 18
    """
    pixels = width * height
    if NVENC_AVAILABLE:
        if pixels > 2560 * 1440:
            return "h264_nvenc", "14", "4K H.264"
        elif pixels > 1920 * 1080:
            return "h264_nvenc", "16", "QHD H.264"
        else:
            return "h264_nvenc", "18", "FHD H.264"
    else:
        if pixels > 2560 * 1440:
            return "libx264", "18", "4K H.264 (CPU)"
        elif pixels > 1920 * 1080:
            return "libx264", "20", "QHD H.264 (CPU)"
        else:
            return "libx264", "22", "FHD H.264 (CPU)"

def _run_video_retalking(video_path: str, audio_path: str, out_path: str, cb) -> str:
    """Ejecuta Video-Retalking para sincronizar los labios antes de los subtítulos."""
    cb("[Lip-Sync] Iniciando Video-Retalking (Esto usará la GPU intensivamente)...", 0.08)
    
    # Asume que clonaste video-retalking en la misma carpeta que la app
    retalking_dir = Path(__file__).parent / "video-retalking"
    if not retalking_dir.exists():
        raise FileNotFoundError(f"No se encontró la carpeta {retalking_dir}. Debes clonar el repositorio de OpenTalker.")

    cmd = [
        sys.executable, "inference.py",
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", out_path
    ]
    
    # Ejecutamos el subproceso dentro de la carpeta de video-retalking
    result = subprocess.run(cmd, cwd=str(retalking_dir), capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Error en Video-Retalking:\n{result.stderr[-1500:]}")
    
    cb("[Lip-Sync] Sincronización labial completada con éxito.", 0.12)
    return out_path




# ─── Función principal ────────────────────────────────────────────────────────

def generate_subtitled_video(
    video_path: str,
    audio_path: str,
    whisper_model,
    out_dir: str,
    test_seconds=None,
    progress_cb=None,
    contrast: float = 0,
    brightness: float = 0,
    sharpness: float = 0,
    do_orig_audio: bool = False,
    orig_audio_path: str = None,
    orig_audio_db: float = -25,
    do_lipsync: bool = False,
) -> tuple:
    """
    Genera un video con subtítulos quemados y audio doblado.

    Soporta cualquier resolución de entrada (1080p, 1440p, 4K, 8K).
    El codec se elige automáticamente según NVENC_AVAILABLE (detectado al inicio).

    Args:
        video_path      : Ruta al video de entrada.
        audio_path      : Ruta al audio doblado (MP3, WAV, etc.)
        whisper_model   : Instancia de faster_whisper.WhisperModel.
        out_dir         : Directorio de salida.
        test_seconds    : Recorta a este número de segundos si se especifica.
        progress_cb     : Callable(msg, pct=None).
        contrast        : −100..100 (eq=contrast)
        brightness      : −100..100 (eq=brightness)
        sharpness       : 0..100 (unsharp luma)
        do_orig_audio   : Mezcla audio original al fondo.
        orig_audio_path : Ruta al audio original EN.
        orig_audio_db   : Nivel dB del audio original (ej. −25).

    Returns:
        (out_video_path: str, status: str)
    """
    def _cb(msg, pct=None):
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}")
        if progress_cb:
            progress_cb(msg, pct)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_label = f"test_{test_seconds}s" if test_seconds else "full"
    out_video  = out_dir / f"output_subbed_{mode_label}.mp4"
    tmp_video  = out_dir / f"_tmp_video_{mode_label}.mp4"
    tmp_audio  = out_dir / f"_tmp_audio_{mode_label}.mp3"
    wav_audio  = out_dir / f"_tmp_audio_{mode_label}.wav"
    ass_path   = out_dir / f"subtitles_{mode_label}.ass"

    # ── Paso 1: Recortar si es modo test ─────────────────────────────────────
    if test_seconds:
        _cb(f"[1/4] Recortando a {test_seconds}s (modo test)...", 0.05)
        _run([
            "ffmpeg", "-y",
            "-ss", "0", "-t", str(test_seconds),
            "-i", video_path,
            "-c:v", "copy", "-an",
            str(tmp_video),
        ], f"Recortar video a {test_seconds}s")
        _run([
            "ffmpeg", "-y",
            "-ss", "0", "-t", str(test_seconds),
            "-i", audio_path,
            "-c:a", "libmp3lame", "-q:a", "3",
            str(tmp_audio),
        ], f"Recortar audio a {test_seconds}s")
        effective_video = str(tmp_video)
        effective_audio = str(tmp_audio)
    else:
        _cb("[1/4] Usando video y audio completos...", 0.05)
        effective_video = video_path
        effective_audio = audio_path

    # ── [NUEVO] Paso 1.5: LIP-SYNC (Video-Retalking) ──────────────────────────
    if do_lipsync:
        retalked_video = out_dir / f"_tmp_retalked_{mode_label}.mp4"
        _run_video_retalking(effective_video, effective_audio, str(retalked_video), _cb)
        effective_video = str(retalked_video) # A partir de aquí, usamos el video con lip-sync

    # ── Detectar resolución y elegir codec ────────────────────────────────────
    w, h = _get_video_resolution(effective_video)
    codec, cq, codec_label = _choose_codec(w, h)
    res_label = f"{w}×{h}" if w else "desconocida"
    _cb(f"[1/4] Resolución detectada: {res_label} → codec {codec_label}", 0.07)

    # ── Paso 2: WAV para Whisper + transcripción ──────────────────────────────
    _cb("[2/4] Extrayendo WAV para Whisper...", 0.10)
    _run([
        "ffmpeg", "-y",
        "-i", effective_audio,
        "-ar", "16000", "-ac", "1", "-vn",
        str(wav_audio),
    ], "Extraer WAV 16kHz mono")

    _cb("[2/4] Transcribiendo subtítulos con Whisper (ES)...", 0.15)
    words  = _transcribe_words(whisper_model, str(wav_audio))
    chunks = _build_subtitle_chunks(words)
    _write_ass(chunks, str(ass_path))
    _cb(f"[2/4] {len(chunks)} líneas de subtítulos generadas.", 0.30)

    # ── Paso 3: Filtros de video ──────────────────────────────────────────────
    _cb("[3/4] Construyendo filtros de video...", 0.35)

    vf_parts = []
    if brightness != 0 or contrast != 0:
        br_ffmpeg = brightness / 100.0       # −1.0 a 1.0
        ct_ffmpeg = 1.0 + contrast / 100.0   # 0.0 a 2.0
        vf_parts.append(f"eq=brightness={br_ffmpeg:.4f}:contrast={ct_ffmpeg:.4f}")
    if sharpness > 0:
        sh_ffmpeg = (sharpness / 100.0) * 1.5  # 0.0 a 1.5
        vf_parts.append(f"unsharp=5:5:{sh_ffmpeg:.4f}:5:5:0.0")

    # Subtítulos: solo nombre de archivo + cwd=out_dir para evitar problema
    # con ':' en rutas Windows interpretado por libass como separador.
    ass_name = ass_path.name
    vf_parts.append(f"ass={ass_name}")
    vf_filter = ",".join(vf_parts)

    # ── Paso 4: Ensamblado final ──────────────────────────────────────────────
    _cb(f"[4/4] Ensamblando {res_label} con {codec_label}...", 0.40)

    has_orig     = do_orig_audio and orig_audio_path and Path(orig_audio_path).exists()
    orig_vol_lin = 10 ** (orig_audio_db / 20.0)

    cmd = ["ffmpeg", "-y"]
    if NVENC_AVAILABLE:
        cmd += ["-hwaccel", "cuda"]   # decodificación en GPU (solo RTX/GTX)
    cmd += [
        "-i", effective_video,   # [0] video
        "-i", effective_audio,   # [1] audio doblado
    ]
    audio_inputs = ["[1:a]volume=1.0[voice]"]
    mix_labels   = ["[voice]"]
    next_idx     = 2

    if has_orig:
        cmd += ["-i", orig_audio_path]
        audio_inputs.append(f"[{next_idx}:a]volume={orig_vol_lin:.4f}[orig]")
        mix_labels.append("[orig]")
        next_idx += 1

    n_audio = len(mix_labels)
    if n_audio == 1:
        filter_complex = audio_inputs[0]
        aout_label     = "[voice]"
    else:
        mix_str        = "".join(mix_labels)
        filter_complex = (
            ";".join(audio_inputs) + ";" +
            f"{mix_str}amix=inputs={n_audio}:duration=first:dropout_transition=2[aout]"
        )
        aout_label = "[aout]"

    # Params de codec: NVENC usa -preset/-rc/-cq; libx264 usa -preset/-crf
    if NVENC_AVAILABLE:
        codec_params = ["-c:v", codec, "-preset", "p4", "-rc", "vbr", "-cq", cq]
    else:
        codec_params = ["-c:v", codec, "-preset", "fast", "-crf", cq]

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", aout_label,
        "-vf", vf_filter,
        "-pix_fmt", "yuv420p",
        *codec_params,
        "-c:a", "aac", "-b:a", "256k",
        "-shortest",
        str(out_video),
    ]

    mode_desc = [f"res={res_label}", f"codec={codec_label}"]
    if has_orig:
        mode_desc.append(f"orig {orig_audio_db}dB")
    if brightness != 0 or contrast != 0:
        mode_desc.append("eq")
    if sharpness > 0:
        mode_desc.append("unsharp")
    mode_desc.append(f"{len(chunks)} subs")

    print(f"  [{time.strftime('%H:%M:%S')}] ffmpeg: {' | '.join(mode_desc)}")
    total_dur = _get_duration(str(effective_video))
    _run(
        cmd,
        cwd=str(out_dir),
        description=f"Ensamblar {res_label} con subtítulos",
        duration_s=total_dur,
        progress_cb=progress_cb,
        pct_start=0.40,
        pct_end=0.98,
    )

    # ── Resultado ─────────────────────────────────────────────────────────────
    size_mb = out_video.stat().st_size / 1e6
    dur     = _get_duration(str(out_video))
    m, s_   = divmod(int(dur), 60)

    _cb(f"[4/4] Listo: {out_video.name} ({size_mb:.1f} MB, {m}m {s_}s)", 1.0)

    status = (
        f"OK [{mode_label}] — {res_label} {codec_label} | "
        f"{len(chunks)} subtítulos | {m}m {s_}s | {size_mb:.1f} MB\n"
        f"Archivo: {out_video.name}"
    )
    return str(out_video), status
