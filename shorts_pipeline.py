"""
Shorts pipeline: genera short videos verticales (1080x1920) con captions
palabra-por-palabra estilo Reels/TikTok a partir de la configuración JSON.
"""

import json
import os
import re
import subprocess
from pathlib import Path

from subtitle_pipeline import NVENC_AVAILABLE


# ─── Timestamp helper (ASS format) ───────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    """Convierte segundos a formato ASS: H:MM:SS.cs"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _slug(text: str, max_len: int = 40) -> str:
    """Convierte un título a slug seguro para nombre de archivo."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:max_len]


# ─── Config ────────────────────────────────────────────────────────────────────

def parse_shorts_config(config_path: str) -> list:
    """Lee el JSON de configuración y retorna la lista de shorts."""
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["video_shorts"]


# ─── Audio segment + transcription ───────────────────────────────────────────

def _audio_hash(path: str) -> str:
    """Hash rápido (MD5 primeros 2 MB + tamaño) para detectar cambios de audio."""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()[:12]


def transcribe_segment_words(
    whisper_model,
    seg_audio: str,
    short_id: int,
    cache_dir: str,
) -> list:
    """
    Transcribe con Whisper el segmento de audio ya extraído (seg_audio)
    y retorna lista de palabras: [{"word": str, "start": float, "end": float}, ...]

    Cachea el resultado en words_short_XX.json junto con un hash del audio.
    Si el audio cambió desde la última transcripción, el caché se invalida
    automáticamente y se vuelve a transcribir.

    Los timestamps son relativos al inicio del short (0-based).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"words_short_{short_id:02d}.json"

    current_hash = _audio_hash(seg_audio)

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        # Formato nuevo: {"audio_hash": ..., "words": [...]}
        if isinstance(cached, dict) and cached.get("audio_hash") == current_hash:
            print(f"  [shorts] Cache hit (audio sin cambios): words_short_{short_id:02d}.json")
            return cached["words"]
        # Hash distinto o formato antiguo → re-transcribir
        if isinstance(cached, dict):
            print(f"  [shorts] Audio cambió — re-transcribiendo short {short_id}")
        else:
            print(f"  [shorts] Cache antiguo sin hash — re-transcribiendo short {short_id}")

    print(f"  [shorts] Transcribiendo con Whisper (es, word_timestamps)...")
    segments_iter, _ = whisper_model.transcribe(
        seg_audio,
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

    print(f"  [shorts] {len(words)} palabras transcritas")

    # Guardar con hash para detectar cambios futuros
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"audio_hash": current_hash, "words": words}, f,
                  ensure_ascii=False, indent=2)

    return words


# ─── Word grouping ─────────────────────────────────────────────────────────────

def build_word_chunks(
    words: list,
    max_words: int = 4,
    max_gap: float = 0.5,
) -> list:
    """
    Agrupa palabras en chunks para las captions estilo Reels.

    Nuevo chunk cuando:
      - gap entre palabras > max_gap segundos, O
      - ya se acumularon max_words palabras

    Retorna: [{"text": str, "start": float, "end": float}, ...]
    """
    if not words:
        return []

    chunks = []
    current = [words[0]]

    for w in words[1:]:
        gap = w["start"] - current[-1]["end"]
        if len(current) >= max_words or gap > max_gap:
            chunks.append({
                "text":  " ".join(c["word"] for c in current),
                "start": current[0]["start"],
                "end":   current[-1]["end"],
            })
            current = [w]
        else:
            current.append(w)

    if current:
        chunks.append({
            "text":  " ".join(c["word"] for c in current),
            "start": current[0]["start"],
            "end":   current[-1]["end"],
        })

    return chunks


# ─── ASS captions for vertical video ─────────────────────────────────────────

ASS_HEADER_VERTICAL = """\
[Script Info]
Title: Shorts Captions
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1080
PlayResY: 1920
PlayDepth: 0
Timer: 100.0000
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,105,&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,5,2,5,60,60,0,1
Style: Title,Arial,79,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,8,50,50,110,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
# Default style — captions centradas en el medio del video:
#   PrimaryColour &H0000FFFF = amarillo (AABBGGRR: A=00, B=00, G=FF, R=FF)
#   Alignment 5 = centro-medio de pantalla
#   Outline 5px negro, Shadow 2px, tamaño 105pt
#
# Title style — título fijo arriba (SIN fondo en ASS):
#   El fondo negro se dibuja con drawbox en FFmpeg (más confiable que BorderStyle 3)
#   PrimaryColour &H0000FFFF = amarillo
#   BorderStyle 1 = texto normal con outline (outline=2, shadow=1)
#   Alignment 8 = top-center
#   Fontsize 79 = 105 * 0.75 (-25%)
#   MarginV 110 = margen desde arriba
#
# El filtro FFmpeg drawbox=x=0:y=0:w=iw:h=310:color=black:t=fill
# dibuja el rectángulo negro detrás del texto del título.

_MIN_WORD_DUR = 0.25  # duración mínima de cada palabra en pantalla (segundos)


def write_ass_captions_vertical(
    chunks: list,
    output_path: str,
    titulo: str = "",
    duration: float = 0.0,
) -> None:
    """
    Escribe un archivo .ass con:
      - Título del short fijo en la parte superior con fondo negro (Style: Title)
      - Captions palabra por palabra en amarillo centradas en el medio (Style: Default)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER_VERTICAL)

        # Captions palabra por palabra en el centro
        for chunk in chunks:
            start = chunk["start"]
            end   = max(chunk["end"], start + _MIN_WORD_DUR)
            text  = chunk["text"].strip().replace("\n", " ")
            f.write(f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},Default,,0,0,0,,{text}\n")


# ─── FFmpeg helpers ────────────────────────────────────────────────────────────

def _run(cmd: list, cwd: str = None, description: str = "") -> None:
    """Ejecuta un comando FFmpeg. Falla con error claro si el comando no tiene éxito."""
    print(f"  [shorts] {description or 'ffmpeg'}")
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"FFmpeg falló ({description}):\n{stderr[-1200:]}") from e


# ─── Single short creation ────────────────────────────────────────────────────

def create_short(
    short_cfg: dict,
    video_src: str,
    audio_src: str,
    whisper_model,
    output_dir: str,
    cache_dir: str,
    bg_music_path: str = None,
    contrast: float = 0,
    brightness: float = 0,
    sharpness: float = 0,
    use_orig_audio: bool = False,
    orig_audio_db: float = -20,
    orig_audio_src: str = None,
    crop_offset_pct: float = 0,
    output_4k: bool = False,
) -> tuple:
    """
    Genera un short vertical completo.

    Resoluciones de salida:
      output_4k=False → 1080×1920 (Full HD vertical, h264_nvenc CQ 15/18)
      output_4k=True  → 2160×3840 (4K vertical, hevc_nvenc CQ 16/18)

    Pasos:
      1. Extrae + convierte video a resolución vertical con crop horizontal.
      2. Transcribe segmento de audio con Whisper (word timestamps).
      3. Agrupa palabras (1 por vez) y genera ASS captions.
      4. Combina video + voz + música de fondo (7.5%) + captions.
         Aplica filtros de imagen (eq, unsharp) y audio original si se indica.

    bg_music_path  : ruta al MP3 de música de fondo. None = sin música.
    contrast       : −100 a +100 (0 = sin cambio)
    brightness     : −100 a +100 (0 = sin cambio)
    sharpness      : 0 a 100 (0 = sin cambio)
    use_orig_audio : mezcla audio original EN a orig_audio_db dB.
    output_4k      : True → salida 2160×3840 4K vertical con hevc_nvenc.

    Retorna (output_path, status_msg).
    """
    short_id   = short_cfg["id"]
    titulo     = short_cfg["titulo"]
    start_sec  = float(short_cfg["start_seconds"])
    end_sec    = float(short_cfg["end_seconds"])
    duration   = end_sec - start_sec

    output_dir = Path(output_dir)
    cache_dir  = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Parámetros de resolución
    # h264_nvenc si GPU lo soporta (RTX/GTX); libx264 en datacenter (A100/RunPod).
    _res_suffix   = "_4k" if output_4k else ""
    _out_h        = 3840  if output_4k else 1920
    _out_w        = 2160  if output_4k else 1080
    _vid_codec    = "h264_nvenc" if NVENC_AVAILABLE else "libx264"
    # CQ/CRF para 4K: valores más altos = archivos más pequeños.
    # CQ 13-15 → ~35-50 Mbps → 1.5+ GB por clip de 5 min → llena el disco.
    # CQ 26    → ~12-18 Mbps → ~400-550 MB por clip de 5 min → calidad social media perfecta.
    _vid_cq_extr  = "20" if output_4k else "15"   # archivo temporal intermedio
    _vid_cq_final = "26" if output_4k else "18"   # salida final (~15 Mbps a 4K)

    slug_name  = _slug(titulo)
    out_name   = f"short_{short_id:02d}_{slug_name}{_res_suffix}.mp4"
    out_path   = output_dir / out_name

    # Determinar si se usan parámetros de mejora (no-default)
    has_enhancements = (contrast != 0 or brightness != 0 or sharpness != 0
                        or use_orig_audio or crop_offset_pct != 0 or output_4k)

    print(f"\n  {'='*50}")
    print(f"  SHORT {short_id}: {titulo}")
    print(f"  Rango  : {start_sec:.1f}s - {end_sec:.1f}s ({duration:.1f}s)")
    print(f"  Res    : {_out_w}×{_out_h} {'(4K)' if output_4k else '(HD)'}")
    print(f"  Output : {out_path.name}")
    if has_enhancements:
        print(f"  Filtros: contraste={contrast} brillo={brightness} nitidez={sharpness}"
              f"{f' audio_orig={orig_audio_db}dB' if use_orig_audio else ''}")

    # ── PREVENTIVO: Validar duración del audio ANTES de intentar extraer ──
    # Esto evita crear archivos MP3 corruptos cuando el rango excede
    # la duración disponible del audio doblado (típico en el último short).
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_src)],
            capture_output=True, text=True, check=True,
        )
        audio_dur = float(probe.stdout.strip())
    except (ValueError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"Short {short_id}: no se pudo determinar duración del audio: {e}")

    # Si el short empieza después de que termina el audio → SALTEAR
    if start_sec >= audio_dur:
        print(f"  [SKIP] Short {short_id}: start_sec ({start_sec:.1f}s) >= duración del audio ({audio_dur:.1f}s). "
              "No hay audio doblado para este rango.")
        return None, f"[SKIP] {titulo} — sin audio doblado (rango fuera del audio)"

    # Si el short termina después del audio → RECORTAR a lo disponible
    if end_sec > audio_dur:
        old_end = end_sec
        end_sec = audio_dur
        duration = end_sec - start_sec
        print(f"  [AJUSTE] Audio termina en {audio_dur:.1f}s, recortando {old_end:.1f}s → {end_sec:.1f}s ({duration:.1f}s)")

    # ── Extraer segmento de audio ─────────────────────────────────────────
    seg_audio = cache_dir / f"seg_audio_{short_id:02d}.mp3"
    print(f"  [0/4] Extrayendo audio {start_sec:.1f}s-{end_sec:.1f}s...")
    _run(
        ["ffmpeg", "-y",
         "-ss", str(start_sec), "-t", str(duration),
         "-i", str(audio_src), "-c:a", "libmp3lame", "-q:a", "4",
         str(seg_audio)],
        description=f"Extraer audio short {short_id}",
    )

    # ── Cache hit (solo si NO hay parámetros de mejora Y el audio no cambió) ─
    if out_path.exists() and not has_enhancements:
        # Leer hash almacenado en el JSON de palabras (si existe)
        words_cache = cache_dir / f"words_short_{short_id:02d}.json"
        audio_unchanged = False
        if words_cache.exists():
            try:
                with open(words_cache, encoding="utf-8") as _f:
                    _cached = json.load(_f)
                if isinstance(_cached, dict) and _cached.get("audio_hash") == _audio_hash(str(seg_audio)):
                    audio_unchanged = True
            except Exception:
                pass
        if audio_unchanged:
            size_mb = out_path.stat().st_size / 1e6
            print(f"  Ya existe y audio sin cambios ({size_mb:.1f} MB), saltando...")
            return str(out_path), f"[CACHE] {titulo} ya existe ({size_mb:.1f} MB)"
        else:
            print(f"  Audio actualizado — regenerando short {short_id}...")

    # ── Paso 1: Extraer video vertical ────────────────────────────────
    # crop_offset_pct: -100=izquierda, 0=centro, +100=derecha
    # Escala a _out_h px de alto, luego recorta _out_w px horizontalmente.
    # El factor desplaza el punto de crop dentro del rango disponible.
    _crop_factor = max(-1.0, min(1.0, crop_offset_pct / 100.0))
    _crop_vf = (
        f"scale=-1:{_out_h},"
        f"crop={_out_w}:{_out_h}:(in_w-{_out_w})/2*(1+{_crop_factor:.4f}):0"
    )
    temp_video = cache_dir / f"temp_video_{short_id:02d}{_res_suffix}.mp4"
    print(f"  [1/4] Extrayendo video vertical {_out_w}×{_out_h} "
          f"(crop_offset={crop_offset_pct}%, {'4K' if output_4k else 'HD'})...")
    _extract_cmd = ["ffmpeg", "-y"]
    if NVENC_AVAILABLE:
        _extract_cmd += ["-hwaccel", "cuda"]
    _extract_cmd += [
        "-ss", str(start_sec), "-t", str(duration),
        "-i", str(video_src),
        "-vf", _crop_vf,
        "-an",
        "-pix_fmt", "yuv420p",
    ]
    if NVENC_AVAILABLE:
        _extract_cmd += ["-c:v", _vid_codec, "-preset", "p4", "-rc", "vbr", "-cq", _vid_cq_extr]
    else:
        _extract_cmd += ["-c:v", _vid_codec, "-preset", "fast", "-crf", _vid_cq_extr]
    _extract_cmd.append(str(temp_video))
    _run(_extract_cmd, description=f"Extraer video vertical short {short_id}")

    # ── Paso 2: Transcribir palabras (usa seg_audio recién extraído) ──
    print(f"  [2/4] Transcribiendo palabras del audio...")
    words = transcribe_segment_words(
        whisper_model=whisper_model,
        seg_audio=str(seg_audio),
        short_id=short_id,
        cache_dir=str(cache_dir),
    )

    # ── Paso 3: Agrupar palabras (1 por vez) y generar ASS ──────────
    print(f"  [3/4] Generando captions ({len(words)} palabras, 1 por vez)...")
    chunks   = build_word_chunks(words, max_words=1, max_gap=0.5)
    ass_path = cache_dir / f"captions_{short_id:02d}.ass"
    write_ass_captions_vertical(
        chunks,
        str(ass_path),
        titulo=titulo,
        duration=duration,
    )
    print(f"  -> {len(chunks)} palabras en captions | título: '{titulo}'")

    # ── Paso 4: Construir filtro de video ─────────────────────────────
    # seg_audio ya fue extraído en el paso 0 (inicio de esta función)
    ass_name  = ass_path.name  # solo nombre para evitar el bug de rutas Windows

    # eq (brillo/contraste) → unsharp (nitidez) → ass (captions)
    # drawbox eliminado: el video ocupa el frame completo sin barra de título
    vf_parts = []
    if brightness != 0 or contrast != 0:
        br_ffmpeg = brightness / 100.0        # −1.0 a 1.0
        ct_ffmpeg = 1.0 + contrast / 100.0    # 0.0 a 2.0
        vf_parts.append(f"eq=brightness={br_ffmpeg:.4f}:contrast={ct_ffmpeg:.4f}")
    if sharpness > 0:
        sh_ffmpeg = (sharpness / 100.0) * 1.5  # 0.0 a 1.5
        vf_parts.append(f"unsharp=5:5:{sh_ffmpeg:.4f}:5:5:0.0")
    vf_parts.append(f"ass={ass_name}")
    vf_filter = ",".join(vf_parts)

    # ── Extraer audio original si se solicita ────────────────────────
    orig_audio_temp = None
    if use_orig_audio:
        # Buscar fuente de audio original: primero en audio_original_en/, si no hay usar video_src
        orig_src_path   = orig_audio_src if orig_audio_src else video_src
        orig_audio_temp = cache_dir / f"orig_audio_{short_id:02d}.mp3"
        print(f"  [4/4-pre] Extrayendo audio original EN de: {Path(orig_src_path).name}")
        _run(
            ["ffmpeg", "-y",
             "-ss", str(start_sec), "-t", str(duration),
             "-i", str(orig_src_path),
             "-vn", "-c:a", "libmp3lame", "-q:a", "4",
             str(orig_audio_temp)],
            description=f"Extraer audio original short {short_id}",
        )

    # ── Paso 4: Combinar video + audio (+ música + audio original) ───
    has_music     = bg_music_path and Path(bg_music_path).exists()
    has_orig      = use_orig_audio and orig_audio_temp and orig_audio_temp.exists()
    orig_vol_lin  = 10 ** (orig_audio_db / 20.0)  # dB → lineal (−20 dB = 0.1)

    # Construir inputs y filter_complex dinámicamente
    cmd = ["ffmpeg", "-y",
           "-i", str(temp_video),   # [0:v]
           "-i", str(seg_audio)]    # [1:a] voz doblada
    audio_inputs = ["[1:a]volume=0.8[voice]"]
    mix_labels   = ["[voice]"]
    next_idx     = 2

    if has_music:
        cmd += ["-stream_loop", "-1", "-i", str(bg_music_path)]  # [2:a]
        audio_inputs.append(f"[{next_idx}:a]volume=0.20[music]")
        mix_labels.append("[music]")
        next_idx += 1

    if has_orig:
        cmd += ["-i", str(orig_audio_temp)]  # [N:a]
        audio_inputs.append(f"[{next_idx}:a]volume={orig_vol_lin:.4f}[orig]")
        mix_labels.append("[orig]")
        next_idx += 1

    n_audio = len(mix_labels)
    if n_audio == 1:
        # Solo voz — no necesita amix
        filter_complex = audio_inputs[0]
        aout_label = "[voice]"
    else:
        mix_str = "".join(mix_labels)
        filter_complex = (
            ";".join(audio_inputs) + ";" +
            f"{mix_str}amix=inputs={n_audio}:duration=first:dropout_transition=2[aout]"
        )
        aout_label = "[aout]"

    mode_desc = [f"{'4K' if output_4k else 'HD'} {_out_w}×{_out_h}"]
    if has_music: mode_desc.append("música")
    if has_orig:  mode_desc.append(f"audio orig {orig_audio_db}dB")
    if brightness != 0 or contrast != 0: mode_desc.append("eq")
    if sharpness > 0: mode_desc.append("unsharp")
    print(f"  [4/4] Ensamblando short ({', '.join(mode_desc)})...")

    if n_audio > 1:
        cmd += ["-filter_complex", filter_complex,
                "-map", "0:v", "-map", aout_label]
    else:
        # Sin amix: solo voz directa
        cmd += ["-map", "0:v", "-map", "1:a"]

    if NVENC_AVAILABLE:
        _final_codec_params = ["-c:v", _vid_codec, "-preset", "p4", "-rc", "vbr", "-cq", _vid_cq_final]
    else:
        _final_codec_params = ["-c:v", _vid_codec, "-preset", "fast", "-crf", _vid_cq_final]

    cmd += ["-vf", vf_filter,
            "-pix_fmt", "yuv420p",
            *_final_codec_params,
            "-c:a", "aac", "-b:a", "256k",
            "-shortest",
            str(out_path)]

    _run(cmd, cwd=str(cache_dir), description=f"Ensamblar short {short_id} ({_out_w}×{_out_h})")

    size_mb = out_path.stat().st_size / 1e6
    print(f"  OK: {out_path.name} ({size_mb:.1f} MB, {_out_w}×{_out_h})")

    return str(out_path), f"OK: {titulo} ({duration:.0f}s, {size_mb:.1f} MB, {_out_w}×{_out_h})"


# ─── Batch generation ─────────────────────────────────────────────────────────

def generate_all_shorts(
    shorts_dir: str,
    whisper_model,
    selected_ids: list = None,
    progress_cb=None,
    contrast: float = 0,
    brightness: float = 0,
    sharpness: float = 0,
    use_orig_audio: bool = False,
    orig_audio_db: float = -20,
    crop_offset_pct: float = 0,
    output_4k: bool = False,
) -> list:
    """
    Genera todos (o los seleccionados) los shorts.

    shorts_dir debe contener:
      - shorts-settings.json
      - short_video/          (video fuente landscape)
      - short_audio_doblado/  (audio doblado en español)
      - audio_original_en/    (audio/video original EN para fondo — opcional)
      - shorts_videos_lists/  (salida)

    Retorna lista de paths generados o existentes.
    """
    shorts_dir = Path(shorts_dir)
    config     = shorts_dir / "shorts-settings.json"
    output_dir = shorts_dir / "shorts_videos_lists"
    cache_dir  = shorts_dir / "_cache"

    # Buscar video y audio fuente (primer archivo .mp4 y .mp3)
    video_src = _find_first(shorts_dir / "short_video", "*.mp4")
    audio_src = _find_first(shorts_dir / "short_audio_doblado", "*.mp3")

    # Audio original EN: buscar en audio_original_en/ (MP4, MP3, WAV, M4A)
    orig_audio_src = None
    orig_audio_dir = shorts_dir / "audio_original_en"
    for ext in ("*.mp4", "*.mp3", "*.wav", "*.m4a", "*.mkv", "*.avi"):
        orig_audio_src = _find_first(orig_audio_dir, ext)
        if orig_audio_src:
            break

    if not video_src:
        raise FileNotFoundError(
            f"No se encontró video en {shorts_dir / 'short_video'}"
        )
    if not audio_src:
        raise FileNotFoundError(
            f"No se encontró audio en {shorts_dir / 'short_audio_doblado'}"
        )

    # Música de fondo — busca primero cinematic_tension.mp3, luego cualquier MP3
    bg_music_path = None
    music_dir = shorts_dir / "background_music"
    specific   = music_dir / "cinematic_tension.mp3"
    if specific.exists():
        bg_music_path = str(specific)
    else:
        bg_music_path = _find_first(music_dir, "*.mp3")

    shorts = parse_shorts_config(str(config))

    if selected_ids:
        shorts = [s for s in shorts if s["id"] in selected_ids]

    print(f"\n[generate_all_shorts] {len(shorts)} shorts a procesar")
    print(f"  Video     : {video_src}")
    print(f"  Audio ES  : {audio_src}")
    print(f"  Audio orig: {orig_audio_src or '(usar video fuente)'}")
    print(f"  Música    : {bg_music_path or 'ninguna'}")
    print(f"  Output    : {output_dir}")
    print(f"  Resolución: {'4K 2160×3840' if output_4k else 'HD 1080×1920'}")

    results = []
    for i, cfg in enumerate(shorts):
        if progress_cb:
            progress_cb(i, len(shorts), cfg["titulo"])
        try:
            out_path, status = create_short(
                short_cfg=cfg,
                video_src=video_src,
                audio_src=audio_src,
                bg_music_path=bg_music_path,
                whisper_model=whisper_model,
                output_dir=str(output_dir),
                cache_dir=str(cache_dir / f"short_{cfg['id']:02d}"),
                contrast=contrast,
                brightness=brightness,
                sharpness=sharpness,
                use_orig_audio=use_orig_audio,
                orig_audio_db=orig_audio_db,
                orig_audio_src=orig_audio_src,
                crop_offset_pct=crop_offset_pct,
                output_4k=output_4k,
            )
            results.append((cfg["id"], cfg["titulo"], out_path, status, None))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  [ERROR] Short {cfg['id']}: {e}\n{tb}")
            results.append((cfg["id"], cfg["titulo"], None, None, str(e)))

    return results


def _find_first(directory: Path, pattern: str):
    """Retorna el primer archivo que coincide con el patrón, o None."""
    directory = Path(directory)
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern))
    return str(files[0]) if files else None


# ─── List existing shorts ─────────────────────────────────────────────────────

def list_existing_shorts(output_dir: str) -> list:
    """Retorna rutas de MP4 existentes en output_dir, ordenadas por nombre."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    return sorted(str(p) for p in output_dir.glob("*.mp4"))
