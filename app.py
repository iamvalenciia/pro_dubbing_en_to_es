#!/usr/bin/env python3
"""
EN -> ES Voice Dubbing — Gradio Web Interface
Con sistema de caché/checkpoint para reanudar trabajos interrumpidos.
"""

# ─── cuDNN DLL fix (Windows) ──────────────────────────────────────────────────
# DEBE ejecutarse ANTES de cualquier import que cargue CUDA DLLs
import os
import sys
import glob as _glob

def _setup_cudnn_dlls():
    """
    Registra en Windows los directorios de DLLs de cuDNN en el orden correcto:
      1. torch/lib  (tiene cudnn64_8.dll 8.9.x — la versión que speechbrain necesita)
      2. nvidia-cudnn-cu12 bin  (tiene cudnn64_9.dll)
    Así cualquier búsqueda de cudnn64_8.dll encuentra la versión correcta (>=8.4)
    en vez de una vieja instalación del sistema que no tiene cudnnGetLibConfig.
    """
    if sys.platform != "win32":
        return

    _added = []

    def _try_add(d):
        if d and os.path.isdir(d) and d not in _added:
            try:
                os.add_dll_directory(d)
                _added.append(d)
                return True
            except Exception:
                pass
        return False

    # ── 1. torch/lib — PRIMERO (tiene cudnn 8.9.x bundleado, versión correcta) ──
    try:
        import torch as _torch
        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if _try_add(_torch_lib):
            _cudnn_in_torch = _glob.glob(os.path.join(_torch_lib, "cudnn*.dll"))
            if _cudnn_in_torch:
                print(f"[cuDNN] torch/lib: {[os.path.basename(d) for d in _cudnn_in_torch]}")
            else:
                print(f"[cuDNN] torch/lib añadido (sin cudnn DLLs propios)")
    except Exception as _e:
        print(f"[cuDNN] torch/lib no disponible: {_e}")

    # ── 2. nvidia-cudnn-cu12 (cudnn64_9.dll) ─────────────────────────────────────
    try:
        import nvidia.cudnn as _pkg
        _file = getattr(_pkg, "__file__", None)
        _paths = list(getattr(_pkg, "__path__", []) or [])
        _base = os.path.dirname(_file) if _file else (_paths[0] if _paths else None)
        if _base:
            for _dll in _glob.glob(os.path.join(_base, "**", "cudnn64_*.dll"), recursive=True):
                _d = os.path.dirname(_dll)
                if _try_add(_d):
                    print(f"[cuDNN] nvidia-cudnn: {_d} ({os.path.basename(_dll)})")
            for _sub in ["", "bin", "lib"]:
                _d = os.path.join(_base, _sub) if _sub else _base
                if _try_add(_d):
                    print(f"[cuDNN] nvidia-cudnn dir: {_d}")
    except ImportError:
        print("[cuDNN] nvidia-cudnn-cu12 no instalado")
    except Exception as _e:
        print(f"[cuDNN] nvidia.cudnn: {_e}")

    print(f"[cuDNN] {len(_added)} directorios registrados en DLL search path")

_setup_cudnn_dlls()

# ─── Imports estándar ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import hashlib
import json
import shutil
import subprocess as _subprocess
import tempfile
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr

# ─── Configuración (sobrescribible via env vars) ──────────────────────────────
BASE_DIR        = Path(os.environ.get("APP_DIR", str(Path(__file__).parent)))
REF_AUDIO       = BASE_DIR / "voice_reference" / "audio_reference_natural.wav"
REF_TEXT_FILE   = BASE_DIR / "voice_reference" / "audio_reference_natural.txt"
CACHE_DIR       = BASE_DIR / "cache"
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL", "base")
TTS_LOCAL_MODEL = os.environ.get("TTS_LOCAL_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
TTS_BATCH_SIZE  = int(os.environ.get("TTS_BATCH_SIZE", "1"))

# ─── Importar funciones del pipeline ─────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from pipeline import (
    extract_audio,
    translate_segments,
    chunk_by_natural_pause,
    generate_tts_local,
    build_final_audio,
    merge_video_with_audio,
    get_duration,
    run_cmd,
    FREE_FLOWING,
    _log,
)
from subtitle_pipeline import generate_subtitled_video
from shorts_pipeline import (
    parse_shorts_config,
    generate_all_shorts,
    list_existing_shorts,
)

# ─── Estado global: cargado una sola vez al arrancar ─────────────────────────
TTS_MODEL_INSTANCE = None
WHISPER_INSTANCE   = None
REF_TEXT           = ""


# ─── Utilidades de caché ──────────────────────────────────────────────────────

def get_file_hash(path: str, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash MD5 del inicio del archivo + tamaño para identificar jobs únicos."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(chunk_bytes))
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()[:16]


def get_cache_status() -> str:
    """Resumen del estado de la caché para mostrar en la UI."""
    if not CACHE_DIR.exists():
        return "Cache vacía"
    jobs = [d for d in CACHE_DIR.iterdir() if d.is_dir()]
    if not jobs:
        return "Cache vacía"
    total_mb = sum(
        f.stat().st_size for j in jobs for f in j.rglob("*") if f.is_file()
    ) / 1e6
    lines = [f"{len(jobs)} job(s) en cache | {total_mb:.1f} MB total"]
    for j in sorted(jobs):
        segs_json  = j / "segments.json"
        final_mp3  = j / "final_es.mp3"
        tts_dir    = j / "tts"
        tts_done   = len(list(tts_dir.glob("seg_*_adj.wav"))) if tts_dir.exists() else 0
        segs_total = 0
        if segs_json.exists():
            try:
                segs_total = len(json.loads(segs_json.read_text(encoding="utf-8")))
            except Exception:
                pass
        state = "COMPLETO" if final_mp3.exists() else (
            f"TTS {tts_done}/{segs_total}" if segs_total else "extrayendo...")
        lines.append(f"  hash={j.name}  [{state}]")
    return "\n".join(lines)


def clear_cache() -> str:
    """Elimina toda la caché de jobs."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        return "Cache eliminada correctamente.\n" + get_cache_status()
    return "No hay cache que eliminar."


def clear_all_caches() -> str:
    """Limpieza TOTAL para nuevo proyecto:
    Borra cache/, subtitle_output/, enhance_output/,
    shorts/_cache/, shorts/shorts_videos_lists/,
    contenido de shorts/short_audio_doblado/, shorts/short_video/,
    shorts/audio_original_en/, archivos de youtube_video/
    y caché PKL + videos de MuseTalk/results/."""
    from pathlib import Path as _P
    removed = []
    freed_mb = 0.0

    def _rmdir(d):
        """Borra el directorio completo."""
        nonlocal freed_mb
        d = _P(d)
        if d.exists():
            freed_mb += sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name + "/")

    def _rmcontents(d, label=None):
        """Borra el contenido de un directorio pero mantiene la carpeta vacía."""
        nonlocal freed_mb
        d = _P(d)
        if not d.exists():
            return
        count = 0
        for item in list(d.iterdir()):
            try:
                if item.is_dir():
                    freed_mb += sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / 1e6
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    freed_mb += item.stat().st_size / 1e6
                    item.unlink()
                count += 1
            except Exception:
                pass
        if count:
            removed.append(f"{label or d.name}/ ({count} item{'s' if count>1 else ''})")

    # 1. Cachés de procesamiento principal
    for _d in [CACHE_DIR, SUBTITLE_OUTPUT_DIR, ENHANCE_OUTPUT_DIR,
               SHORTS_DIR / "_cache", SHORTS_OUTPUT_DIR]:
        _rmdir(_d)

    # 2. Archivos de fuente/salida del proyecto shorts
    _rmcontents(SHORTS_DIR / "short_audio_doblado", "short_audio_doblado")
    _rmcontents(SHORTS_DIR / "short_video",         "short_video")
    _rmcontents(SHORTS_DIR / "audio_original_en",   "audio_original_en")

    # 3. Archivos de video/audio en youtube_video/
    YOUTUBE_VIDEO_DIR.mkdir(exist_ok=True)
    yt_files = [f for f in YOUTUBE_VIDEO_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS]
    for f in yt_files:
        freed_mb += f.stat().st_size / 1e6
        try:
            f.unlink()
        except Exception:
            pass
    if yt_files:
        removed.append(f"youtube_video/ ({len(yt_files)} archivos)")

    # 4. MuseTalk: PKL de landmarks + carpetas de resultados
    if MUSETALK_DIR.exists():
        musetalk_results = MUSETALK_DIR / "results"
        if musetalk_results.exists():
            n_pkl = 0
            n_subdirs = 0
            for pkl in list(musetalk_results.glob("*.pkl")):
                try:
                    freed_mb += pkl.stat().st_size / 1e6
                    pkl.unlink()
                    n_pkl += 1
                except Exception:
                    pass
            for subdir in list(musetalk_results.iterdir()):
                if subdir.is_dir():
                    freed_mb += sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file()) / 1e6
                    shutil.rmtree(subdir, ignore_errors=True)
                    n_subdirs += 1
            if n_pkl or n_subdirs:
                removed.append(f"MuseTalk/results/ ({n_pkl} pkl, {n_subdirs} carpetas)")

    if not removed:
        return "No hay nada que limpiar."
    return (f"Limpiado: {', '.join(removed)}\n"
            f"{freed_mb:.1f} MB liberados — listo para nuevo proyecto.")


def list_youtube_files() -> str:
    """Lista archivos en youtube_video/ con tamaño y tipo."""
    YOUTUBE_VIDEO_DIR.mkdir(exist_ok=True)
    files = [f for f in sorted(YOUTUBE_VIDEO_DIR.iterdir()) if f.is_file()]
    if not files:
        return "No hay archivos en youtube_video/"
    lines = [f"{len(files)} archivo(s) en youtube_video/\n"]
    for f in files:
        size_mb = f.stat().st_size / 1e6
        ext = f.suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            tipo = "VIDEO"
        elif ext in AUDIO_EXTENSIONS:
            tipo = "AUDIO"
        else:
            tipo = "OTRO"
        lines.append(f"  [{tipo}]  {f.name}  ({size_mb:.1f} MB)")
    return "\n".join(lines)


def get_subtitle_output_files() -> list:
    """
    Busca todos los output_subbed.mp4 (y variantes) dentro de subtitle_output/{hash}/.
    Retorna lista de (label_legible, path_absoluto) para gr.Dropdown.
    """
    if not SUBTITLE_OUTPUT_DIR.exists():
        return []
    choices = []
    for job_dir in sorted(SUBTITLE_OUTPUT_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        for mp4 in sorted(job_dir.glob("*.mp4")):
            size_mb = mp4.stat().st_size / 1e6
            label = f"{mp4.name}  [{job_dir.name[:8]}…]  ({size_mb:.0f} MB)"
            choices.append((label, str(mp4)))
    return choices


def get_youtube_file_choices(include_video=True, include_audio=True) -> list:
    """Retorna lista de paths (strings) en youtube_video/ según filtro."""
    YOUTUBE_VIDEO_DIR.mkdir(exist_ok=True)
    result = []
    for f in sorted(YOUTUBE_VIDEO_DIR.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if include_video and ext in VIDEO_EXTENSIONS:
            result.append(str(f))
        elif include_audio and ext in AUDIO_EXTENSIONS:
            result.append(str(f))
    return result


def fetch_youtube_formats(url: str, cookies_file=None):
    """
    Usa yt-dlp -j para obtener JSON estructurado de formatos.
    Retorna (video_rows, audio_rows, status_msg).
    video_rows  → lista de listas para gr.Dataframe VIDEO
    audio_rows  → lista de listas para gr.Dataframe AUDIO
    """
    import subprocess as _sp
    import json as _json

    if not url.strip():
        return [], [], "Ingresa una URL de YouTube."

    cmd = ["yt-dlp", "-j", "--no-playlist", "--quiet", url.strip()]
    if cookies_file:
        cmd.extend(["--cookies", cookies_file if isinstance(cookies_file, str) else cookies_file.name])

    try:
        result = _sp.run(
            cmd,
            capture_output=True, text=True, timeout=45,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout)[-800:]
            return [], [], f"Error yt-dlp (código {result.returncode}):\n{err}"

        data = _json.loads(result.stdout)
        formats = data.get("formats", [])

        video_rows = []
        audio_rows = []

        for f in formats:
            fid    = f.get("format_id", "")
            ext    = f.get("ext", "")
            vcodec = f.get("vcodec") or "none"
            acodec = f.get("acodec") or "none"
            width  = f.get("width")
            height = f.get("height")
            fps    = f.get("fps")
            fsize  = f.get("filesize") or f.get("filesize_approx") or 0
            size_s = f"{fsize/1e6:.0f} MB" if fsize else "-"
            tbr    = f.get("tbr") or 0
            abr    = f.get("abr") or 0
            vbr    = f.get("vbr") or 0
            note   = f.get("format_note") or ""
            lang   = f.get("language") or ""

            has_video = vcodec != "none"
            has_audio = acodec != "none"

            if has_video and not has_audio:
                res    = f"{width}×{height}" if width and height else "?"
                fps_s  = f"{fps:.0f}" if fps else "-"
                br_s   = f"{int(vbr or tbr)}k" if (vbr or tbr) else "-"
                video_rows.append([fid, ext.upper(), res, fps_s, vcodec[:12], br_s, size_s])

            elif has_audio and not has_video:
                br_s  = f"{int(abr or tbr)}k" if (abr or tbr) else "-"
                audio_rows.append([fid, ext.upper(), br_s, acodec[:12], size_s, lang or "-"])

        # Mejor calidad primero: video por altura desc, audio por bitrate desc
        def _h(row):
            try:
                return int(row[2].split("×")[1])
            except Exception:
                return 0

        def _br(row):
            try:
                return int(row[2].replace("k", ""))
            except Exception:
                return 0

        video_rows.sort(key=_h, reverse=True)
        audio_rows.sort(key=_br, reverse=True)

        status = (f"OK — título: {data.get('title','')[:60]}\n"
                  f"{len(video_rows)} formatos de video · {len(audio_rows)} formatos de audio")
        return video_rows, audio_rows, status

    except FileNotFoundError:
        return [], [], "Error: yt-dlp no está instalado."
    except _json.JSONDecodeError as e:
        return [], [], f"Error parseando JSON de yt-dlp: {e}"
    except Exception as e:
        return [], [], f"Error: {e}"


def download_youtube_format(url: str, format_id: str, cookies_file=None):
    """Descarga el formato indicado a youtube_video/. Retorna (msg, video_choices, audio_choices)."""
    import subprocess as _sp
    if not url.strip() or not format_id.strip():
        return "Ingresa URL y format-ID.", [], []
    YOUTUBE_VIDEO_DIR.mkdir(exist_ok=True)
    
    cmd = ["yt-dlp", "-f", format_id.strip(), "--no-playlist", "-o", str(YOUTUBE_VIDEO_DIR / "%(title)s.%(ext)s"), url.strip()]
    if cookies_file:
        cmd.extend(["--cookies", cookies_file if isinstance(cookies_file, str) else cookies_file.name])
        
    try:
        result = _sp.run(
            cmd,
            capture_output=True, text=True, timeout=600,
        )
        out = (result.stdout + result.stderr)[-2000:]
        if result.returncode == 0:
            msg = f"Descargado correctamente.\n\n{out}"
        else:
            msg = f"Error (código {result.returncode}):\n{out}"
    except Exception as e:
        msg = f"Error: {e}"
    choices_v = get_youtube_file_choices(include_video=True, include_audio=False)
    choices_a = get_youtube_file_choices(include_video=False, include_audio=True)
    return msg, choices_v, choices_a


def load_shorts_json() -> str:
    """Lee shorts-settings.json y retorna su contenido pretty-printed."""
    if not SHORTS_CONFIG.exists():
        return '{\n  "video_shorts": []\n}'
    try:
        data = json.loads(SHORTS_CONFIG.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error leyendo config: {e}"


def save_shorts_json(content: str):
    """Valida y guarda el JSON. Retorna (mensaje, nuevas choices para selector)."""
    try:
        data = json.loads(content)
        if "video_shorts" not in data:
            return "Error: falta la clave 'video_shorts'.", get_shorts_choices()
        SHORTS_CONFIG.parent.mkdir(exist_ok=True)
        SHORTS_CONFIG.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n = len(data["video_shorts"])
        return f"Guardado correctamente. {n} short(s) en config.", get_shorts_choices()
    except json.JSONDecodeError as e:
        return f"JSON inválido: {e}", get_shorts_choices()
    except Exception as e:
        return f"Error guardando: {e}", get_shorts_choices()


def get_cache_job_choices() -> list:
    """Lista los job hashes disponibles en cache/."""
    if not CACHE_DIR.exists():
        return []
    return [d.name for d in sorted(CACHE_DIR.iterdir()) if d.is_dir()]


def get_shorts_video_choices() -> list:
    """Video files in youtube_video/ suitable as shorts source."""
    if not YOUTUBE_VIDEO_DIR.exists():
        return []
    return sorted(
        str(f) for f in YOUTUBE_VIDEO_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    )


def get_cache_final_audio_choices() -> list:
    """(label, path) tuples for each cache/{hash}/final_es.mp3 found, newest first."""
    if not CACHE_DIR.exists():
        return []
    choices = []
    for job_dir in sorted(CACHE_DIR.iterdir(), key=lambda d: -d.stat().st_mtime):
        if not job_dir.is_dir():
            continue
        final = job_dir / "final_es.mp3"
        if final.exists():
            size_mb = final.stat().st_size / (1024 * 1024)
            label = f"{job_dir.name[:12]}  {size_mb:.1f} MB"
            choices.append((label, str(final)))
    return choices


def route_shorts_files(
    video_path: str | None,
    audio_path: str | None,
    orig_audio_path: str | None = None,
) -> str:
    """Hard-link (or copy) selected files into the expected SHORTS_DIR subfolders."""
    import shutil as _shutil

    def _link_or_copy(src: Path, dst: Path) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            try:
                os.symlink(src.resolve(), dst)
                return "symlinked"
            except OSError:
                _shutil.copy2(src, dst)
                return "copied"

    def _clear_dir(d: Path):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.is_file() or f.is_symlink():
                f.unlink()

    msgs = []

    if video_path and Path(video_path).exists():
        src = Path(video_path)
        sv_dir = SHORTS_DIR / "short_video"
        _clear_dir(sv_dir)
        mode = _link_or_copy(src, sv_dir / src.name)
        msgs.append(f"OK Video ({mode}) → short_video/{src.name}")

    if audio_path and Path(audio_path).exists():
        src = Path(audio_path)
        sa_dir = SHORTS_DIR / "short_audio_doblado"
        _clear_dir(sa_dir)
        mode = _link_or_copy(src, sa_dir / "final_es.mp3")
        msgs.append(f"OK Audio ES ({mode}) → short_audio_doblado/final_es.mp3")

    if orig_audio_path and Path(orig_audio_path).exists():
        src = Path(orig_audio_path)
        oa_dir = SHORTS_DIR / "audio_original_en"
        _clear_dir(oa_dir)
        mode = _link_or_copy(src, oa_dir / src.name)
        msgs.append(f"OK Audio EN ({mode}) → audio_original_en/{src.name}")

    return "\n".join(msgs) if msgs else ""


def _recalc_dubbed_timestamps(segments: list) -> list:
    """Recalcula timestamps basándose en las duraciones del audio doblado (tts_adjusted_dur).

    Replica la lógica de build_final_audio en modo free-flowing:
      - Pausas naturales entre segmentos capeadas a MAX_FREE_GAP (0.5s)
      - Acumulación secuencial de duraciones reales del TTS

    Retorna lista de dicts con: start_es, end_es, dur_es, text_es, text (original EN).
    """
    MAX_FREE_GAP = 0.50  # debe coincidir con pipeline.py
    result = []
    pos = 0.0

    for i, seg in enumerate(segments):
        tts_dur = seg.get("tts_adjusted_dur")
        if not tts_dur or tts_dur <= 0:
            continue

        # Calcular pausa antes de este segmento (misma lógica que build_final_audio)
        if i == 0:
            leading = min(seg.get("start", 0.0), MAX_FREE_GAP)
        else:
            natural_gap = max(0.0, seg.get("start", 0.0) - segments[i - 1].get("end", 0.0))
            leading = min(natural_gap, MAX_FREE_GAP)

        pos += leading
        start_es = round(pos, 3)
        end_es = round(pos + tts_dur, 3)
        pos = end_es

        result.append({
            "seg_index": i,
            "start_es": start_es,
            "end_es": end_es,
            "dur_es": round(tts_dur, 2),
            "start_en": seg.get("start", 0.0),
            "end_en": seg.get("end", 0.0),
            "text_es": seg.get("text_es", ""),
        })

    return result


def load_segments_json(job_hash: str) -> str:
    """Lee cache/{job_hash}/segments.json y retorna los segmentos con timestamps
    recalculados para el audio doblado en español (final_es.mp3).

    Los campos start_es/end_es reflejan la posición real en el audio doblado,
    que es lo que necesita el JSON de shorts.
    """
    if not job_hash:
        return "Selecciona un job."
    segs_path = CACHE_DIR / job_hash / "segments.json"
    if not segs_path.exists():
        return f"No existe segments.json para el job '{job_hash}'"
    try:
        segments = json.loads(segs_path.read_text(encoding="utf-8"))
        dubbed = _recalc_dubbed_timestamps(segments)
        return json.dumps(dubbed, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: {e}"


# ─── Precarga de modelos ──────────────────────────────────────────────────────

def startup():
    """Precarga Whisper y TTS en GPU antes de abrir el servidor."""
    global TTS_MODEL_INSTANCE, WHISPER_INSTANCE, REF_TEXT

    print("=" * 62)
    print("  EN->ES Dubbing App  |  startup()")
    print("=" * 62)

    REF_TEXT = REF_TEXT_FILE.read_text(encoding="utf-8").strip()
    print(f"  Voz de referencia : {REF_AUDIO.name}")
    print(f"  Texto referencia  : {REF_TEXT[:80]}...")
    print(f"  Cache dir         : {CACHE_DIR}")
    print(f"  Whisper model     : {WHISPER_MODEL}")
    print(f"  TTS model         : {TTS_LOCAL_MODEL}")
    print(f"  TTS batch size    : {TTS_BATCH_SIZE}")

    # ── Cargar Whisper ────────────────────────────────────────────
    print(f"\n  [startup] Cargando Whisper '{WHISPER_MODEL}' en GPU (float16)...")
    from faster_whisper import WhisperModel
    WHISPER_INSTANCE = WhisperModel(
        WHISPER_MODEL, device="cuda", compute_type="float16"
    )
    print("  [startup] OK Whisper listo")

    # ── Cargar Qwen3-TTS ─────────────────────────────────────────
    import torch
    from qwen_tts import Qwen3TTSModel
    print(f"\n  [startup] Cargando {TTS_LOCAL_MODEL} en GPU (bfloat16)...")
    print("  [startup] (primera vez: hasta 60s si descarga desde cache HF)")

    # Precargar explícitamente cudnn DLLs desde torch/lib ANTES de cargar TTS
    # Esto garantiza que Windows cachea la versión correcta antes de que
    # speechbrain/qwen-tts la busque y encuentre una versión vieja del sistema.
    import ctypes
    _preloaded = []
    # Intentar desde torch/lib primero (tiene versión correcta 8.9.x)
    try:
        import torch as _t
        _tlib = os.path.join(os.path.dirname(_t.__file__), "lib")
        for _name in ["cudnn64_8.dll", "cudnn64_9.dll", "cudnn_ops_infer64_8.dll",
                       "cudnn_cnn_infer64_8.dll", "cudnn_adv_infer64_8.dll"]:
            _p = os.path.join(_tlib, _name)
            if os.path.exists(_p):
                try:
                    ctypes.CDLL(_p)
                    _preloaded.append(_name)
                except OSError:
                    pass
    except Exception:
        pass
    # Fallback: cargar por nombre (usa os.add_dll_directory registrado arriba)
    for _name in ["cudnn64_8.dll", "cudnn64_9.dll"]:
        if _name not in _preloaded:
            try:
                ctypes.CDLL(_name)
                _preloaded.append(_name)
            except OSError:
                pass
    if _preloaded:
        print(f"  [startup] cuDNN pre-cargadas: {_preloaded}")
    else:
        print("  [startup] WARN: No se pre-cargaron cudnn DLLs")

    TTS_MODEL_INSTANCE = Qwen3TTSModel.from_pretrained(
        TTS_LOCAL_MODEL,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    # torch.compile reduce el overhead Python entre forward passes autoregresivos
    # (el GPU estaba al 0% por CPU bottleneck; reduce-overhead cachea kernels CUDA)
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        TTS_MODEL_INSTANCE = torch.compile(TTS_MODEL_INSTANCE, mode="reduce-overhead")
        print("  [startup] torch.compile OK (reduce-overhead)")
    except Exception as _e:
        print(f"  [startup] WARN torch.compile no disponible: {_e}")
    print("  [startup] OK TTS listo\n")
    print(f"  {get_cache_status()}")
    print("  Servidor iniciando en 0.0.0.0:7860 ...")
    print("=" * 62)


# ─── Pipeline principal con checkpoints ──────────────────────────────────────

def _compute_segs_from_json(segments_json) -> str | None:
    """Lee segments_json y retorna el JSON de timestamps del audio doblado, o None."""
    try:
        raw_segs = json.loads(Path(segments_json).read_text(encoding="utf-8"))
        dubbed = _recalc_dubbed_timestamps(raw_segs)
        if not dubbed:
            return None
        return json.dumps(dubbed, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  WARN _compute_segs_from_json: {e}")
        return None


def run_dubbing(audio_file, progress=gr.Progress(track_tqdm=False)):
    """
    Pipeline completo con sistema de caché/checkpoint.
    Cada job se guarda en cache/<hash>/ y se puede reanudar si falla.
    """
    if audio_file is None:
        yield None, "Por favor sube un archivo de audio / Please upload an audio file.", None
        return

    # ── Calcular hash del archivo e inicializar job dir ───────────
    print(f"\n{'='*55}")
    file_hash = get_file_hash(audio_file)
    job_dir   = CACHE_DIR / file_hash
    job_dir.mkdir(parents=True, exist_ok=True)

    tts_dir        = job_dir / "tts"
    segments_json  = job_dir / "segments.json"
    final_mp3_path = job_dir / "final_es.mp3"
    final_wav_path = job_dir / "final_es.wav"
    wav_path       = job_dir / "input.wav"

    print(f"  JOB INICIO")
    print(f"  Input     : {audio_file}")
    print(f"  Hash      : {file_hash}")
    print(f"  Job dir   : {job_dir}")
    print(f"  WAV existe: {wav_path.exists()}")
    print(f"  Segs JSON : {segments_json.exists()}")
    print(f"  Final MP3 : {final_mp3_path.exists()}")

    try:
        # ── CACHE HIT TOTAL: resultado final ya existe ────────────
        if final_mp3_path.exists():
            total_dur = get_duration(str(wav_path)) if wav_path.exists() else 0
            m, s = divmod(int(total_dur), 60)
            print(f"  CACHE HIT COMPLETO: devolviendo resultado previo")
            segs_content = _compute_segs_from_json(segments_json)
            yield str(final_mp3_path), (
                f"[CACHE] Job completado anteriormente — {m}m {s}s | hash={file_hash}"
            ), segs_content
            return

        # ── PASO 1: Extraer audio ─────────────────────────────────
        if wav_path.exists():
            print(f"\n  [1/5] WAV ya en cache ({wav_path.stat().st_size/1e6:.1f} MB), saltando...")
            progress(0.05, desc="[1/5] WAV en cache, saltando extracción...")
        else:
            progress(0.05, desc="[1/5] Extrayendo audio WAV...")
            print(f"\n  [1/5] Extrayendo audio WAV 16kHz mono...")
            extract_audio(audio_file, str(job_dir))
            print(f"  -> WAV guardado: {wav_path} ({wav_path.stat().st_size/1e6:.1f} MB)")

        total_dur = get_duration(str(wav_path))
        m_dur, s_dur = divmod(int(total_dur), 60)
        print(f"  -> Duración total: {total_dur:.1f}s ({m_dur}m {s_dur}s)")

        # ── PASO 2: Transcribir ───────────────────────────────────
        if segments_json.exists():
            print(f"\n  [2-3/5] CHECKPOINT: cargando segmentos de {segments_json.name}...")
            progress(0.20, desc="[2-3/5] Cargando segmentos del checkpoint...")
            with open(segments_json, encoding="utf-8") as f:
                segments = json.load(f)
            print(f"  -> {len(segments)} bloques cargados")
        else:
            # Transcribir con Whisper
            progress(0.15, desc="[2/5] Transcribiendo con Whisper...")
            print(f"\n  [2/5] Transcribiendo con Whisper '{WHISPER_MODEL}'...")
            print(f"  -> Iniciando transcripción sobre {wav_path.name}...")
            sys.stdout.flush()

            diarization_file = job_dir / "diarization.json"
            diarization_data = []
            if diarization_file.exists():
                try:
                    with open(diarization_file, "r", encoding="utf-8") as f:
                        diarization_data = json.load(f).get("results", [])
                except Exception:
                    pass

            segments_iter, info = WHISPER_INSTANCE.transcribe(
                str(wav_path),
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            segments = []
            for s in segments_iter:
                if not s.text.strip():
                    continue
                    
                start = round(s.start, 3)
                end = round(s.end, 3)
                
                speaker = "SPEAKER_UNKNOWN"
                if diarization_data:
                    max_overlap = 0.0
                    for d in diarization_data:
                        overlap = max(0, min(end, d["end"]) - max(start, d["start"]))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            speaker = d["speaker"]
                            
                seg = {"start": start, "end": end, "text": s.text.strip(), "speaker": speaker}
                segments.append(seg)
                _log(f"  Whisper [{start:6.1f}s→{end:5.1f}s] [{speaker}] {s.text.strip()[:60]}")
            if not segments:
                yield None, "No se detectó habla en el audio / No speech detected.", None
                return

            print(f"  -> {len(segments)} segmentos detectados")
            print(f"  -> Idioma: {info.language} ({info.language_probability:.0%})")
            print(f"  -> Primer seg : [{segments[0]['start']:.1f}s - {segments[0]['end']:.1f}s]"
                  f" '{segments[0]['text'][:70]}'")
            print(f"  -> Ultimo seg : [{segments[-1]['start']:.1f}s - {segments[-1]['end']:.1f}s]"
                  f" '{segments[-1]['text'][:70]}'")

        # ── PASO 3: Chunk primero, luego traducir (orden correcto) ──────
        # Si el checkpoint ya tiene chunks con traducción lista, saltar todo.
        tts_dir.mkdir(exist_ok=True)
        already_done = segments and all(
            s.get("_chunked") and s.get("text_es") for s in segments
        )

        if already_done:
            progress(0.35, desc="[3/5] Chunks y traducción cacheados...")
            print(f"\n  [3/5] Chunks y traducción cacheados ({len(segments)} chunks)")
        else:
            # 3a. Chunk: agrupa segmentos en párrafos con texto EN bien puntuado.
            # Debe ir ANTES de la traducción para que el traductor reciba párrafos
            # completos en lugar de frases fragmentadas (mejora calidad de traducción).
            old_count = len(segments)
            segments  = chunk_by_natural_pause(segments)
            new_count = len(segments)
            print(f"\n  [3/5] Chunks: {old_count} segs → {new_count} chunks "
                  f"(~{old_count // max(new_count, 1)} segs/chunk)")
            # Borrar TTS obsoleto si cambió el número de chunks
            for stale in list(tts_dir.glob("seg_*_adj.wav")) + list(tts_dir.glob("seg_*_raw.wav")):
                stale.unlink(missing_ok=True)

            # 3b. Traducir cada chunk como unidad completa → traducción contextual.
            n_pending = sum(1 for s in segments if not s.get("text_es"))
            if n_pending:
                progress(0.30, desc=f"[3/5] Traduciendo {n_pending} chunks EN->ES...")
                print(f"  -> Traduciendo {n_pending} chunks (párrafo completo por chunk)...")
                segments = translate_segments(segments)

            with open(segments_json, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            print(f"  -> CHECKPOINT GUARDADO: {segments_json}")

        # ── PASO 4: TTS ───────────────────────────────────────────
        tts_dir.mkdir(exist_ok=True)

        # pipeline.py ya revisa si seg_XXXX_adj.wav existe y lo salta
        done_tts  = sum(1 for i in range(len(segments))
                        if (tts_dir / f"seg_{i:04d}_adj.wav").exists())
        total_seg = len(segments)
        pend_tts  = total_seg - done_tts

        print(f"\n  [4/5] TTS estado: {done_tts}/{total_seg} segmentos ya generados en cache")

        if pend_tts > 0:
            progress(0.45, desc=f"[4/5] TTS: {done_tts} en cache, generando {pend_tts}...")
            print(f"  -> Pendientes : {pend_tts} segmentos nuevos a generar")
            print(f"  -> ref_audio  : {REF_AUDIO.name}")
            print(f"  -> ref_text   : {REF_TEXT[:60]}...")
            print(f"  -> batch_size : {TTS_BATCH_SIZE}")
            print(f"  -> tts_dir    : {tts_dir}")
            print(f"  -> Llamando model.create_voice_clone_prompt() ...")
            sys.stdout.flush()

            def _tts_progress(done: int, total: int) -> None:
                frac = 0.45 + 0.40 * (done / max(total, 1))
                progress(frac, desc=f"[4/5] TTS: {done}/{total} chunks generados...")

            # Load speaker references from voice_reference folder if available
            speaker_refs = {}
            ref_dir = BASE_DIR / "voice_reference"
            if ref_dir.exists():
                for spk_wav in ref_dir.glob("*.wav"):
                    spk_name = spk_wav.stem
                    spk_txt = ref_dir / f"{spk_name}.txt"
                    if spk_txt.exists():
                        speaker_refs[spk_name] = {
                            "audio": str(spk_wav),
                            "text": spk_txt.read_text(encoding="utf-8").strip()
                        }

            segments = generate_tts_local(
                model=TTS_MODEL_INSTANCE,
                segments=segments,
                ref_audio_path=str(REF_AUDIO),
                ref_text=REF_TEXT,
                tts_dir=str(tts_dir),
                language="Spanish",
                batch_size=TTS_BATCH_SIZE,
                progress_cb=_tts_progress,
                checkpoint_path=str(segments_json),
                speaker_references=speaker_refs,
            )

            # Actualizar checkpoint con tts_adjusted_path
            with open(segments_json, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

            done_after = sum(1 for i in range(len(segments))
                             if (tts_dir / f"seg_{i:04d}_adj.wav").exists())
            print(f"\n  -> TTS completados: {done_after}/{total_seg} segmentos")

        else:
            print(f"  -> CACHE HIT TTS completo: todos los archivos adj.wav existen")
            progress(0.88, desc="[4/5] TTS completo (cache)...")
            for i, seg in enumerate(segments):
                adj = tts_dir / f"seg_{i:04d}_adj.wav"
                if adj.exists():
                    seg["tts_adjusted_path"] = str(adj)
                    if not seg.get("tts_adjusted_dur"):
                        seg["tts_adjusted_dur"] = get_duration(str(adj))

        segs_content = _compute_segs_from_json(segments_json)
        if segs_content:
            yield None, "TTS completado — ensamblando audio...", segs_content

        # ── PASO 5: Ensamblar y exportar MP3 ─────────────────────
        total_pasos = "6" if FREE_FLOWING else "5"
        progress(0.88, desc=f"[5/{total_pasos}] Ensamblando y exportando MP3...")
        print(f"\n  [5/{total_pasos}] Ensamblando audio final...")
        print(f"  -> WAV de salida: {final_wav_path}")
        build_final_audio(segments, total_dur, str(final_wav_path))

        print(f"  -> Convirtiendo a MP3: {final_mp3_path}")
        run_cmd(["ffmpeg", "-y", "-i", str(final_wav_path),
                 "-q:a", "2", "-ar", "44100", str(final_mp3_path)])

        # ── PASO 6: Fusionar video + audio doblado (FREE_FLOWING) ──
        final_dubbed_mp4_path = None
        if FREE_FLOWING:
            input_ext = Path(audio_file).suffix.lower()
            if input_ext in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
                progress(0.95, desc=f"[6/{total_pasos}] Fusionando video + audio...")
                print(f"\n  [6/{total_pasos}] Fusionando video + audio doblado...")
                final_dubbed_mp4_path = job_dir / "final_dubbed.mp4"
                if not final_dubbed_mp4_path.exists():
                    merge_video_with_audio(
                        audio_file, str(final_wav_path), str(final_dubbed_mp4_path)
                    )
                print(f"  -> MP4 doblado: {final_dubbed_mp4_path}")

        progress(1.0, desc="Listo / Done!")

        status = (
            f"Doblaje completado — {m_dur}m {s_dur}s | "
            f"{total_seg} segmentos | cache: {file_hash}"
        )
        if final_dubbed_mp4_path:
            status += f" | MP4: {final_dubbed_mp4_path}"
        print(f"\n  DONE: {status}")
        print(f"{'='*55}\n")
        segs_content = _compute_segs_from_json(segments_json)
        yield str(final_mp3_path), status, segs_content

    except Exception:
        tb = traceback.format_exc()
        print(f"\n  [ERROR CRITICO]\n{tb}")
        print(f"  Progreso guardado en: {job_dir}")
        print(f"  Al reiniciar, el job reanuda desde el ultimo checkpoint.")
        yield None, f"Error (progreso guardado en cache/{file_hash}):\n{tb[-1500:]}", None


def run_dubbing_test(audio_file, progress=gr.Progress(track_tqdm=False)):
    """Dobla solo los primeros 5 minutos del audio para prueba rápida."""
    if audio_file is None:
        yield None, "Por favor sube un archivo de audio.", None
        return
    import tempfile
    import subprocess as _sp
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        _sp.run(
            ["ffmpeg", "-y", "-t", "300", "-i", audio_file,
             "-c:a", "libmp3lame", "-q:a", "2", tmp.name],
            check=True, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
        )
        
        # Copiar diarization si existe en la cache original
        orig_hash = get_file_hash(audio_file)
        orig_diarization = CACHE_DIR / orig_hash / "diarization.json"
        
        test_hash = get_file_hash(tmp.name)
        test_dir = CACHE_DIR / test_hash
        test_dir.mkdir(parents=True, exist_ok=True)
        
        if orig_diarization.exists():
            import shutil
            shutil.copy(orig_diarization, test_dir / "diarization.json")
            
        yield from run_dubbing(tmp.name, progress=progress)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ─── Interfaz Gradio ──────────────────────────────────────────────────────────

SUBTITLE_OUTPUT_DIR = BASE_DIR / "subtitle_output"
ENHANCE_OUTPUT_DIR  = BASE_DIR / "enhance_output"
SHORTS_DIR          = BASE_DIR / "shorts"
SHORTS_OUTPUT_DIR   = SHORTS_DIR / "shorts_videos_lists"
SHORTS_CONFIG       = SHORTS_DIR / "shorts-settings.json"
YOUTUBE_VIDEO_DIR   = BASE_DIR / "youtube_video"
ARCHIVES_DIR        = BASE_DIR / "archives"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}


def _extract_file_path(val) -> str | None:
    """Normalize file values returned by gr.File / gr.Audio / gr.Video to a plain path string.

    Gradio 6.x may return:
      - None             → no file
      - str              → already a path
      - dict             → {"name": "/tmp/...", ...}  (older Gradio)
      - FileData object  → .path or .name attribute    (newer Gradio)
    """
    if val is None:
        return None
    if isinstance(val, str):
        return val if val else None
    if isinstance(val, dict):
        return val.get("path") or val.get("name") or None
    # FileData / NamedString / similar objects
    for attr in ("path", "name"):
        v = getattr(val, attr, None)
        if v and isinstance(v, str):
            return v
    return None

RUNPOD_GUIDE_MD = """
## Guia RunPod — Turn ON / Turn OFF del servidor

### Configuracion inicial (una sola vez)

**1. Construir y publicar la imagen Docker**
```bash
docker build -t tuusuario/dubbing-app:latest .
docker push tuusuario/dubbing-app:latest
```

**2. Crear Network Volume en RunPod** (Storage → Create Volume)
- Nombre: `dubbing-volume` · Tamaño: **20 GB** · misma región que tu pod

**3. Crear Pod Template** (Templates → New Template)
- Container image: `tuusuario/dubbing-app:latest`
- Volume mount: `/runpod-volume` → tu Network Volume
- Expose HTTP port: **7860**
- Env vars: `WHISPER_MODEL=base` · `TTS_BATCH_SIZE=8`

---

### Turn ON — Encender el servidor

1. **Pods → tu pod → Start**
2. Espera ~2 min (primera vez descarga modelos ~5 GB al Network Volume)
3. **Connect → HTTP Service [Port 7860]**

---

### Turn OFF — Apagar para ahorrar dinero

1. **Pods → tu pod → Stop** → costo **$0/hora** mientras está apagado
2. Tus archivos en `/runpod-volume` **se conservan** siempre
3. El disco del pod (cache, youtube_video, etc.) también persiste al hacer Stop

> Si haces **Terminate** (borrar pod), pierdes el disco local pero NO el Network Volume.

---

### Persistencia de archivos

| Directorio | Dónde vive | Persiste al Stop | Persiste al Terminate |
|---|---|:---:|:---:|
| `/runpod-volume/huggingface/` | Network Volume | Si | Si |
| `/app/cache/` | Disco del pod | Si | No |
| `/app/youtube_video/` | Disco del pod | Si | No |
| `/app/shorts/` | Disco del pod | Si | No |
| `/app/archives/` | Disco del pod | Si | No |

**Tip:** Usa el boton **"Archivar Proyecto"** antes de apagar para guardar tus archivos en un ZIP.

---

### Tips de costo

- **GPU recomendada:** RTX 3090 (24 GB VRAM) ~$0.69/hr Community Cloud
- **Apaga siempre** que no uses el servidor (1 dia olvidado aprox. $16+)
- **Spot/Interruptible:** reduce el precio ~40% (el pod puede reiniciarse)
- **Network Volume:** ~$0.07/GB/mes (20 GB = $1.40/mes, siempre disponible)
"""

APP_CSS = """
/* ── Premium Minimalist B&W Theme ───────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    /* Pure Light Mode */
    --nav-bg: #ffffff;
    --nav-border: #e5e5e5;
    --nav-label-color: #888888;
    --nav-text: #111111;
    --nav-hover-bg: #f5f5f5;
    --nav-hover-text: #000000;
    --nav-active-bg: #000000;
    --nav-active-text: #ffffff;
    --ft-divider: #f0f0f0;
    --ft-file-color: #333333;
    --ft-hover-bg: #f5f5f5;
    --ft-hover-text: #000000;
    --ft-name-color: #000000;
    --ft-meta-color: #888888;
}

.dark {
    /* Pure Dark Mode (Deep Black) */
    --nav-bg: #000000;
    --nav-border: #222222;
    --nav-label-color: #666666;
    --nav-text: #aaaaaa;
    --nav-hover-bg: #111111;
    --nav-hover-text: #ffffff;
    --nav-active-bg: #ffffff;
    --nav-active-text: #000000;
    --ft-divider: #111111;
    --ft-file-color: #aaaaaa;
    --ft-hover-bg: #111111;
    --ft-hover-text: #ffffff;
    --ft-name-color: #ffffff;
    --ft-meta-color: #666666;
}

body, .gradio-container, * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }

/* ── Hide native Gradio tab bar */
#main-tabs > .tab-nav { display: none !important; }

/* ── App Grid & Shadows */
#app-layout { gap: 0 !important; align-items: stretch !important; min-height: 100vh; }
#left-nav {
    min-width: 260px !important;
    max-width: 260px !important;
    border-right: 1px solid var(--nav-border) !important;
    padding: 30px 20px !important;
    background: var(--nav-bg) !important;
    border-radius: 0 !important;
    min-height: 100vh;
}
#right-spacer { display: none !important; }

/* ── Nav labels & Buttons ── */
.nav-label p {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--nav-label-color) !important;
    margin: 24px 0 8px !important;
    padding: 0 8px !important;
}
.nav-btn button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 10px 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    border: none !important;
    background: transparent !important;
    color: var(--nav-text) !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    margin-bottom: 2px !important;
}
.nav-btn button:hover {
    background: var(--nav-hover-bg) !important;
    color: var(--nav-hover-text) !important;
    transform: translateX(4px);
}
.nav-btn-active button {
    background: var(--nav-active-bg) !important;
    color: var(--nav-active-text) !important;
    font-weight: 600 !important;
}

/* ── File Tree ── */
.file-tree {
    font-family: 'Inter', monospace;
    font-size: 11.5px;
    line-height: 1.6;
    overflow-y: auto;
    max-height: 350px;
}
.ft-section {
    font-weight: 600;
    color: var(--nav-label-color);
    text-transform: uppercase;
    font-size: 9.5px;
    letter-spacing: 0.1em;
    margin-top: 14px;
    padding: 4px 8px 2px;
    border-bottom: 1px solid var(--ft-divider);
}
.ft-file {
    padding: 4px 8px;
    color: var(--ft-file-color);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
    margin-top: 2px;
}
.ft-file:hover { background: var(--ft-hover-bg); color: var(--ft-hover-text); }
.ft-name { color: var(--ft-name-color); }
.ft-size { color: var(--ft-meta-color); font-size: 10px; float: right; }
.ft-count { font-weight: normal; font-size: 9px; opacity: 0.7; }
.ft-empty, .ft-more { color: var(--ft-meta-color); font-style: italic; padding: 4px 8px; }

/* ── Main Layout & Padding ── */
#main-content { padding: 44px 5% !important; display: flex; flex-direction: column; gap: 20px;}

/* Override thick borders on boxes to be elegant */
.box, .block {
    border: 1px solid var(--nav-border) !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.02) !important;
}
.dark .box, .dark .block {
    box-shadow: 0 4px 16px rgba(0,0,0,0.4) !important;
    background: #0a0a0a !important;
}
input, textarea, select {
    border-radius: 6px !important;
    border: 1px solid var(--nav-border) !important;
    transition: all 0.2s ease;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--nav-text) !important;
    box-shadow: 0 0 0 1px var(--nav-text) !important;
}
button { transition: all 0.2s ease !important; }
button:active { transform: scale(0.98); }
button.primary { letter-spacing: 0.05em; text-transform: uppercase; font-size: 12px !important; }

/* ── Strict B&W overrides: kill ALL color from Gradio theme ── */

/* Primary buttons → solid black / white text */
button.primary, .primary button, .gr-button-primary button {
    background: #000000 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}
button.primary:hover, .primary button:hover {
    background: #222222 !important;
}

/* Secondary buttons → white / black border */
button.secondary, .secondary button, .gr-button-secondary button {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1.5px solid #000000 !important;
    box-shadow: none !important;
}
button.secondary:hover, .secondary button:hover {
    background: #f5f5f5 !important;
}

/* Stop/danger buttons (Gradio default = rojo) → neutral */
button.stop, .stop button, .gr-button-stop button {
    background: #ffffff !important;
    color: #333333 !important;
    border: 1.5px solid #cccccc !important;
    box-shadow: none !important;
}
button.stop:hover, .stop button:hover {
    background: #f0f0f0 !important;
    border-color: #666666 !important;
    color: #000000 !important;
}

/* Tabs: selected → black underline/text */
.tab-nav button.selected {
    color: #000000 !important;
    border-color: #000000 !important;
    font-weight: 600 !important;
}
.tab-nav button { color: #666666 !important; }
.tab-nav button:hover { color: #000000 !important; }

/* Links → black */
a, a:visited, a:active { color: #000000 !important; }
a:hover { color: #444444 !important; text-decoration: underline; }

/* Checkboxes, radios, sliders → black accent */
input[type="checkbox"], input[type="radio"], input[type="range"] {
    accent-color: #000000 !important;
}

/* Progress bar → black */
.progress-bar-fill, [class*="progress"][class*="fill"] {
    background: #000000 !important;
}

/* Focus ring → black instead of colored */
*:focus-visible {
    outline: 2px solid #000000 !important;
    outline-offset: 2px !important;
}
.block:focus-within {
    box-shadow: 0 0 0 1px #000000 !important;
}

/* Remove any orange/colored loader or spinner */
[class*="loader"], [class*="spinner"] {
    border-top-color: #000000 !important;
    border-left-color: #000000 !important;
}
"""


# ─── Subtitle backend functions ───────────────────────────────────────────────

def _run_subtitles(
    video_file, audio_file, test_seconds,
    contrast=0, brightness=0, sharpness=0,
    do_orig_audio=False, orig_audio_path=None, orig_audio_db=-25,
    progress=gr.Progress(track_tqdm=False)
):
    if video_file is None:
        return None, "Por favor sube un archivo de video."
    if audio_file is None:
        return None, "Por favor sube el archivo de audio doblado."

    # video_file from gr.File comes as a temp path string
    video_path = video_file if isinstance(video_file, str) else video_file.name
    audio_path = audio_file  # gr.Audio type="filepath" gives a string
    orig_path = (orig_audio_path if isinstance(orig_audio_path, str) else orig_audio_path.name) if orig_audio_path is not None else None

    import hashlib
    h = hashlib.md5()
    with open(audio_path, "rb") as f:
        h.update(f.read(8 * 1024 * 1024))
    h.update(str(os.path.getsize(audio_path)).encode())
    with open(video_path, "rb") as f:
        h.update(f.read(8 * 1024 * 1024))
    h.update(str(os.path.getsize(video_path)).encode())
    if orig_path:
        with open(orig_path, "rb") as f:
            h.update(f.read(1024 * 1024))
    job_hash = h.hexdigest()[:16]

    job_dir = SUBTITLE_OUTPUT_DIR / job_hash
    job_dir.mkdir(parents=True, exist_ok=True)

    mode_label = f"test_{test_seconds}s" if test_seconds else "full"
    print(f"\n{'='*55}")
    print(f"  SUBTITULOS [{mode_label.upper()}]")
    print(f"  Video   : {video_path}")
    print(f"  Audio   : {audio_path}")
    print(f"  Orig bg : {orig_path if do_orig_audio and orig_path else 'NO'}")
    print(f"  JobDir  : {job_dir}")
    print(f"{'='*55}")

    try:
        status_msgs = []

        def _progress_cb(msg, pct=None):
            status_msgs.append(msg)
            frac = pct if pct is not None else 0.15
            progress(frac, desc=msg)

        progress(0.05, desc="Preparando...")
        current_video_path = video_path

        out_video, status = generate_subtitled_video(
            video_path=current_video_path,
            audio_path=audio_path,
            whisper_model=WHISPER_INSTANCE,
            out_dir=str(job_dir),
            test_seconds=test_seconds,
            progress_cb=_progress_cb,
            contrast=contrast,
            brightness=brightness,
            sharpness=sharpness,
            do_orig_audio=do_orig_audio,
            orig_audio_path=orig_path,
            orig_audio_db=orig_audio_db,
        )

        progress(1.0, desc="Listo!")
        
        steps_done = ["subtítulos"]
        if do_orig_audio and orig_path:
            steps_done.append(f"audio fondo {orig_audio_db}dB")
            
        final_status = (
            f"{status}\n"
            f"Pasos extra: {' + '.join(steps_done)}\n"
            f"Filtros - C:{contrast} B:{brightness} S:{sharpness}"
        )
        return out_video, final_status

    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(f"\n  [ERROR SUBTITULOS/MEJORA]\n{tb}")
        return None, f"Error:\n{tb[-1500:]}"


def run_subtitles_full(
    video_file, audio_file, contrast, brightness, sharpness,
    do_orig_audio, orig_audio_file, orig_audio_db,
    progress=gr.Progress(track_tqdm=False)
):
    return _run_subtitles(
        video_file, audio_file, None, contrast, brightness, sharpness,
        do_orig_audio, orig_audio_file, orig_audio_db, progress=progress
    )


def run_subtitles_test(
    video_file, audio_file, contrast, brightness, sharpness,
    do_orig_audio, orig_audio_file, orig_audio_db,
    progress=gr.Progress(track_tqdm=False)
):
    return _run_subtitles(
        video_file, audio_file, 20, contrast, brightness, sharpness,
        do_orig_audio, orig_audio_file, orig_audio_db, progress=progress
    )


# ─── Shorts backend ───────────────────────────────────────────────────────────

def get_shorts_choices() -> list:
    """Lee la config y retorna opciones para el CheckboxGroup."""
    if not SHORTS_CONFIG.exists():
        return []
    try:
        shorts = parse_shorts_config(str(SHORTS_CONFIG))
        return [
            f"{s['id']:02d} — {s['titulo']} ({s['duracion_segundos']}s)"
            for s in shorts
        ]
    except Exception:
        return []


def refresh_shorts_list() -> str:
    """Devuelve texto con los shorts ya generados en shorts_videos_lists/."""
    files = list_existing_shorts(str(SHORTS_OUTPUT_DIR))
    if not files:
        return "No hay shorts generados aún."
    lines = [f"{len(files)} short(s) en {SHORTS_OUTPUT_DIR.name}/"]
    for p in files:
        size_mb = Path(p).stat().st_size / 1e6
        lines.append(f"  {Path(p).name}  ({size_mb:.1f} MB)")
    return "\n".join(lines)


def run_generate_shorts(
    selected_choices,
    contrast,
    brightness,
    sharpness,
    use_orig_audio,
    orig_audio_db,
    crop_offset_pct=0,
    resolution="1080p",
    video_src=None,
    audio_src=None,
    orig_audio_src=None,
    video_upload=None,
    audio_upload=None,
    orig_upload=None,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Genera los shorts seleccionados aplicando los filtros de imagen/audio
    directamente durante la creación (un solo video de salida).
    selected_choices: lista de strings "ID — Titulo (Xs)"
    Uploaded files (video_upload/audio_upload/orig_upload) take priority over
    dropdown server paths (video_src/audio_src/orig_audio_src).
    """
    if not selected_choices:
        return "Selecciona al menos un short.", refresh_shorts_list(), gr.update()

    # ── Resolve: uploaded file takes priority over server dropdown ──
    video_src      = _extract_file_path(video_upload) or video_src
    audio_src      = _extract_file_path(audio_upload) or audio_src
    orig_audio_src = _extract_file_path(orig_upload)  or orig_audio_src

    # ── Route source files into expected subfolders ───────────────
    _oa = orig_audio_src if use_orig_audio else None
    if video_src or audio_src or _oa:
        route_msg = route_shorts_files(video_src, audio_src, _oa)
        if route_msg:
            print(f"  [shorts routing]\n{route_msg}")

    selected_ids = []
    for choice in selected_choices:
        try:
            selected_ids.append(int(choice.split("—")[0].strip()))
        except (ValueError, IndexError):
            pass

    if not selected_ids:
        return "No se pudieron parsear los IDs.", refresh_shorts_list(), gr.update()

    status_lines = [f"Generando {len(selected_ids)} short(s)...\n"]

    def _progress_cb(i, total, titulo):
        msg = f"[{i+1}/{total}] {titulo}"
        status_lines.append(msg)
        progress((i + 1) / max(total, 1), desc=msg)
        print(f"  [shorts UI] {msg}")

    try:
        results = generate_all_shorts(
            shorts_dir=str(SHORTS_DIR),
            whisper_model=WHISPER_INSTANCE,
            selected_ids=selected_ids,
            progress_cb=_progress_cb,
            contrast=contrast,
            brightness=brightness,
            sharpness=sharpness,
            use_orig_audio=use_orig_audio,
            orig_audio_db=orig_audio_db,
            crop_offset_pct=crop_offset_pct,
            output_4k=(str(resolution).lower() == "4k"),
        )

        ok_count  = sum(1 for r in results if r[2] is not None)
        err_count = len(results) - ok_count
        status_lines.append(f"\nListo: {ok_count} generados, {err_count} errores")

        for _, titulo, path, status, err in results:
            if err:
                status_lines.append(f"  ERROR — {titulo}: {err[:120]}")
            else:
                status_lines.append(f"  OK — {status}")

    except Exception:
        tb = traceback.format_exc()
        status_lines.append(f"\n[ERROR CRÍTICO]\n{tb[-1200:]}")

    progress(1.0, desc="Listo!")
    return "\n".join(status_lines), refresh_shorts_list(), gr.update(choices=get_generated_shorts_for_preview())


def run_generate_all_shorts(
    contrast,
    brightness,
    sharpness,
    use_orig_audio,
    orig_audio_db,
    crop_offset_pct=0,
    resolution="1080p",
    video_src=None,
    audio_src=None,
    orig_audio_src=None,
    video_upload=None,
    audio_upload=None,
    orig_upload=None,
    progress=gr.Progress(track_tqdm=False),
):
    """Genera todos los shorts de la configuración."""
    choices = get_shorts_choices()
    return run_generate_shorts(
        choices, contrast, brightness, sharpness, use_orig_audio, orig_audio_db,
        crop_offset_pct=crop_offset_pct,
        resolution=resolution,
        video_src=video_src, audio_src=audio_src, orig_audio_src=orig_audio_src,
        video_upload=video_upload, audio_upload=audio_upload, orig_upload=orig_upload,
        progress=progress,
    )


# ─── Project management ──────────────────────────────────────────────────────

def archive_project(name: str) -> str:
    """Comprime todas las salidas del proyecto en un ZIP en archives/."""
    ARCHIVES_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "proyecto"))
    zip_path = ARCHIVES_DIR / f"{ts}_{safe}.zip"

    dirs_to_archive = [
        (CACHE_DIR,          "cache"),
        (SUBTITLE_OUTPUT_DIR,"subtitle_output"),
        (ENHANCE_OUTPUT_DIR, "enhance_output"),
        (SHORTS_OUTPUT_DIR,  "shorts_videos_lists"),
        (YOUTUBE_VIDEO_DIR,  "youtube_video"),
    ]
    total = 0
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for src_dir, arc_prefix in dirs_to_archive:
                if not src_dir.exists():
                    continue
                for f in src_dir.rglob("*"):
                    if f.is_file():
                        zf.write(f, arc_prefix + "/" + f.relative_to(src_dir).as_posix())
                        total += 1
        size_mb = zip_path.stat().st_size / 1e6
        return f"Archivado: {zip_path.name} ({size_mb:.1f} MB, {total} archivos)"
    except Exception as e:
        return f"Error archivando: {e}"


def new_project() -> str:
    """Archiva automáticamente el proyecto actual y limpia el workspace."""
    # Archive first
    archive_msg = archive_project("auto_before_new")
    # Then clear everything
    clear_msg = clear_all_caches()
    return f"{archive_msg}\n\n{clear_msg}"


# ─── ZIP download ─────────────────────────────────────────────────────────────

def download_all_shorts_zip():
    """Empaca todos los shorts generados en un ZIP y lo devuelve para descarga."""
    files = sorted(SHORTS_OUTPUT_DIR.glob("*.mp4")) if SHORTS_OUTPUT_DIR.exists() else []
    if not files:
        return None, "No hay shorts generados todavía."
    zip_path = BASE_DIR / "shorts_all.zip"
    try:
        # ZIP_STORED: los MP4 ya están comprimidos con H.264; DEFLATE no reduce
        # el tamaño pero sí añade mucho tiempo de CPU innecesario.
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for f in files:
                zf.write(f, f.name)
        size_mb = zip_path.stat().st_size / 1e6
        return str(zip_path), f"ZIP con {len(files)} shorts ({size_mb:.1f} MB) listo para descargar."
    except Exception as e:
        return None, f"Error creando ZIP: {e}"


# ─── Sidebar file tree ────────────────────────────────────────────────────────

def get_file_tree_html() -> str:
    """Genera HTML con árbol de archivos del proyecto."""
    sections = [
        ("youtube_video/",  YOUTUBE_VIDEO_DIR, VIDEO_EXTENSIONS | AUDIO_EXTENSIONS),
        ("cache/",          CACHE_DIR,          {".mp3", ".wav", ".json", ".mp4"}),
        ("subtitle_output/",SUBTITLE_OUTPUT_DIR,{".mp4"}),
        ("enhance_output/", ENHANCE_OUTPUT_DIR, {".mp4"}),
        ("shorts_videos/",  SHORTS_OUTPUT_DIR,  {".mp4"}),
        ("archives/",       ARCHIVES_DIR,        {".zip"}),
    ]

    rows = []
    for label, directory, exts in sections:
        if not directory.exists():
            rows.append(f'<div class="ft-section">{label} <span class="ft-empty">(vacio)</span></div>')
            continue
        files = [f for f in sorted(directory.rglob("*")) if f.is_file() and f.suffix.lower() in exts]
        count_label = f"({len(files)} archivo{'s' if len(files)!=1 else ''})" if files else "(vacio)"
        rows.append(f'<div class="ft-section">{label} <span class="ft-count">{count_label}</span></div>')
        for f in files[:20]:  # cap at 20 per folder
            size_mb = f.stat().st_size / 1e6
            rows.append(
                f'<div class="ft-file">'
                f'<span class="ft-name" title="{f}">{f.name}</span>'
                f' <span class="ft-size">{size_mb:.1f}MB</span></div>'
            )
        if len(files) > 20:
            rows.append(f'<div class="ft-file ft-more">... +{len(files)-20} más</div>')

    body = "\n".join(rows) if rows else "<div class='ft-empty'>Sin archivos</div>"
    return f'<div class="file-tree">{body}</div>'


def get_all_output_files_for_preview() -> list:
    """Todos los MP4/MP3/WAV disponibles para previsualizar en la barra lateral."""
    choices = []
    for directory in [SHORTS_OUTPUT_DIR, SUBTITLE_OUTPUT_DIR, ENHANCE_OUTPUT_DIR, YOUTUBE_VIDEO_DIR]:
        if not directory.exists():
            continue
        for f in sorted(directory.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".mp4", ".mp3", ".wav", ".m4a"}:
                label = f"{f.parent.name}/{f.name}" if f.parent != directory else f.name
                choices.append((label, str(f)))
    return choices


def get_generated_shorts_for_preview() -> list:
    """Lista de shorts MP4 generados para el dropdown de preview."""
    if not SHORTS_OUTPUT_DIR.exists():
        return []
    return [(f.name, str(f)) for f in sorted(SHORTS_OUTPUT_DIR.glob("*.mp4"))]


# ─── Frame preview helpers ────────────────────────────────────────────────────

def preview_enhance_frame(video_path, contrast=0, brightness=0, sharpness=0):
    """Extrae el primer frame del video y aplica los filtros de mejora. Retorna path JPEG."""
    if not video_path:
        return None
    vpath = video_path if isinstance(video_path, str) else getattr(video_path, "name", None)
    if not vpath or not Path(vpath).exists():
        return None
    out = BASE_DIR / "_preview_enhance.jpg"
    br_ffmpeg = brightness / 100.0
    ct_ffmpeg = 1.0 + contrast / 100.0
    sh_ffmpeg = (sharpness / 100.0) * 1.5
    vf_parts = [f"eq=brightness={br_ffmpeg:.4f}:contrast={ct_ffmpeg:.4f}"]
    if sh_ffmpeg > 0:
        vf_parts.append(f"unsharp=5:5:{sh_ffmpeg:.4f}:5:5:0.0")
    vf = ",".join(vf_parts)
    try:
        _subprocess.run(
            ["ffmpeg", "-y", "-i", vpath, "-vframes", "1", "-vf", vf, "-q:v", "2", str(out)],
            capture_output=True, timeout=20,
        )
        return str(out) if out.exists() else None
    except Exception:
        return None


def preview_short_crop_frame(
    crop_offset_pct: float,
    video_path: str = None,
    video_upload=None,
    resolution: str = "1080p",
):
    """
    Extrae el primer frame del video fuente con el crop aplicado.

    resolution: "1080p" → crop 1080×1920 | "4k" → crop 2160×3840
    La previsualización siempre se escala a 360×640 para respuesta rápida.
    """
    from shorts_pipeline import _find_first
    _up = _extract_file_path(video_upload)
    video_src = None
    effective = _up or video_path
    if effective and Path(effective).exists():
        video_src = effective
    if not video_src:
        video_src = _find_first(SHORTS_DIR / "short_video", "*.mp4")
    if not video_src:
        return None

    is_4k  = str(resolution).lower() == "4k"
    out_h  = 3840 if is_4k else 1920
    out_w  = 2160 if is_4k else 1080
    factor = max(-1.0, min(1.0, crop_offset_pct / 100.0))
    vf = (
        f"scale=-1:{out_h},"
        f"crop={out_w}:{out_h}:(in_w-{out_w})/2*(1+{factor:.4f}):0,"
        f"scale=360:640"
    )
    out = BASE_DIR / "_preview_crop.jpg"
    try:
        _subprocess.run(
            ["ffmpeg", "-y", "-i", video_src, "-vframes", "1", "-vf", vf, "-q:v", "3", str(out)],
            capture_output=True, timeout=20,
        )
        return str(out) if out.exists() else None
    except Exception:
        return None





def build_ui(init=None):  # noqa: C901
    # init: dict of pre-computed values to avoid firing callables on page load
    if init is None:
        init = {}

    def _v(key, fn, *args, **kwargs):
        """Return pre-computed value from init dict, or compute it now."""
        return init[key] if key in init else fn(*args, **kwargs)

    with gr.Blocks(
        title="EN->ES Voice Dubbing",
        theme=gr.themes.Monochrome(),
        css=APP_CSS,
    ) as demo:

        with gr.Row(elem_id="app-layout"):

            # ─────────────────────── LEFT NAV ────────────────────────────
            with gr.Column(scale=1, min_width=220, elem_id="left-nav"):

                gr.Markdown("## EN→ES\nVoice Dubbing")

                gr.Markdown("---")
                gr.Markdown("PROYECTO", elem_classes=["nav-label"])
                proj_name_input = gr.Textbox(
                    placeholder="Nombre del proyecto...",
                    show_label=False, lines=1,
                )
                archive_btn  = gr.Button("Archivar Proyecto", variant="secondary",
                                         size="sm", elem_classes=["nav-btn"])
                new_proj_btn = gr.Button("Nuevo Proyecto",    variant="stop",
                                         size="sm", elem_classes=["nav-btn"])
                proj_status = gr.Textbox(
                    show_label=False, interactive=False, lines=1,
                    placeholder="Estado...",
                )

                gr.Markdown("---")
                gr.Markdown("NAVEGACION", elem_classes=["nav-label"])
                nav_archivos_btn  = gr.Button("Archivos / YouTube",    elem_classes=["nav-btn"])
                nav_doblaje_btn   = gr.Button("Doblaje EN → ES",       elem_classes=["nav-btn"])
                nav_subtitulos_btn = gr.Button("Subtitulos y Mejoras",  elem_classes=["nav-btn"])
                nav_shorts_btn    = gr.Button("Shorts",                elem_classes=["nav-btn"])

                gr.Markdown("---")
                with gr.Accordion("Guia RunPod", open=False) as runpod_accordion:
                    gr.Markdown(RUNPOD_GUIDE_MD)

                gr.Markdown("---")
                gr.Markdown("ARCHIVOS", elem_classes=["nav-label"])
                sidebar_refresh_btn = gr.Button("Actualizar", size="sm", elem_classes=["nav-btn"])
                sidebar_tree = gr.HTML(value=_v("file_tree", get_file_tree_html))

                gr.Markdown("VISTA PREVIA", elem_classes=["nav-label"])
                sidebar_file_dd = gr.Dropdown(
                    choices=_v("preview_files", get_all_output_files_for_preview),
                    label="Seleccionar archivo",
                    value=None,
                )
                sidebar_preview_video = gr.Video(
                    label="", interactive=False, visible=False,
                )
                sidebar_preview_audio = gr.Audio(
                    label="", interactive=False, visible=False, type="filepath",
                )

            # ─────────────────────── MAIN CONTENT ────────────────────────
            with gr.Column(scale=3, elem_id="main-content"):
                with gr.Tabs(selected=0, elem_id="main-tabs") as main_tabs:

                    # ── TAB 0: Archivos / YouTube ─────────────────────────
                    with gr.Tab("Archivos / YouTube", id=0):

                        with gr.Accordion("Limpiar Todos los Caches", open=False):
                            gr.Markdown(
                                "Elimina `cache/`, `subtitle_output/`, `enhance_output/`, `shorts/`, `youtube_video/`.\n"
                                "Mantiene `voice_reference/`, `shorts-settings.json`, `background_music/`."
                            )
                            clean_all_btn = gr.Button(
                                "Limpiar Todo — Nuevo Proyecto",
                                variant="stop", size="lg",
                            )
                            clean_all_status = gr.Textbox(
                                label="Estado", interactive=False, lines=3,
                            )
                            clean_all_btn.click(fn=clear_all_caches, outputs=[clean_all_status])

                        gr.Markdown("---\n## Descargar desde YouTube")

                        yt_url_input = gr.Textbox(
                            label="URL de YouTube",
                            placeholder="https://www.youtube.com/watch?v=...",
                            lines=1,
                        )
                        yt_cookies_upload = gr.File(
                            label="Opcional: Subir cookies.txt (Requerido para RunPod)",
                            file_types=[".txt"],
                            height=100,
                        )

                        with gr.Row():
                            yt_search_btn    = gr.Button("Buscar Formatos", variant="primary", scale=2)
                            yt_clear_yt_btn  = gr.Button("Limpiar youtube_video/", variant="secondary", scale=1)

                        yt_search_status = gr.Textbox(
                            label="Estado", interactive=False, lines=2,
                            placeholder="Aquí aparecerá el título y conteo de formatos...",

                        )

                        yt_video_formats = gr.Dataframe(
                            headers=["ID", "EXT", "Resolución", "FPS", "Codec", "Bitrate", "Tamaño"],
                            datatype=["str"] * 7,
                            label="Formatos de VIDEO (mejor calidad primero)",
                            interactive=False, wrap=False,
                            column_widths=["60px","60px","110px","55px","130px","75px","75px"],
                        )
                        yt_audio_formats = gr.Dataframe(
                            headers=["ID", "EXT", "Bitrate", "Codec", "Tamaño", "Idioma"],
                            datatype=["str"] * 6,
                            label="Formatos de AUDIO (mayor bitrate primero)",
                            interactive=False, wrap=False,
                            column_widths=["60px","60px","80px","130px","75px","80px"],
                        )

                        with gr.Row():
                            yt_format_id    = gr.Textbox(
                                label="Format-ID a descargar",
                                placeholder="Ej: 137 (video 1080p) ó 251 (audio opus)",
                                scale=2,
                            )
                            yt_download_btn = gr.Button("Descargar Formato", variant="primary", scale=1)

                        yt_download_status = gr.Textbox(
                            label="Progreso de descarga", interactive=False, lines=3,
                            placeholder="El resultado aparecerá aquí...",

                        )

                        gr.Markdown("---\n## Archivos en youtube_video/")
                        yt_files_display = gr.Textbox(
                            label="Archivos disponibles",
                            value=_v("yt_files", list_youtube_files),
                            interactive=False, lines=5,
                        )
                        yt_refresh_files_btn = gr.Button("Refrescar lista", size="sm")

                        _yt_video_dd_choices = gr.State(_v("yt_vid_choices", get_youtube_file_choices, include_video=True, include_audio=False))
                        _yt_audio_dd_choices = gr.State(_v("yt_aud_choices", get_youtube_file_choices, include_video=False, include_audio=True))

                        def _search_formats(url, cookies):
                            v_rows, a_rows, status = fetch_youtube_formats(url, cookies_file=cookies)
                            return v_rows, a_rows, status

                        def _download_format(url, fmt_id, cookies):
                            msg, cv, ca = download_youtube_format(url, fmt_id, cookies_file=cookies)
                            return msg, list_youtube_files(), cv, ca

                        def _clear_yt_files():
                            YOUTUBE_VIDEO_DIR.mkdir(exist_ok=True)
                            for f in list(YOUTUBE_VIDEO_DIR.iterdir()):
                                if f.is_file():
                                    try: f.unlink()
                                    except Exception: pass
                            cv = get_youtube_file_choices(include_video=True,  include_audio=False)
                            ca = get_youtube_file_choices(include_video=False, include_audio=True)
                            return list_youtube_files(), cv, ca

                        def _refresh_yt():
                            cv = get_youtube_file_choices(include_video=True,  include_audio=False)
                            ca = get_youtube_file_choices(include_video=False, include_audio=True)
                            return list_youtube_files(), cv, ca

                        yt_search_btn.click(
                            fn=_search_formats,
                            inputs=[yt_url_input, yt_cookies_upload],
                            outputs=[yt_video_formats, yt_audio_formats, yt_search_status],
                        )
                        yt_download_btn.click(
                            fn=_download_format,
                            inputs=[yt_url_input, yt_format_id, yt_cookies_upload],
                            outputs=[yt_download_status, yt_files_display,
                                     _yt_video_dd_choices, _yt_audio_dd_choices],
                        )
                        yt_clear_yt_btn.click(
                            fn=_clear_yt_files,
                            outputs=[yt_files_display, _yt_video_dd_choices, _yt_audio_dd_choices],
                        )
                        yt_refresh_files_btn.click(
                            fn=_refresh_yt,
                            outputs=[yt_files_display, _yt_video_dd_choices, _yt_audio_dd_choices],
                        )

                    # ── TAB 1: Auditoría de Audio en Inglés ──────────────────────
                    with gr.Tab("Auditoría de Audio en Inglés", id=1):
                        gr.Markdown("## Auditoría y Detección de Voces (Diarización)")
                        gr.Markdown("Extrae información de los locutores (VAD) y te muestra qué referencias de voz necesitas para el doblaje.")
                        
                        audit_audio_input = gr.Audio(
                            label="Archivo de Audio/Video en Inglés (MP3 · WAV · MP4 · M4A)",
                            type="filepath", sources=["upload"],
                        )
                        hf_token_input = gr.Textbox(
                            label="HuggingFace Token (hf_...)",
                            placeholder="Pega aquí tu token de HuggingFace para usar Pyannote",
                            lines=1,
                            type="password",
                            value="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"
                        )
                        audit_btn = gr.Button("Ejecutar Auditoría", variant="primary", size="lg")
                        
                        audit_status = gr.Textbox(
                            label="Estado de la Auditoría", interactive=False, lines=3,
                        )
                        
                        audit_results = gr.Dataframe(
                            headers=["Speaker", "Tiempo Total (s)", "Archivo Audio", "Archivo Texto"],
                            datatype=["str", "number", "str", "str"],
                            label="Voces Detectadas (Referencias Necesarias en voice_reference/)",
                            interactive=False, wrap=False,
                        )
                        
                        gr.Markdown("### Cargar Archivos de Referencia")
                        gr.Markdown("Sube un archivo de audio (.wav) y el texto (.txt) para cada locutor que aparezca Listado en la tabla. Luego ve a Doblaje.")
                        
                        with gr.Row():
                            ref_spk_input = gr.Textbox(label="Locutor (ej: SPEAKER_00)")
                            ref_audio_upload = gr.Audio(label="Audio (.wav)", type="filepath", sources=["upload"])
                            ref_text_upload = gr.File(label="Texto (.txt)", file_types=[".txt"])
                        
                        ref_upload_btn = gr.Button("Guardar Referencia", variant="primary")
                        ref_upload_status = gr.Textbox(label="Estado", interactive=False, lines=1)
                            
                        def _run_audit_ui(audio_file, hf_token):
                            if not audio_file:
                                return "Sube un archivo de audio", []
                            
                            if hf_token:
                                os.environ["HF_TOKEN"] = hf_token.strip()
                            
                            file_hash = get_file_hash(audio_file)
                            job_dir   = CACHE_DIR / file_hash
                            job_dir.mkdir(parents=True, exist_ok=True)
                            
                            wav_path = str(job_dir / "input.wav")
                            if not os.path.exists(wav_path):
                                extract_audio(audio_file, str(job_dir))
                            
                            from pipeline import audit_audio
                            out_json = audit_audio(wav_path, str(job_dir))
                            if not out_json:
                                return "Error en auditoría", []
                                
                            with open(out_json, "r", encoding="utf-8") as f:
                                data = json.load(f).get("results", [])
                                
                            speakers = {}
                            for d in data:
                                spk = d["speaker"]
                                dur = d["end"] - d["start"]
                                speakers[spk] = speakers.get(spk, 0.0) + dur
                                
                            rows = []
                            for spk, dur in speakers.items():
                                wav_file = f"voice_reference/{spk}.wav"
                                txt_file = f"voice_reference/{spk}.txt"
                                
                                wav_exists = os.path.exists(BASE_DIR / wav_file)
                                txt_exists = os.path.exists(BASE_DIR / txt_file)
                                
                                wav_status = wav_file if wav_exists else f"FALTA: {wav_file}"
                                txt_status = txt_file if txt_exists else f"FALTA: {txt_file}"
                                
                                rows.append([spk, round(dur, 1), wav_status, txt_status])
                                
                            return f"Auditoría terminada. JSON guardado.", rows
                            
                        def _save_reference(spk, audio_file, text_file):
                            if not spk or not audio_file or not text_file:
                                return "Faltan archivos o nombre de locutor."
                            spk = spk.strip()
                            ref_dir = BASE_DIR / "voice_reference"
                            ref_dir.mkdir(exist_ok=True)
                            import shutil
                            dst_audio = ref_dir / f"{spk}.wav"
                            dst_text = ref_dir / f"{spk}.txt"
                            
                            audio_src = _extract_file_path(audio_file)
                            text_src = _extract_file_path(text_file)
                            
                            if audio_src: shutil.copy(audio_src, dst_audio)
                            if text_src: shutil.copy(text_src, dst_text)
                            return f"Guardado {dst_audio.name} y {dst_text.name}"
                            
                        audit_btn.click(
                            fn=_run_audit_ui,
                            inputs=[audit_audio_input, hf_token_input],
                            outputs=[audit_status, audit_results],
                        )
                        ref_upload_btn.click(
                            fn=_save_reference,
                            inputs=[ref_spk_input, ref_audio_upload, ref_text_upload],
                            outputs=[ref_upload_status]
                        )

                    # ── TAB 2: Doblaje EN→ES ──────────────────────────────
                    with gr.Tab("Doblaje EN → ES", id=2):

                        with gr.Row():
                            dob_yt_dd = gr.Dropdown(
                                choices=_v("yt_all_choices", get_youtube_file_choices, include_video=True, include_audio=True),
                                label="Seleccionar desde youtube_video/",
                                value=None, scale=2,
                            )
                            dob_yt_refresh = gr.Button("↺", size="sm", scale=0)

                        audio_input = gr.Audio(
                            label="Archivo en inglés (MP3 · WAV · MP4 · M4A)",
                            type="filepath", sources=["upload"],
                        )

                        with gr.Row():
                            dub_btn      = gr.Button("Doblar al Español / Dub to Spanish",
                                                     variant="primary", size="lg", scale=2)
                            dub_test_btn = gr.Button("Test 5 minutos",
                                                     variant="secondary", size="lg", scale=1)

                        audio_output = gr.Audio(
                            label="Audio doblado / Dubbed audio (ES)",
                            type="filepath", interactive=False,
                        )
                        dub_download = gr.File(
                            label="Descargar audio doblado (MP3)", interactive=False,
                        )

                        status_box = gr.Textbox(
                            label="Estado / Status", interactive=False, lines=3,
                            placeholder="El resultado aparecerá aquí...",

                        )

                        gr.Markdown("#### Segmentos con timestamps del audio doblado ES — copia esto para generar el JSON de shorts")
                        segs_auto_viewer = gr.Code(
                            language="json",
                            label="Segmentos (timestamps de final_es.mp3)",
                            lines=10, interactive=False,
                            visible=False,
                        )

                        with gr.Accordion("Cache / Checkpoints", open=False):
                            gr.Markdown(
                                "Los checkpoints se guardan automáticamente en `cache/`. "
                                "Si falla el TTS, al subir el mismo archivo de nuevo saltará "
                                "directamente al TTS sin re-transcribir ni re-traducir."
                            )
                            cache_status_box = gr.Textbox(
                                label="Estado del cache",
                                value=_v("cache_status", get_cache_status),
                                interactive=False, lines=4,
                            )
                            with gr.Row():
                                refresh_btn = gr.Button("Actualizar", size="sm")
                                clear_btn   = gr.Button("Limpiar todo el cache", variant="stop", size="sm")

                            gr.Markdown("**Ver / copiar segments.json**")
                            with gr.Row():
                                segs_job_dd = gr.Dropdown(
                                    choices=_v("cache_jobs", get_cache_job_choices),
                                    label="Job (hash)", value=None, scale=3,
                                )
                                segs_refresh_dd_btn = gr.Button("↺", size="sm", scale=0)
                                segs_load_btn       = gr.Button("Ver segments.json", size="sm", scale=1)
                            segs_viewer = gr.Code(
                                language="json",
                                label="segments.json (usa el botón de copia ↗)",
                                lines=20, interactive=False,
                            )

                        def _dub_and_download(audio_file, progress=gr.Progress(track_tqdm=False)):
                            for result_path, status, segs_content in run_dubbing(audio_file, progress=progress):
                                segs_upd = gr.update(value=segs_content or "", visible=bool(segs_content))
                                if result_path:
                                    yield result_path, result_path, status, segs_upd
                                else:
                                    yield gr.update(), gr.update(), status, segs_upd

                        def _dub_test_and_download(audio_file, progress=gr.Progress(track_tqdm=False)):
                            for result_path, status, segs_content in run_dubbing_test(audio_file, progress=progress):
                                segs_upd = gr.update(value=segs_content or "", visible=bool(segs_content))
                                if result_path:
                                    yield result_path, result_path, status, segs_upd
                                else:
                                    yield gr.update(), gr.update(), status, segs_upd

                        dub_btn.click(
                            fn=_dub_and_download, inputs=[audio_input],
                            outputs=[audio_output, dub_download, status_box, segs_auto_viewer],
                            show_progress="full",
                        )
                        dub_test_btn.click(
                            fn=_dub_test_and_download, inputs=[audio_input],
                            outputs=[audio_output, dub_download, status_box, segs_auto_viewer],
                            show_progress="full",
                        )
                        refresh_btn.click(fn=get_cache_status, outputs=[cache_status_box])
                        clear_btn.click(fn=clear_all_caches, outputs=[cache_status_box])
                        segs_refresh_dd_btn.click(
                            fn=get_cache_job_choices, outputs=[segs_job_dd],
                        )
                        segs_load_btn.click(
                            fn=load_segments_json,
                            inputs=[segs_job_dd], outputs=[segs_viewer],
                        )
                        dob_yt_dd.change(
                            fn=lambda p: gr.update(value=p) if p else gr.update(),
                            inputs=[dob_yt_dd], outputs=[audio_input],
                        )
                        dob_yt_refresh.click(
                            fn=lambda: gr.update(
                                choices=get_youtube_file_choices(include_video=True, include_audio=True)
                            ),
                            outputs=[dob_yt_dd],
                        )


                    # ── TAB 3: Subtítulos ─────────────────────────────────
                    with gr.Tab("Subtítulos", id=3):

                        gr.Markdown(
                            "Selecciona archivos del servidor **o** sube desde tu computadora. "
                            "Si subes un archivo, tiene prioridad sobre el dropdown."
                        )

                        # ── Video source ──────────────────────────────────
                        with gr.Row():
                            sub_yt_video_dd = gr.Dropdown(
                                choices=_v("yt_vid_choices", get_youtube_file_choices, include_video=True, include_audio=False),
                                label="Video desde el servidor (youtube_video/)",
                                value=None, scale=3,
                            )
                            sub_yt_video_ref = gr.Button("↺", size="sm", scale=0)

                        sub_video_upload = gr.File(
                            label="O sube video desde tu computadora (.mp4 .mkv .avi .mov .webm)",
                            file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"],
                            file_count="single",
                        )

                        # ── Audio source ──────────────────────────────────
                        with gr.Row():
                            sub_audio_dd = gr.Dropdown(
                                choices=_v("shorts_src_audio", get_cache_final_audio_choices),
                                label="Audio doblado ES desde el servidor (cache/.../final_es.mp3)",
                                value=None, scale=3,
                            )
                            sub_audio_ref = gr.Button("↺", size="sm", scale=0)

                        sub_audio_upload = gr.Audio(
                            label="O sube audio doblado ES desde tu computadora",
                            type="filepath", sources=["upload"],
                        )

                        gr.Markdown("---\n**Filtros de imagen**")
                        with gr.Row():
                            sub_contrast   = gr.Slider(-100, 100, value=48,  step=1, label="Contraste")
                            sub_brightness = gr.Slider(-100, 100, value=21,  step=1, label="Brillo")
                            sub_sharpness  = gr.Slider(0, 100,    value=100, step=1, label="Nitidez")

                        sub_frame_preview = gr.Image(
                            label="Vista previa — primer frame",
                            interactive=False, type="filepath", visible=True,
                        )

                        gr.Markdown("---\n**Opciones Adicionales**")
                        with gr.Row():
                            sub_do_orig_audio = gr.Checkbox(label="Audio original de fondo", value=False, scale=1)

                        with gr.Group(visible=False) as sub_orig_audio_group:
                            gr.Markdown("**Audio original EN como fondo ambiental**")
                            with gr.Row():
                                sub_yt_orig_dd  = gr.Dropdown(
                                    choices=_v("yt_all_choices", get_youtube_file_choices, include_video=True, include_audio=True),
                                    label="Video/audio original EN desde el servidor (youtube_video/)",
                                    value=None, scale=3,
                                )
                                sub_yt_orig_ref = gr.Button("↺", size="sm", scale=0)
                            sub_orig_video = gr.File(
                                label="O sube video/audio original EN desde tu computadora",
                                file_types=[".mp4", ".mkv", ".avi", ".mov", ".m4a", ".mp3", ".wav", ".webm"],
                            )
                            sub_audio_db = gr.Slider(-40, 0, value=-25, step=1, label="Volumen audio original (dB)")

                        gr.Markdown("---")
                        with gr.Row():
                            sub_full_btn = gr.Button("Generar video con subtitulos y mejoras",
                                                     variant="primary", size="lg", scale=1)
                            sub_test_btn = gr.Button("Test 20 segundos",
                                                     variant="secondary", size="lg", scale=1)

                        sub_video_output = gr.Video(
                            label="Video final mejorado", interactive=False,
                        )
                        sub_download = gr.File(
                            label="Descargar video final (MP4)", interactive=False,
                        )

                        sub_status_box = gr.Textbox(
                            label="Estado", interactive=False, lines=3,
                            placeholder="El resultado aparecerá aquí...",
                        )

                        def _sub_full(video_dd, audio_dd, video_up, audio_up, contrast, brightness, sharpness, do_orig_audio, orig_dd, orig_up, audio_db, progress=gr.Progress(track_tqdm=False)):
                            video_path = _extract_file_path(video_up) or video_dd
                            audio_path = _extract_file_path(audio_up) or audio_dd
                            orig_path = _extract_file_path(orig_up) or orig_dd
                            out, status = run_subtitles_full(video_path, audio_path, contrast, brightness, sharpness, do_orig_audio, orig_path, audio_db, progress=progress)
                            return out, out, status

                        def _sub_test(video_dd, audio_dd, video_up, audio_up, contrast, brightness, sharpness, do_orig_audio, orig_dd, orig_up, audio_db, progress=gr.Progress(track_tqdm=False)):
                            video_path = _extract_file_path(video_up) or video_dd
                            audio_path = _extract_file_path(audio_up) or audio_dd
                            orig_path = _extract_file_path(orig_up) or orig_dd
                            out, status = run_subtitles_test(video_path, audio_path, contrast, brightness, sharpness, do_orig_audio, orig_path, audio_db, progress=progress)
                            return out, out, status

                        _sub_inputs = [
                            sub_yt_video_dd, sub_audio_dd, sub_video_upload, sub_audio_upload,
                            sub_contrast, sub_brightness, sub_sharpness,
                            sub_do_orig_audio, sub_yt_orig_dd, sub_orig_video, sub_audio_db,
                        ]

                        sub_full_btn.click(
                            fn=_sub_full,
                            inputs=_sub_inputs,
                            outputs=[sub_video_output, sub_download, sub_status_box],
                            show_progress="full",
                        )
                        sub_test_btn.click(
                            fn=_sub_test,
                            inputs=_sub_inputs,
                            outputs=[sub_video_output, sub_download, sub_status_box],
                            show_progress="full",
                        )
                        sub_yt_video_ref.click(
                            fn=lambda: gr.update(
                                choices=get_youtube_file_choices(include_video=True, include_audio=False)
                            ),
                            outputs=[sub_yt_video_dd],
                        )
                        sub_audio_ref.click(
                            fn=lambda: gr.update(choices=get_cache_final_audio_choices()),
                            outputs=[sub_audio_dd],
                        )
                        sub_yt_orig_ref.click(
                            fn=lambda: gr.update(choices=get_youtube_file_choices(include_video=True, include_audio=True)),
                            outputs=[sub_yt_orig_dd],
                        )
                        sub_do_orig_audio.change(
                            fn=lambda v: gr.update(visible=v),
                            inputs=[sub_do_orig_audio], outputs=[sub_orig_audio_group],
                        )

                        def _sub_preview(video_up, contrast, brightness, sharpness, video_dd=None):
                            video_path = _extract_file_path(video_up) or video_dd
                            return preview_enhance_frame(video_path, contrast, brightness, sharpness)
                        
                        _sub_preview_inputs = [
                            sub_video_upload, sub_contrast, sub_brightness, sub_sharpness,
                            sub_yt_video_dd,
                        ]
                        _sub_preview_kwargs = dict(
                            fn=_sub_preview,
                            inputs=_sub_preview_inputs,
                            outputs=[sub_frame_preview],
                            show_progress="hidden",
                            queue=False,
                        )
                        sub_video_upload.change(**_sub_preview_kwargs)
                        sub_contrast.release(**_sub_preview_kwargs)
                        sub_brightness.release(**_sub_preview_kwargs)
                        sub_sharpness.release(**_sub_preview_kwargs)
                        sub_yt_video_dd.change(**_sub_preview_kwargs)


                    # ── TAB 4: Shorts ─────────────────────────────────────
                    with gr.Tab("Shorts", id=4):

                        # ── Archivos fuente ────────────────────────────────
                        gr.Markdown(
                            "### Archivos fuente\n"
                            "Selecciona desde el servidor **o** sube desde tu computadora. "
                            "El archivo subido tiene prioridad sobre el dropdown."
                        )
                        with gr.Row():
                            sh_src_video_dd = gr.Dropdown(
                                choices=_v("shorts_src_video", get_shorts_video_choices),
                                label="Video 4K desde el servidor (youtube_video/)",
                                value=None, scale=3,
                            )
                            sh_src_video_ref = gr.Button("↺", size="sm", scale=0)

                        sh_src_video_upload = gr.File(
                            label="O sube video 4K desde tu computadora (.mp4 .mkv .avi .mov .webm)",
                            file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"],
                            file_count="single",
                        )

                        with gr.Row():
                            sh_src_audio_dd = gr.Dropdown(
                                choices=_v("shorts_src_audio", get_cache_final_audio_choices),
                                label="Audio doblado ES desde el servidor (cache/…/final_es.mp3)",
                                value=None, scale=3,
                            )
                            sh_src_audio_ref = gr.Button("↺", size="sm", scale=0)

                        sh_src_audio_upload = gr.Audio(
                            label="O sube audio doblado ES desde tu computadora",
                            type="filepath", sources=["upload"],
                        )

                        with gr.Row():
                            sh_src_orig_dd = gr.Dropdown(
                                choices=get_youtube_file_choices(include_video=True, include_audio=True),
                                label="Audio original EN de fondo desde el servidor (youtube_video/)",
                                value=None, scale=3,
                            )
                            sh_src_orig_ref = gr.Button("↺", size="sm", scale=0)

                        sh_src_orig_upload = gr.Audio(
                            label="O sube audio original EN desde tu computadora",
                            type="filepath", sources=["upload"],
                        )

                        gr.Markdown("---")
                        with gr.Accordion("Editar shorts-settings.json", open=False):
                            shorts_json_editor = gr.Code(
                                language="json",
                                label="shorts-settings.json",
                                value=_v("shorts_json", load_shorts_json), lines=20, interactive=True,
                            )
                            with gr.Row():
                                shorts_json_load_btn = gr.Button("Recargar desde archivo", size="sm", scale=1)
                                shorts_json_save_btn = gr.Button("Guardar y validar", variant="primary", size="sm", scale=1)
                            shorts_json_status = gr.Textbox(
                                label="Estado JSON", interactive=False, lines=2,
                            )

                        shorts_selector = gr.CheckboxGroup(
                            choices=_v("shorts_choices", get_shorts_choices),
                            label="Shorts disponibles — selecciona los que quieres generar",
                            value=[],
                        )

                        gr.Markdown("---\n**Configuración de imagen y audio**")

                        gr.Markdown("**Resolución de salida**")
                        sh_resolution = gr.Radio(
                            choices=["1080p", "4k"],
                            value="1080p",
                            label="Resolución vertical",
                            info="1080p = 1080×1920 HD · 4k = 2160×3840 (requiere GPU con NVENC o CPU potente)",
                        )

                        gr.Markdown("**Imagen**")
                        with gr.Row():
                            sh_contrast   = gr.Slider(-100, 100, value=48,  step=1, label="Contraste")
                            sh_brightness = gr.Slider(-100, 100, value=21,  step=1, label="Brillo")
                            sh_sharpness  = gr.Slider(0, 100,    value=100, step=1, label="Nitidez")

                        # ── Crop offset ────────────────────────────────────
                        gr.Markdown("**Encuadre horizontal** — 0 = centro · − = izquierda · + = derecha")
                        with gr.Row():
                            sh_crop_offset = gr.Slider(
                                -100, 100, value=0, step=1,
                                label="Desplazamiento horizontal (%)",
                                scale=3,
                            )
                            sh_crop_preview_btn = gr.Button("Ver primer frame", size="sm", scale=1)

                        sh_crop_preview_img = gr.Image(
                            label="Vista previa del encuadre",
                            interactive=False, visible=True, type="filepath",
                            height=300,
                        )

                        gr.Markdown("**Audio original EN de fondo** — coloca el archivo en `shorts/audio_original_en/`")
                        with gr.Row():
                            sh_use_orig_audio = gr.Checkbox(
                                label="Agregar audio original EN como fondo ambiental",
                                value=False,
                            )
                            sh_audio_db = gr.Slider(
                                -40, 0, value=-25, step=1,
                                label="Volumen audio original (dB)",
                            )

                        gr.Markdown("---")
                        with gr.Row():
                            shorts_gen_btn     = gr.Button("Generar Seleccionados",
                                                           variant="primary", size="lg", scale=2)
                            shorts_all_btn     = gr.Button("Generar Todos",
                                                           variant="secondary", size="lg", scale=1)
                            shorts_refresh_btn = gr.Button("Actualizar Lista", size="sm", scale=1)

                        shorts_status = gr.Textbox(
                            label="Estado / Progreso", interactive=False, lines=5,
                            placeholder="Los logs de proceso aparecerán aquí...",

                        )
                        shorts_list = gr.Textbox(
                            label="Shorts generados en shorts_videos_lists/",
                            interactive=False, lines=4,
                            value=_v("shorts_list", refresh_shorts_list),
                        )

                        # ── ZIP download ────────────────────────────────
                        with gr.Row():
                            shorts_zip_btn  = gr.Button("Descargar Todos los Shorts (ZIP)",
                                                        variant="secondary", scale=1)
                            shorts_zip_status = gr.Textbox(
                                show_label=False, interactive=False, lines=1,
                                placeholder="Estado ZIP...", scale=2,
                            )
                        shorts_zip_file = gr.File(
                            label="ZIP de todos los shorts", interactive=False,
                        )

                        # ── Shorts preview ──────────────────────────────
                        gr.Markdown("---\n**Vista previa de shorts generados**")
                        with gr.Row():
                            shorts_preview_dd = gr.Dropdown(
                                choices=_v("shorts_preview", get_generated_shorts_for_preview),
                                label="Seleccionar short para previsualizar",
                                value=None, scale=3,
                            )
                            shorts_preview_ref = gr.Button("↺", size="sm", scale=0)
                        shorts_preview_video = gr.Video(
                            label="Vista previa del short",
                            interactive=False,
                        )

                        _sh_enhance_inputs = [
                            sh_contrast, sh_brightness, sh_sharpness,
                            sh_use_orig_audio, sh_audio_db, sh_crop_offset,
                            sh_resolution,
                            sh_src_video_dd, sh_src_audio_dd, sh_src_orig_dd,
                            # uploads — take priority over dropdown values inside the function
                            sh_src_video_upload, sh_src_audio_upload, sh_src_orig_upload,
                        ]

                        shorts_gen_btn.click(
                            fn=run_generate_shorts,
                            inputs=[shorts_selector] + _sh_enhance_inputs,
                            outputs=[shorts_status, shorts_list, shorts_preview_dd],
                            show_progress="full",
                        )
                        shorts_all_btn.click(
                            fn=run_generate_all_shorts,
                            inputs=_sh_enhance_inputs,
                            outputs=[shorts_status, shorts_list, shorts_preview_dd],
                            show_progress="full",
                        )

                        # ── Source file refresh buttons ────────────────────
                        sh_src_video_ref.click(
                            fn=lambda: gr.update(choices=get_shorts_video_choices()),
                            outputs=[sh_src_video_dd],
                        )
                        sh_src_audio_ref.click(
                            fn=lambda: gr.update(choices=get_cache_final_audio_choices()),
                            outputs=[sh_src_audio_dd],
                        )
                        sh_src_orig_ref.click(
                            fn=lambda: gr.update(
                                choices=get_youtube_file_choices(include_video=True, include_audio=True)
                            ),
                            outputs=[sh_src_orig_dd],
                        )
                        shorts_refresh_btn.click(
                            fn=refresh_shorts_list, outputs=[shorts_list],
                        )
                        shorts_zip_btn.click(
                            fn=download_all_shorts_zip,
                            outputs=[shorts_zip_file, shorts_zip_status],
                        )
                        _crop_preview_kwargs = dict(
                            fn=preview_short_crop_frame,
                            inputs=[sh_crop_offset, sh_src_video_dd, sh_src_video_upload, sh_resolution],
                            outputs=[sh_crop_preview_img],
                            show_progress="hidden",
                            queue=False,
                        )
                        sh_crop_preview_btn.click(**_crop_preview_kwargs)
                        sh_crop_offset.release(**_crop_preview_kwargs)
                        sh_resolution.change(**_crop_preview_kwargs)
                        shorts_json_load_btn.click(
                            fn=load_shorts_json, outputs=[shorts_json_editor],
                        )

                        def _save_and_refresh_shorts(content):
                            msg, new_choices = save_shorts_json(content)
                            return msg, gr.update(choices=new_choices, value=[])

                        shorts_json_save_btn.click(
                            fn=_save_and_refresh_shorts,
                            inputs=[shorts_json_editor],
                            outputs=[shorts_json_status, shorts_selector],
                        )
                        shorts_preview_dd.change(
                            fn=lambda p: p,
                            inputs=[shorts_preview_dd],
                            outputs=[shorts_preview_video],
                        )
                        shorts_preview_ref.click(
                            fn=lambda: gr.update(choices=get_generated_shorts_for_preview()),
                            outputs=[shorts_preview_dd],
                        )

                        gr.Markdown(
                            """
                            ---
                            **Formatos de salida (MP4 · vertical 9:16 · AAC 256k · `shorts/shorts_videos_lists/`)**

                            | Resolución | Píxeles | Codec | CQ |
                            |------------|---------|-------|----|
                            | 1080p (HD) | 1080×1920 | h264_nvenc | 15/18 |
                            | 4K | 2160×3840 | hevc_nvenc | 16/18 |

                            **Crop:** El desplazamiento horizontal se aplica al crear el short — no escala ni degrada la calidad.
                            Fallback automático a libx264/libx265 (CPU) si la GPU no tiene NVENC.
                            """
                        )

            # ─────────────────────── RIGHT SPACER ────────────────────────
            with gr.Column(scale=1, elem_id="right-spacer"):
                pass

        # ── Global event wiring (outside inner columns/rows) ─────────────────

        # Navigation buttons → switch tab
        nav_archivos_btn.click(fn=lambda: gr.update(selected=0), outputs=[main_tabs])
        nav_doblaje_btn.click(fn=lambda: gr.update(selected=2), outputs=[main_tabs])
        nav_subtitulos_btn.click(fn=lambda: gr.update(selected=3), outputs=[main_tabs])
        nav_shorts_btn.click(fn=lambda: gr.update(selected=4), outputs=[main_tabs])

        archive_btn.click(
            fn=archive_project,
            inputs=[proj_name_input],
            outputs=[proj_status],
        )
        new_proj_btn.click(
            fn=new_project,
            outputs=[proj_status],
        )
        sidebar_refresh_btn.click(
            fn=lambda: (
                get_file_tree_html(),
                gr.update(choices=get_all_output_files_for_preview()),
            ),
            outputs=[sidebar_tree, sidebar_file_dd],
        )

        def _sidebar_select(file_path):
            if not file_path:
                return gr.update(visible=False), gr.update(visible=False)
            ext = Path(file_path).suffix.lower()
            if ext in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
                return gr.update(value=file_path, visible=True), gr.update(visible=False)
            elif ext in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}:
                return gr.update(visible=False), gr.update(value=file_path, visible=True)
            return gr.update(visible=False), gr.update(visible=False)

        sidebar_file_dd.change(
            fn=_sidebar_select,
            inputs=[sidebar_file_dd],
            outputs=[sidebar_preview_video, sidebar_preview_audio],
        )

    return demo



# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    startup()
    # Pre-compute initial UI values so page-load doesn't fire multiple
    # callables through the queue simultaneously (causes "busy" errors).
    _init = {
        "file_tree":      get_file_tree_html(),
        "yt_files":       list_youtube_files(),
        "cache_status":   get_cache_status(),
        "shorts_json":    load_shorts_json(),
        "shorts_choices": get_shorts_choices(),
        "yt_vid_choices": get_youtube_file_choices(include_video=True, include_audio=False),
        "yt_aud_choices": get_youtube_file_choices(include_video=False, include_audio=True),
        "yt_all_choices": get_youtube_file_choices(include_video=True, include_audio=True),
        "sub_out_files":  get_subtitle_output_files(),
        "cache_jobs":     get_cache_job_choices(),
        "shorts_list":    refresh_shorts_list(),
        "preview_files":  get_all_output_files_for_preview(),
        "shorts_preview": get_generated_shorts_for_preview(),
        "shorts_src_video": get_shorts_video_choices(),
        "shorts_src_audio": get_cache_final_audio_choices(),
    }
    demo = build_ui(init=_init)

    demo.queue(
        max_size=3,
        default_concurrency_limit=1,
    )

    # Gradio 6+ requires theme/css in launch(); Gradio 4/5 accepts them in Blocks()
    # We pass them here so both versions work (Gradio ignores unknown launch kwargs gracefully)
    _launch_kwargs = dict(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        max_threads=4,
        allowed_paths=[str(BASE_DIR)],  # permite a Gradio servir el ZIP y otros archivos de BASE_DIR
    )
    demo.launch(**_launch_kwargs)
