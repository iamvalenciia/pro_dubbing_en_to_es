import os
import re
import json
import time
import shutil
import threading
import subprocess
import gdown

from src.logger import get_logger, phase_timer, step_timer, log_file_info

try:
    from src.hw_autotune import autotune, detect_gpu_name
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("input")


# Threads de ffmpeg para la normalizacion inicial. En stream-copy no importan
# (es I/O bound), pero si tenemos que re-encodear por incompat de codec, si.
# Escalado por CPU count con cap a 16 (ffmpeg no escala bien mas alla).
if _HAS_AUTOTUNE:
    PHASE0_FFMPEG_THREADS = autotune(
        "PHASE0_FFMPEG_THREADS", baseline=8, scale_with="cpu",
        baseline_ref=8, min_v=2, max_v=16,
    )
else:
    PHASE0_FFMPEG_THREADS = int(os.environ.get("PHASE0_FFMPEG_THREADS", "8"))


def _detect_nvenc_available() -> bool:
    """True si ffmpeg tiene h264_nvenc y el driver NVIDIA responde.
    Usado como preferencia en el re-encode fallback (stream-copy sigue siendo
    el primer intento porque es instantaneo)."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-h", "encoder=h264_nvenc"],
            capture_output=True, text=True, timeout=5,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return "h264_nvenc" in out and "not recognized" not in out.lower()
    except (subprocess.TimeoutExpired, OSError):
        return False


def extract_drive_id(url: str) -> str:
    """Extrae el ID real de un link de Google Drive."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match: return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match: return match.group(1)
    return url.strip()


def _looks_like_local_path(s: str) -> bool:
    """True si el string es path local y el archivo existe.
    Acepta rutas estilo Unix (/runpod-volume/...), Windows (C:\\...) y relativas.
    """
    if not s:
        return False
    s = s.strip()
    if s.startswith(("http://", "https://", "drive.google.com")):
        return False
    return os.path.exists(s)


def _fetch_source(spec: str, dest_raw: str, label: str) -> str:
    """Resuelve un input a un archivo local. spec puede ser:
    - URL de Google Drive (con /d/ID o id=ID) -> gdown
    - Path local existente -> copy a dest_raw
    Retorna el path del archivo crudo listo para ffmpeg.
    """
    if _looks_like_local_path(spec):
        log.info(f"  [{label}] detectado path local: {spec}")
        with step_timer(log, f"symlink local -> {dest_raw}"):
            try:
                # Elimina si ya existe de una corrida previa
                if os.path.exists(dest_raw):
                    os.remove(dest_raw)
                # Crea un acceso directo (symlink) que funciona entre discos distintos
                os.symlink(os.path.abspath(spec), dest_raw)
            except OSError:
                # Fallback a copy tradicional si symlink falla
                shutil.copyfile(spec, dest_raw)
        log_file_info(log, dest_raw, f"raw_{label}")
        return dest_raw

    drive_id = extract_drive_id(spec)
    if not drive_id:
        raise RuntimeError(f"[{label}] no es path local ni URL de Drive parseable: {spec!r}")
    log.info(f"  [{label}] Drive ID extraido: {drive_id}")
    with step_timer(log, f"gdown: descargando {label} {drive_id}"):
        gdown.download(id=drive_id, output=dest_raw, quiet=False)
    if not os.path.exists(dest_raw):
        raise RuntimeError(
            f"[{label}] gdown no produjo archivo. Puede ser rate-limit de Drive. "
            f"Sube el archivo manualmente al volumen (/runpod-volume/user_input/...) "
            f"y pega el path local en la UI."
        )
    log_file_info(log, dest_raw, f"raw_{label}")
    return dest_raw


def _probe_streams(path: str) -> dict:
    """Usa ffprobe para leer container + codecs + duracion + tamaño.
    Retorna {} si ffprobe no existe o falla. Usado para decidir si podemos
    saltarnos la normalizacion completamente.
    """
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return {}
    try:
        result = subprocess.run(
            [ffprobe_bin, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout or "{}")
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        return {}

    fmt = data.get("format") or {}
    info = {
        "format_name": fmt.get("format_name", ""),
        "duration_s": float(fmt.get("duration", "0") or "0"),
        "size_bytes": int(fmt.get("size", "0") or "0"),
        "video_codec": None,
        "audio_codec": None,
    }
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and info["video_codec"] is None:
            info["video_codec"] = s.get("codec_name")
        elif s.get("codec_type") == "audio" and info["audio_codec"] is None:
            info["audio_codec"] = s.get("codec_name")
    return info


def _should_skip_video_normalize(info: dict, test_mode: bool) -> bool:
    """True si el video ya es MP4/H.264 (o H.265) y no hay que recortar.
    En ese caso podemos saltarnos ffmpeg completamente y solo renombrar el
    symlink — ahorra procesar 5GB innecesariamente sobre network volume.
    """
    if test_mode:
        return False  # test_mode requiere trim a 30s
    if not info:
        return False
    fmt = info.get("format_name", "") or ""
    vcodec = info.get("video_codec", "") or ""
    # ffprobe devuelve 'mov,mp4,m4a,3gp,3g2,mj2' para contenedores MP4-like.
    is_mp4_container = any(x in fmt for x in ("mp4", "mov", "m4v"))
    is_compatible_codec = vcodec in ("h264", "hevc")  # H.265 tambien es playable
    return is_mp4_container and is_compatible_codec


def _run_ffmpeg_with_progress(cmd: list, label: str, total_duration_s: float = 0.0,
                              log_every_s: float = 5.0, timeout_s: float = 7200):
    """Reemplaza `.run(quiet=True)` de ffmpeg-python. Loguea progreso cada `log_every_s`
    segundos para que el usuario vea que no esta colgado.

    El `cmd` DEBE tener '-progress pipe:1 -nostats -loglevel warning -y' incluidos en el
    orden correcto (-y antes del output, -progress antes del output). El helper solo drena
    stdout (progreso) y stderr (warnings) sin bloquear.
    """
    short_cmd = (cmd[0] + " " + " ".join(cmd[-4:])) if len(cmd) > 6 else " ".join(cmd)
    log.info(f"  $ {short_cmd}")

    t_start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    # Drain stderr en thread para que no sature el pipe y bloquee ffmpeg.
    stderr_buf: list[str] = []

    def _drain_stderr():
        for errline in iter(proc.stderr.readline, ""):
            if errline:
                stderr_buf.append(errline)

    t_err = threading.Thread(target=_drain_stderr, daemon=True)
    t_err.start()

    last_log_t = t_start
    last_out_time_us = 0
    last_size = 0
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            line = line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "out_time_us":
                try:
                    last_out_time_us = int(v)
                except ValueError:
                    pass
            elif k == "total_size":
                try:
                    last_size = int(v)
                except ValueError:
                    pass

            now = time.time()
            if now - last_log_t >= log_every_s:
                pct = ""
                if total_duration_s > 0:
                    pct = f"{100.0 * (last_out_time_us / 1e6) / total_duration_s:5.1f}% "
                log.info(
                    f"    [{label}] {pct}"
                    f"t_out={last_out_time_us/1e6:7.1f}s "
                    f"size={last_size/1e6:7.1f}MB "
                    f"elapsed={now-t_start:5.1f}s"
                )
                last_log_t = now

            if time.time() - t_start > timeout_s:
                proc.kill()
                raise RuntimeError(f"ffmpeg [{label}] timeout {timeout_s}s")

        rc = proc.wait(timeout=60)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
        raise

    t_err.join(timeout=2)

    if rc != 0:
        tail = "".join(stderr_buf[-30:])
        raise RuntimeError(f"ffmpeg [{label}] exit={rc}:\n{tail}")

    dt = time.time() - t_start
    log.info(
        f"    [{label}] OK en {dt:.1f}s "
        f"(size={last_size/1e6:.1f}MB, duration_out={last_out_time_us/1e6:.1f}s)"
    )


def _normalize_video(raw_video: str, final_video: str, test_mode: bool) -> None:
    """Normaliza video a MP4. Estrategia en 3 niveles:
      0) SKIP (fast path real): si el video ya es MP4/H.264 (o H.265) y no
         hay trim, renombramos el symlink y no tocamos ffmpeg. Ahorra
         procesar 5GB innecesariamente sobre network volume.
      1) Stream-copy SIN `+faststart` (era la causa del "stuck"). faststart
         obliga a reescribir todo el archivo para mover el moov atom; como
         este video es intermedio (Phase 5b re-muxea el final con faststart),
         aca no hace falta.
      2) Re-encode con NVENC (GPU) o libx264 (CPU) como fallback.

    Todas las llamadas pasan por `_run_ffmpeg_with_progress()` — el usuario
    ve % y MB procesados cada 5s en vez de silencio durante minutos.
    """
    info = _probe_streams(raw_video)
    duration_s = info.get("duration_s", 0.0)
    log.info(
        f"  ffprobe: fmt={info.get('format_name', '?')} "
        f"vcodec={info.get('video_codec', '?')} "
        f"acodec={info.get('audio_codec', '?')} "
        f"dur={duration_s:.1f}s size={info.get('size_bytes', 0)/1e9:.2f}GB"
    )

    # --- PATH 0: fast path real. Ya es MP4/H.264 -> solo renombrar. ---
    if _should_skip_video_normalize(info, test_mode):
        log.info(
            "  [fast-path] video ya es MP4/H.264 compatible. "
            "Saltando ffmpeg completo (rename del symlink — 0s)."
        )
        with step_timer(log, f"rename {raw_video} -> {final_video}"):
            # os.replace funciona con symlinks: mueve el enlace, no el target.
            # Atomico y cross-platform.
            os.replace(raw_video, final_video)
        return

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    in_args: list[str] = []
    if test_mode:
        # -ss/-t como input args: ffmpeg los aplica antes del decode (mas rapido).
        in_args = ["-ss", "0", "-t", "30"]
        log.info("  modo TEST: recortando video a 30s")

    # --- PATH 1: stream-copy (sin +faststart). ---
    copy_cmd = [
        ffmpeg_bin, "-hide_banner", "-y",
        "-nostats", "-loglevel", "warning",
        *in_args,
        "-i", raw_video,
        "-c:v", "copy", "-c:a", "copy",
        "-threads", str(PHASE0_FFMPEG_THREADS),
        "-progress", "pipe:1",
        final_video,
    ]
    try:
        with step_timer(log, f"ffmpeg stream-copy [threads={PHASE0_FFMPEG_THREADS}]"):
            _run_ffmpeg_with_progress(copy_cmd, "stream-copy", duration_s, timeout_s=1800)
        return
    except RuntimeError as e:
        log.warning(f"Stream-copy del video fallo: {e}. Re-intentando con re-encode.")

    # --- PATH 2: re-encode (NVENC preferido, libx264 fallback). ---
    use_nvenc = _detect_nvenc_available()
    vcodec = os.environ.get("PHASE0_VCODEC") or ("h264_nvenc" if use_nvenc else "libx264")
    enc_args = ["-c:v", vcodec, "-c:a", "aac", "-b:a", "192k",
                "-threads", str(PHASE0_FFMPEG_THREADS)]
    if vcodec == "h264_nvenc":
        enc_args += [
            "-preset", os.environ.get("PHASE0_NVENC_PRESET", "p4"),
            "-cq", str(os.environ.get("PHASE0_NVENC_CQ", "20")),
            "-b:v", "0",
        ]
        log.info(f"Re-encode GPU (NVENC) [GPU={detect_gpu_name() if _HAS_AUTOTUNE else 'cuda'}]")
    else:
        enc_args += ["-preset", "medium",
                     "-crf", str(os.environ.get("PHASE0_X264_CRF", "20"))]
        log.info("Re-encode CPU (libx264)")

    enc_cmd = [
        ffmpeg_bin, "-hide_banner", "-y",
        "-nostats", "-loglevel", "warning",
        *in_args,
        "-i", raw_video,
        *enc_args,
        "-progress", "pipe:1",
        final_video,
    ]
    with step_timer(log, f"ffmpeg re-encode ({vcodec})"):
        _run_ffmpeg_with_progress(enc_cmd, f"encode-{vcodec}", duration_s, timeout_s=7200)


def download_and_prepare_media(video_url: str, audio_url: str, test_mode: bool, output_dir: str = "input"):
    """Resuelve video+audio (URL Drive o path local) y normaliza con ffmpeg.
    test_mode=True recorta a 30s.

    Video: stream-copy primero (instantaneo); si falla, re-encode con NVENC/libx264.
    Audio: siempre PCM s16le (formato que espera Demucs en Fase 1).
    """
    with phase_timer(log, "FASE 0 — Descarga y Normalizacion de Media"):
        log.info(f"video_input: {video_url}")
        log.info(f"audio_input: {audio_url}")
        log.info(f"test_mode: {test_mode}")
        log.info(f"output_dir: {output_dir}")
        log.info(f"PHASE0_FFMPEG_THREADS={PHASE0_FFMPEG_THREADS} "
                 f"nvenc_available={_detect_nvenc_available()}")

        # Cleanup de residuos de runs previos. `output_dir` solo contiene
        # intermediarios de Fase 0 (raw_* symlinks, master_*/test_*.mp4/wav).
        # NUNCA archivos del usuario — esos viven en /runpod-volume/user_input/
        # y no los tocamos. Importante para evitar que un symlink viejo o un
        # master_video.mp4 parcial de un run interrumpido confunda al siguiente.
        if os.path.isdir(output_dir):
            try:
                entries = os.listdir(output_dir)
            except OSError:
                entries = []
            if entries:
                log.info(
                    f"  Limpiando {len(entries)} residuo(s) previo(s) en "
                    f"{output_dir}/ ({', '.join(entries[:5])}"
                    f"{'...' if len(entries) > 5 else ''})"
                )
            try:
                shutil.rmtree(output_dir)
            except OSError as e:
                log.warning(f"  rmtree({output_dir}) fallo: {e}. Intentando continuar.")
        os.makedirs(output_dir, exist_ok=True)

        raw_video = os.path.join(output_dir, "raw_video_temp")
        raw_audio = os.path.join(output_dir, "raw_audio_temp")

        _fetch_source(video_url, raw_video, "video")
        _fetch_source(audio_url, raw_audio, "audio")

        final_video = os.path.join(output_dir, "test_video.mp4" if test_mode else "master_video.mp4")
        final_audio = os.path.join(output_dir, "test_audio.wav" if test_mode else "master_audio.wav")

        _normalize_video(raw_video, final_video, test_mode)
        # _normalize_video puede haber: (a) renombrado el symlink al fast-path, o
        # (b) creado un nuevo archivo via ffmpeg. En (a) raw_video ya no existe,
        # el cleanup de abajo es defensivo (no-op). En (b) el symlink sigue ahi
        # y lo removemos para liberar el FS de basura.
        try:
            if os.path.exists(raw_video) or os.path.islink(raw_video):
                os.remove(raw_video)
        except OSError:
            pass
        log_file_info(log, final_video, "final_video")

        # --- Audio normalize: siempre re-transcodear a PCM_16LE mono/stereo ---
        # Demucs (Fase 1) requiere PCM WAV; no podemos saltarnos esto como el video.
        # Pero SI podemos loguear progreso para que el usuario vea que no esta colgado.
        audio_info = _probe_streams(raw_audio)
        audio_dur = audio_info.get("duration_s", 0.0)
        log.info(
            f"  ffprobe audio: fmt={audio_info.get('format_name', '?')} "
            f"acodec={audio_info.get('audio_codec', '?')} "
            f"dur={audio_dur:.1f}s size={audio_info.get('size_bytes', 0)/1e6:.1f}MB"
        )

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        a_in: list[str] = []
        if test_mode:
            a_in = ["-ss", "0", "-t", "30"]
            log.info("  modo TEST: recortando audio a 30s")
        audio_cmd = [
            ffmpeg_bin, "-hide_banner", "-y",
            "-nostats", "-loglevel", "warning",
            *a_in,
            "-i", raw_audio,
            "-acodec", "pcm_s16le",
            "-threads", str(PHASE0_FFMPEG_THREADS),
            "-progress", "pipe:1",
            final_audio,
        ]
        with step_timer(log, f"ffmpeg audio -> {final_audio} (pcm_s16le)"):
            _run_ffmpeg_with_progress(audio_cmd, "audio-pcm", audio_dur, timeout_s=1800)
        try:
            if os.path.exists(raw_audio) or os.path.islink(raw_audio):
                os.remove(raw_audio)
        except OSError:
            pass
        log_file_info(log, final_audio, "final_audio")

        if not os.path.exists(final_video) or not os.path.exists(final_audio):
            raise RuntimeError(
                f"Input handler no produjo media final. video={final_video} audio={final_audio}."
            )

    return final_video, final_audio
