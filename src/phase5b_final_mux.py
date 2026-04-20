"""Fase 5b: Mux rapido video+audio doblado -> MP4 descargable.

Objetivo: que al terminar el doblaje (fase 5) aparezca YA un MP4 final listo
para descargar desde la UI, sin tener que abrir Tab 2 ni hacer lipsync.
Esto evita que el usuario tenga que hacer scp/ffmpeg manual desde el pod.

Estrategia:
  - DEFAULT: stream-copy del video (-c:v copy) + AAC 192k del audio doblado.
    Es casi instantaneo (segundos, no minutos) porque no re-encodea el video.
    Perfecto cuando el video ya viene en H.264/MP4 (caso 99%).
  - FALLBACK: si el mux por stream-copy falla (incompat codec/container),
    reintenta con h264_nvenc en GPU (o libx264 CPU si no hay NVENC).
  - `-shortest` para que el mux se corte al mas corto — protege contra
    audio doblado ligeramente mas largo que el video.
  - `-threads` autotuneado con CPU count del host.

NVENC rationale: en los pods con A100/H100 el ffmpeg tiene libnvenc y el
re-encode NVENC corre ~10x mas rapido que libx264 medium, con calidad
equivalente a CRF 20. Si NVENC no esta disponible (pod sin GPU/driver
incompleto), caemos a libx264 sin drama.
"""

from __future__ import annotations

import os
import subprocess
import shutil

from src.logger import get_logger, phase_timer, step_timer, log_file_info

try:
    from src.hw_autotune import autotune, detect_gpu_name
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("phase5b")


# Threads de ffmpeg para el mux. En stream-copy no importa mucho (es I/O bound),
# pero en re-encode si — escalamos por CPU count. Cap a 16 porque ffmpeg no
# escala linealmente mas alla de eso.
if _HAS_AUTOTUNE:
    FFMPEG_THREADS = autotune(
        "FFMPEG_THREADS", baseline=8, scale_with="cpu",
        baseline_ref=8, min_v=2, max_v=16,
    )
else:
    FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", "8"))


# Bitrate del audio AAC. 192k es el sweet-spot para contenido hablado +
# musica ambiente: inaudible vs 320k pero ocupa 40% menos.
FINAL_MUX_AUDIO_BITRATE = os.environ.get("FINAL_MUX_AUDIO_BITRATE", "192k")


def _detect_nvenc_available() -> bool:
    """True si el ffmpeg instalado tiene h264_nvenc y el driver NVIDIA esta OK.

    La verificacion real es si ffmpeg puede abrir el encoder sin crashear;
    listar encoders con -encoders no garantiza que el driver este cargado.
    Aca usamos una heuristica barata: probamos que ffmpeg -h encoder=h264_nvenc
    no falle. Si el ffmpeg tiene nvenc compilado pero el driver no esta,
    lanzara error al encodear realmente, y el caller hace fallback a libx264.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-h", "encoder=h264_nvenc"],
            capture_output=True, text=True, timeout=5,
        )
        # Si el encoder NO existe, ffmpeg responde con "Codec 'h264_nvenc' is not recognized"
        out = (result.stdout or "") + (result.stderr or "")
        return "h264_nvenc" in out and "not recognized" not in out.lower()
    except (subprocess.TimeoutExpired, OSError):
        return False


def _pick_vcodec_for_reencode() -> str:
    """Elige codec de video para re-encode (fallback path). NVENC > libx264."""
    override = os.environ.get("FINAL_MUX_VCODEC", "").strip()
    if override:
        log.info(f"FINAL_MUX_VCODEC override: {override}")
        return override
    if _detect_nvenc_available():
        gpu_name = detect_gpu_name() if _HAS_AUTOTUNE else "cuda"
        log.info(f"NVENC disponible (GPU={gpu_name}), re-encode via h264_nvenc")
        return "h264_nvenc"
    log.info("NVENC no disponible, re-encode via libx264 (CPU)")
    return "libx264"


def _run_ffmpeg(cmd: list[str], label: str) -> None:
    """Corre ffmpeg capturando stderr. Si falla, loguea ultimas lineas y raisea."""
    log.info(f"  $ {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=3600,
        )
    except subprocess.CalledProcessError as e:
        stderr_tail = "\n".join((e.stderr or "").splitlines()[-15:])
        log.error(f"ffmpeg [{label}] fallo (exit={e.returncode}):\n{stderr_tail}")
        raise RuntimeError(f"ffmpeg {label} fallo: exit={e.returncode}") from e
    except subprocess.TimeoutExpired as e:
        log.error(f"ffmpeg [{label}] timeout 3600s")
        raise RuntimeError(f"ffmpeg {label} timeout") from e
    # Log ultimas lineas de ffmpeg para ver bitrate/speed en el pipeline.log
    if proc.stderr:
        tail = "\n    ".join(proc.stderr.strip().splitlines()[-3:])
        log.info(f"  ffmpeg ok. tail:\n    {tail}")


def _mux_stream_copy(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux rapido: copia stream de video + re-encodea audio a AAC. No toca el video.
    Falla si el contenedor/codec no soporta copy (raro en MP4 H.264)."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-threads", str(FFMPEG_THREADS),
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", FINAL_MUX_AUDIO_BITRATE,
        "-shortest",
        "-movflags", "+faststart",  # metadata al principio -> playable mientras descarga
        output_path,
    ]
    _run_ffmpeg(cmd, "stream-copy mux")


def _mux_reencode(video_path: str, audio_path: str, output_path: str) -> None:
    """Fallback: re-encode video con NVENC (o libx264). Mas lento pero robusto
    ante containers/codecs raros."""
    vcodec = _pick_vcodec_for_reencode()

    if vcodec == "h264_nvenc":
        # NVENC preset p4 = balanced. cq=20 calidad equivalente a x264 CRF 20.
        v_args = [
            "-c:v", "h264_nvenc",
            "-preset", os.environ.get("FINAL_MUX_NVENC_PRESET", "p4"),
            "-cq", os.environ.get("FINAL_MUX_NVENC_CQ", "20"),
            "-b:v", "0",
        ]
    else:
        v_args = [
            "-c:v", vcodec,
            "-preset", "medium",
            "-crf", os.environ.get("FINAL_MUX_X264_CRF", "20"),
        ]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-threads", str(FFMPEG_THREADS),
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        *v_args,
        "-c:a", "aac", "-b:a", FINAL_MUX_AUDIO_BITRATE,
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    _run_ffmpeg(cmd, f"reencode mux ({vcodec})")


def run_phase5b_final_mux(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """Muxea `video_path` + `audio_path` -> `output_path` (MP4 descargable).

    Args:
      video_path: video original (o graded) con imagen que queremos conservar.
      audio_path: WAV doblado de la fase 5 (output/final_audio_dubbed.wav).
      output_path: destino MP4 final, tipicamente output/FINAL_DUBBED_VIDEO.mp4.

    Estrategia:
      1) Intento rapido: -c:v copy -c:a aac. Casi instantaneo.
      2) Si (1) falla (rarisimo): fallback NVENC/libx264.

    Returns:
      output_path (mismo string que el argumento), para encadenar en pipelines.
    """
    with phase_timer(log, "FASE 5b — Mux video+audio (descarga lista)"):
        log.info(f"Input video : {video_path}")
        log.info(f"Input audio : {audio_path}")
        log.info(f"Output MP4  : {output_path}")
        log.info(
            f"FFMPEG_THREADS={FFMPEG_THREADS} "
            f"FINAL_MUX_AUDIO_BITRATE={FINAL_MUX_AUDIO_BITRATE}"
        )

        for label, p in [("video", video_path), ("audio", audio_path)]:
            if not p or not os.path.exists(p):
                raise RuntimeError(f"[phase5b] {label} no existe: {p!r}")
            log_file_info(log, p, f"mux_in_{label}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 1) Fast path: stream-copy (segundos para un video de 90 min)
        try:
            with step_timer(log, "Mux rapido (stream-copy del video)"):
                _mux_stream_copy(video_path, audio_path, output_path)
        except RuntimeError as e:
            log.warning(
                f"Stream-copy mux fallo ({e}). Reintentando con re-encode "
                f"(NVENC si disponible, libx264 si no)."
            )
            # 2) Fallback: re-encode. Mas lento pero siempre funciona.
            with step_timer(log, "Mux con re-encode (fallback)"):
                _mux_reencode(video_path, audio_path, output_path)

        log_file_info(log, output_path, "final_mux_out")

    return output_path
