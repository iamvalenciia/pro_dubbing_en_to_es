"""
LatentSync wrapper — video-to-video lipsync nativo (Bytedance/LatentSync).

DISEÑO (IMPORTANTE):

- LatentSync es OBLIGATORIO. Cada vez que se genera un master, sus labios
  se re-sincronizan al audio ES. No hay modo "placeholder" / "rapido".
  Si el bootstrap falla, la funcion lanza RuntimeError — nada de fallbacks
  silenciosos.

- BOOTSTRAP AUTOMATICO: la primera vez que se llama a
  `lipsync_video_to_video` (o `ensure_latentsync_ready` explicitamente),
  si el repo o los pesos faltan, se clonan/descargan automaticamente al
  VOLUMEN PERSISTENTE de RunPod (`/runpod-volume/LatentSync/` por default):

      /runpod-volume/LatentSync/               ← repo clonado (persistente)
      ├── scripts/inference.py                 ← entrypoint llamado via subprocess
      ├── configs/unet/stage2_512.yaml         ← config default (512x512, 18 GB VRAM)
      ├── configs/unet/stage2.yaml             ← alternativa (256x256, 8 GB VRAM)
      ├── requirements.txt
      └── checkpoints/                         ← pesos (persistentes)
          ├── latentsync_unet.pt               ← ~1.8 GB
          └── whisper/tiny.pt                  ← ~75 MB

  Como viven en /runpod-volume, sobreviven al restart del pod. Solo los
  pip packages del entorno del pod (efimero) se re-instalan la primera
  vez que el pod levanta — eso lo maneja `_install_requirements()` y se
  marca con un sentinel `.deps_installed` para saltarlo despues.

- LatentSync TOMA un video + un audio y devuelve el MISMO video con los
  labios re-sincronizados. NO regenera los frames completos — solo edita
  el area de la boca. Esto preserva gestos, expresiones y movimientos
  de camara del original. VRAM: 8 GB (v1.5 @ 256x256) / 18 GB (v1.6 @
  512x512, default).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import ffmpeg

from src.logger import get_logger

log = get_logger("latentsync")


# =========================================================================
# Configuration
# =========================================================================

LATENTSYNC_REPO_URL = "https://github.com/bytedance/LatentSync.git"
LATENTSYNC_HF_REPO = "ByteDance/LatentSync-1.6"
LATENTSYNC_ALLOW_PATTERNS = ["latentsync_unet.pt", "whisper/tiny.pt"]


def _repo_path() -> str:
    return os.environ.get("LATENTSYNC_PATH", "/app/LatentSync")

def _weights_path() -> str:
    return os.environ.get("LATENTSYNC_WEIGHTS_PATH", "/runpod-volume/models/LatentSync")

def _unet_ckpt() -> str:
    return os.path.join(_weights_path(), "latentsync_unet.pt")


def _whisper_ckpt() -> str:
    return os.path.join(_weights_path(), "whisper", "tiny.pt")


def _default_unet_config() -> str:
    """
    Config por defecto. v1.6 usa stage2_512.yaml (512x512, 18GB VRAM).
    Para GPUs de 8-12 GB VRAM, forza stage2.yaml (256x256) via
    env LATENTSYNC_UNET_CONFIG.
    """
    return os.environ.get(
        "LATENTSYNC_UNET_CONFIG",
        "configs/unet/stage2_512.yaml",  # relativo al repo
    )


# =========================================================================
# Availability checks (diagnostico + usado por ensure_latentsync_ready)
# =========================================================================

def is_code_available() -> bool:
    repo = _repo_path()
    return (
        os.path.isdir(repo)
        and os.path.isfile(os.path.join(repo, "scripts", "inference.py"))
    )


def is_models_available() -> bool:
    unet = _unet_ckpt()
    whisper = _whisper_ckpt()
    if not os.path.isfile(unet) or not os.path.isfile(whisper):
        return False
    # latentsync_unet.pt es ~1-2 GB, whisper/tiny.pt es ~75 MB
    if os.path.getsize(unet) < 500 * 1024 * 1024:
        return False
    if os.path.getsize(whisper) < 10 * 1024 * 1024:
        return False
    return True


def status_report() -> dict:
    unet = _unet_ckpt()
    whisper = _whisper_ckpt()
    return {
        "repo_path": _repo_path(),
        "repo_present": is_code_available(),
        "unet_ckpt": {
            "path": unet,
            "present": os.path.isfile(unet),
            "size_mb": round(os.path.getsize(unet) / 1024 / 1024, 1)
            if os.path.isfile(unet) else 0,
        },
        "whisper_ckpt": {
            "path": whisper,
            "present": os.path.isfile(whisper),
            "size_mb": round(os.path.getsize(whisper) / 1024 / 1024, 1)
            if os.path.isfile(whisper) else 0,
        },
        "unet_config": _default_unet_config(),
        "ready": is_code_available() and is_models_available(),
    }


# =========================================================================
# Bootstrap (download weights if missing) — AUTOMATICO Y SIN PIP
# =========================================================================

def _download_weights():
    from huggingface_hub import snapshot_download

    ckpt_dir = _weights_path()
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "whisper"), exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    log.info(
        f"[LatentSync] Descargando pesos desde "
        f"{LATENTSYNC_HF_REPO} -> {ckpt_dir} (~2 GB, solo la primera vez)"
    )
    snapshot_download(
        repo_id=LATENTSYNC_HF_REPO,
        local_dir=ckpt_dir,
        token=token,
        max_workers=4,
        allow_patterns=LATENTSYNC_ALLOW_PATTERNS,
    )


def ensure_latentsync_ready(install_deps: bool = True) -> None:
    """
    Verifica código y pesos sin intervenir paquetes.
    """
    target_repo = _repo_path()
    if not is_code_available():
        if os.path.exists("/app/LatentSync/scripts/inference.py") and target_repo != "/app/LatentSync":
            log.info(f"Copiando código pre-empaquetado de /app/LatentSync a {target_repo}")
            os.makedirs(target_repo, exist_ok=True)
            shutil.copytree("/app/LatentSync", target_repo, dirs_exist_ok=True)
        else:
            raise RuntimeError(f"El código de LatentSync no se encuentra pre-empaquetado. Reconstruye la imagen.")

    # Paso 2: descargar pesos si faltan en volumen de red
    if not is_models_available():
        try:
            _download_weights()
        except Exception as e:
            raise RuntimeError(
                f"Fallo descargando pesos LatentSync desde "
                f"{LATENTSYNC_HF_REPO}: {e}. Verifica HF_TOKEN si el repo "
                f"requiere autenticacion."
            ) from e
        if not is_models_available():
            raise RuntimeError(
                f"Descarga completada pero pesos faltan o son invalidos:\n"
                f"  unet:    {_unet_ckpt()}\n"
                f"  whisper: {_whisper_ckpt()}"
            )

    # Crear symlink a los pesos dentro del repo si es necesario para inference.py
    ckpt_symlink = os.path.join(target_repo, "checkpoints")
    if not os.path.exists(ckpt_symlink):
        try:
            os.symlink(_weights_path(), ckpt_symlink)
            log.info(f"Symlink de checkpoints creado en {ckpt_symlink}")
        except OSError as e:
            log.warning(f"No se pudo crear symlink de checkpoints: {e}")

    log.info("LatentSync listo (código y pesos verificados).")


# =========================================================================
# Helpers (ffmpeg probes + input normalization)
# =========================================================================

def _video_duration(video_path: str) -> float:
    probe = ffmpeg.probe(video_path)
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            try:
                return float(s.get("duration") or probe["format"]["duration"])
            except (KeyError, ValueError):
                pass
    return float(probe["format"]["duration"])


def _normalize_inputs(video_path: str, audio_path: str, workdir: str) -> tuple[str, str]:
    """
    LatentSync espera video a 25fps y audio a 16kHz mono.
    Normalizamos por si vienen distintos.
    """
    v_out = os.path.join(workdir, "ls_input_video.mp4")
    a_out = os.path.join(workdir, "ls_input_audio.wav")

    # NVENC default en A100 (env LATENTSYNC_NORM_VCODEC permite override).
    # preset=p1 es el mas rapido; esta pre-normalizacion es solo setup, no calidad final.
    vcodec = os.environ.get("LATENTSYNC_NORM_VCODEC", "h264_nvenc")
    preset = "p1" if vcodec.endswith("_nvenc") else "ultrafast"

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    # 1. Normalizar Video con Fallback (Si NVENC falla por version de ffmpeg o driver)
    cmd_v = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-i", video_path, "-r", "25", "-c:v", vcodec, "-preset", preset, "-c:a", "aac", v_out]
    res_v = subprocess.run(cmd_v, capture_output=True, text=True)
    if res_v.returncode != 0:
        log.warning(f"NVENC falló en LatentSync norm: {res_v.stderr}")
        log.info("Usando fallback libx264 silenciosamente (esto tardará mucho en videos largos)...")
        cmd_v_fallback = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-i", video_path, "-r", "25", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", v_out]
        res_v2 = subprocess.run(cmd_v_fallback, capture_output=True, text=True)
        if res_v2.returncode != 0:
            raise RuntimeError(f"Fallo normalizando video LatentSync (incluso con libx264): {res_v2.stderr}")

    # 2. Normalizar Audio
    cmd_a = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", a_out]
    res_a = subprocess.run(cmd_a, capture_output=True, text=True)
    if res_a.returncode != 0:
        raise RuntimeError(f"Fallo normalizando audio LatentSync: {res_a.stderr}")

    return v_out, a_out


# =========================================================================
# Core inference (OBLIGATORIO — no hay fallback)
# =========================================================================

def lipsync_video_to_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    inference_steps: int | None = None,
    guidance_scale: float | None = None,
    enable_deepcache: bool = True,
    seed: int | None = None,
    timeout_s: int = 3600,
    normalize_inputs: bool = True,
) -> str:
    """
    Corre LatentSync sobre un video completo (video-to-video nativo).

    OBLIGATORIO: si el setup no esta listo, lo instala automaticamente
    antes de correr. Si algo falla en cualquier etapa (bootstrap o
    inference), lanza RuntimeError. NO hay fallback silencioso.

    - video_path: video original a re-sincronizar
    - audio_path: audio doblado (ES) que va a drivear los labios
    - output_path: destino .mp4
    - inference_steps: 10-50, mayor = mejor calidad, mas lento.
      Default: env LATENTSYNC_STEPS (default 10 — 2x mas rapido que 20 con
      DeepCache ON, calidad de labios prac. igual para dubbing).
    - guidance_scale: 1.0-3.0, mayor = mejor sync pero mas distorsion.
      Default: env LATENTSYNC_GUIDANCE (default 1.5).
    - enable_deepcache: acelera el denoising loop (recomendado ON)

    LatentSync maneja internamente videos de cualquier duracion: no
    necesitamos chunking manual. Si el video es muy largo la inference
    toma mas tiempo pero no hace falta dividirlo a mano.
    """
    if inference_steps is None:
        inference_steps = int(os.environ.get("LATENTSYNC_STEPS", "10"))
    if guidance_scale is None:
        guidance_scale = float(os.environ.get("LATENTSYNC_GUIDANCE", "1.5"))
    # Bootstrap automatico — noop si ya esta listo
    ensure_latentsync_ready()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    temp_base = os.environ.get("LOCAL_TEMP_DIR", "/workspace/qdp_data/temp_workspace")
    os.makedirs(temp_base, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="latentsync_", dir=temp_base)
    try:
        if normalize_inputs:
            v_in, a_in = _normalize_inputs(video_path, audio_path, workdir)
        else:
            v_in = os.path.abspath(video_path)
            a_in = os.path.abspath(audio_path)

        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--unet_config_path", _default_unet_config(),
            "--inference_ckpt_path", _unet_ckpt(),
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--video_path", os.path.abspath(v_in),
            "--audio_path", os.path.abspath(a_in),
            "--video_out_path", os.path.abspath(output_path),
        ]
        if enable_deepcache:
            cmd.append("--enable_deepcache")
        if seed is not None:
            cmd += ["--seed", str(seed)]

        log.info(
            f"LatentSync infer: video={video_path} audio={audio_path} "
            f"steps={inference_steps} gs={guidance_scale} deepcache={enable_deepcache}"
        )
        result = subprocess.run(
            cmd, cwd=_repo_path(), capture_output=True, text=True, timeout=timeout_s,
        )
        if result.returncode != 0:
            log.error(f"LatentSync stdout:\n{result.stdout[-3000:]}")
            log.error(f"LatentSync stderr:\n{result.stderr[-3000:]}")
            raise RuntimeError(
                f"LatentSync fallo (code={result.returncode}). "
                f"Ultimas lineas de stderr:\n{result.stderr[-600:]}"
            )

        if not os.path.isfile(output_path):
            raise RuntimeError(
                f"LatentSync corrio sin error pero no produjo {output_path}"
            )

        log.info(
            f"LatentSync OK: {output_path} "
            f"({os.path.getsize(output_path)/1024/1024:.1f} MB)"
        )
        return output_path
    finally:
        if os.environ.get("LATENTSYNC_KEEP_WORKDIR") != "1":
            shutil.rmtree(workdir, ignore_errors=True)
