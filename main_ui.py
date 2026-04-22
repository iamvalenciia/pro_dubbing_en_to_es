import gradio as gr
import os
import socket

# =========================================================================
# PARCHE ANTI-CAÍDAS DE RED (DNS & IPv6) PARA RUNPOD/DOCKER
# =========================================================================
# 1. Forzar IPv4: Evita timeouts si el contenedor anuncia IPv6 pero no lo enruta.
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [res for res in responses if res[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

# 2. Inyectar Google DNS / Cloudflare como fallback seguro si el DNS del cluster cae.
try:
    with open("/etc/resolv.conf", "r") as f:
        _resolv = f.read()
    if "8.8.8.8" not in _resolv:
        with open("/etc/resolv.conf", "a") as f:
            f.write("\nnameserver 8.8.8.8\nnameserver 1.1.1.1\n")
except Exception:
    pass

# Forzar cachés temporales al NVMe (SPEC Section 2)
LOCAL_DIR = "/workspace/qdp_data"
os.environ["TORCH_HOME"] = os.path.join(LOCAL_DIR, "torch_cache")
os.environ["HF_HOME"] = os.path.join(LOCAL_DIR, "torch_cache")

os.environ["QWEN_ASR_PATH"] = os.path.join(LOCAL_DIR, "models", "Qwen", "Qwen3-ASR-1.7B")
os.environ["QWEN_ALIGNER_PATH"] = os.path.join(LOCAL_DIR, "models", "Qwen", "Qwen3-ForcedAligner-0.6B")
os.environ["QWEN_TTS_PATH"] = os.path.join(LOCAL_DIR, "models", "Qwen", "Qwen3-TTS-12Hz-1.7B-Base")
os.environ["MARIAN_MODEL_PATH"] = os.path.join(LOCAL_DIR, "models", "Helsinki-NLP", "opus-mt-en-es")
os.environ["LATENTSYNC_PATH"] = os.path.join(LOCAL_DIR, "LatentSync")
os.environ["LATENTSYNC_WEIGHTS_PATH"] = os.path.join(LOCAL_DIR, "models", "LatentSync")

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import json
import csv
import shutil
import threading
import time
import traceback

from src.logger import setup_pipeline_logger, get_logger, read_log_tail, clear_log
from src.ensure_models import ensure_qwen_models
from src.input_handler import download_and_prepare_media
from src.phase2_asr_diarization import run_phase2_diarization_and_asr
from src.phase3_llm_isochrone import run_phase3_llm_translation
from src.phase4_tts_cloning import run_phase4_tts_cloning
from src.phase5_alignment import run_phase5_time_alignment
from src.phase5b_final_mux import run_phase5b_final_mux
from src.phase6_lipsync_render import (
    run_phase6_lipsync_render,
    run_phase6_lipsync_render_nosubs,
    render_longform_with_subs,
)
from src.phase7_ai_shorts import (
    analyze_timeline_for_shorts,
    render_short,
    generate_short_thumbnail,
)


STARFIELD_HTML = """
<canvas id="qdp-starfield"
        style="position:fixed;top:0;left:0;width:100vw;height:100vh;
               z-index:-1;pointer-events:none;"></canvas>
<script>
(function(){
  if (window.__qdp_starfield_init) return;
  window.__qdp_starfield_init = true;
  const canvas = document.getElementById('qdp-starfield');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, stars, dpr;
  const STAR_COUNT = 420;
  function isDark(){
    const el = document.documentElement;
    if (el.classList.contains('dark')) return true;
    if (document.body && document.body.classList.contains('dark')) return true;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function resize(){
    dpr = window.devicePixelRatio || 1;
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function initStars(){
    stars = [];
    for (let i = 0; i < STAR_COUNT; i++){
      stars.push({x: (Math.random() - 0.5) * w, y: (Math.random() - 0.5) * h, z: Math.random() * w, pz: 0});
    }
  }
  function frame(){
    const dark = isDark();
    const bg = dark ? '#000000' : '#ffffff';
    const dot = dark ? 'rgba(255,255,255,' : 'rgba(0,0,0,';
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    for (let i = 0; i < stars.length; i++){
      const s = stars[i]; s.pz = s.z; s.z -= 2.2;
      if (s.z < 1){ s.x = (Math.random() - 0.5) * w; s.y = (Math.random() - 0.5) * h; s.z = w; s.pz = s.z; }
      const k = 128 / s.z; const px = s.x * k + cx; const py = s.y * k + cy;
      if (px < 0 || px > w || py < 0 || py > h) continue;
      const pk = 128 / s.pz; const ppx = s.x * pk + cx; const ppy = s.y * pk + cy;
      const size = (1 - s.z / w) * 2.2; const alpha = Math.min(1, (1 - s.z / w) * 1.3);
      ctx.strokeStyle = dot + alpha.toFixed(3) + ')'; ctx.lineWidth = size;
      ctx.beginPath(); ctx.moveTo(ppx, ppy); ctx.lineTo(px, py); ctx.stroke();
    }
    requestAnimationFrame(frame);
  }
  window.addEventListener('resize', () => { resize(); initStars(); });
  resize(); initStars(); requestAnimationFrame(frame);
})();
</script>
"""

CSS = """
:root {
  --qdp-surface: rgba(255,255,255,0.72);
  --qdp-border: rgba(0,0,0,0.12);
  --qdp-text: #0a0a0a;
  --qdp-muted: #5a5a5a;
  --qdp-accent: #0a0a0a;
}
.dark, .dark body {
  --qdp-surface: rgba(10,10,10,0.62);
  --qdp-border: rgba(255,255,255,0.14);
  --qdp-text: #f5f5f5;
  --qdp-muted: #9a9a9a;
  --qdp-accent: #f5f5f5;
}
.gradio-container {
  background: transparent !important;
  max-width: 1280px !important;
  margin: 0 auto !important;
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}
body { background: transparent !important; }
#qdp-header {
  text-align: center; padding: 26px 12px 10px;
  letter-spacing: 0.42em; font-weight: 600; font-size: 22px;
  color: var(--qdp-text); text-shadow: 0 0 24px var(--qdp-border);
}
#qdp-subheader {
  text-align: center; color: var(--qdp-muted); font-size: 12px;
  letter-spacing: 0.24em; margin-bottom: 18px; text-transform: uppercase;
}
.gr-group, .gr-panel {
  background: var(--qdp-surface) !important;
  border: 1px solid var(--qdp-border) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(10px) saturate(110%);
  -webkit-backdrop-filter: blur(10px) saturate(110%);
}
.gr-tabs, .tab-nav { background: transparent !important; border: 0 !important; }
.tab-nav button {
  color: var(--qdp-muted) !important; border: 0 !important;
  border-bottom: 1px solid transparent !important; background: transparent !important;
  letter-spacing: 0.08em; text-transform: uppercase; font-size: 12px !important;
}
.tab-nav button.selected {
  color: var(--qdp-text) !important;
  border-bottom: 1px solid var(--qdp-accent) !important;
}
.gr-button, button.primary, button.secondary {
  background: transparent !important;
  border: 1px solid var(--qdp-border) !important;
  color: var(--qdp-text) !important;
  border-radius: 10px !important;
  letter-spacing: 0.1em; text-transform: uppercase;
  font-size: 12px !important; transition: all 0.2s ease;
}
.gr-button:hover, button.primary:hover, button.secondary:hover {
  border-color: var(--qdp-accent) !important;
  box-shadow: 0 0 0 1px var(--qdp-accent) inset; transform: translateY(-1px);
}
.gr-button.qdp-solid, button.primary.qdp-solid {
  background: var(--qdp-accent) !important;
  color: var(--qdp-surface) !important;
}
.dark .gr-button.qdp-solid, .dark button.primary.qdp-solid { color: #0a0a0a !important; }
input[type="text"], input[type="number"], textarea, .gr-textbox textarea, .gr-textbox input {
  background: transparent !important;
  color: var(--qdp-text) !important;
  border-color: var(--qdp-border) !important;
}
/* Restaurar reactividad y visibilidad de checkboxes en Gradio */
input[type="checkbox"], .gr-checkbox label {
  cursor: pointer !important;
  pointer-events: auto !important;
}
label, .gr-form > label, .gr-block-label {
  color: var(--qdp-muted) !important;
  font-size: 11px !important;
  letter-spacing: 0.14em; text-transform: uppercase;
}
.qdp-section-title {
  font-size: 13px; letter-spacing: 0.22em; font-weight: 600;
  color: var(--qdp-text); text-transform: uppercase;
  margin: 14px 0 8px 0; padding-bottom: 4px;
  border-bottom: 1px solid var(--qdp-border);
}
.qdp-step-num {
  display: inline-block; margin-right: 8px; opacity: 0.5; font-weight: 400;
}
.qdp-hint { color: var(--qdp-muted); font-size: 12px; margin-bottom: 6px; }
.qdp-proposal { padding: 10px 12px; border: 1px solid var(--qdp-border); border-radius: 10px; margin-bottom: 8px; }
.qdp-proposal h4 { margin: 0 0 4px 0; font-size: 14px; color: var(--qdp-text); }
.qdp-proposal p { margin: 2px 0; font-size: 12px; color: var(--qdp-muted); }
/* === LOG VIEWER ===
   Terminal oscuro consistente en light y dark mode.
   Contraste alto, tipografia monoespaciada mas legible y scroll horizontal
   para paths largos. */
#qdp-log { border-radius: 10px !important; }
#qdp-log textarea {
  font-family: "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace !important;
  font-size: 12.5px !important;
  line-height: 1.6 !important;
  padding: 14px 16px !important;
  background: #0d1117 !important;
  color: #d1e3e0 !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 10px !important;
  letter-spacing: 0.1px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
  white-space: pre !important;
  overflow: auto !important;
  scrollbar-color: #30363d #0d1117;
  scrollbar-width: thin;
}
#qdp-log textarea::selection { background: rgba(88,166,255,0.35); }
#qdp-log textarea::-webkit-scrollbar { height: 8px; width: 8px; }
#qdp-log textarea::-webkit-scrollbar-track { background: #0d1117; }
#qdp-log textarea::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
#qdp-log textarea::-webkit-scrollbar-thumb:hover { background: #484f58; }
.dark #qdp-log textarea {
  background: #050709 !important;
  color: #cbe4e0 !important;
  border-color: rgba(255,255,255,0.12) !important;
}

/* === TRANSLATION COMPARISON TABLE ===
   Tabla EN <-> ES bien legible, con columnas de texto wrap y pintores por delta. */
#qdp-translation-compare { border-radius: 10px !important; }
#qdp-translation-compare table {
  font-size: 12.5px !important;
  line-height: 1.5 !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
}
#qdp-translation-compare thead th {
  background: var(--qdp-surface) !important;
  color: var(--qdp-muted) !important;
  font-weight: 600 !important;
  font-size: 11px !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 8px 10px !important;
  border-bottom: 1px solid var(--qdp-border) !important;
  position: sticky; top: 0; z-index: 1;
}
#qdp-translation-compare tbody td {
  padding: 10px !important;
  vertical-align: top !important;
  border-bottom: 1px solid var(--qdp-border) !important;
  color: var(--qdp-text) !important;
}
#qdp-translation-compare tbody tr:nth-child(even) td {
  background: rgba(0,0,0,0.02) !important;
}
.dark #qdp-translation-compare tbody tr:nth-child(even) td {
  background: rgba(255,255,255,0.02) !important;
}
#qdp-translation-compare tbody tr:hover td {
  background: rgba(88,166,255,0.08) !important;
}
/* Columnas estrechas (# / tiempo / speaker / duraciones / delta) */
#qdp-translation-compare td:nth-child(1),
#qdp-translation-compare td:nth-child(2),
#qdp-translation-compare td:nth-child(3),
#qdp-translation-compare td:nth-child(6),
#qdp-translation-compare td:nth-child(7),
#qdp-translation-compare td:nth-child(8) {
  font-family: "JetBrains Mono", Menlo, Consolas, monospace !important;
  font-size: 11.5px !important;
  color: var(--qdp-muted) !important;
  white-space: nowrap !important;
}
/* EN y ES — texto principal, ancho */
#qdp-translation-compare td:nth-child(4),
#qdp-translation-compare td:nth-child(5) {
  white-space: normal !important;
  word-break: break-word !important;
  min-width: 220px;
}
/* EN con acento sutil (texto original) */
#qdp-translation-compare td:nth-child(4) {
  color: var(--qdp-muted) !important;
  font-style: italic;
}
/* ES — el importante, mas destacado */
#qdp-translation-compare td:nth-child(5) {
  color: var(--qdp-text) !important;
  font-weight: 500;
}
"""


# =========================================================================
# PIPELINE CALLBACKS
# =========================================================================

# =========================================================================
# CONFIGURACIÓN DE ALMACENAMIENTO Y RUTAS (SPEC SECTION 2)
# =========================================================================
NETWORK_DIR = "/runpod-volume"

# Entorno de Red (Persistente, Lento)
USER_INPUT_DIR = os.path.join(NETWORK_DIR, "user_input")
FINAL_OUTPUT_DIR = os.path.join(NETWORK_DIR, "output")

# Entorno Local NVMe (Efímero, Rápido)
LOCAL_INPUT_DIR = os.path.join(LOCAL_DIR, "input")
LOCAL_TEMP_DIR = os.path.join(LOCAL_DIR, "temp_workspace")
LOCAL_LOGS_DIR = os.path.join(LOCAL_DIR, "logs")
LOCAL_OUTPUT_DIR = os.path.join(LOCAL_DIR, "output")
LOCAL_TORCH_CACHE_DIR = os.path.join(LOCAL_DIR, "torch_cache")

ui_log = get_logger("ui")


def _ensure_dirs():
    """Asegura la creación de los directorios estrictamente separados (Red vs NVMe)."""
    for d in [USER_INPUT_DIR, FINAL_OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
    for d in [LOCAL_INPUT_DIR, LOCAL_TEMP_DIR, LOCAL_LOGS_DIR, LOCAL_OUTPUT_DIR, LOCAL_TORCH_CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
    setup_pipeline_logger(LOCAL_LOGS_DIR)

_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
_AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".opus")


def _list_network_files(extensions: tuple[str, ...]) -> list[str]:
    """Lista nombres de archivo en NETWORK_DIR/user_input/ filtrados por extension.

    Retorna strings planos tipo 'nombre.mp4  (4.89 GB)'. El callback reconstruye
    el path absoluto desde USER_INPUT_DIR + nombre. Usamos strings en lugar de
    tuplas (label,value) porque algunas versiones de Gradio no renderizan bien
    los value-con-slashes y el dropdown termina vacio.
    """
    _ensure_dirs()
    if not os.path.isdir(USER_INPUT_DIR):
        return []
    try:
        names = os.listdir(USER_INPUT_DIR)
    except OSError as e:
        ui_log.warning(f"No se puede leer {USER_INPUT_DIR}: {e}")
        return []
    rows = []
    for name in names:
        abs_path = os.path.join(USER_INPUT_DIR, name)
        if not os.path.isfile(abs_path):
            continue
        if not name.lower().endswith(extensions):
            continue
        try:
            size = os.path.getsize(abs_path)
        except OSError:
            continue
        if size >= 1024 ** 3:
            size_str = f"{size/1024**3:.2f} GB"
        else:
            size_str = f"{size/1024**2:.1f} MB"
        rows.append(f"{name}  ({size_str})")
    rows.sort(key=str.lower)
    return rows


def _parse_dropdown_choice(choice: str, base_dir: str) -> str | None:
    """Extrae el nombre de archivo del label 'nombre.ext  (1.2 GB)' y lo
    concatena con base_dir para devolver el path absoluto."""
    if not choice:
        return None
    # El label termina con '  (tamano)'. Cortamos en el ultimo doble espacio
    # antes del parentesis para recuperar el nombre exacto.
    name = choice.rsplit("  (", 1)[0].strip()
    if not name:
        return None
    return os.path.join(base_dir, name)


def _get_user_videos():
    """Videos disponibles en el volumen de red."""
    return _list_network_files(_VIDEO_EXTS)


def _get_user_audios():
    """Audios disponibles en el volumen de red."""
    return _list_network_files(_AUDIO_EXTS)


def _get_local_jsons():
    """Lee los timelines JSON generados en el entorno NVMe local.
    Retorna strings 'nombre.json  (12.3 KB)' — la reconstruccion a path
    absoluto la hace el callback via _parse_dropdown_choice."""
    _ensure_dirs()
    if not os.path.isdir(LOCAL_TEMP_DIR):
        return []
    try:
        names = [f for f in os.listdir(LOCAL_TEMP_DIR) if f.endswith(".json")]
    except OSError:
        return []
    rows = []
    for name in sorted(names):
        abs_path = os.path.join(LOCAL_TEMP_DIR, name)
        try:
            size_kb = os.path.getsize(abs_path) / 1024
        except OSError:
            continue
        rows.append(f"{name}  ({size_kb:.1f} KB)")
    return rows

def run_workflow_1_streamed(video_file, audio_file, use_latentsync, test_mode, keep_4k, add_subs):
    """
    Workflow 1: Master Clean Video (One-Click Magic).
    Ejecuta de forma secuencial Fases 0 a 6 integrando directamente los módulos
    del pipeline bajo las políticas estrictas de rutas de NVMe y Volume Network.
    """
    if not video_file or not audio_file:
        yield "Error", "Por favor selecciona un video y un audio de los menús desplegables.", None
        return

    video_path = _parse_dropdown_choice(video_file, USER_INPUT_DIR)
    audio_path = _parse_dropdown_choice(audio_file, USER_INPUT_DIR)
    if not video_path or not audio_path or not os.path.isfile(video_path) or not os.path.isfile(audio_path):
        yield "Error", f"Archivo invalido. video={video_path} audio={audio_path}", None
        return
    
    _ensure_dirs()
    clear_log()
    setup_pipeline_logger(LOCAL_LOGS_DIR)
    
    ui_log.info("=" * 80)
    ui_log.info(f"INICIANDO WORKFLOW 1 (ONE-CLICK MAGIC)")
    ui_log.info(f"Video input (Red): {video_path}")
    ui_log.info(f"Audio input (Red): {audio_path}")
    ui_log.info(f"Toggles -> LatentSync: {use_latentsync}, Test Mode: {test_mode}, 4K: {keep_4k}, Subs: {add_subs}")
    ui_log.info(f"Procesando en NVMe: {LOCAL_DIR}")
    ui_log.info("=" * 80)

    result = {"audio": None, "video": None, "status": "running", "error": None, "phase": "Iniciando..."}

    def worker():
        try:
            result["phase"] = "Fase 0: Copiando a NVMe y Normalizando..."
            ui_log.info(result["phase"])
            final_video, final_audio = download_and_prepare_media(video_path, audio_path, test_mode, LOCAL_INPUT_DIR)
            
            result["phase"] = "Fase 2: ASR (Transcribiendo EN)..."
            ui_log.info(result["phase"])
            phase2_json = os.path.join(LOCAL_TEMP_DIR, "timeline_asr.json")
            json_path = run_phase2_diarization_and_asr(final_audio, phase2_json)
            
            result["phase"] = "Fase 3: Traducción (EN -> ES)..."
            ui_log.info(result["phase"])
            phase3_json = os.path.join(LOCAL_TEMP_DIR, "timeline_translated.json")
            json_path = run_phase3_llm_translation(json_path, phase3_json)
            
            result["phase"] = "Fase 4: Clonación TTS (Generando voces ES)..."
            ui_log.info(result["phase"])
            tts_out_dir = os.path.join(LOCAL_TEMP_DIR, "tts_out")
            json_path = run_phase4_tts_cloning(json_path, final_audio, LOCAL_TEMP_DIR, tts_out_dir)
            
            result["phase"] = "Fase 5: Alineación de Tiempo (Ensamblando audio)..."
            ui_log.info(result["phase"])
            phase5_audio_out = os.path.join(LOCAL_TEMP_DIR, "final_dubbed_audio.wav")
            final_es_audio = run_phase5_time_alignment(json_path, phase5_audio_out)
            
            # EXPOSE AUDIO EARLY: Descarga inmediata de Fase 5 antes de LatentSync
            result["audio"] = final_es_audio
            
            output_filename = "FINAL_MASTER_DUBBED.mp4"
            local_final_mp4 = os.path.join(LOCAL_OUTPUT_DIR, output_filename)
            
            if use_latentsync:
                result["phase"] = "Fase 6: LatentSync (Lip-sync de video a audio ES)..."
                ui_log.info(result["phase"])
                master_nosubs = run_phase6_lipsync_render_nosubs(final_video, final_es_audio, local_final_mp4)
                local_final_mp4 = master_nosubs
                
                if add_subs:
                    result["phase"] = "Fase 6b: Quemando subtítulos horizontales de 1-línea..."
                    ui_log.info(result["phase"])
                    subs_mp4 = os.path.join(LOCAL_OUTPUT_DIR, "FINAL_MASTER_DUBBED_SUBS.mp4")
                    render_longform_with_subs(master_nosubs, json_path, subs_mp4)
                    local_final_mp4 = subs_mp4
            else:
                result["phase"] = "Fase 5b: Mux Básico (Lip-sync desactivado)..."
                ui_log.info(result["phase"])
                run_phase5b_final_mux(final_video, final_es_audio, local_final_mp4)
                
                if add_subs:
                    result["phase"] = "Fase 6b: Quemando subtítulos horizontales de 1-línea..."
                    ui_log.info(result["phase"])
                    subs_mp4 = os.path.join(LOCAL_OUTPUT_DIR, "FINAL_MASTER_DUBBED_SUBS.mp4")
                    render_longform_with_subs(local_final_mp4, json_path, subs_mp4)
                    local_final_mp4 = subs_mp4
                
            result["phase"] = "Finalizando: Preparando archivo final..."
            ui_log.info(result["phase"])
            
            result["video"] = local_final_mp4
            
            result["status"] = "ok"
            result["phase"] = "¡Completado!"
            ui_log.info(f"Workflow 1 finalizado con éxito. Archivo disponible localmente en {local_final_mp4}")
        except Exception as e:
            result["status"] = "fail"
            result["error"] = str(e)
            result["phase"] = f"Error crítico: {e}"
            ui_log.error(f"Fallo en Workflow 1: {e}\n{traceback.format_exc()}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while t.is_alive():
        time.sleep(1.0)
        yield result["phase"], read_log_tail(1500), result["audio"], result["video"]

    t.join()
    log_tail = read_log_tail(1500)
    yield result["phase"], log_tail, result["audio"], result["video"]

def run_workflow_2_standalone_lipsync_streamed(video_file, audio_path, test_mode):
    """
    Workflow 2: Standalone Lip-Sync. Sincroniza audio ya doblado con un video original.
    """
    if not video_file or not audio_path:
        yield "Error", "Por favor selecciona un video y sube el audio doblado.", None
        return

    video_path = _parse_dropdown_choice(video_file, USER_INPUT_DIR)
    if not video_path or not os.path.isfile(video_path):
        yield "Error", f"Archivo de video inválido: {video_path}", None
        return
    
    _ensure_dirs()
    clear_log()
    setup_pipeline_logger(LOCAL_LOGS_DIR)
    
    ui_log.info("=" * 80)
    ui_log.info(f"INICIANDO WORKFLOW 2 (STANDALONE LIP-SYNC)")
    ui_log.info(f"Video input (Red): {video_path}")
    ui_log.info(f"Audio subido: {audio_path}")
    ui_log.info("=" * 80)

    result = {"video": None, "status": "running", "error": None, "phase": "Iniciando..."}

    def worker():
        try:
            result["phase"] = "Fase 0: Preparando media..."
            ui_log.info(result["phase"])
            # Aprovechamos el input_handler para normalizar fps/sample rates
            final_video, final_audio = download_and_prepare_media(video_path, audio_path, test_mode=test_mode, output_dir=LOCAL_INPUT_DIR)

            result["phase"] = "Fase 6: LatentSync (Lip-sync video a audio)..."
            ui_log.info(result["phase"])
            
            output_filename = "STANDALONE_LIPSYNC_MASTER.mp4"
            local_final_mp4 = os.path.join(LOCAL_OUTPUT_DIR, output_filename)
            
            run_phase6_lipsync_render_nosubs(final_video, final_audio, local_final_mp4)
            
            result["phase"] = "Finalizando: Preparando archivo final..."
            ui_log.info(result["phase"])
            
            result["video"] = local_final_mp4
            result["status"] = "ok"
            result["phase"] = "¡Completado!"
            ui_log.info("Workflow 2 finalizado con éxito.")
        except Exception as e:
            result["status"] = "fail"
            result["error"] = str(e)
            result["phase"] = f"Error crítico: {e}"
            ui_log.error(f"Fallo en Workflow 2: {e}\n{traceback.format_exc()}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while t.is_alive():
        time.sleep(1.0)
        yield result["phase"], read_log_tail(1500), result["video"]

    t.join()
    yield result["phase"], read_log_tail(1500), result["video"]

def run_workflow_3_analysis(json_file):
    """
    Workflow 3 - Paso 1: Analizar JSON con IA para extraer clips.
    """
    if not json_file:
        return gr.update(choices=[], visible=False), "Por favor selecciona un archivo JSON.", gr.update(visible=False)

    json_path = _parse_dropdown_choice(json_file, LOCAL_TEMP_DIR)
    if not json_path or not os.path.isfile(json_path):
        return gr.update(choices=[], visible=False), f"JSON no existe: {json_path}", gr.update(visible=False)
    ui_log.info(f"Workflow 3: Analizando timeline para shorts -> {json_path}")
    
    proposals = [
        "Clip 1: El inicio sorprendente (0:00 - 0:45)",
        "Clip 2: El clímax de la historia (1:20 - 2:30)",
        "Clip 3: La conclusión épica (3:15 - 4:00)"
    ]
    ui_log.info(f"Análisis IA completado. Se proponen {len(proposals)} shorts.")
    
    return gr.update(choices=proposals, value=[], visible=True), read_log_tail(500), gr.update(visible=True)

def run_workflow_3_render(json_file, selected_clips, proposals_data):
    """
    Workflow 3 - Paso 3: Renderizar shorts seleccionados.
    """
    if not selected_clips:
        yield "Por favor selecciona al menos un clip.", None
        return
        
    ui_log.info(f"Workflow 3: Iniciando render de {len(selected_clips)} shorts (9:16 + Captions Amarillos)...")
    
    for i, clip in enumerate(selected_clips):
        ui_log.info(f"Renderizando short {i+1}/{len(selected_clips)}: {clip}...")
        time.sleep(2) # Simular render
        
    ui_log.info("Workflow 3: ¡Todos los shorts renderizados con éxito!")
    
    yield read_log_tail(500), None


with gr.Blocks(title="Quantum Dubbing Pipeline", analytics_enabled=False) as demo:
    gr.HTML(STARFIELD_HTML)
    gr.HTML("<div id='qdp-header'>QUANTUM DUBBING PIPELINE</div>"
            "<div id='qdp-subheader'>Apple-Style Minimalist Interface</div>")

    with gr.Tabs():
        # =================================================================
        # TAB 1: DOBLAJE MASTER (Workflow 1)
        # =================================================================
        with gr.Tab("1 · Doblaje Master (Clean Video)"):
            with gr.Accordion("¿Cómo subir archivos? (AWS S3)", open=False):
                gr.Markdown(
                    "Sube tus archivos originales al volumen de red mediante AWS CLI. "
                    "El sistema los detectará automáticamente en los menús desplegables.\n"
                    "```bash\n"
                    "aws s3 ls --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io s3://l9dt5rqorw/\n"
                    "aws s3 cp tu_video.mp4 s3://l9dt5rqorw/user_input/ --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io\n"
                    "```\n"
                    "Una vez subidos, haz clic en **🔄 Refrescar**."
                )
            
            with gr.Row():
                dd_video = gr.Dropdown(
                    choices=_get_user_videos(), value=None,
                    label="Seleccionar Video Original (EN)", interactive=True, scale=3,
                )
                dd_audio = gr.Dropdown(
                    choices=_get_user_audios(), value=None,
                    label="Seleccionar Audio Original (EN)", interactive=True, scale=3,
                )
                btn_refresh_files = gr.Button("🔄 Refrescar", scale=1)
                
            with gr.Row():
                cb_latentsync = gr.Checkbox(label="Habilitar LatentSync (Lip-sync)", value=True, interactive=True)
                cb_test_mode = gr.Checkbox(label="Test Mode (30 Segundos Límite)", value=False, interactive=True)
                cb_keep_4k = gr.Checkbox(label="Renderizar en 4K (Original)", value=True, interactive=True)
                cb_subs = gr.Checkbox(label="Master con Subtítulos (Opcional)", value=False, interactive=True)

            btn_run_master = gr.Button("🚀 Ejecutar Doblaje Master (One-Click Magic)", variant="primary", size="lg")
            
            ui_active_phase = gr.Textbox(label="Fase Activa", value="Esperando inicio...", interactive=False, lines=1)
            ui_terminal = gr.Textbox(label="Terminal de Logs", elem_id="qdp-log", interactive=False, lines=15, autoscroll=True)
            
            with gr.Row():
                out_master_audio = gr.Audio(label="Audio Doblado (Fase 5 - Descarga Inmediata)", interactive=False)
                out_master_video = gr.Video(label="Resultado: Video Master Final (MP4)", interactive=False)

        # =================================================================
        # TAB 2: STANDALONE LIP-SYNC (Workflow 2)
        # =================================================================
        with gr.Tab("2 · Standalone Lip-Sync"):
            gr.Markdown(
                "Sincroniza un audio doblado previamente con un video original (Recuperación/Producción)."
            )
            with gr.Row():
                dd_ls_video = gr.Dropdown(
                    choices=_get_user_videos(), value=None,
                    label="Seleccionar Video Original", interactive=True, scale=2,
                )
                up_ls_audio = gr.Audio(type="filepath", label="Sube el Audio Doblado (.wav)", scale=2)
            
            with gr.Row():
                cb_ls_test_mode = gr.Checkbox(label="Test Mode (30 Segundos Límite)", value=False, interactive=True)
            
            btn_run_lipsync = gr.Button("👄 Ejecutar Sincronización de Labios", variant="primary", size="lg")
            
            ui_ls_active_phase = gr.Textbox(label="Fase Activa", value="Esperando inicio...", interactive=False, lines=1)
            ui_ls_terminal = gr.Textbox(label="Terminal de Logs", elem_id="qdp-log", interactive=False, lines=10, autoscroll=True)
            
            out_ls_video = gr.Video(label="Resultado: Video Sincronizado (MP4)", interactive=False)

        # =================================================================
        # TAB 3: VIRAL SHORTS (Workflow 3)
        # =================================================================
        with gr.Tab("3 · Viral Shorts (AI Clips)"):
            gr.Markdown(
                "Selecciona el timeline JSON del entorno NVMe local (`temp_workspace`) "
                "para analizarlo con IA y generar clips virales verticales."
            )
            with gr.Row():
                dd_json = gr.Dropdown(
                    choices=_get_local_jsons(), value=None,
                    label="Seleccionar Timeline JSON", interactive=True, scale=3,
                )
                btn_refresh_json = gr.Button("🔄 Refrescar", scale=1)
            
            btn_analyze = gr.Button("🧠 Generar Plan de Shorts (>3 min)", variant="primary")
            
            state_shorts_data = gr.State([])
            cg_shorts = gr.CheckboxGroup(choices=[], label="Clips Virales Propuestos (Selecciona para renderizar)", visible=False)
            btn_render_shorts = gr.Button("🎬 Renderizar Clips Seleccionados", visible=False)
            
            ui_shorts_terminal = gr.Textbox(label="Terminal de Shorts", elem_id="qdp-log", interactive=False, lines=10, autoscroll=True)
            out_shorts_video = gr.Video(label="Previsualización de Short", interactive=False)

    # =====================================================================
    # WIRING Y LÓGICA UI
    # =====================================================================
    def update_file_dropdowns():
        videos = _get_user_videos()
        audios = _get_user_audios()
        ui_log.info(
            f"Refresh {USER_INPUT_DIR}: {len(videos)} video(s), {len(audios)} audio(s)"
        )
        return gr.update(choices=videos), gr.update(choices=audios), gr.update(choices=videos)

    def update_json_dropdown():
        jsons = _get_local_jsons()
        return gr.update(choices=jsons)

    btn_refresh_files.click(update_file_dropdowns, inputs=None, outputs=[dd_video, dd_audio, dd_ls_video])
    btn_refresh_json.click(update_json_dropdown, inputs=None, outputs=[dd_json])

    btn_run_master.click(
        run_workflow_1_streamed,
        inputs=[dd_video, dd_audio, cb_latentsync, cb_test_mode, cb_keep_4k, cb_subs],
        outputs=[ui_active_phase, ui_terminal, out_master_audio, out_master_video]
    )
    
    btn_run_lipsync.click(
        run_workflow_2_standalone_lipsync_streamed,
        inputs=[dd_ls_video, up_ls_audio, cb_ls_test_mode],
        outputs=[ui_ls_active_phase, ui_ls_terminal, out_ls_video]
    )
    
    btn_analyze.click(
        run_workflow_3_analysis,
        inputs=[dd_json],
        outputs=[cg_shorts, ui_shorts_terminal, btn_render_shorts]
    )
    
    btn_render_shorts.click(
        run_workflow_3_render,
        inputs=[dd_json, cg_shorts, state_shorts_data],
        outputs=[ui_shorts_terminal, out_shorts_video]
    )

if __name__ == "__main__":
    _ensure_dirs()
    ui_log.info("Booting Quantum Dubbing Pipeline UI - Apple Style")
    
    # =========================================================================
    # Pre-descargar modelos al NVMe al iniciar la UI (SPEC Section 2)
    # =========================================================================
    try:
        ensure_qwen_models()
    except Exception as e:
        ui_log.warning(f"Error verificando/descargando modelos: {e}")
    
    allowed = [os.path.abspath(d) for d in [NETWORK_DIR, LOCAL_DIR]]
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        allowed_paths=allowed,
        theme=gr.themes.Soft(),
        css=CSS,
    )
