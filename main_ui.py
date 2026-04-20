import gradio as gr
import os
import json
import csv
import shutil
import threading
import time
import traceback

from src.logger import setup_pipeline_logger, get_logger, read_log_tail, clear_log
from src.ensure_models import ensure_qwen_models
from src.input_handler import download_and_prepare_media
from src.phase1_audio_separation import run_phase1_audio_separation
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
.gr-group, .gr-panel, .gr-box, .block {
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
input, textarea, .gr-textbox textarea, .gr-textbox input {
  background: transparent !important;
  color: var(--qdp-text) !important;
  border-color: var(--qdp-border) !important;
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

WORKSPACES = ["input", "temp_workspace", "output"]

ui_log = get_logger("ui")


def _ensure_dirs():
    """Asegura APP_DATA_DIR como cwd (para que input/temp/output aterricen en volumen)."""
    data_dir = os.environ.get("APP_DATA_DIR")
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        os.chdir(data_dir)
    for d in WORKSPACES:
        os.makedirs(d, exist_ok=True)
    os.makedirs("user_input", exist_ok=True)
    setup_pipeline_logger("logs")


# -------------------------------------------------------------------------
# Pipeline de doblaje como generator con thread en background
# -------------------------------------------------------------------------

PHASE_DEFINITIONS = [
    ("FASE 1", "Separacion Demucs", 0.10, 0.22),
    ("FASE 2", "ASR + Segmentacion", 0.22, 0.48),
    ("FASE 3", "Traduccion Gemini", 0.48, 0.58),
    ("FASE 4", "Clonacion TTS Qwen3", 0.58, 0.88),
    ("FASE 5", "Alineamiento + Mezcla", 0.88, 0.96),
    ("FASE 5b", "Mux video+audio final", 0.96, 1.00),
]


def _detect_phase_from_log(log_text: str) -> tuple[str, float]:
    """Detecta fase actual a partir del log tail. Retorna (descripcion, fraccion [0-1])."""
    if not log_text:
        return "Iniciando...", 0.0
    last_phase = ("Iniciando...", 0.05)
    for tag, desc, frac_start, frac_end in PHASE_DEFINITIONS:
        if f"### {tag}" in log_text:
            last_phase = (f"{tag}: {desc}", frac_start)
            if f"### {tag}" in log_text and f"— OK" in log_text.split(f"### {tag}")[-1]:
                last_phase = (f"{tag} OK: {desc}", frac_end)
    return last_phase


def _list_workspace_files() -> list[list[str]]:
    """Lista archivos generados en temp_workspace/, output/ y equivalentes absolutos
    (APP_DATA_DIR/output, /runpod-volume/output, etc.) para que siempre se vean,
    independiente del cwd actual."""
    rows = []
    seen = set()
    bases = ["temp_workspace", "output", "input", "logs"]
    data_dir = os.environ.get("APP_DATA_DIR", "")
    abs_bases = []
    if data_dir:
        for d in ["temp_workspace", "output", "input", "logs"]:
            abs_bases.append(os.path.join(data_dir, d))
    # Siempre intenta /runpod-volume por si APP_DATA_DIR no esta seteado
    for d in ["temp_workspace", "output", "input", "logs"]:
        abs_bases.append(f"/runpod-volume/{d}")

    for base in bases + abs_bases:
        if not base or not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                p = os.path.join(root, f)
                try:
                    real = os.path.realpath(p)
                except OSError:
                    real = p
                if real in seen:
                    continue
                seen.add(real)
                try:
                    size_mb = os.path.getsize(p) / 1024 ** 2
                    mtime = time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(os.path.getmtime(p)))
                    rows.append([p, f"{size_mb:.2f} MB", mtime])
                except OSError:
                    continue
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def _audio_dur_fast(path: str) -> float:
    """Devuelve la duracion de un audio leyendo solo el header (sin decodificar)."""
    if not path or not os.path.exists(path):
        return 0.0
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.duration)
    except Exception:
        return 0.0


def _build_translation_comparison(json_path: str) -> list[list[str]]:
    """Construye tabla de comparacion EN <-> ES a partir del timeline JSON.

    Columnas: #, Tiempo, Speaker, EN (original), ES (traduccion), dur EN, dur TTS ES, delta.
    El JSON puede ser el de fase 3 (solo text_es) o el de fase 4 (incluye cloned_audio_path).
    """
    if not json_path or not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            timeline = json.load(f)
    except Exception as e:
        ui_log.warning(f"_build_translation_comparison: no pude leer {json_path}: {e}")
        return []
    rows = []
    for seg in timeline:
        seg_id = seg.get("segment_id", "?")
        speaker = seg.get("speaker", "?")
        try:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
        except (TypeError, ValueError):
            start = end = 0.0
        dur_en = max(0.0, end - start)
        text_en = (seg.get("text_en") or "").strip()
        text_es = (seg.get("text_es") or "").strip()

        cloned = seg.get("cloned_audio_path") or ""
        dur_es = _audio_dur_fast(cloned) if cloned else 0.0
        delta = dur_es - dur_en if dur_es > 0 else 0.0
        delta_str = f"{delta:+.2f}s" if dur_es > 0 else "—"

        rows.append([
            str(seg_id),
            f"{start:.2f} – {end:.2f}",
            speaker,
            text_en,
            text_es,
            f"{dur_en:.2f}s",
            f"{dur_es:.2f}s" if dur_es > 0 else "—",
            delta_str,
        ])
    return rows


def _upload_to_path(file_obj) -> str:
    """Convierte un gr.File upload a path local persistente en user_input/."""
    if file_obj is None:
        return ""
    _ensure_dirs()
    src = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    base = os.path.basename(src)
    dest = os.path.join("user_input", base)
    try:
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copyfile(src, dest)
    except OSError as e:
        ui_log.error(f"_upload_to_path fallo copiando {src} -> {dest}: {e}")
        return src
    ui_log.info(f"Upload recibido -> {dest}")
    return dest


def _resolve_input(textbox_val: str, upload_val) -> str:
    """Decide que path usar: upload gana sobre textbox si hay upload."""
    if upload_val is not None:
        return _upload_to_path(upload_val)
    return (textbox_val or "").strip()


def prepare_data(v_spec, a_spec, v_up, a_up, is_test):
    """Resuelve video+audio (URL, path local o upload) y corre input_handler."""
    _ensure_dirs()
    video_spec = _resolve_input(v_spec, v_up)
    audio_spec = _resolve_input(a_spec, a_up)
    if not video_spec or not audio_spec:
        return ("", "",
                "Faltan video y/o audio. Pega URL de Drive, path local o sube archivo.",
                _list_workspace_files(), read_log_tail(300))
    ui_log.info(f"prepare_data: video={video_spec} audio={audio_spec} test={is_test}")
    try:
        final_video, final_audio = download_and_prepare_media(video_spec, audio_spec, is_test)
    except Exception as e:
        ui_log.error(f"prepare_data FAIL: {e}")
        ui_log.error(traceback.format_exc())
        return ("", "", f"❌ FAIL descarga: {e}",
                _list_workspace_files(), read_log_tail(400))
    mode = "TEST 30s" if is_test else "COMPLETO"
    return (final_video, final_audio,
            f"✅ Listo. Modo: {mode}. Video: {final_video}. Audio: {final_audio}",
            _list_workspace_files(), read_log_tail(400))


def _latest_comparison_json() -> str:
    """Devuelve el JSON mas avanzado disponible durante la corrida:
    prefiere p4 (con TTS) > p3 (con traduccion) > p2 (solo EN)."""
    for candidate in (
        "temp_workspace/p4/master_timeline_with_audio.json",
        "temp_workspace/p3_data.json",
        "temp_workspace/p2_data.json",
    ):
        if os.path.exists(candidate):
            return candidate
    return ""


def _run_dubbing_streamed(audio_path, video_path, progress_desc: str):
    """Generator que corre el pipeline en thread y yielda updates del UI cada segundo.

    Encadena: Fase 1 -> 2 -> 3 -> 4 -> 5 (audio doblado) -> 5b (mux final MP4).
    Al terminar, la UI recibe tanto el WAV doblado como el MP4 listo para descargar
    sin pasar por Tab 2. Si falta el video fuente, fase 5b se omite y el usuario
    solo recibe el WAV.
    """
    if not audio_path or not os.path.exists(audio_path):
        yield (None, None, None, "Audio origen no disponible. Sincroniza primero.",
               _list_workspace_files(), read_log_tail(400), 0.0, [])
        return

    _ensure_dirs()
    clear_log()
    setup_pipeline_logger("logs")
    ui_log.info("=" * 80)
    ui_log.info(f"NUEVA CORRIDA DE DOBLAJE ({progress_desc})")
    ui_log.info(f"Audio input: {audio_path}")
    ui_log.info(f"Video input: {video_path or '(no disponible — fase 5b se omitira)'}")
    ui_log.info("=" * 80)

    result = {
        "audio": None, "video": None, "json": None,
        "status": "running", "error": None,
    }

    def worker():
        try:
            v_path, a_path = run_phase1_audio_separation(audio_path, "temp_workspace/p1")
            json_path = run_phase2_diarization_and_asr(v_path, "temp_workspace/p2_data.json")
            json_path = run_phase3_llm_translation(json_path, "temp_workspace/p3_data.json")
            json_path = run_phase4_tts_cloning(json_path, v_path, "temp_workspace/p4", "temp_workspace/p4/out")
            final_audio = run_phase5_time_alignment(json_path, a_path, "output/final_audio_dubbed.wav")
            result["audio"] = final_audio
            result["json"] = json_path

            # Fase 5b: mux rapido video+audio -> MP4 descargable. Se corre
            # siempre que tengamos el video fuente (URL/upload ya resuelto por Fase 0).
            if video_path and os.path.exists(video_path):
                try:
                    final_video = run_phase5b_final_mux(
                        video_path=video_path,
                        audio_path=final_audio,
                        output_path="output/FINAL_DUBBED_VIDEO.mp4",
                    )
                    result["video"] = final_video
                except Exception as e:
                    # Mux es opcional: si falla, el audio ya esta listo igual
                    ui_log.error(f"Fase 5b mux fallo (no crash pipeline): {e}")
                    ui_log.error(traceback.format_exc())
            else:
                ui_log.warning(
                    f"Fase 5b omitida: no hay video fuente disponible ({video_path!r}). "
                    "Solo se entrega el WAV doblado."
                )

            result["status"] = "ok"
        except Exception as e:
            tb = traceback.format_exc()
            ui_log.error(f"PIPELINE FAIL: {type(e).__name__}: {e}")
            ui_log.error(tb)
            result["error"] = f"{type(e).__name__}: {e}"
            result["status"] = "fail"

    t = threading.Thread(target=worker, daemon=True, name="dubbing-pipeline")
    t.start()

    while t.is_alive():
        time.sleep(1.0)
        log_tail = read_log_tail(400)
        phase_desc, frac = _detect_phase_from_log(log_tail)
        # Comparacion progresiva: aparece en cuanto fase 2/3/4 produce JSON
        partial_compare = _build_translation_comparison(_latest_comparison_json())
        yield (None, None, None, f"⏳ {phase_desc}",
               _list_workspace_files(), log_tail, frac, partial_compare)

    t.join()
    log_tail = read_log_tail(600)
    files = _list_workspace_files()
    compare_rows = _build_translation_comparison(result["json"] or _latest_comparison_json())

    if result["status"] == "ok":
        if result["video"]:
            status_msg = (
                f"✅ Doblaje OK. MP4 listo: {result['video']} | "
                f"Audio: {result['audio']}"
            )
        else:
            status_msg = (
                f"✅ Doblaje OK (solo audio). Audio: {result['audio']} "
                "(video fuente no disponible para mux automatico)"
            )
        yield (result["video"], result["audio"], result["json"],
               status_msg, files, log_tail, 1.0, compare_rows)
    else:
        yield (None, None, None,
               f"❌ FAIL: {result['error']}",
               files, log_tail, 0.0, compare_rows)


def pipe_doblar_audio(audio_path, video_path):
    # Guard: si el audio preparado viene del modo TEST (Fase 0 lo recorto a 30s),
    # el doblaje "completo" va a correr sobre ese recorte y el resultado tambien
    # va a ser de 30s. Abortamos con un mensaje claro para que el usuario
    # vuelva a sincronizar con el checkbox "Modo prueba" DESMARCADO.
    if audio_path and os.path.basename(audio_path).startswith("test_"):
        msg = (
            "⚠️ El audio preparado esta en MODO PRUEBA (recortado a 30s en Fase 0). "
            "Doblaje completo corre el pipeline entero pero sobre lo que hay preparado — "
            "es decir, se doblarian solo esos 30s. "
            "Desmarca el checkbox 'Modo prueba' en el paso 1 y volve a clickear "
            "'Sincronizar / preparar media' para procesar el video completo."
        )
        ui_log.warning(msg)
        yield (None, None, None, msg, _list_workspace_files(), read_log_tail(400), 0.0, [])
        return
    yield from _run_dubbing_streamed(audio_path, video_path, "completo")


def pipe_doblar_audio_test30(audio_path, video_path):
    """Modo TEST: recorta 30s del audio Y del video antes de doblar.
    Asi el mux de Fase 5b sale coherente (ambos tracks de 30s)."""
    if not audio_path or not os.path.exists(audio_path):
        yield (None, None, None, "Audio origen no disponible.",
               _list_workspace_files(), read_log_tail(400), 0.0, [])
        return
    _ensure_dirs()
    setup_pipeline_logger("logs")
    ui_log.info("Recortando 30s de prueba (audio + video)")

    import ffmpeg
    trimmed_audio = "temp_workspace/test30_audio.wav"
    trimmed_video = "temp_workspace/test30_video.mp4" if video_path else None

    try:
        (ffmpeg
            .input(audio_path, t=30)
            .output(trimmed_audio, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True))
        ui_log.info(f"Recorte audio OK: {trimmed_audio}")

        # Video opcional: si no esta, igual procedemos con el dub solo audio
        if video_path and os.path.exists(video_path):
            (ffmpeg
                .input(video_path, t=30)
                .output(trimmed_video, c="copy")
                .overwrite_output()
                .run(quiet=True))
            ui_log.info(f"Recorte video OK: {trimmed_video}")
        else:
            trimmed_video = None
    except Exception as e:
        ui_log.error(f"Recorte 30s fallo: {e}")
        yield (None, None, None, f"Recorte 30s fallo: {e}",
               _list_workspace_files(), read_log_tail(400), 0.0, [])
        return

    yield from _run_dubbing_streamed(trimmed_audio, trimmed_video, "test 30s")


def clear_workspace():
    _ensure_dirs()
    ui_log.info("Limpiando workspace (input/, temp_workspace/, output/)")
    for d in WORKSPACES:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    clear_log()
    setup_pipeline_logger("logs")
    ui_log.info("Workspace purgado")
    return "Workspace purgado.", _list_workspace_files(), read_log_tail(100)


def refresh_logs_and_files():
    return _list_workspace_files(), read_log_tail(500)


def manual_download(path: str):
    """Trae CUALQUIER archivo por path absoluto al gr.File para descargar desde el navegador.
    No depende de la tabla ni del auto-populate. Funciona con /runpod-volume/, /app/, etc."""
    if not path or not path.strip():
        return None, "Pega una ruta absoluta."
    path = path.strip()
    if not os.path.isabs(path):
        return None, f"La ruta debe ser absoluta (empezar con /). Recibi: {path}"
    if not os.path.exists(path):
        # Intento listar el directorio padre para ayudar al usuario
        parent = os.path.dirname(path) or "/"
        hint = ""
        if os.path.isdir(parent):
            try:
                kids = sorted(os.listdir(parent))[:20]
                hint = f" | Archivos en {parent}: {', '.join(kids) if kids else '(vacio)'}"
            except OSError:
                pass
        return None, f"No existe: {path}{hint}"
    if os.path.isdir(path):
        try:
            kids = sorted(os.listdir(path))[:30]
            return None, f"{path} es un directorio. Contiene: {', '.join(kids) if kids else '(vacio)'}"
        except OSError as e:
            return None, f"Error leyendo directorio: {e}"
    try:
        size_mb = os.path.getsize(path) / 1024 ** 2
        return path, f"OK — {path} ({size_mb:.2f} MB). Click en el archivo para descargar."
    except OSError as e:
        return None, f"Error: {e}"


def preview_selected_file(file_path: str):
    """Dado un path seleccionado en la lista, decide que preview mostrar."""
    if not file_path or not os.path.exists(file_path):
        return None, None, "(selecciona un archivo en la tabla)", None
    ext = os.path.splitext(file_path)[1].lower()
    audio_p, video_p, text_p, dl_p = None, None, None, file_path

    if ext in [".wav", ".mp3", ".flac", ".ogg"]:
        audio_p = file_path
        text_p = f"Audio: {file_path}\nTamano: {os.path.getsize(file_path)/1024**2:.2f} MB"
    elif ext in [".mp4", ".mov", ".mkv", ".webm"]:
        video_p = file_path
        text_p = f"Video: {file_path}\nTamano: {os.path.getsize(file_path)/1024**2:.2f} MB"
    elif ext in [".json", ".txt", ".log", ".ass"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if len(content) > 20000:
                content = content[:20000] + f"\n\n... (truncado, total {len(content)} chars)"
            text_p = content
        except OSError as e:
            text_p = f"Error leyendo {file_path}: {e}"
    else:
        text_p = f"{file_path}\nTipo no previsualizable. Usa el boton descargar."

    return audio_p, video_p, text_p, dl_p


def on_files_df_select(evt: gr.SelectData, files_data):
    try:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if files_data is None:
            return None, None, "(sin datos)", None
        if hasattr(files_data, "values"):
            rows = files_data.values.tolist()
        else:
            rows = files_data
        if row_idx < 0 or row_idx >= len(rows):
            return None, None, "(indice fuera de rango)", None
        file_path = rows[row_idx][0]
        return preview_selected_file(file_path)
    except (AttributeError, IndexError, TypeError) as e:
        return None, None, f"Error seleccion: {e}", None


# -------------------------------------------------------------------------
# Video render
# -------------------------------------------------------------------------

def _graded_video(video_path, nitidez, brillo, contraste):
    """Aplica color grading con ffmpeg. Encode con NVENC (h264_nvenc) si la env
    UI_VCODEC lo indica (default en A100). En CPU-only exportar UI_VCODEC=libx264."""
    import ffmpeg
    graded = "temp_workspace/graded.mp4"
    vf = f"eq=brightness={brillo}:contrast={contraste},unsharp=5:5:{nitidez}:5:5:0.0"

    vcodec = os.environ.get("UI_VCODEC", "h264_nvenc")
    if vcodec.endswith("_nvenc"):
        # NVENC: preset p1..p7 + cq (constant quality). p4 ~ libx264 medium, p2 es mas rapido.
        out_kwargs = {
            "vf": vf,
            "vcodec": vcodec,
            "acodec": "copy",
            "preset": os.environ.get("UI_NVENC_PRESET", "p4"),
            "cq": int(os.environ.get("UI_NVENC_CQ", "20")),
            "b:v": "0",
        }
    else:
        # libx264 CPU fallback
        out_kwargs = {
            "vf": vf,
            "vcodec": vcodec,
            "acodec": "copy",
            "preset": "medium",
            "crf": 20,
        }

    (ffmpeg
        .input(video_path)
        .output(graded, **out_kwargs)
        .overwrite_output()
        .run(quiet=True))
    return graded


def pipe_generar_video(v_spec, a_spec, j_spec, v_up, a_up, j_up, nitidez, brillo, contraste,
                      lipsync_steps, lipsync_guidance):
    """
    Render longform: produce DOS archivos:
      - output/master_lipsynced_nosubs.mp4  (insumo para shorts, sin subs, YA con LatentSync)
      - output/FINAL_DUBBED_VIDEO.mp4       (longform publicable con subs pro)
    Devuelve también el path del master para que Tab 3 lo autoconsuma.

    LatentSync es OBLIGATORIO. La primera vez el wrapper descarga los pesos
    (~2 GB) al volumen automaticamente. Si falla → RuntimeError arriba.
    """
    _ensure_dirs()
    setup_pipeline_logger("logs")
    video_path = _resolve_input(v_spec, v_up)
    audio_final = _resolve_input(a_spec, a_up)
    json_path = _resolve_input(j_spec, j_up)
    ui_log.info(f"pipe_generar_video: video={video_path} audio={audio_final} json={json_path} steps={lipsync_steps} gs={lipsync_guidance}")
    if not (video_path and audio_final and json_path):
        return None, None, "Faltan entradas (video, audio doblado o JSON).", _list_workspace_files(), read_log_tail(300)
    for label, p in [("video", video_path), ("audio", audio_final), ("json", json_path)]:
        if not os.path.exists(p):
            return None, None, f"❌ {label} no existe: {p}", _list_workspace_files(), read_log_tail(300)
    try:
        graded = _graded_video(video_path, nitidez, brillo, contraste)
        ui_log.info(f"Video graded: {graded}")
        master_path = "output/master_lipsynced_nosubs.mp4"
        run_phase6_lipsync_render_nosubs(
            graded, audio_final, master_path,
            lipsync_steps=int(lipsync_steps),
            lipsync_guidance=float(lipsync_guidance),
        )
        out_vid = render_longform_with_subs(master_path, json_path, "output/FINAL_DUBBED_VIDEO.mp4")
        return out_vid, master_path, "✅ Render longform completo + master LatentSync listo para shorts.", _list_workspace_files(), read_log_tail(500)
    except Exception as e:
        ui_log.error(f"pipe_generar_video FAIL: {e}")
        ui_log.error(traceback.format_exc())
        return None, None, f"❌ FAIL: {e}", _list_workspace_files(), read_log_tail(500)


def pipe_preview_video(v_spec, a_spec, j_spec, v_up, a_up, j_up, nitidez, brillo, contraste,
                      lipsync_steps, lipsync_guidance):
    """Preview 30s del longform (master LatentSync + subs pro, recortado)."""
    _ensure_dirs()
    setup_pipeline_logger("logs")
    video_path = _resolve_input(v_spec, v_up)
    audio_final = _resolve_input(a_spec, a_up)
    json_path = _resolve_input(j_spec, j_up)
    ui_log.info(f"pipe_preview_video 30s: video={video_path} audio={audio_final} json={json_path} steps={lipsync_steps} gs={lipsync_guidance}")
    if not (video_path and audio_final and json_path):
        return None, None, "Faltan entradas para preview.", _list_workspace_files(), read_log_tail(300)
    for label, p in [("video", video_path), ("audio", audio_final), ("json", json_path)]:
        if not os.path.exists(p):
            return None, None, f"❌ {label} no existe: {p}", _list_workspace_files(), read_log_tail(300)
    try:
        graded = _graded_video(video_path, nitidez, brillo, contraste)
        master_preview = "output/master_lipsynced_nosubs_PREVIEW.mp4"
        run_phase6_lipsync_render_nosubs(
            graded, audio_final, master_preview, duration_s=30.0,
            lipsync_steps=int(lipsync_steps),
            lipsync_guidance=float(lipsync_guidance),
        )
        out_vid = render_longform_with_subs(master_preview, json_path, "output/PREVIEW_30s.mp4", duration_s=30.0)
        return out_vid, master_preview, "✅ Preview 30s listo (LatentSync aplicado).", _list_workspace_files(), read_log_tail(500)
    except Exception as e:
        ui_log.error(f"pipe_preview_video FAIL: {e}")
        ui_log.error(traceback.format_exc())
        return None, None, f"❌ FAIL: {e}", _list_workspace_files(), read_log_tail(500)


# -------------------------------------------------------------------------
# AI SHORTS
# -------------------------------------------------------------------------

def _format_proposal_label(p, idx):
    dur = p["end"] - p["start"]
    m, s = divmod(int(dur), 60)
    return f"#{idx+1} [{m:02d}:{s:02d}] {p['title']}"


def pipe_analyze_shorts(j_spec, j_up, min_min, max_min):
    """
    Analiza timeline ES con Gemini.
    min_min / max_min vienen en MINUTOS desde los sliders UI.
    """
    _ensure_dirs()
    setup_pipeline_logger("logs")
    json_path = _resolve_input(j_spec, j_up)
    if not json_path or not os.path.exists(json_path):
        return gr.update(choices=[], value=[]), [], f"Falta JSON de timeline: {json_path!r}"
    try:
        min_s = float(min_min) * 60.0
        max_s = float(max_min) * 60.0
        if max_s <= min_s:
            return gr.update(choices=[], value=[]), [], "Duración máxima debe ser mayor que la mínima."
        proposals = analyze_timeline_for_shorts(json_path, min_duration_s=min_s, max_duration_s=max_s)
    except Exception as e:
        ui_log.error(f"analyze_timeline FAIL: {e}")
        ui_log.error(traceback.format_exc())
        return gr.update(choices=[], value=[]), [], f"❌ FAIL: {e}"
    if not proposals:
        return gr.update(choices=[], value=[]), [], "Gemini no devolvio propuestas validas."
    labels = [_format_proposal_label(p, i) for i, p in enumerate(proposals)]
    summary_lines = []
    for i, p in enumerate(proposals):
        summary_lines.append(f"{_format_proposal_label(p, i)}\n    hook: {p['hook']}\n    idea: {p['description']}")
    return gr.update(choices=labels, value=labels), proposals, "\n\n".join(summary_lines)


def _label_to_index(label):
    try:
        tag = label.split(" ", 1)[0]
        return int(tag.lstrip("#")) - 1
    except Exception:
        return None


def pipe_preview_thumbnail(m_spec, m_up, proposals_state, selected_labels):
    """
    Extrae un thumbnail JPG 9:16 del master para el primer short seleccionado.
    Rápido: solo 1 frame de ffmpeg, no renderiza video.
    """
    if not proposals_state or not selected_labels:
        return None, "Selecciona un short para generar thumbnail."
    master_path = _resolve_input(m_spec, m_up)
    if not master_path or not os.path.exists(master_path):
        return None, f"Falta master lip-sync: {master_path!r}"
    idx = _label_to_index(selected_labels[0])
    if idx is None or idx >= len(proposals_state):
        return None, "Propuesta no encontrada."
    p = proposals_state[idx]
    out_jpg = f"output/thumb_short_{idx+1}.jpg"
    try:
        generate_short_thumbnail(master_path, float(p["start"]), out_jpg)
    except Exception as e:
        ui_log.error(f"preview_thumbnail FAIL: {e}")
        return None, f"❌ FAIL: {e}"
    return out_jpg, f"Thumbnail short #{idx+1}: {p['title']}"


def pipe_preview_short(m_spec, j_spec, m_up, j_up, proposals_state, selected_labels,
                      nitidez, brillo, contraste, captions_wbw, chunk_words):
    """
    Preview 10s del primer short seleccionado, desde el master lip-sync.
    El master ya tiene LatentSync aplicado (viene de Tab 2); aqui solo
    recortamos + 9:16 + captions + color grading.
    """
    if not proposals_state or not selected_labels:
        return None, "Selecciona al menos un short para previsualizar."
    master_path = _resolve_input(m_spec, m_up)
    json_path = _resolve_input(j_spec, j_up)
    if not (master_path and json_path):
        return None, "Faltan master lip-sync o JSON."
    idx = _label_to_index(selected_labels[0])
    if idx is None or idx >= len(proposals_state):
        return None, "Propuesta no encontrada."
    p = proposals_state[idx]
    out = f"output/PREVIEW_SHORT_{idx+1}.mp4"
    try:
        render_short(
            source_video=master_path, json_path=json_path,
            start_s=float(p["start"]), end_s=float(p["end"]),
            output_path=out, sharpness=nitidez, brightness=brillo, contrast=contraste,
            preview_seconds=10.0,
            captions_word_by_word=bool(captions_wbw),
            chunk_words=int(chunk_words),
        )
    except Exception as e:
        ui_log.error(f"preview_short FAIL: {e}")
        ui_log.error(traceback.format_exc())
        return None, f"❌ FAIL: {e}"
    return out, f"Preview 10s del short #{idx+1}: {p['title']}"


def pipe_render_selected_shorts(m_spec, j_spec, m_up, j_up, proposals_state, selected_labels,
                                nitidez, brillo, contraste, captions_wbw, chunk_words):
    if not proposals_state:
        return [], "Primero analiza el timeline con IA."
    if not selected_labels:
        return [], "No seleccionaste ningun short."
    master_path = _resolve_input(m_spec, m_up)
    json_path = _resolve_input(j_spec, j_up)
    if not (master_path and json_path):
        return [], "Faltan master lip-sync o JSON."
    outputs = []
    for label in selected_labels:
        idx = _label_to_index(label)
        if idx is None or idx >= len(proposals_state):
            continue
        p = proposals_state[idx]
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in p["title"])[:40]
        out = f"output/SHORT_{idx+1}_{safe_title}.mp4"
        try:
            render_short(
                source_video=master_path, json_path=json_path,
                start_s=float(p["start"]), end_s=float(p["end"]),
                output_path=out, sharpness=nitidez, brightness=brillo, contrast=contraste,
                captions_word_by_word=bool(captions_wbw),
                chunk_words=int(chunk_words),
            )
            outputs.append(out)
        except Exception as e:
            ui_log.error(f"render_short #{idx+1} FAIL: {e}")
            ui_log.error(traceback.format_exc())
    return outputs, f"{len(outputs)} shorts generados."


# =========================================================================
# UI
# =========================================================================

def _input_row(label_prefix: str, tb_placeholder: str, file_types: list):
    """Helper: genera un par (textbox path/URL, file uploader) con change handler."""
    tb = gr.Textbox(
        label=f"{label_prefix} — URL Drive o path local",
        placeholder=tb_placeholder,
        lines=1,
    )
    up = gr.File(label=f"O subir {label_prefix.lower()}", file_types=file_types)
    up.change(lambda f: f.name if f else gr.update(), inputs=[up], outputs=[tb])
    return tb, up


with gr.Blocks(theme=gr.themes.Monochrome(), css=CSS, title="Quantum Dubbing Pipeline") as demo:
    gr.HTML(STARFIELD_HTML)
    gr.HTML("<div id='qdp-header'>QUANTUM DUBBING PIPELINE</div>"
            "<div id='qdp-subheader'>English to Spanish voice dubbing / AI shorts</div>")

    # Global proposals state (compartido entre shorts)
    sys_proposals = gr.State([])

    # ---- Global controls
    with gr.Row():
        btn_clear = gr.Button("Limpiar workspace")
        ui_global_status = gr.Textbox(label="Estado global", interactive=False, lines=1, scale=4)

    # ---- Tabs
    with gr.Tabs():

        # =================================================================
        # TAB 1: DOBLAJE DE AUDIO
        # =================================================================
        with gr.Tab("1 · Doblaje de audio"):

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>1</span>Subir media fuente</div>")
            gr.HTML("<div class='qdp-hint'>Video HQ (canal visual, puede venir muteado) + audio original en ingles. "
                    "Pega URL de Drive, path local del volumen (<code>/runpod-volume/user_input/...</code>) o sube archivo.</div>")
            with gr.Row():
                with gr.Column():
                    dub_v_tb, dub_v_up = _input_row(
                        "Video HQ (mute)",
                        "https://drive.google.com/... o /runpod-volume/user_input/video.mp4",
                        [".mp4", ".mov", ".mkv", ".webm"],
                    )
                with gr.Column():
                    dub_a_tb, dub_a_up = _input_row(
                        "Audio original EN",
                        "https://drive.google.com/... o /runpod-volume/user_input/audio.wav",
                        [".wav", ".mp3", ".flac", ".m4a", ".ogg"],
                    )
            with gr.Row():
                dub_is_test = gr.Checkbox(
                    label="Modo prueba: recortar video+audio a 30s (tildar SOLO para tests rapidos)",
                    value=False,
                )
                btn_dub_prep = gr.Button("Sincronizar / preparar media", variant="primary")
            dub_prep_status = gr.Textbox(label="Estado sincronizacion", interactive=False, lines=2)

            # Internal state for prepared paths (used by doblaje buttons)
            dub_video_prepared = gr.State("")
            dub_audio_prepared = gr.State("")

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>2</span>Ejecutar doblaje</div>")
            with gr.Row():
                btn_dub_test = gr.Button("Prueba 30 s")
                btn_dub_full = gr.Button("Doblaje completo", variant="primary")
            ui_dub_progress = gr.Slider(label="Progreso estimado", minimum=0, maximum=1, value=0,
                                         step=0.01, interactive=False)
            ui_dub_status = gr.Textbox(label="Fase actual", interactive=False, lines=1)

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>3</span>Progreso en vivo</div>")
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML("<div class='qdp-hint'>Log (pipeline.log)</div>")
                    ui_log_live = gr.Textbox(
                        label="", interactive=False, lines=22,
                        elem_id="qdp-log", autoscroll=True, show_copy_button=True,
                    )
                with gr.Column(scale=2):
                    gr.HTML("<div class='qdp-hint'>Archivos generados</div>")
                    ui_files_table = gr.DataFrame(
                        headers=["Path", "Tamano", "Modif"],
                        value=[], interactive=False, wrap=False,
                        row_count=(10, "dynamic"), col_count=(3, "fixed"),
                    )
                    btn_refresh = gr.Button("Refrescar log + archivos", size="sm")

            # ============================================================
            # DESCARGA MANUAL POR RUTA (siempre funciona, no depende del
            # auto-populate ni de la tabla)
            # ============================================================
            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>3b</span>Descargar archivo por ruta absoluta</div>")
            gr.HTML("<div class='qdp-hint'>Si los archivos finales no aparecen automaticamente, "
                    "pega aqui la ruta absoluta (ej: <code>/runpod-volume/output/FINAL_DUBBED_VIDEO.mp4</code>) "
                    "y dale al boton. Funciona con cualquier archivo dentro de <code>/runpod-volume/</code> o <code>/app/</code>.</div>")
            with gr.Row():
                manual_dl_path = gr.Textbox(
                    label="Ruta absoluta del archivo",
                    value="/runpod-volume/output/FINAL_DUBBED_VIDEO.mp4",
                    lines=1,
                )
                btn_manual_dl = gr.Button("⬇ Traer archivo", variant="primary")
            manual_dl_file = gr.File(label="Archivo listo para descargar", interactive=False)
            manual_dl_status = gr.Textbox(label="Estado", interactive=False, lines=1)

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>4</span>Resultado descargable</div>")
            gr.HTML("<div class='qdp-hint'>Al terminar el doblaje, <b>aparecen aca listos para descargar</b> "
                    "el MP4 final (video original + audio ES) y el WAV doblado, sin necesidad de SSH ni scp al pod. "
                    "El MP4 se arma con <code>ffmpeg -c:v copy</code> (segundos, no minutos) y cae a NVENC/libx264 "
                    "solo si el container no soporta stream-copy.</div>")
            out_dubbed_video = gr.Video(label="Video doblado final (MP4 descargable)", interactive=False)
            with gr.Row():
                out_dubbed_video_file = gr.File(
                    label="⬇ Descargar MP4 (FINAL_DUBBED_VIDEO.mp4)",
                    interactive=False,
                )
                out_dubbed = gr.Audio(label="Audio doblado final (WAV descargable)", interactive=False)
            out_json_path_display = gr.Textbox(label="JSON timeline generado (path)", interactive=False, lines=1)

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>5</span>Comparacion de traducciones (EN &harr; ES)</div>")
            gr.HTML("<div class='qdp-hint'>Se va llenando en vivo conforme el pipeline avanza: "
                    "al terminar la FASE 3 aparecen las traducciones, y al terminar la FASE 4 "
                    "se agrega la duracion del TTS ES y el delta respecto al EN original.</div>")
            ui_translation_compare = gr.DataFrame(
                headers=["#", "Tiempo (s)", "Speaker", "EN (original)", "ES (traduccion)",
                         "dur EN", "dur TTS ES", "delta"],
                value=[], interactive=False, wrap=True,
                row_count=(8, "dynamic"), col_count=(8, "fixed"),
                elem_id="qdp-translation-compare",
            )

            with gr.Accordion("Preview del archivo seleccionado (click fila de la tabla)", open=False):
                preview_audio = gr.Audio(label="Preview audio", interactive=False)
                preview_video = gr.Video(label="Preview video", interactive=False)
                preview_text = gr.Textbox(label="Preview texto/JSON", interactive=False, lines=15,
                                          show_copy_button=True)
                preview_download = gr.File(label="Descargar archivo seleccionado", interactive=False)

        # =================================================================
        # TAB 2: RENDER DE VIDEO
        # =================================================================
        with gr.Tab("2 · Render de video"):

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>1</span>Inputs para el render</div>")
            gr.HTML("<div class='qdp-hint'>Audio YA DOBLADO (ES) + video original + JSON timeline. "
                    "Si vienes de Tab 1, los campos se auto-rellenan al terminar el doblaje.</div>")
            with gr.Row():
                with gr.Column():
                    vid_v_tb, vid_v_up = _input_row(
                        "Video original",
                        "/runpod-volume/user_input/video.mp4",
                        [".mp4", ".mov", ".mkv"],
                    )
                with gr.Column():
                    vid_dub_tb, vid_dub_up = _input_row(
                        "Audio doblado ES",
                        "/runpod-volume/output/final_audio_dubbed.wav",
                        [".wav", ".mp3", ".flac"],
                    )
                with gr.Column():
                    vid_json_tb, vid_json_up = _input_row(
                        "JSON timeline",
                        "/runpod-volume/temp_workspace/p4_data.json",
                        [".json"],
                    )

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>2</span>Color grading</div>")
            with gr.Row():
                sl_nitidez = gr.Slider(-1.0, 2.0, value=0.5, step=0.1, label="Nitidez (unsharp)")
                sl_brillo = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Brillo")
                sl_contraste = gr.Slider(0.5, 2.0, value=1.05, step=0.05, label="Contraste")

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>3</span>Calidad del lip-sync (LatentSync)</div>")
            gr.HTML("<div class='qdp-hint'>LatentSync se aplica <b>siempre</b> al generar el master: re-sincroniza "
                    "los labios al audio ES preservando gestos, escenas y movimientos de cámara. "
                    "La primera vez que uses la app, los pesos (~2 GB) se descargan automáticamente a "
                    "<code>/runpod-volume/LatentSync/</code> (persistente, una sola vez por volumen). "
                    "Ajusta aqui la calidad del lipsync.</div>")
            with gr.Row():
                sl_lipsync_steps = gr.Slider(10, 50, value=20, step=1, label="Inference steps (20=default, +steps=+calidad/+tiempo)")
                sl_lipsync_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Guidance scale (1.5=default, +gs=+sync/+distorsión)")

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>4</span>Renderizar</div>")
            with gr.Row():
                btn_vid_prev = gr.Button("Previsualizar 30 s")
                btn_vid_full = gr.Button("Renderizar completo", variant="primary")
            ui_vid_status = gr.Textbox(label="Estado del render", interactive=False, lines=2)

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>5</span>Resultado</div>")
            out_vid = gr.Video(label="Longform final (16:9, subs pro quemados)", interactive=False)
            out_master = gr.Textbox(
                label="Master lip-sync sin subs (insumo para shorts)",
                interactive=False, lines=1,
            )
            gr.HTML("<div class='qdp-hint'>El master sin subs es el que Tab 3 consume para "
                    "generar shorts verticales con captions amarillos.</div>")
            with gr.Accordion("Log del render + archivos", open=False):
                ui_log_render = gr.Textbox(label="Log", interactive=False, lines=15, elem_id="qdp-log")
                ui_files_table_vid = gr.DataFrame(
                    headers=["Path", "Tamano", "Modif"], value=[], interactive=False, wrap=False,
                )

        # =================================================================
        # TAB 3: AI SHORTS
        # =================================================================
        with gr.Tab("3 · Shorts con IA"):

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>1</span>Inputs para los shorts</div>")
            gr.HTML("<div class='qdp-hint'>Los shorts se generan a partir del <b>master lip-sync sin subs</b> "
                    "(el que produce Tab 2). Tab 3 recorta, hace crop 9:16 y agrega captions amarillos en español. "
                    "Si vienes de Tab 2, el master se auto-rellena.</div>")
            with gr.Row():
                with gr.Column():
                    sh_master_tb, sh_master_up = _input_row(
                        "Master lip-sync (sin subs)",
                        "/runpod-volume/output/master_lipsynced_nosubs.mp4",
                        [".mp4", ".mov", ".mkv"],
                    )
                with gr.Column():
                    sh_json_tb, sh_json_up = _input_row(
                        "JSON timeline (con text_es y timestamps)",
                        "/runpod-volume/temp_workspace/p4_data.json",
                        [".json"],
                    )

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>2</span>Duración de los shorts</div>")
            gr.HTML("<div class='qdp-hint'>Rango de duración permitido (minutos). "
                    "Gemini solo propondrá shorts dentro de este rango.</div>")
            with gr.Row():
                sl_min_min = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Duración mínima (min)")
                sl_max_min = gr.Slider(2.0, 20.0, value=15.0, step=0.5, label="Duración máxima (min)")

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>3</span>Analizar con IA</div>")
            btn_analyze = gr.Button("Analizar timeline ES con Gemini", variant="primary")
            shorts_summary = gr.Textbox(label="Propuestas de Gemini", interactive=False, lines=10)
            cg_shorts = gr.CheckboxGroup(choices=[], label="Shorts a generar (selecciona varios)")

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>4</span>Captions + color grading</div>")
            with gr.Row():
                cb_captions_wbw = gr.Checkbox(label="Captions word-by-word (amarillo)", value=True)
                sl_chunk_words = gr.Slider(1, 6, value=3, step=1, label="Palabras por caption")
            with gr.Row():
                sl_nitidez_s = gr.Slider(-1.0, 2.0, value=0.5, step=0.1, label="Nitidez")
                sl_brillo_s = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Brillo")
                sl_contraste_s = gr.Slider(0.5, 2.0, value=1.05, step=0.05, label="Contraste")

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>5</span>Preview + renderizado</div>")
            gr.HTML("<div class='qdp-hint'>Los shorts heredan el lip-sync del master (ya fue aplicado en Tab 2 "
                    "con LatentSync). Aqui solo recortamos, hacemos crop 9:16 y quemamos los captions amarillos.</div>")
            with gr.Row():
                btn_short_thumb = gr.Button("Thumbnail del primer seleccionado (rapido)")
                btn_short_prev = gr.Button("Preview video 10s del primer seleccionado")
                btn_short_all = gr.Button("Generar todos los seleccionados", variant="primary")
            ui_short_status = gr.Textbox(label="Estado", interactive=False, lines=2)

            gr.HTML("<div class='qdp-section-title'><span class='qdp-step-num'>6</span>Resultado</div>")
            with gr.Row():
                out_short_thumb = gr.Image(label="Thumbnail (9:16)", interactive=False, height=400)
                out_short_preview = gr.Video(label="Preview del short", interactive=False)
            out_shorts_files = gr.File(label="Shorts generados (descargables)", file_count="multiple", interactive=False)

    # =====================================================================
    # WIRING
    # =====================================================================

    btn_clear.click(
        clear_workspace, inputs=[],
        outputs=[ui_global_status, ui_files_table, ui_log_live],
    )

    # --- Tab 1 wiring ---

    btn_dub_prep.click(
        prepare_data,
        inputs=[dub_v_tb, dub_a_tb, dub_v_up, dub_a_up, dub_is_test],
        outputs=[dub_video_prepared, dub_audio_prepared, dub_prep_status,
                 ui_files_table, ui_log_live],
    )

    btn_dub_full.click(
        pipe_doblar_audio,
        inputs=[dub_audio_prepared, dub_video_prepared],
        outputs=[out_dubbed_video, out_dubbed, out_json_path_display, ui_dub_status,
                 ui_files_table, ui_log_live, ui_dub_progress,
                 ui_translation_compare],
    ).then(
        # El gr.File descargable refleja el mismo MP4 que el gr.Video (si existe).
        lambda v: v,
        inputs=[out_dubbed_video],
        outputs=[out_dubbed_video_file],
    ).then(
        # Auto-populate Tab 2 (video + audio + json) y Tab 3 (json).
        # El master_lipsynced se auto-popula en Tab 3 DESPUÉS de Tab 2.
        lambda v, a, j: (v, a, j, j),
        inputs=[dub_video_prepared, out_dubbed, out_json_path_display],
        outputs=[vid_v_tb, vid_dub_tb, vid_json_tb, sh_json_tb],
    )

    btn_dub_test.click(
        pipe_doblar_audio_test30,
        inputs=[dub_audio_prepared, dub_video_prepared],
        outputs=[out_dubbed_video, out_dubbed, out_json_path_display, ui_dub_status,
                 ui_files_table, ui_log_live, ui_dub_progress,
                 ui_translation_compare],
    ).then(
        lambda v: v,
        inputs=[out_dubbed_video],
        outputs=[out_dubbed_video_file],
    ).then(
        lambda v, a, j: (v, a, j, j),
        inputs=[dub_video_prepared, out_dubbed, out_json_path_display],
        outputs=[vid_v_tb, vid_dub_tb, vid_json_tb, sh_json_tb],
    )

    btn_refresh.click(
        refresh_logs_and_files, inputs=[],
        outputs=[ui_files_table, ui_log_live],
    )

    btn_manual_dl.click(
        manual_download,
        inputs=[manual_dl_path],
        outputs=[manual_dl_file, manual_dl_status],
    )
    # Enter en el textbox tambien dispara la descarga
    manual_dl_path.submit(
        manual_download,
        inputs=[manual_dl_path],
        outputs=[manual_dl_file, manual_dl_status],
    )

    ui_files_table.select(
        on_files_df_select,
        inputs=[ui_files_table],
        outputs=[preview_audio, preview_video, preview_text, preview_download],
    )

    # --- Tab 2 wiring ---

    btn_vid_full.click(
        pipe_generar_video,
        inputs=[vid_v_tb, vid_dub_tb, vid_json_tb, vid_v_up, vid_dub_up, vid_json_up,
                sl_nitidez, sl_brillo, sl_contraste,
                sl_lipsync_steps, sl_lipsync_guidance],
        outputs=[out_vid, out_master, ui_vid_status, ui_files_table_vid, ui_log_render],
    ).then(
        # Auto-populate Tab 3 con el master recién generado (ya tiene LatentSync)
        lambda m: m,
        inputs=[out_master],
        outputs=[sh_master_tb],
    )
    btn_vid_prev.click(
        pipe_preview_video,
        inputs=[vid_v_tb, vid_dub_tb, vid_json_tb, vid_v_up, vid_dub_up, vid_json_up,
                sl_nitidez, sl_brillo, sl_contraste,
                sl_lipsync_steps, sl_lipsync_guidance],
        outputs=[out_vid, out_master, ui_vid_status, ui_files_table_vid, ui_log_render],
    ).then(
        lambda m: m,
        inputs=[out_master],
        outputs=[sh_master_tb],
    )

    # --- Tab 3 wiring ---

    btn_analyze.click(
        pipe_analyze_shorts,
        inputs=[sh_json_tb, sh_json_up, sl_min_min, sl_max_min],
        outputs=[cg_shorts, sys_proposals, shorts_summary],
    )
    btn_short_thumb.click(
        pipe_preview_thumbnail,
        inputs=[sh_master_tb, sh_master_up, sys_proposals, cg_shorts],
        outputs=[out_short_thumb, ui_short_status],
    )
    btn_short_prev.click(
        pipe_preview_short,
        inputs=[sh_master_tb, sh_json_tb, sh_master_up, sh_json_up, sys_proposals, cg_shorts,
                sl_nitidez_s, sl_brillo_s, sl_contraste_s,
                cb_captions_wbw, sl_chunk_words],
        outputs=[out_short_preview, ui_short_status],
    )
    btn_short_all.click(
        pipe_render_selected_shorts,
        inputs=[sh_master_tb, sh_json_tb, sh_master_up, sh_json_up, sys_proposals, cg_shorts,
                sl_nitidez_s, sl_brillo_s, sl_contraste_s,
                cb_captions_wbw, sl_chunk_words],
        outputs=[out_shorts_files, ui_short_status],
    )


if __name__ == "__main__":
    _ensure_dirs()
    setup_pipeline_logger("logs")
    ui_log.info("Booting Quantum Dubbing Pipeline UI")
    ensure_qwen_models()
    # allowed_paths: permite que gr.Audio/gr.Video/gr.File sirvan archivos
    # desde estos directorios (incluye user_input/ para uploads persistentes).
    allowed = [os.path.abspath(d) for d in ["input", "temp_workspace", "output", "logs", "user_input"]]
    data_dir = os.environ.get("APP_DATA_DIR")
    if data_dir:
        allowed.append(os.path.abspath(data_dir))
    # Permitir servir archivos desde /runpod-volume y /app directamente,
    # asi el usuario puede descargar CUALQUIER archivo pegando su ruta absoluta
    # en el campo "Descargar archivo por ruta absoluta" del Tab 1.
    for extra in ["/runpod-volume", "/app"]:
        if os.path.isdir(extra) and extra not in allowed:
            allowed.append(extra)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_api=False,
        allowed_paths=allowed,
    )
