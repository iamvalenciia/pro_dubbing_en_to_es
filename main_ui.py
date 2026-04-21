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

# =========================================================================
# CONFIGURACIÓN DE ALMACENAMIENTO Y RUTAS (SPEC SECTION 2)
# =========================================================================
NETWORK_DIR = "/runpod-volume"
LOCAL_DIR = "/workspace/qdp_data"

# Entorno de Red (Persistente, Lento)
USER_INPUT_DIR = os.path.join(NETWORK_DIR, "user_input")
FINAL_OUTPUT_DIR = os.path.join(NETWORK_DIR, "output")

# Entorno Local NVMe (Efímero, Rápido)
LOCAL_INPUT_DIR = os.path.join(LOCAL_DIR, "input")
LOCAL_TEMP_DIR = os.path.join(LOCAL_DIR, "temp_workspace")
LOCAL_LOGS_DIR = os.path.join(LOCAL_DIR, "logs")

ui_log = get_logger("ui")


def _ensure_dirs():
    """Asegura la creación de los directorios estrictamente separados (Red vs NVMe)."""
    for d in [USER_INPUT_DIR, FINAL_OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
    for d in [LOCAL_INPUT_DIR, LOCAL_TEMP_DIR, LOCAL_LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    setup_pipeline_logger(LOCAL_LOGS_DIR)

def _get_user_files():
    """Lee archivos subidos al volumen de red de forma directa para los dropdowns."""
    _ensure_dirs()
    try:
        files = [f for f in os.listdir(USER_INPUT_DIR) if os.path.isfile(os.path.join(USER_INPUT_DIR, f))]
        return sorted(files)
    except OSError:
        return []

def _get_local_jsons():
    """Lee los timelines JSON generados en el entorno NVMe local."""
    _ensure_dirs()
    try:
        files = [f for f in os.listdir(LOCAL_TEMP_DIR) if f.endswith(".json")]
        return sorted(files)
    except OSError:
        return []

def run_workflow_1_streamed(video_file, audio_file, use_latentsync, test_mode, keep_4k, add_subs):
    """
    Workflow 1: Master Clean Video (One-Click Magic).
    Mockup estructural guiado por los toggles de la UI respetando NVMe vs Network Volume.
    """
    if not video_file or not audio_file:
        yield "Error", "Por favor selecciona un video y un audio de los menús desplegables.", None
        return
        
    video_path = os.path.join(USER_INPUT_DIR, video_file)
    audio_path = os.path.join(USER_INPUT_DIR, audio_file)
    
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

    result = {"video": None, "status": "running", "error": None, "phase": "Iniciando..."}

    def worker():
        try:
            # ------------------------------------------------------------------
            # MOCKUP DEL WORKFLOW 1
            # Aquí se llamarían las funciones reales de src/ respetando rutas
            # ------------------------------------------------------------------
            result["phase"] = "Fase 0: Copiando a NVMe y Normalizando..."
            ui_log.info(result["phase"])
            # Simulamos el trabajo de Fase 0 (input_handler.py)
            time.sleep(2)
            
            result["phase"] = "Fase 2: ASR (Transcribiendo EN)..."
            ui_log.info(result["phase"])
            # Simulamos Fase 2 (phase2_asr_diarization.py)
            time.sleep(2)
            
            result["phase"] = "Fase 3: Traducción (EN -> ES)..."
            ui_log.info(result["phase"])
            # Simulamos Fase 3 (phase3_llm_isochrone.py)
            time.sleep(2)
            
            result["phase"] = "Fase 4: Clonación TTS (Generando voces ES)..."
            ui_log.info(result["phase"])
            ui_log.info(">>> Aplicando restricción dura de isocronía (truncado numpy)")
            # Simulamos Fase 4 (phase4_tts_cloning.py)
            time.sleep(2)
            
            result["phase"] = "Fase 5: Alineación de Tiempo (Ensamblando audio)..."
            ui_log.info(result["phase"])
            # Simulamos Fase 5 (phase5_alignment.py)
            time.sleep(2)
            
            if use_latentsync:
                result["phase"] = "Fase 6: LatentSync (Lip-sync de video a audio ES)..."
                ui_log.info(result["phase"])
                # Simulamos Fase 6 (phase6_lipsync_render.py)
                time.sleep(3)
            else:
                result["phase"] = "Fase 5b: Mux Básico (Lip-sync desactivado)..."
                ui_log.info(result["phase"])
                time.sleep(2)
                
            if add_subs:
                result["phase"] = "Fase 6b: Quemando subtítulos horizontales de 1-línea..."
                ui_log.info(result["phase"])
                time.sleep(2)
                
            result["phase"] = "Finalizando: Copiando resultado al volumen de red..."
            ui_log.info(result["phase"])
            
            # En producción, esto apuntaría al MP4 real generado por el pipeline
            final_mp4 = os.path.join(FINAL_OUTPUT_DIR, "FINAL_MASTER_DUBBED.mp4")
            # Dejamos el path como None en el mock para no romper el reproductor del frontend.
            result["video"] = None 
            
            result["status"] = "ok"
            result["phase"] = "¡Completado!"
            ui_log.info("Workflow 1 finalizado con éxito. Archivo disponible en NETWORK_DIR/output/")
        except Exception as e:
            result["status"] = "fail"
            result["error"] = str(e)
            result["phase"] = f"Error crítico: {e}"
            ui_log.error(f"Fallo en Workflow 1: {e}\n{traceback.format_exc()}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while t.is_alive():
        time.sleep(1.0)
        yield result["phase"], read_log_tail(1500), None

    t.join()
    log_tail = read_log_tail(1500)
    
    if result["status"] == "ok":
        yield result["phase"], log_tail, result["video"]
    else:
        yield result["phase"], log_tail, None

def run_workflow_2_analysis(json_file):
    """
    Workflow 2 - Paso 1: Analizar JSON con IA para extraer clips.
    """
    if not json_file:
        return gr.update(choices=[]), "Por favor selecciona un archivo JSON."
        
    json_path = os.path.join(LOCAL_TEMP_DIR, json_file)
    ui_log.info(f"Workflow 2: Analizando timeline para shorts -> {json_path}")
    
    # Mockup: Proponer clips ficticios
    proposals = [
        "Clip 1: El inicio sorprendente (0:00 - 0:45)",
        "Clip 2: El clímax de la historia (1:20 - 2:30)",
        "Clip 3: La conclusión épica (3:15 - 4:00)"
    ]
    ui_log.info(f"Análisis IA completado. Se proponen {len(proposals)} shorts.")
    
    return gr.update(choices=proposals, value=[]), read_log_tail(500)

def run_workflow_2_render(selected_clips):
    """
    Workflow 2 - Paso 3: Renderizar shorts seleccionados.
    """
    if not selected_clips:
        yield "Por favor selecciona al menos un clip.", None
        return
        
    ui_log.info(f"Workflow 2: Iniciando render de {len(selected_clips)} shorts (9:16 + Captions Amarillos)...")
    
    for i, clip in enumerate(selected_clips):
        ui_log.info(f"Renderizando short {i+1}/{len(selected_clips)}: {clip}...")
        time.sleep(2) # Simular render
        
    ui_log.info("Workflow 2: ¡Todos los shorts renderizados con éxito!")
    
    # Igual que arriba, retornamos None para no romper Gradio con archivos inexistentes en el mockup
    yield read_log_tail(500), None


with gr.Blocks(theme=gr.themes.Monochrome(), css=CSS, title="Quantum Dubbing Pipeline") as demo:
    gr.HTML(STARFIELD_HTML)
    gr.HTML("<div id='qdp-header'>QUANTUM DUBBING PIPELINE</div>"
            "<div id='qdp-subheader'>Apple-Style Minimalist Interface</div>")

    gr.Markdown("### ⚙️ Ajustes Globales (Workflows)")
    with gr.Row():
        cb_latentsync = gr.Checkbox(label="Habilitar LatentSync (Lip-sync)", value=True, interactive=True)
        cb_test_mode = gr.Checkbox(label="Test Mode (30 Segundos Límite)", value=False, interactive=True)
        cb_keep_4k = gr.Checkbox(label="Renderizar en 4K (Original)", value=True, interactive=True)
        cb_subs = gr.Checkbox(label="Workflow 1.1: Master con Subtítulos (Opcional)", value=False, interactive=True)

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
                    "aws s3 ls --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io s3://x73d6lzlpq/\n"
                    "aws s3 cp tu_video.mp4 s3://x73d6lzlpq/user_input/ --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io\n"
                    "```\n"
                    "Una vez subidos, haz clic en **🔄 Refrescar**."
                )
            
            with gr.Row():
                dd_video = gr.Dropdown(choices=[], label="Seleccionar Video Original (EN)", interactive=True, scale=3)
                dd_audio = gr.Dropdown(choices=[], label="Seleccionar Audio Original (EN)", interactive=True, scale=3)
                btn_refresh_files = gr.Button("🔄 Refrescar", scale=1)
                
            btn_run_master = gr.Button("🚀 Ejecutar Doblaje (One-Click Magic)", variant="primary", size="lg")
            
            ui_active_phase = gr.Textbox(label="Fase Activa", value="Esperando inicio...", interactive=False, lines=1)
            ui_terminal = gr.Textbox(label="Terminal de Logs", elem_id="qdp-log", interactive=False, lines=15, autoscroll=True)
            
            out_master_video = gr.Video(label="Resultado: Video Master Final (MP4)", interactive=False)

        # =================================================================
        # TAB 2: GENERADOR DE SHORTS (Workflow 2)
        # =================================================================
        with gr.Tab("2 · Generador de Shorts (AI Viral Clips)"):
            gr.Markdown(
                "Selecciona el archivo JSON generado en el entorno NVMe local (`temp_workspace`) "
                "para analizarlo con IA y generar clips virales verticales."
            )
            with gr.Row():
                dd_json = gr.Dropdown(choices=[], label="Seleccionar Timeline JSON", interactive=True, scale=3)
                btn_refresh_json = gr.Button("🔄 Refrescar", scale=1)
            
            btn_analyze = gr.Button("🧠 Analizar Timeline con IA (>3 min)", variant="primary")
            
            cg_shorts = gr.CheckboxGroup(choices=[], label="Clips Virales Propuestos (Selecciona para renderizar)")
            btn_render_shorts = gr.Button("🎬 Renderizar Shorts Seleccionados (9:16 + Captions)")
            
            ui_shorts_terminal = gr.Textbox(label="Terminal de Shorts", elem_id="qdp-log", interactive=False, lines=10, autoscroll=True)
            out_shorts_video = gr.Video(label="Previsualización de Short", interactive=False)

    # =====================================================================
    # WIRING Y LÓGICA UI
    # =====================================================================
    def update_file_dropdowns():
        files = _get_user_files()
        return gr.update(choices=files), gr.update(choices=files)
        
    def update_json_dropdown():
        jsons = _get_local_jsons()
        return gr.update(choices=jsons)

    btn_refresh_files.click(update_file_dropdowns, inputs=[], outputs=[dd_video, dd_audio])
    btn_refresh_json.click(update_json_dropdown, inputs=[], outputs=[dd_json])
    
    # Carga inicial
    demo.load(update_file_dropdowns, inputs=[], outputs=[dd_video, dd_audio])
    demo.load(update_json_dropdown, inputs=[], outputs=[dd_json])

    # Ejecutar Doblaje Master
    btn_run_master.click(
        run_workflow_1_streamed,
        inputs=[dd_video, dd_audio, cb_latentsync, cb_test_mode, cb_keep_4k, cb_subs],
        outputs=[ui_active_phase, ui_terminal, out_master_video]
    )
    
    # Shorts
    btn_analyze.click(
        run_workflow_2_analysis,
        inputs=[dd_json],
        outputs=[cg_shorts, ui_shorts_terminal]
    )
    
    btn_render_shorts.click(
        run_workflow_2_render,
        inputs=[cg_shorts],
        outputs=[ui_shorts_terminal, out_shorts_video]
    )

if __name__ == "__main__":
    _ensure_dirs()
    ui_log.info("Booting Quantum Dubbing Pipeline UI - Apple Style")
    
    allowed = [os.path.abspath(d) for d in [NETWORK_DIR, LOCAL_DIR]]
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_api=False,
        allowed_paths=allowed,
    )
