import json
import importlib
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import ffmpeg
import gradio as gr
from dotenv import load_dotenv
from frame_editor import extract_frame, apply_frame_adjustments, build_video_filter

from src.logger import clear_log, get_logger, read_log_tail, setup_pipeline_logger
from src.paths import (
    LOCAL_DIR,
    LOCAL_LOGS,
    LOCAL_TEMP,
    NETWORK_DIR,
    NETWORK_OUTPUT,
    NETWORK_USER_INPUT,
    ensure_local_dirs,
)

PYVIDEOTRANS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyvideotrans")
PYVIDEOTRANS_CLI = os.path.join(PYVIDEOTRANS_ROOT, "cli.py")
PYVIDEOTRANS_PYTHON = os.environ.get("PYVIDEOTRANS_PYTHON", sys.executable)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VOICE_REFS_ROOT = os.path.join(REPO_ROOT, "input", "voice_refs")
VOICE_REFS_NORMALIZED = os.path.join(VOICE_REFS_ROOT, "normalized")
VOICE_REFS_PREVIEWS = os.path.join(VOICE_REFS_ROOT, "previews")
VOICE_REFS_CATALOG = os.path.join(VOICE_REFS_ROOT, "voice_refs_catalog.json")
VOICE_SPEAKER_MAP = os.path.join(VOICE_REFS_ROOT, "speaker_voice_map.json")
VOICE_REFS_SOURCE = os.path.join(REPO_ROOT, "reference_voices")

STATIC_VOICE_OPTIONS = [
    {
        "label": "Discovery Channel (Hombre)",
        "ref_id": "discovery_channel-hombre-discovery_channel_clean",
    },
    {
        "label": "Donald Trump (Hombre)",
        "ref_id": "donald_trump-hombre-president_trump-enhanced-v2",
    },
    {
        "label": "DW Documentales (Mujer)",
        "ref_id": "dw_documentales-mujer-dw_documental_DeepFilterNet3_clean",
    },
    {
        "label": "Professor Jiang (Hombre)",
        "ref_id": "professor_jiang-hombre-professor_jiang-enhanced-v2",
    },
    {
        "label": "Veritasium (Hombre)",
        "ref_id": "veritasium-hombre-veritasium",
    },
]
STATIC_VOICE_LABELS = [it["label"] for it in STATIC_VOICE_OPTIONS]
STATIC_VOICE_REF_BY_LABEL = {it["label"]: it["ref_id"] for it in STATIC_VOICE_OPTIONS}

if PYVIDEOTRANS_ROOT not in sys.path:
    sys.path.insert(0, PYVIDEOTRANS_ROOT)

try:
    from subtitle_ui_wrapper import ui_generate_subtitles, ui_render_subtitles
except ImportError:
    print("Warning: subtitle_ui_wrapper not available")
    ui_generate_subtitles = None
    ui_render_subtitles = None

try:
    from reel_ui_wrapper import ui_analyze_reels, ui_render_reel
except ImportError:
    print("Warning: reel_ui_wrapper not available")
    ui_analyze_reels = None
    ui_render_reel = None

load_dotenv()  # Carga GEMINI_MODEL, HF_TOKEN, API_GOOGLE_STUDIO desde tu .env local


def _parse_local_dotenv(dotenv_path: str) -> dict:
    env_map = {}
    try:
        if not os.path.isfile(dotenv_path):
            return env_map
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if key:
                    env_map[key] = val
    except Exception:
        return {}
    return env_map


def _inject_ai_env_to_child(child_env: dict) -> dict:
    """Ensure subprocess gets AI env vars even if parent process missed dotenv load."""
    repo_env = _parse_local_dotenv(os.path.join(REPO_ROOT, ".env"))
    forced_gemini_model = "gemini-3.1-flash-lite-preview"
    for k in ("API_GOOGLE_STUDIO", "GEMINI_MODEL", "HF_TOKEN"):
        val = os.environ.get(k) or repo_env.get(k)
        if val:
            child_env[k] = str(val).strip().strip('"').strip("'")
    # Hard requirement for this pipeline: always use the configured Gemini model.
    child_env["GEMINI_MODEL"] = forced_gemini_model
    return child_env


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

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


STARFIELD_HTML = """
<div id="qdp-atmosphere" aria-hidden="true">
        <canvas id="qdp-stars"></canvas>
        <div class="qdp-orb qdp-orb-a"></div>
        <div class="qdp-orb qdp-orb-b"></div>
</div>
<script>
(function() {
    if (window.__qdp_stars_init) return;
    window.__qdp_stars_init = true;
    const canvas = document.getElementById('qdp-stars');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let w = 0;
    let h = 0;
    let dpr = 1;
    let stars = [];
    const STAR_COUNT = 180;

    function isDark() {
        const el = document.documentElement;
        if (el.classList.contains('dark')) return true;
        if (document.body && document.body.classList.contains('dark')) return true;
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    function resize() {
        dpr = window.devicePixelRatio || 1;
        w = window.innerWidth;
        h = window.innerHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function initStars() {
        stars = Array.from({ length: STAR_COUNT }, () => ({
            x: Math.random() * w,
            y: Math.random() * h,
            r: Math.random() * 1.8 + 0.2,
            a: Math.random() * 0.7 + 0.1,
            s: Math.random() * 0.35 + 0.05
        }));
    }

    function draw() {
        const dark = isDark();
        const color = dark ? '255,215,140' : '120,95,35';
        ctx.clearRect(0, 0, w, h);
        for (const st of stars) {
            st.a += (Math.random() - 0.5) * st.s;
            st.a = Math.max(0.08, Math.min(0.85, st.a));
            ctx.fillStyle = 'rgba(' + color + ',' + st.a.toFixed(3) + ')';
            ctx.beginPath();
            ctx.arc(st.x, st.y, st.r, 0, Math.PI * 2);
            ctx.fill();
        }
        requestAnimationFrame(draw);
    }

    window.addEventListener('resize', () => {
        resize();
        initStars();
    });

    resize();
    initStars();
    draw();
})();
</script>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --qdp-bg: #f4f4f2;
    --qdp-bg-elev: rgba(255, 255, 255, 0.74);
    --qdp-text: #111111;
    --qdp-muted: #4d4d4d;
    --qdp-border: rgba(17, 17, 17, 0.16);
    --qdp-border-strong: rgba(17, 17, 17, 0.28);
    --qdp-shadow: 0 14px 50px rgba(0, 0, 0, 0.08);
    --qdp-focus: #9a7b2f;
    --qdp-pill: rgba(255, 255, 255, 0.88);
    --qdp-log-bg: rgba(255, 255, 255, 0.9);
    --qdp-log-fg: #222222;
    --qdp-log-border: rgba(17, 17, 17, 0.16);
}

@media (prefers-color-scheme: dark) {
    :root {
        --qdp-bg: #080808;
        --qdp-bg-elev: rgba(18, 18, 18, 0.78);
        --qdp-text: #f4f4f4;
        --qdp-muted: #aaaaaa;
        --qdp-border: rgba(255, 255, 255, 0.14);
        --qdp-border-strong: rgba(255, 255, 255, 0.28);
        --qdp-shadow: 0 18px 52px rgba(0, 0, 0, 0.38);
        --qdp-focus: #d6b46a;
        --qdp-pill: rgba(24, 24, 24, 0.85);
        --qdp-log-bg: #050505;
        --qdp-log-fg: #f0f0f0;
        --qdp-log-border: rgba(255, 255, 255, 0.18);
    }
}

.dark,
.dark body {
    --qdp-bg: #080808;
    --qdp-bg-elev: rgba(18, 18, 18, 0.78);
    --qdp-text: #f4f4f4;
    --qdp-muted: #aaaaaa;
    --qdp-border: rgba(255, 255, 255, 0.14);
    --qdp-border-strong: rgba(255, 255, 255, 0.28);
    --qdp-shadow: 0 18px 52px rgba(0, 0, 0, 0.38);
    --qdp-focus: #d6b46a;
    --qdp-pill: rgba(24, 24, 24, 0.85);
    --qdp-log-bg: #050505;
    --qdp-log-fg: #f0f0f0;
    --qdp-log-border: rgba(255, 255, 255, 0.18);
}

body,
.gradio-container {
    background: var(--qdp-bg) !important;
    color: var(--qdp-text) !important;
}

#qdp-atmosphere {
    position: fixed;
    inset: 0;
    z-index: -1;
    overflow: hidden;
    pointer-events: none;
    background:
        radial-gradient(circle at 20% 15%, rgba(0, 0, 0, 0.07), transparent 42%),
        radial-gradient(circle at 80% 85%, rgba(0, 0, 0, 0.05), transparent 44%),
        linear-gradient(160deg, rgba(255, 255, 255, 0.35), transparent 55%),
        var(--qdp-bg);
}

#qdp-stars {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
}

@media (prefers-color-scheme: dark) {
    #qdp-atmosphere {
        background:
            radial-gradient(circle at 18% 12%, rgba(255, 255, 255, 0.09), transparent 48%),
            radial-gradient(circle at 82% 88%, rgba(255, 255, 255, 0.06), transparent 50%),
            linear-gradient(150deg, rgba(255, 255, 255, 0.03), transparent 55%),
            var(--qdp-bg);
    }
}

.qdp-orb {
    position: absolute;
    border-radius: 999px;
    filter: blur(18px);
    animation: qdpFloat 18s ease-in-out infinite;
}

.qdp-orb-a {
    width: min(38vw, 440px);
    height: min(38vw, 440px);
    left: -8vw;
    top: -14vh;
    background: rgba(0, 0, 0, 0.08);
}

.qdp-orb-b {
    width: min(44vw, 520px);
    height: min(44vw, 520px);
    right: -14vw;
    bottom: -20vh;
    background: rgba(0, 0, 0, 0.06);
    animation-delay: -7s;
}

@media (prefers-color-scheme: dark) {
    .qdp-orb-a,
    .qdp-orb-b {
        background: rgba(255, 255, 255, 0.08);
    }
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 0 12px 18px !important;
    font-family: "Space Grotesk", "Bahnschrift", "Segoe UI", sans-serif !important;
}

#qdp-header {
    text-align: center;
    margin-top: 14px;
    padding: 24px 12px 6px;
    color: var(--qdp-text);
    font-family: "Cormorant Garamond", "Times New Roman", serif;
    font-size: clamp(28px, 3.2vw, 44px);
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    animation: qdpRiseIn 0.75s ease-out both;
}

#qdp-subheader {
    text-align: center;
    color: var(--qdp-muted);
    font-size: 11px;
    letter-spacing: 0.24em;
    margin-bottom: 18px;
    text-transform: uppercase;
    animation: qdpRiseIn 0.9s ease-out both;
}

.gr-group,
.gr-panel {
    background: var(--qdp-bg-elev) !important;
    border: 1px solid var(--qdp-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--qdp-shadow) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}

.speaker-voice-dd {
    position: relative !important;
    z-index: 1500 !important;
    overflow: visible !important;
}

.speaker-voice-dd .wrap {
    overflow: visible !important;
}

.speaker-voice-dd .wrap-inner,
.speaker-voice-dd .secondary-wrap,
.speaker-voice-dd .reference {
    overflow: visible !important;
}

.speaker-voice-dd .choices {
    z-index: 20000 !important;
}

.speaker-voice-dd ul.options {
    top: calc(100% + 6px) !important;
    bottom: auto !important;
    z-index: 20001 !important;
    max-height: min(260px, 42vh) !important;
    overflow-y: auto !important;
}

.gr-group,
.gr-panel,
.gr-column {
    overflow: visible !important;
}

.gr-tabs,
.tab-nav {
    background: transparent !important;
    border: 0 !important;
}

.tab-nav button {
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    padding: 7px 16px !important;
    margin-right: 6px !important;
    background: var(--qdp-pill) !important;
    color: var(--qdp-muted) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
    font-size: 11px !important;
    transition: all 0.2s ease;
}

.tab-nav button.selected {
    color: var(--qdp-text) !important;
    border-color: var(--qdp-border-strong) !important;
    transform: translateY(-1px);
}

.gr-button,
button.primary,
button.secondary {
    background: transparent !important;
    border: 1px solid var(--qdp-border) !important;
    color: var(--qdp-text) !important;
    border-radius: 11px !important;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    font-size: 11px !important;
    font-weight: 600 !important;
    transition: transform 0.16s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.gr-button:hover,
button.primary:hover,
button.secondary:hover {
    border-color: var(--qdp-border-strong) !important;
    box-shadow: 0 0 0 1px var(--qdp-focus) inset;
    transform: translateY(-1px);
}

input[type="text"],
input[type="number"],
textarea,
.gr-textbox textarea,
.gr-textbox input,
.gr-dropdown input,
.gr-dropdown select {
    background: rgba(255, 255, 255, 0.3) !important;
    color: var(--qdp-text) !important;
    border: 1px solid var(--qdp-border) !important;
}

@media (prefers-color-scheme: dark) {
    input[type="text"],
    input[type="number"],
    textarea,
    .gr-textbox textarea,
    .gr-textbox input,
    .gr-dropdown input,
    .gr-dropdown select {
        background: rgba(0, 0, 0, 0.32) !important;
    }
}

input[type="checkbox"],
.gr-checkbox label {
    cursor: pointer !important;
    pointer-events: auto !important;
}

label,
.gr-form > label,
.gr-block-label {
    color: var(--qdp-muted) !important;
    font-size: 10.5px !important;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

#qdp-log,
#qdp-reels-log {
    border-radius: 11px !important;
}

#qdp-log textarea,
#qdp-reels-log textarea {
    font-family: "JetBrains Mono", "Cascadia Mono", Consolas, monospace !important;
    font-size: 12.4px !important;
    line-height: 1.62 !important;
    padding: 14px 16px !important;
    background: var(--qdp-log-bg) !important;
    color: var(--qdp-log-fg) !important;
    border: 1px solid var(--qdp-log-border) !important;
    border-radius: 11px !important;
    white-space: pre !important;
    overflow: auto !important;
    scrollbar-width: thin;
}

#qdp-log textarea::-webkit-scrollbar,
#qdp-reels-log textarea::-webkit-scrollbar {
    height: 8px;
    width: 8px;
}

#qdp-log textarea::-webkit-scrollbar-thumb,
#qdp-reels-log textarea::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.24);
    border-radius: 4px;
}

.gr-image,
.gr-video,
.gr-audio {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid var(--qdp-border) !important;
}

.gr-file {
    border-radius: 10px !important;
}

.gr-group,
.gr-panel,
.tab-nav button,
.gr-button {
    animation: qdpRiseIn 0.42s ease both;
}

@keyframes qdpRiseIn {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes qdpFloat {
    0%,
    100% {
        transform: translate3d(0, 0, 0) scale(1);
    }
    50% {
        transform: translate3d(0, -14px, 0) scale(1.03);
    }
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 0 8px 14px !important;
    }

    #qdp-header {
        padding-top: 14px;
        font-size: clamp(24px, 8vw, 34px);
        letter-spacing: 0.08em;
    }

    #qdp-subheader {
        font-size: 10px;
        letter-spacing: 0.14em;
    }

    .tab-nav button {
        padding: 7px 10px !important;
        font-size: 10px !important;
        margin-bottom: 6px !important;
    }
}
"""


# =========================================================================
# PIPELINE CALLBACKS
# =========================================================================

# =========================================================================
# CONFIGURACIÓN DE ALMACENAMIENTO Y RUTAS (SPEC SECTION 2)
# =========================================================================

ui_log = get_logger("ui")


# =========================================================================
# ADAPTADOR DE DATOS (SPEC SECTION 3) : SRT -> JSON
# =========================================================================
def _srt_time_to_seconds(time_str: str) -> float:
    """Convierte un timestamp de SRT '00:00:05,000' a float (segundos)."""
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def convert_srt_to_timestamps_json(srt_path: str, json_path: str):
    """Lee el SRT exportado por pyvideotrans y genera el timestamps.json para los Shorts."""
    if not os.path.exists(srt_path):
        return False
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    blocks = re.split(r'\n\s*\n', content.strip())
    timeline = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3 and " --> " in lines[1]:
            start_str, end_str = lines[1].split(" --> ")
            timeline.append({
                "start": _srt_time_to_seconds(start_str),
                "end": _srt_time_to_seconds(end_str),
                "text_es": " ".join(lines[2:]),
                "speaker": "SPEAKER_00"
            })
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    return True

def _ensure_dirs():
    ensure_local_dirs()
    setup_pipeline_logger(LOCAL_LOGS)
    os.makedirs(VOICE_REFS_ROOT, exist_ok=True)
    os.makedirs(VOICE_REFS_NORMALIZED, exist_ok=True)
    os.makedirs(VOICE_REFS_PREVIEWS, exist_ok=True)


def _voice_ref_audio_files(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        return []
    out: list[str] = []
    for base, _, files in os.walk(root_dir):
        for f in files:
            low = f.lower()
            if low.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")):
                out.append(os.path.join(base, f))
    out.sort()
    return out


def _prepare_voice_references() -> tuple[str, list[str], str | None]:
    _ensure_dirs()
    candidates = _voice_ref_audio_files(VOICE_REFS_SOURCE)
    if not candidates:
        candidates = _voice_ref_audio_files(VOICE_REFS_ROOT)
    if not candidates:
        return "❌ No se encontraron audios de referencia en reference_voices ni input/voice_refs", [], None

    catalog = []
    log_lines = [f"Preparando {len(candidates)} referencia(s)..."]
    for src in candidates:
        base_name = os.path.splitext(os.path.basename(src))[0]
        safe = re.sub(r"[^a-zA-Z0-9_-]", "-", base_name).strip("-") or "voice-ref"
        parent = os.path.basename(os.path.dirname(src)).strip()
        speaker_id = re.sub(r"[^a-zA-Z0-9_-]", "-", parent).strip("-") if parent else "unknown"
        ref_id = f"{speaker_id}-{safe}" if speaker_id and speaker_id != safe else safe

        normalized = os.path.join(VOICE_REFS_NORMALIZED, f"{ref_id}.wav")
        preview = os.path.join(VOICE_REFS_PREVIEWS, f"{ref_id}_preview.wav")

        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", src,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            normalized,
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", normalized,
            "-t", "12",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            preview,
        ], check=True)

        dur = _media_duration_seconds(normalized)
        catalog.append({
            "ref_id": ref_id,
            "speaker_hint": speaker_id,
            "source_path": src,
            "normalized_path": normalized,
            "preview_path": preview,
            "duration_sec": round(dur, 2),
        })
        log_lines.append(f"✅ {ref_id} ({dur:.1f}s)")

    with open(VOICE_REFS_CATALOG, "w", encoding="utf-8") as f:
        json.dump({"voices": catalog}, f, indent=2, ensure_ascii=False)

    choices = [f"{it['ref_id']}  ({it['duration_sec']:.1f}s)" for it in catalog]
    default_preview = catalog[0]["preview_path"] if catalog else None
    log_lines.append(f"Catálogo guardado en: {VOICE_REFS_CATALOG}")
    return "\n".join(log_lines), choices, default_preview


def _load_voice_catalog() -> list[dict]:
    if not os.path.isfile(VOICE_REFS_CATALOG):
        return []
    try:
        data = json.loads(open(VOICE_REFS_CATALOG, "r", encoding="utf-8").read())
        arr = data.get("voices") if isinstance(data, dict) else []
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def _voice_choice_to_preview(choice: str) -> str | None:
    if not choice:
        return None
    ref_id = choice.rsplit("  (", 1)[0].strip()
    for it in _load_voice_catalog():
        if it.get("ref_id") == ref_id:
            p = it.get("preview_path")
            if p and os.path.isfile(p):
                return p
    return None


def _latest_speaker_artifact(suffix: str) -> str | None:
    if not os.path.isdir(NETWORK_OUTPUT):
        return None
    files = []
    for f in os.listdir(NETWORK_OUTPUT):
        if f.endswith(suffix):
            p = os.path.join(NETWORK_OUTPUT, f)
            if os.path.isfile(p):
                files.append(p)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _load_speaker_context_for_ui() -> tuple[str, str]:
    profile_path = _latest_speaker_artifact("_speaker_profile.json")
    identity_path = _latest_speaker_artifact("_speaker_identity.json")
    if not profile_path:
        return "❌ No hay speaker_profile reciente. Ejecuta primero un doblaje con Speaker Clone activo.", "{}"

    profile = json.loads(open(profile_path, "r", encoding="utf-8").read())
    identity_map = {}
    if identity_path and os.path.isfile(identity_path):
        try:
            identity = json.loads(open(identity_path, "r", encoding="utf-8").read())
            for it in (identity.get("speakers") or []):
                k = str(it.get("speaker_id", "")).strip()
                if k:
                    identity_map[k] = it
        except Exception:
            identity_map = {}

    voices = _load_voice_catalog()
    voice_ids = [v.get("ref_id") for v in voices if v.get("ref_id")]

    lines = [f"Perfil: {os.path.basename(profile_path)}"]
    out_map = {}
    for spk in (profile.get("speakers") or []):
        sid = str(spk.get("speaker_id", "")).strip()
        if not sid:
            continue
        ident = identity_map.get(sid, {})
        role = ident.get("role_label", "-")
        gender = ident.get("gender_label", "-")
        conf = ident.get("confidence", 0)
        head = (spk.get("sample_sentences_head") or [])[:1]
        lines.append(
            f"- {sid} | rol={role} | genero={gender} | conf={conf} | ejemplo={head[0] if head else '-'}"
        )

        suggested = None
        if voice_ids:
            # Heurística ligera para sugerencia inicial editable.
            lower_sid = sid.lower()
            if "trump" in lower_sid:
                suggested = next((v for v in voice_ids if "trump" in v.lower()), voice_ids[0])
            elif "jiang" in lower_sid or role == "profesor":
                suggested = next((v for v in voice_ids if "jiang" in v.lower()), voice_ids[0])
            elif gender == "mujer":
                suggested = next((v for v in voice_ids if "mujer" in v.lower() or "dw" in v.lower()), voice_ids[0])
            else:
                suggested = voice_ids[0]

        out_map[sid] = suggested

    return "\n".join(lines), json.dumps(out_map, ensure_ascii=False, indent=2)


def _save_speaker_map_from_json(mapping_text: str) -> str:
    _ensure_dirs()
    try:
        payload = json.loads(mapping_text or "{}")
    except Exception as exc:
        return f"❌ JSON inválido: {exc}"

    if not isinstance(payload, dict):
        return "❌ El mapping debe ser un objeto JSON: {\"spk0\": \"ref_id\"}"

    catalog = _load_voice_catalog()
    by_ref = {it.get("ref_id"): it for it in catalog}
    if not by_ref:
        return "❌ Catálogo de voces vacío. Ejecuta primero 'Preparar referencias'"

    saved = {}
    for spk, ref_id in payload.items():
        sid = re.sub(r"[^a-zA-Z0-9_-]", "-", str(spk)).strip()
        if not sid:
            continue
        if not ref_id:
            continue
        ref = by_ref.get(str(ref_id).strip())
        if not ref:
            return f"❌ ref_id no existe en catálogo: {ref_id}"
        saved[sid] = {
            "ref": ref.get("normalized_path"),
            "ref_id": ref.get("ref_id"),
        }

    with open(VOICE_SPEAKER_MAP, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)
    return f"✅ Mapping guardado en {VOICE_SPEAKER_MAP} con {len(saved)} speaker(s)"


def _warmup_nllb200():
    if os.environ.get("QDP_WARMUP_NLLB", "1").lower() in {"0", "false", "no"}:
        ui_log.info("NLLB warm-up deshabilitado por QDP_WARMUP_NLLB")
        return

    try:
        ui_log.info("Iniciando warm-up de NLLB-200(Local)...")
        NLLB200Trans = None
        for mod_name in ("videotrans.translator._nllb200", "pyvideotrans.videotrans.translator._nllb200"):
            try:
                NLLB200Trans = importlib.import_module(mod_name).NLLB200Trans
                break
            except Exception:
                continue
        if NLLB200Trans is None:
            raise RuntimeError("No se pudo importar NLLB200Trans desde videotrans")

        warmup = NLLB200Trans(
            translate_type=2,
            text_list=[],
            source_code="en",
            target_code="es",
            target_language_name="Spanish",
            is_test=True,
        )
        warmup._download()
        warmup._unload()
        ui_log.info("Warm-up de NLLB-200(Local) completado")
    except Exception as exc:
        ui_log.warning(f"Warm-up de NLLB-200(Local) falló: {exc}")


def _start_background_warmups():
    threading.Thread(target=_warmup_nllb200, name="nllb-warmup", daemon=True).start()

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
    if not os.path.isdir(NETWORK_USER_INPUT):
        return []
    try:
        names = os.listdir(NETWORK_USER_INPUT)
    except OSError as e:
        ui_log.warning(f"No se puede leer {NETWORK_USER_INPUT}: {e}")
        return []
    rows = []
    for name in names:
        abs_path = os.path.join(NETWORK_USER_INPUT, name)
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
    if not os.path.isdir(LOCAL_TEMP):
        return []
    try:
        names = [f for f in os.listdir(LOCAL_TEMP) if f.endswith(".json")]
    except OSError:
        return []
    rows = []
    for name in sorted(names):
        abs_path = os.path.join(LOCAL_TEMP, name)
        try:
            size_kb = os.path.getsize(abs_path) / 1024
        except OSError:
            continue
        rows.append(f"{name}  ({size_kb:.1f} KB)")
    return rows


def _get_output_videos():
    """Videos finales disponibles en NETWORK_OUTPUT para generar reels."""
    _ensure_dirs()
    if not os.path.isdir(NETWORK_OUTPUT):
        return []
    rows = []
    try:
        names = os.listdir(NETWORK_OUTPUT)
    except OSError:
        return []

    for name in sorted(names):
        if not name.lower().endswith(_VIDEO_EXTS):
            continue
        abs_path = os.path.join(NETWORK_OUTPUT, name)
        if not os.path.isfile(abs_path):
            continue
        try:
            size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        except OSError:
            continue
        rows.append(f"{name}  ({size_mb:.1f} MB)")
    return rows


# pyvideotrans CLI espera IDs enteros para estos canales.
ASR_TYPE_MAP = {
    "faster": 0,
    "openai": 1,
    "qwen": 2,
}

TRANSLATE_TYPE_MAP = {
    "nllb_local": 2,
    "m2m100_local": 2,
    "google": 0,
    "microsoft": 1,
    "chatgpt": 3,
    "gemini": 5,
    "deepseek": 4,
    "openrouter": 9,
    "qwenmt": 12,
    "deepl": 16,
    "deeplx": 17,
    "ollama": 8,
    "minimax": 23,
}

TTS_TYPE_MAP = {
    "edge": 0,
    "qwen_local_clone": 1,
    "f5": 10,
    "cosyvoice": 12,
    "openai": 15,
}


def _find_pyvideotrans_output_dir(candidate_names: list[str], min_mtime: float | None = None) -> str | None:
    """Busca carpeta de salida del run actual de pyvideotrans de forma robusta."""
    output_root = os.path.join(PYVIDEOTRANS_ROOT, "output")
    if not os.path.isdir(output_root):
        return None

    def _has_artifacts(path: str) -> bool:
        try:
            names = os.listdir(path)
        except OSError:
            return False
        return any(n.lower().endswith((".mp4", ".srt")) for n in names)

    for name in candidate_names:
        if not name:
            continue
        path = os.path.join(output_root, name)
        if os.path.isdir(path) and _has_artifacts(path):
            return path

    folders = []
    for name in os.listdir(output_root):
        abs_path = os.path.join(output_root, name)
        if not os.path.isdir(abs_path):
            continue
        if min_mtime is not None:
            try:
                if os.path.getmtime(abs_path) < min_mtime:
                    continue
            except OSError:
                continue
        if _has_artifacts(abs_path):
            folders.append(abs_path)

    if not folders:
        return None

    folders.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return folders[0]


def _probe_media(path: str) -> dict | None:
    try:
        return ffmpeg.probe(path)
    except Exception:
        return None


def _media_duration_seconds(path: str) -> float:
    info = _probe_media(path)
    if not info:
        return 0.0
    try:
        return float((info.get("format") or {}).get("duration") or 0.0)
    except Exception:
        return 0.0


def _mp4_has_audio_stream(path: str) -> bool:
    info = _probe_media(path)
    if not info:
        return False
    streams = info.get("streams") or []
    return any((s.get("codec_type") or "").lower() == "audio" for s in streams)


def _pick_best_output_mp4(vt_output_dir: str, ui_log) -> str:
    mp4_files = [f for f in os.listdir(vt_output_dir) if f.lower().endswith('.mp4')]
    if not mp4_files:
        raise RuntimeError("PyVideoTrans no generó el video final.")

    ranked = []
    for name in mp4_files:
        abs_path = os.path.join(vt_output_dir, name)
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0
        has_audio = _mp4_has_audio_stream(abs_path)
        ranked.append((1 if has_audio else 0, mtime, abs_path))

    # Prefer files that actually contain an audio stream, then newest mtime.
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = ranked[0][2]
    if ranked[0][0] == 0:
        raise RuntimeError("Ningún MP4 de salida tiene pista de audio; se aborta para evitar publicar una salida degradada.")
    return best


def _build_video_filter(brightness: float, contrast: float, saturation: float, sharpness: float) -> str:
    return build_video_filter(brightness, contrast, saturation, sharpness)


def _apply_video_enhancements(
    input_video: str,
    output_video: str,
    brightness: float,
    contrast: float,
    saturation: float,
    sharpness: float,
):
    video_filter = _build_video_filter(brightness, contrast, saturation, sharpness)

    # High-quality path only: hardware encode, fail fast on errors.
    gpu_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_video,
        "-vf", video_filter,
        "-c:v", "h264_nvenc", "-preset", "p4",
        "-c:a", "copy",
        output_video,
    ]
    subprocess.run(gpu_cmd, check=True)

def run_phase1(
    video_file,
    test_mode,
):
    """
    Fase 1: ASR + traducción + diarización + análisis de speakers.
    Yields (phase_text, log, spk_section_update, *row_updates×6, btn_phase2_update, state_speakers_value).
    """
    N = 6  # max speaker rows

    def _empty_updates():
        """Return updates that hide everything."""
        base = [gr.update(visible=False), gr.update(value=""), gr.update(value=""), gr.update(value=None)]
        rows_flat = base * N
        return rows_flat

    def _yield_error(msg):
        empty_rows = []
        for _ in range(N):
            empty_rows += [gr.update(visible=False), gr.update(value=""), gr.update(value=""), gr.update(value=None)]
        yield (msg, "", gr.update(visible=False), *empty_rows, gr.update(visible=False), [],
               gr.update(interactive=True, value="Analizar video (Fase 1)"))

    if not video_file:
        yield from _yield_error("❌ Selecciona un video primero.")
        return

    video_path = _parse_dropdown_choice(video_file, NETWORK_USER_INPUT)
    if not video_path or not os.path.isfile(video_path):
        yield from _yield_error(f"❌ Archivo inválido: {video_path}")
        return

    llm_model = os.environ.get("QDP_TRANSLATE_MODEL", "nllb_local").strip().lower()
    translate_type = TRANSLATE_TYPE_MAP.get(llm_model)
    if translate_type is None:
        yield from _yield_error(f"❌ Translate model desconocido: {llm_model}")
        return

    asr_type = ASR_TYPE_MAP.get("faster")
    asr_model_name = os.environ.get("QDP_ASR_MODEL", "large-v3-turbo").strip() or "large-v3-turbo"
    _ensure_dirs()
    clear_log()
    setup_pipeline_logger(LOCAL_LOGS)

    result = {"phase": "Iniciando Fase 1 (análisis)...", "target_dir": None, "done": False, "error": None}

    def worker():
        try:
            current_video_path = video_path

            if test_mode:
                result["phase"] = "Modo Test: Recortando video a 30s..."
                ui_log.info(result["phase"])
                source_name, source_ext = os.path.splitext(os.path.basename(current_video_path))
                safe_name = re.sub(r'[\s\. #*?!:"]', '-', source_name)
                test_video_path = os.path.join(LOCAL_TEMP, f"test30s_{safe_name}{source_ext or '.mp4'}")
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                    "-i", current_video_path, "-t", "30", "-c", "copy", test_video_path
                ], check=True)
                current_video_path = test_video_path

            cmd = [
                PYVIDEOTRANS_PYTHON, PYVIDEOTRANS_CLI,
                "--task", "analyze",
                "--name", current_video_path,
                "--source_language_code", "en",
                "--target_language_code", "es",
                "--recogn_type", str(asr_type),
                "--model_name", asr_model_name,
                "--translate_type", str(translate_type),
                "--cuda",
                "--enable_diariz",
                "--nums_diariz", "0",
            ]

            result["phase"] = "Fase 1: Transcripción + traducción + diarización..."
            ui_log.info(f"Ejecutando Fase 1: {' '.join(cmd)}")

            child_env = os.environ.copy()
            child_env.setdefault("PYTHONUTF8", "1")
            child_env.setdefault("PYTHONIOENCODING", "utf-8")
            child_env.setdefault("TRANSFORMERS_VERBOSITY", "error")
            child_env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            child_env.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
            child_env.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
            child_env = _inject_ai_env_to_child(child_env)
            ui_log.info(
                f"[ENV-PHASE1] gemini_api_set={bool(child_env.get('API_GOOGLE_STUDIO'))} "
                f"gemini_model={child_env.get('GEMINI_MODEL', '')}"
            )

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=PYVIDEOTRANS_ROOT,
                env=child_env,
            )
            analyze_re = re.compile(r"^ANALYZE_DONE:(.+)$")
            for line in iter(process.stdout.readline, ""):
                if line:
                    line_clean = line.strip()
                    ui_log.info(f"[VT-PHASE1] {line_clean}")
                    m = analyze_re.match(line_clean)
                    if m:
                        result["target_dir"] = m.group(1).strip()

            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"Fase 1 falló con código {process.returncode}")

            result["done"] = True
            result["phase"] = "✅ Fase 1 completada. Asigna voces y arranca Fase 2."
            ui_log.info(result["phase"])

        except Exception as exc:
            result["error"] = str(exc)
            result["phase"] = f"❌ Error Fase 1: {exc}"
            ui_log.error(f"Fase 1 error: {exc}\n{traceback.format_exc()}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Stream progress while Phase 1 runs
    empty_rows = []
    for _ in range(N):
        empty_rows += [gr.update(visible=False), gr.update(value=""), gr.update(value=""), gr.update(value=None)]

    while t.is_alive():
        time.sleep(1.0)
        yield (result["phase"], read_log_tail(800), gr.update(visible=False), *empty_rows, gr.update(visible=False), [],
               gr.update(interactive=False, value="⏳ Analizando..."))

    t.join()

    if result["error"] or not result["done"]:
        yield from _yield_error(result["phase"])
        return

    # --- Parse artifacts from target_dir ---
    target_dir = result.get("target_dir")
    if not target_dir:
        # Fallback: scan pyvideotrans/output for newest dir with speaker_profile.json
        vt_output = os.path.join(PYVIDEOTRANS_ROOT, "output")
        candidates = []
        if os.path.isdir(vt_output):
            for name in os.listdir(vt_output):
                p = os.path.join(vt_output, name)
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, "speaker_profile.json")):
                    candidates.append(p)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            target_dir = candidates[0]

    speakers = []
    if target_dir:
        profile_path = os.path.join(target_dir, "speaker_profile.json")
        identity_path = os.path.join(target_dir, "speaker_identity.json")
        profile = {}
        identity_map = {}
        if os.path.isfile(profile_path):
            try:
                profile = json.loads(open(profile_path, "r", encoding="utf-8").read())
            except Exception:
                profile = {}
        if os.path.isfile(identity_path):
            try:
                raw = json.loads(open(identity_path, "r", encoding="utf-8").read())
                for item in (raw.get("speakers") or []):
                    k = str(item.get("speaker_id", "")).strip()
                    if k:
                        identity_map[k] = item
            except Exception:
                identity_map = {}

        for spk in (profile.get("speakers") or []):
            sid = str(spk.get("speaker_id", "")).strip()
            if not sid:
                continue
            ident = identity_map.get(sid, {})
            role = ident.get("role_label", "")
            gender_raw = ident.get("gender_label", "")
            # Build compact 1-2 word label
            label_parts = [p for p in [role, gender_raw] if p and p not in ("-", "unknown")]
            ai_label = " ".join(label_parts[:2]) if label_parts else sid
            head_sents = (spk.get("sample_sentences_head") or [])
            speakers.append({
                "speaker_id": sid,
                "ai_label": ai_label,
                "sample": head_sents[0] if head_sents else "",
                "target_dir": target_dir,
            })

    if not speakers:
        # No diarization data — still allow Phase 2 with a single default row
        ui_log.warning("Fase 1: no se detectaron speakers individuales (diarización no activa o sin marcas).")
        speakers = [{"speaker_id": "default", "ai_label": "narrador", "sample": "", "target_dir": target_dir or ""}]

    # Fixed choices to avoid dynamic value/label mismatches in Gradio updates.
    voice_choices = STATIC_VOICE_LABELS[:]

    # Build per-row updates (N_SPK_MAX = 6 rows)
    row_updates = []
    for i in range(N):
        if i < len(speakers):
            spk = speakers[i]
            default_voice = voice_choices[0] if voice_choices else None
            row_updates += [
                gr.update(visible=True),                # group visibility
                gr.update(value=spk["speaker_id"]),     # id textbox
                gr.update(value=spk["ai_label"]),        # label textbox
                gr.update(value=default_voice),  # voice dd
            ]
        else:
            row_updates += [
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=None),
            ]

    yield (
        result["phase"],
        read_log_tail(800),
        gr.update(visible=True),   # spk_section
        *row_updates,              # 6×4 = 24 items
        gr.update(visible=True),   # btn_phase2
        speakers,                  # state_speakers
        gr.update(interactive=True, value="Analizar video (Fase 1)"),  # btn_phase1
    )


def run_phase2(
    video_file,
    test_mode,
    state_speakers,
    spk_voice_0, spk_voice_1, spk_voice_2, spk_voice_3, spk_voice_4, spk_voice_5,
    spk_label_0, spk_label_1, spk_label_2, spk_label_3, spk_label_4, spk_label_5,
    apply_video_fx,
    brightness,
    contrast,
    color,
    sharpness,
):
    """
    Fase 2: Doblaje completo usando el mapa de voz elegido en la UI.
    Yields same 7-tuple as old run_pyvideotrans_pipeline.
    """
    if not video_file:
        yield "Error", "Por favor selecciona un video.", None, None, None, None, None, gr.update(interactive=True, value="Doblar con voces asignadas (Fase 2)")
        return

    video_path = _parse_dropdown_choice(video_file, NETWORK_USER_INPUT)
    if not video_path or not os.path.isfile(video_path):
        yield "Error", f"Archivo inválido: {video_path}", None, None, None, None, None, gr.update(interactive=True, value="Doblar con voces asignadas (Fase 2)")
        return

    # --- Write speaker voice map from UI assignments ---
    voice_inputs = [spk_voice_0, spk_voice_1, spk_voice_2, spk_voice_3, spk_voice_4, spk_voice_5]
    active_speakers = state_speakers or []
    catalog = _load_voice_catalog()
    by_ref = {it["ref_id"]: it for it in catalog}
    speaker_map = {}
    for i, spk_data in enumerate(active_speakers[:6]):
        choice = voice_inputs[i]
        if not choice:
            continue
        ref_id = STATIC_VOICE_REF_BY_LABEL.get(str(choice).strip())
        if not ref_id:
            # Backward compatibility for old dynamic values.
            ref_id = str(choice).rsplit("  (", 1)[0].strip()
        ref_entry = by_ref.get(ref_id)
        if ref_entry:
            sid = spk_data["speaker_id"]
            speaker_map[sid] = {
                "ref": ref_entry["normalized_path"],
                "ref_id": ref_id,
            }
    _ensure_dirs()
    with open(VOICE_SPEAKER_MAP, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2, ensure_ascii=False)
    ui_log.info(f"Voice map escrito: {len(speaker_map)} speaker(s) → {VOICE_SPEAKER_MAP}")

    # --- Now run the full pipeline (Fase 2) ---
    llm_model = os.environ.get("QDP_TRANSLATE_MODEL", "nllb_local").strip().lower()
    tts_model = "qwen_local_clone"

    asr_type = ASR_TYPE_MAP.get("faster")
    asr_model_name = os.environ.get("QDP_ASR_MODEL", "large-v3-turbo").strip() or "large-v3-turbo"
    translate_type = TRANSLATE_TYPE_MAP.get(llm_model)
    tts_type = TTS_TYPE_MAP.get(tts_model)
    if asr_type is None or translate_type is None or tts_type is None:
        yield "Error", f"Config inválida: LLM={llm_model}, TTS={tts_model}", None, None, None, None, None, gr.update(interactive=True, value="Doblar con voces asignadas (Fase 2)")
        return

    clear_log()
    setup_pipeline_logger(LOCAL_LOGS)

    ui_log.info("=" * 80)
    ui_log.info("INICIANDO MOTOR PYVIDEOTRANS — FASE 2 (DOBLAJE)")
    ui_log.info(f"Video: {video_path}")
    ui_log.info(f"Speaker map: {len(speaker_map)} speaker(s) asignados")
    ui_log.info("=" * 80)

    result = {"audio": None, "video": None, "json_timestamps": None, "srt_file": None,
              "download_audio": None, "status": "running", "error": None,
              "phase": "Fase 2: Iniciando doblaje con voces asignadas..."}

    def worker():
        try:
            current_video_path = video_path

            if test_mode:
                result["phase"] = "Modo Test: Recortando video a 30s..."
                ui_log.info(result["phase"])
                source_name, source_ext = os.path.splitext(os.path.basename(current_video_path))
                safe_name = re.sub(r'[\s\. #*?!:"]', '-', source_name)
                test_video_path = os.path.join(LOCAL_TEMP, f"test30s_{safe_name}{source_ext or '.mp4'}")
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                    "-i", current_video_path, "-t", "30", "-c", "copy", test_video_path
                ], check=True)
                current_video_path = test_video_path

            cmd = [
                PYVIDEOTRANS_PYTHON, PYVIDEOTRANS_CLI,
                "--task", "vtv",
                "--name", current_video_path,
                "--source_language_code", "en",
                "--target_language_code", "es",
                "--recogn_type", str(asr_type),
                "--model_name", asr_model_name,
                "--translate_type", str(translate_type),
                "--tts_type", str(tts_type),
                "--cuda",
                "--subtitle_type", "0",
                "--video_autorate",
                "--voice_role", "clone",
                "--enable_diariz",
                "--nums_diariz", "0",
            ]

            result["phase"] = "Fase 2: Pipeline de doblaje activo..."
            ui_log.info(f"Ejecutando Fase 2: {' '.join(cmd)}")
            run_started_at = time.time()

            child_env = os.environ.copy()
            child_env.setdefault("PYTHONUTF8", "1")
            child_env.setdefault("PYTHONIOENCODING", "utf-8")
            child_env.setdefault("TRANSFORMERS_VERBOSITY", "error")
            child_env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            child_env.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
            child_env.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
            child_env = _inject_ai_env_to_child(child_env)
            ui_log.info(
                f"[ENV-PHASE2] gemini_api_set={bool(child_env.get('API_GOOGLE_STUDIO'))} "
                f"gemini_model={child_env.get('GEMINI_MODEL', '')}"
            )
            sox_candidates = [
                os.path.join(PYVIDEOTRANS_ROOT, "ffmpeg", "sox"),
                os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe\sox-14.4.2"),
                r"C:\Program Files\sox-14-4-2",
            ]
            sox_paths = [p for p in sox_candidates if os.path.isdir(p)]
            if sox_paths:
                child_env["PATH"] = os.pathsep.join(sox_paths + [child_env.get("PATH", "")])

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=PYVIDEOTRANS_ROOT,
                env=child_env,
            )
            vt_recent_lines = []
            qwen_re = re.compile(r"\[qwen3tts\]\s+(batch_start|batch_done)\s+([^\n]*)", re.IGNORECASE)
            qwen_generated_re = re.compile(r"generated=(\d+)/(\d+)", re.IGNORECASE)
            qwen_eta_re = re.compile(r"eta=(\d{2}:\d{2})", re.IGNORECASE)

            for line in iter(process.stdout.readline, ""):
                if line:
                    line_clean = line.strip()
                    ui_log.info(f"[VT-PHASE2] {line_clean}")
                    qwen_match = qwen_re.search(line_clean)
                    if qwen_match:
                        generated = qwen_generated_re.search(line_clean)
                        eta = qwen_eta_re.search(line_clean)
                        if generated:
                            done_n = int(generated.group(1))
                            total_n = int(generated.group(2))
                            pct = (done_n / max(total_n, 1)) * 100.0
                            eta_txt = eta.group(1) if eta else "--:--"
                            result["phase"] = f"Qwen TTS: {done_n}/{total_n} ({pct:.1f}%) · ETA {eta_txt}"
                        else:
                            result["phase"] = "Qwen TTS en progreso..."
                    vt_recent_lines.append(line_clean)
                    if len(vt_recent_lines) > 120:
                        vt_recent_lines.pop(0)

            process.wait()
            if process.returncode != 0:
                joined = "\n".join(vt_recent_lines)
                hint = None
                if "Kernel size can't be greater than actual input size" in joined:
                    hint = "Referencia de voz demasiado corta para Qwen clone. Usa refs ≥1.2s."
                raise RuntimeError(
                    f"Fase 2 falló con código {process.returncode}. {hint or ''}"
                )

            joined = "\n".join(vt_recent_lines)
            m = re.search(r"Success:\s*(\d+)\s*,\s*Failed:\s*(\d+)", joined)
            if m:
                ok_n = int(m.group(1))
                fail_n = int(m.group(2))
                if fail_n > 0 and ok_n <= fail_n:
                    raise RuntimeError(
                        f"TTS con demasiados fallos (Success={ok_n}, Failed={fail_n}). "
                        "Revisa las referencias de voz."
                    )

            result["phase"] = "Recolectando outputs..."
            ui_log.info(result["phase"])

            original_base = os.path.splitext(os.path.basename(video_path))[0]
            original_safe = re.sub(r'[\s\. #*?!:"]', '-', original_base)
            current_base = os.path.splitext(os.path.basename(current_video_path))[0]
            current_safe = re.sub(r'[\s\. #*?!:"]', '-', current_base)

            vt_output_dir = _find_pyvideotrans_output_dir(
                candidate_names=[current_safe, current_base, original_safe, original_base],
                min_mtime=run_started_at - 120,
            )
            if not vt_output_dir:
                raise RuntimeError("Fase 2 terminó pero no se encontró la carpeta de salida.")

            safe_name = original_safe
            vt_mp4 = _pick_best_output_mp4(vt_output_dir, ui_log)
            final_mp4 = os.path.join(NETWORK_OUTPUT, f"{safe_name}_dubbed.mp4")
            shutil.copy2(vt_mp4, final_mp4)

            if apply_video_fx:
                result["phase"] = "Aplicando ajustes visuales..."
                enhanced_tmp = os.path.join(NETWORK_OUTPUT, f"{safe_name}_dubbed_enhanced_tmp.mp4")
                _apply_video_enhancements(final_mp4, enhanced_tmp, brightness, contrast, color, sharpness)
                os.replace(enhanced_tmp, final_mp4)

            result["video"] = final_mp4

            final_wav = os.path.join(NETWORK_OUTPUT, f"{safe_name}_dubbed.wav")
            (
                ffmpeg.input(final_mp4)
                .output(final_wav, acodec="pcm_s16le", ac=1, ar="16000")
                .overwrite_output()
                .run(quiet=True)
            )
            video_sec = _media_duration_seconds(final_mp4)
            audio_sec = _media_duration_seconds(final_wav)
            min_ratio = float(os.environ.get("QDP_MIN_DUB_AUDIO_RATIO", "0.20") or "0.20")
            if video_sec > 0 and audio_sec > 0 and (audio_sec / video_sec) < min_ratio:
                raise RuntimeError(
                    f"Audio doblado demasiado corto (audio={audio_sec:.1f}s, video={video_sec:.1f}s). "
                    "Indica fallos masivos en TTS. Revisa referencias de voz."
                )
            result["audio"] = final_wav
            result["download_audio"] = final_wav

            srt_files = [f for f in os.listdir(vt_output_dir) if f.lower().endswith(".srt")]
            if srt_files:
                srt_files.sort(key=lambda f: os.path.getmtime(os.path.join(vt_output_dir, f)), reverse=True)
                srt_path = os.path.join(vt_output_dir, srt_files[0])
                final_srt = os.path.join(NETWORK_OUTPUT, f"{safe_name}_subtitles.srt")
                shutil.copy2(srt_path, final_srt)
                result["srt_file"] = final_srt
                json_out = os.path.join(NETWORK_OUTPUT, f"{safe_name}_timestamps.json")
                convert_srt_to_timestamps_json(srt_path, json_out)
                result["json_timestamps"] = json_out

            result["status"] = "ok"
            result["phase"] = "¡Doblaje Fase 2 Completado!"
            ui_log.info(result["phase"])

        except Exception as e:
            result["status"] = "fail"
            result["error"] = str(e)
            result["phase"] = f"❌ Error Fase 2: {e}"
            ui_log.error(f"Fase 2 falló: {e}\n{traceback.format_exc()}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while t.is_alive():
        time.sleep(1.0)
        yield (
            result["phase"], read_log_tail(1500),
            result["audio"], result["video"],
            result.get("json_timestamps"), result.get("srt_file"), result.get("download_audio"),
            gr.update(interactive=False, value="⏳ Doblando..."),
        )

    t.join()
    yield (
        result["phase"], read_log_tail(1500),
        result["audio"], result["video"],
        result.get("json_timestamps"), result.get("srt_file"), result.get("download_audio"),
        gr.update(interactive=True, value="Doblar con voces asignadas (Fase 2)"),
    )


def run_pyvideotrans_pipeline(
    video_file,
    test_mode,
    speaker_clone_mode,
    apply_video_fx,
    brightness,
    contrast,
    color,
    sharpness,
):
    """
    Ejecuta el CLI nativo de pyvideotrans y procesa sus outputs para nuestro sistema.
    """
    if not video_file:
        yield "Error", "Por favor selecciona un video.", None, None, None, None, None
        return

    video_path = _parse_dropdown_choice(video_file, NETWORK_USER_INPUT)
    if not video_path or not os.path.isfile(video_path):
        yield "Error", f"Archivo invalido. video={video_path}", None, None, None, None, None
        return

    # Defaults fijos para simplificar UX (sin 4 selectores técnicos en UI).
    asr_model = "faster"
    asr_model_name = os.environ.get("QDP_ASR_MODEL", "large-v3-turbo").strip() or "large-v3-turbo"
    target_lang = "es"
    llm_model = os.environ.get("QDP_TRANSLATE_MODEL", "nllb_local").strip().lower()
    tts_model = "edge"

    effective_tts_model = tts_model
    if speaker_clone_mode and tts_model != "qwen_local_clone":
        # Guardrail: speaker clone only works with local Qwen clone path.
        effective_tts_model = "qwen_local_clone"

    asr_type = ASR_TYPE_MAP.get(asr_model)
    translate_type = TRANSLATE_TYPE_MAP.get(llm_model)
    tts_type = TTS_TYPE_MAP.get(effective_tts_model)
    if asr_type is None or translate_type is None or tts_type is None:
        yield "Error", (
            "Configuracion de modelos no valida para pyvideotrans: "
            f"ASR={asr_model}, LLM={llm_model}, TTS={effective_tts_model}"
        ), None, None, None, None, None
        return
    
    _ensure_dirs()
    clear_log()
    setup_pipeline_logger(LOCAL_LOGS)
    
    ui_log.info("=" * 80)
    ui_log.info("INICIANDO MOTOR PYVIDEOTRANS")
    ui_log.info(f"Video input (Red): {video_path}")
    if speaker_clone_mode and tts_model != "qwen_local_clone":
        ui_log.warning(
            f"SpeakerClone activo con TTS={tts_model}; forzando qwen_local_clone para clonado de voz real."
        )
    ui_log.info(
        f"Config fija -> ASR: {asr_model}({asr_type}), Target: {target_lang}, "
        f"TTS: {effective_tts_model}({tts_type}), LLM: {llm_model}({translate_type}), "
        f"SpeakerClone: {speaker_clone_mode}, ASR model: {asr_model_name}"
    )
    ui_log.info(f"Procesando en NVMe: {LOCAL_DIR}")
    ui_log.info("=" * 80)

    result = {"audio": None, "video": None, "json_timestamps": None, "srt_file": None, "download_audio": None, "status": "running", "error": None, "phase": "Preparando motor pyvideotrans..."}

    def worker():
        try:
            current_video_path = video_path
            
            if test_mode:
                result["phase"] = "Modo Test: Recortando video a 30 segundos..."
                ui_log.info(result["phase"])
                source_name, source_ext = os.path.splitext(os.path.basename(current_video_path))
                safe_name = re.sub(r'[\s\. #*?!:"]', '-', source_name)
                test_video_path = os.path.join(LOCAL_TEMP, f"test30s_{safe_name}{source_ext or '.mp4'}")
                
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                    "-i", current_video_path, "-t", "30", "-c", "copy", test_video_path
                ], check=True)
                current_video_path = test_video_path

            cmd = [
                PYVIDEOTRANS_PYTHON, PYVIDEOTRANS_CLI,
                "--task", "vtv",
                "--name", current_video_path,
                "--source_language_code", "en",
                "--target_language_code", target_lang,
                "--recogn_type", str(asr_type),
                "--model_name", asr_model_name,
                "--translate_type", str(translate_type),
                "--tts_type", str(tts_type),
                "--cuda",
                "--subtitle_type", "0",  # Sin subs quemados (como requiere el SPEC)
                "--video_autorate"       # Delega la isocronía a pyvideotrans
            ]

            if speaker_clone_mode and effective_tts_model == "qwen_local_clone":
                # Flujo speaker-aware: diarizar -> detectar speakers -> clone por speaker.
                cmd.extend([
                    "--voice_role", "clone",
                    "--enable_diariz",
                    "--nums_diariz", "0",
                ])
            elif effective_tts_model == "edge":
                voice_role = "es-ES-AlvaroNeural" if target_lang == "es" else "en-US-ChristopherNeural"
                cmd.extend(["--voice_role", voice_role])
            elif effective_tts_model == "qwen_local_clone":
                # Sin diarización explícita, Qwen local puede clonar por línea.
                cmd.extend(["--voice_role", "clone"])
            
            result["phase"] = "Fase Activa: Procesando pipeline pesado de pyvideotrans..."
            ui_log.info(f"Ejecutando subproceso: {' '.join(cmd)}")
            run_started_at = time.time()

            # On Windows, pyvideotrans may print CJK chars while the host console
            # default codec is cp1252; force UTF-8 in the child process to avoid
            # UnicodeEncodeError during model download checks.
            child_env = os.environ.copy()
            child_env.setdefault("PYTHONUTF8", "1")
            child_env.setdefault("PYTHONIOENCODING", "utf-8")
            # Reduce noisy HF warnings like repeated pad_token_id messages.
            child_env.setdefault("TRANSFORMERS_VERBOSITY", "error")
            # Ensure ctranslate2 finds models in standard HuggingFace cache
            child_env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            child_env.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
            child_env.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
            child_env = _inject_ai_env_to_child(child_env)
            ui_log.info(
                f"[ENV-PIPELINE] gemini_api_set={bool(child_env.get('API_GOOGLE_STUDIO'))} "
                f"gemini_model={child_env.get('GEMINI_MODEL', '')}"
            )
            # Ensure SoX is discoverable (Winget/Program Files/local vendored paths).
            sox_candidates = [
                os.path.join(PYVIDEOTRANS_ROOT, "ffmpeg", "sox"),
                os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe\sox-14.4.2"),
                r"C:\Program Files\sox-14-4-2",
                r"C:\Program Files (x86)\sox-14-4-2",
            ]
            sox_paths = [p for p in sox_candidates if os.path.isdir(p)]
            if sox_paths:
                child_env["PATH"] = os.pathsep.join(sox_paths + [child_env.get("PATH", "")])
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=PYVIDEOTRANS_ROOT,
                env=child_env,
            )
            vt_recent_lines = []
            qwen_re = re.compile(
                r"\[qwen3tts\]\s+(batch_start|batch_done)\s+([^\n]*)",
                re.IGNORECASE,
            )
            qwen_generated_re = re.compile(r"generated=(\d+)/(\d+)", re.IGNORECASE)
            qwen_eta_re = re.compile(r"eta=(\d{2}:\d{2})", re.IGNORECASE)
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    line_clean = line.strip()
                    ui_log.info(f"[VT-CORE] {line_clean}")
                    qwen_match = qwen_re.search(line_clean)
                    if qwen_match:
                        generated = qwen_generated_re.search(line_clean)
                        eta = qwen_eta_re.search(line_clean)
                        if generated:
                            done_n = int(generated.group(1))
                            total_n = int(generated.group(2))
                            pct = (done_n / max(total_n, 1)) * 100.0
                            eta_txt = eta.group(1) if eta else "--:--"
                            result["phase"] = (
                                f"Qwen TTS en progreso: {done_n}/{total_n} "
                                f"({pct:.1f}%) · ETA {eta_txt}"
                            )
                        else:
                            result["phase"] = "Qwen TTS en progreso..."
                    vt_recent_lines.append(line_clean)
                    if len(vt_recent_lines) > 120:
                        vt_recent_lines.pop(0)
                    
            process.wait()
            
            if process.returncode != 0:
                hint = None
                joined = "\n".join(vt_recent_lines)
                if "Kernel size can't be greater than actual input size" in joined:
                    hint = (
                        "Qwen clone failed: reference audio is too short for voice prompt encoding. "
                        "Use longer clone refs (>=1.2s), or disable speaker clone for this run."
                    )
                elif "Torch 2.5.1+cu121 es incompatible" in joined:
                    hint = "NLLB warm-up failed due Torch version mismatch (requires torch>=2.6)."

                if hint:
                    raise RuntimeError(
                        f"PyVideoTrans falló con código de salida {process.returncode}. {hint}"
                    )
                raise RuntimeError(f"PyVideoTrans falló con código de salida {process.returncode}")

            joined = "\n".join(vt_recent_lines)
            # If TTS generated mostly failed lines, stop here to avoid publishing a mostly silent "success" video.
            m = re.search(r"Success:\s*(\d+)\s*,\s*Failed:\s*(\d+)", joined)
            if m:
                ok_n = int(m.group(1))
                fail_n = int(m.group(2))
                if fail_n > 0 and ok_n <= fail_n:
                    raise RuntimeError(
                        "TTS generó demasiados fallos "
                        f"(Success={ok_n}, Failed={fail_n}). "
                        "No se publica video para evitar salida sin audio útil. "
                        "Prueba con referencia de voz más limpia/larga o reduce diarización."
                    )
            
            # === RECOLECTAR Y ADAPTAR OUTPUTS ===
            result["phase"] = "Adaptando datos e incrustando en el ecosistema QDP..."
            ui_log.info(result["phase"])
            
            original_base = os.path.splitext(os.path.basename(video_path))[0]
            original_safe = re.sub(r'[\s\. #*?!:"]', '-', original_base)
            current_base = os.path.splitext(os.path.basename(current_video_path))[0]
            current_safe = re.sub(r'[\s\. #*?!:"]', '-', current_base)

            vt_output_dir = _find_pyvideotrans_output_dir(
                candidate_names=[current_safe, current_base, original_safe, original_base],
                min_mtime=run_started_at - 120,
            )
            if not vt_output_dir:
                raise RuntimeError("PyVideoTrans terminó pero no se encontró carpeta de salida con artefactos.")

            safe_name = original_safe
            
            # 1. Encontrar y mover Video Doblado Final
            vt_mp4 = _pick_best_output_mp4(vt_output_dir, ui_log)
            final_mp4 = os.path.join(NETWORK_OUTPUT, f"{safe_name}_dubbed.mp4")
            shutil.copy2(vt_mp4, final_mp4)

            if apply_video_fx:
                result["phase"] = "Aplicando ajustes visuales al video final..."
                ui_log.info(
                    "Aplicando filtros de video: "
                    f"brightness={brightness}, contrast={contrast}, color={color}, sharpness={sharpness}"
                )
                enhanced_tmp = os.path.join(NETWORK_OUTPUT, f"{safe_name}_dubbed_enhanced_tmp.mp4")
                _apply_video_enhancements(
                    input_video=final_mp4,
                    output_video=enhanced_tmp,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=color,
                    sharpness=sharpness,
                )
                os.replace(enhanced_tmp, final_mp4)

            result["video"] = final_mp4
            
            # 2. Extraer Audio Aislado para LipSync (Pestaña 2)
            final_wav = os.path.join(NETWORK_OUTPUT, f"{safe_name}_dubbed.wav")
            (
                ffmpeg.input(final_mp4)
                .output(final_wav, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )

            # Guardrail: reject outputs where dubbed audio is unrealistically short.
            video_sec = _media_duration_seconds(final_mp4)
            audio_sec = _media_duration_seconds(final_wav)
            min_ratio = float(os.environ.get("QDP_MIN_DUB_AUDIO_RATIO", "0.20") or "0.20")
            if video_sec > 0 and audio_sec > 0 and (audio_sec / video_sec) < min_ratio:
                raise RuntimeError(
                    "El audio doblado quedó demasiado corto para el video "
                    f"(audio={audio_sec:.2f}s, video={video_sec:.2f}s, ratio={audio_sec/video_sec:.2f} < {min_ratio:.2f}). "
                    "Esto suele indicar fallos masivos en TTS/diarización o referencias inválidas."
                )
            result["audio"] = final_wav
            result["download_audio"] = final_wav  # Para descarga
            
            # 3. Construir el Contrato de Datos (Adaptador de JSON)
            srt_path = None
            srt_files = [f for f in os.listdir(vt_output_dir) if f.lower().endswith('.srt')]
            if srt_files:
                srt_files.sort(
                    key=lambda f: os.path.getmtime(os.path.join(vt_output_dir, f)),
                    reverse=True,
                )
                srt_path = os.path.join(vt_output_dir, srt_files[0])
                # Copiar SRT a output para descarga
                final_srt = os.path.join(NETWORK_OUTPUT, f"{safe_name}_subtitles.srt")
                shutil.copy2(srt_path, final_srt)
                result["srt_file"] = final_srt
                
                # Generar JSON timestamps
                json_out = os.path.join(NETWORK_OUTPUT, f"{safe_name}_timestamps.json")
                convert_srt_to_timestamps_json(srt_path, json_out)
                result["json_timestamps"] = json_out
                ui_log.info(f"Contrato de Datos JSON construido exitosamente en: {json_out}")
            else:
                ui_log.warning("PyVideoTrans no generó SRT; se omite timestamps.json en este run.")

            # 4. Exponer artefactos de speaker intelligence para mapeo manual en UI.
            for artifact_name in ("speaker_profile.json", "speaker_identity.json"):
                src_art = os.path.join(vt_output_dir, artifact_name)
                if not os.path.isfile(src_art):
                    continue
                dst_art = os.path.join(NETWORK_OUTPUT, f"{safe_name}_{artifact_name}")
                shutil.copy2(src_art, dst_art)
                ui_log.info(f"Artefacto speaker copiado: {dst_art}")
            
            result["status"] = "ok"
            result["phase"] = "¡Doblaje Master Completado!"
            ui_log.info(result["phase"])
            
        except Exception as e:
            result["status"] = "fail"
            result["error"] = str(e)
            result["phase"] = f"Error crítico: {e}"
            ui_log.error(f"Fallo en motor PyVideoTrans: {e}\n{traceback.format_exc()}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while t.is_alive():
        time.sleep(1.0)
        yield (
            result["phase"], 
            read_log_tail(1500), 
            result["audio"], 
            result["video"],
            result.get("json_timestamps"),
            result.get("srt_file"),
            result.get("download_audio")
        )

    t.join()
    log_tail = read_log_tail(1500)
    yield (
        result["phase"], 
        log_tail, 
        result["audio"], 
        result["video"],
        result.get("json_timestamps"),
        result.get("srt_file"),
        result.get("download_audio")
    )

with gr.Blocks(
    title="Quantum Dubbing Pipeline",
    analytics_enabled=False,
    theme=gr.themes.Base(primary_hue="amber", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.HTML(STARFIELD_HTML)
    gr.HTML("<div id='qdp-header'>QUANTUM DUBBING PIPELINE</div>"
            "<div id='qdp-subheader'>Apple-Style Minimalist Interface</div>")

    with gr.Tabs():
        # =================================================================
        # TAB 1: DOBLAJE MASTER (Workflow 1)
        # =================================================================
        with gr.Tab("1 · Doblaje Master"):
            with gr.Accordion("¿Cómo subir archivos? (AWS S3)", open=False):
                gr.Markdown(
                    "Sube tus archivos originales al volumen de red mediante AWS CLI. "
                    "El sistema los detectará automáticamente en los menús desplegables.\n"
                    "```bash\n"
                    "aws s3 ls --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io s3://l9dt5rqorw/\n"
                    "aws s3 cp tu_video.mp4 s3://l9dt5rqorw/user_input/ --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io\n"
                    "```\n"
                    "Una vez subidos, haz clic en **Refrescar videos**."
                )
            
            with gr.Row():
                dd_video = gr.Dropdown(
                    choices=_get_user_videos(), value=None,
                    label="Seleccionar video original", interactive=True, scale=3,
                )
                btn_refresh_files = gr.Button("Refrescar videos", scale=1, min_width=120)
                up_video = gr.File(label="Subir video desde PC", file_types=["video"], type="filepath", scale=2)
                
            with gr.Row():
                cb_test_mode = gr.Checkbox(label="Modo de prueba (cortar a 30s)", value=False, interactive=True)

            # ── FASE 1: Análisis de Speakers ──────────────────────────────────────
            gr.Markdown("### 🎙️ Fase 1 · Analizar video")
            gr.Markdown(
                "Transcribe, traduce y detecta quiénes hablan. "
                "Después asigna una voz de referencia a cada speaker y lanza la Fase 2."
            )
            btn_phase1 = gr.Button("Analizar video (Fase 1)", variant="secondary", size="lg")

            # ── SPEAKER SECTION (oculto hasta completar Fase 1) ─────────────────
            N_SPK_MAX = 6
            _initial_voice_choices = STATIC_VOICE_LABELS[:]

            with gr.Column(visible=False) as spk_section:
                gr.Markdown("### 🗣️ Speakers detectados — Asigna una voz a cada speaker")

                # Pre-build N_SPK_MAX rows; Phase 1 callback makes them visible
                spk_groups = []
                spk_id_boxes = []
                spk_label_boxes = []
                spk_voice_dds = []
                for _i in range(N_SPK_MAX):
                    with gr.Group(visible=False) as _grp:
                        with gr.Row():
                            _id_box = gr.Textbox(
                                label="ID", value=f"spk{_i}", interactive=False,
                                scale=1, min_width=80,
                            )
                            _label_box = gr.Textbox(
                                label="Identidad IA (editable)", interactive=True,
                                scale=2,
                            )
                            _voice_dd = gr.Dropdown(
                                label="Voz de referencia",
                                choices=_initial_voice_choices,
                                value=_initial_voice_choices[0] if _initial_voice_choices else None,
                                interactive=True,
                                scale=3,
                                elem_classes=["speaker-voice-dd"],
                            )
                    spk_groups.append(_grp)
                    spk_id_boxes.append(_id_box)
                    spk_label_boxes.append(_label_box)
                    spk_voice_dds.append(_voice_dd)

            # State to pass detected speakers to Phase 2
            state_speakers = gr.State([])

            # ── FASE 2 ────────────────────────────────────────────────────────────
            btn_phase2 = gr.Button(
                "Doblar con voces asignadas (Fase 2)", variant="primary", size="lg", visible=False,
            )

            gr.Markdown("### 🖼️ Editor de Frame (Referencia de Color)")
            with gr.Row():
                frame_second = gr.Slider(
                    minimum=0,
                    maximum=300,
                    step=1,
                    value=10,
                    label="Segundo del video para extraer frame",
                    interactive=True,
                )
                btn_extract_frame = gr.Button("Extraer frame", variant="secondary")

            with gr.Row():
                slider_brightness = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="Brillo")
                slider_contrast = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="Contraste")
                slider_color = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="Color")
                slider_sharpness = gr.Slider(0.5, 2.5, value=1.0, step=0.05, label="Nitidez")

            with gr.Row():
                frame_original = gr.Image(label="Frame Original", interactive=False, type="filepath")
                frame_adjusted = gr.Image(label="Frame Ajustado", interactive=False, type="filepath")

            btn_apply_frame = gr.Button("Aplicar ajustes al frame", variant="secondary")
            ui_frame_log = gr.Textbox(
                label="Log Editor de Frame",
                interactive=False,
                lines=2,
                value="Esperando extracción de frame...",
            )

            cb_apply_video_fx = gr.Checkbox(
                label="Aplicar estos ajustes al video final doblado",
                value=True,
                interactive=True,
            )

            ui_active_phase = gr.Textbox(label="Fase Activa", value="Esperando inicio...", interactive=False, lines=1)
            ui_terminal = gr.Textbox(label="Terminal de Logs", elem_id="qdp-log", interactive=False, lines=15, autoscroll=True)
            
            with gr.Row():
                out_master_audio = gr.Audio(label="Audio Doblado", interactive=False)
                out_master_video = gr.Video(label="Resultado: Video Master Final (MP4)", interactive=False)
            
            with gr.Row():
                file_master_json = gr.File(label="Descargar JSON de timestamps", interactive=False)
                file_master_srt = gr.File(label="Descargar subtitulos SRT", interactive=False)
                file_master_audio = gr.File(label="Descargar audio WAV", interactive=False)

        # =================================================================
        # TAB 2: AGREGAR SUBTÍTULOS (Workflow 2)
        # =================================================================
        with gr.Tab("2 · Agregar Subtítulos"):
            gr.Markdown(
                "### Generador de subtitulos para video doblado\n"
                "Carga el video ya doblado y genera automáticamente subtítulos estilizados "
                "en español usando transcripción con IA (Whisper).\n"
                "El resultado será un video con subtítulos profesionales estilo documental."
            )
            
            with gr.Row():
                up_video_subs = gr.File(
                    label="Subir video o audio doblado", 
                    file_types=["video", "audio", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"],
                    type="filepath",
                    scale=2
                )
                dd_video_subs = gr.Dropdown(
                    choices=(_get_user_videos() + _get_user_audios()),
                    value=None,
                    label="O seleccionar de red",
                    interactive=True,
                    scale=2
                )
                btn_refresh_subs = gr.Button("Refrescar videos", scale=1, min_width=120)
            
            dd_subs_lang = gr.Dropdown(
                choices=["es", "en", "fr", "de", "pt", "zh"],
                value="es",
                label="Idioma para subtitulos",
                interactive=True
            )
            
            with gr.Row():
                btn_gen_subs = gr.Button("Generar subtitulos", variant="primary", scale=1)
                btn_render_subs = gr.Button("Renderizar video con subtitulos", scale=1)
            
            ui_subs_log = gr.Textbox(
                label="Terminal de Subtítulos",
                elem_id="qdp-log",
                interactive=False,
                lines=8,
                autoscroll=True,
                value="Esperando acción..."
            )
            
            # File outputs
            file_srt = gr.File(label="Descargar SRT", interactive=False)
            file_json = gr.File(label="Descargar JSON", interactive=False)
            
            with gr.Row():
                out_video_subs = gr.Video(label="Previsualización: Video con Subtítulos", interactive=False)

        with gr.Tab("3 · Viral Shorts"):
            gr.Markdown(
                "## Generador de reels virales con hooks\n\n"
                "Analiza el timeline JSON con Gemini AI para detectar segmentos con hooks fuertes. "
                "Genera reels verticales de 3+ minutos optimizados para redes sociales."
            )

            with gr.Row():
                up_video_reels = gr.File(
                    label="Subir video doblado final",
                    file_types=["video"],
                    type="filepath",
                    scale=2,
                )
                dd_video_reels = gr.Dropdown(
                    choices=_get_output_videos(),
                    value=None,
                    label="O seleccionar video final de salida",
                    interactive=True,
                    scale=2,
                )
                btn_refresh_video_reels = gr.Button("Refrescar videos", scale=1, min_width=120)
            
            # Row 1: Input JSON
            with gr.Row():
                up_json_reels = gr.File(
                    label="Subir JSON de timestamps (o usar el del doblaje)",
                    file_types=[".json"],
                    scale=2
                )
                dd_json_reels = gr.Dropdown(
                    choices=_get_local_jsons(),
                    label="O seleccionar JSON existente",
                    scale=2,
                    interactive=True
                )
                btn_refresh_json_reels = gr.Button("Refrescar JSON", scale=1, min_width=120)
            
            # Row 2: Analyze & Render buttons
            with gr.Row():
                btn_analyze_reels = gr.Button(
                    "Analizar/cargar JSON de reels",
                    variant="primary",
                    size="lg",
                    scale=2
                )
                btn_render_reels = gr.Button(
                    "Generar reels seleccionados",
                    variant="secondary",
                    size="lg",
                    scale=1,
                    visible=False
                )
            
            # Row 3: Reel specifications display
            ui_reels_analysis = gr.Textbox(
                label="Analisis Gemini",
                interactive=False,
                lines=3,
                visible=False
            )
            
            # Row 4: Reel selector
            cg_reels_select = gr.CheckboxGroup(
                choices=[],
                label="Selecciona reels para generar",
                interactive=True,
                visible=False,
            )
            
            # Row 5: Processing logs
            ui_reels_terminal = gr.Textbox(
                label="Terminal de reels",
                elem_id="qdp-reels-log",
                interactive=False,
                lines=8,
                autoscroll=True
            )
            
            # Row 6: Output preview
            with gr.Row():
                out_reel_video = gr.Video(
                    label="Previsualizacion de reel vertical",
                    interactive=False
                )
                file_reel_json = gr.File(
                    label="Descargar JSON de reels",
                    interactive=False
                )
            files_reels_output = gr.Files(
                label="Descargar reels generados",
                interactive=False,
            )
            
            # Hidden state for JSON specs
            state_reel_specs = gr.State(None)
            state_reel_video = gr.State(None)

    # =====================================================================
    # WIRING Y LÓGICA UI
    # =====================================================================
    
    # Helpers para subtítulos
    def handle_subtitle_generation(video_input, lang):
        """Generar subtítulos del video doblado."""
        if not video_input:
            return "", "", "❌ Selecciona un video primero"
        
        try:
            if ui_generate_subtitles is None:
                return "", "", "❌ Módulo de subtítulos no disponible"
            
            # Determinar si es un archivo o una ruta
            if isinstance(video_input, str) and os.path.exists(video_input):
                video_path = video_input
            else:
                return "", "", "❌ Video no encontrado"
            
            # Ejecutar en background
            def update_log(msg):
                ui_log.info(msg)
            
            srt, json_file, msg = ui_generate_subtitles(video_path, lang, update_log)
            
            # Preparar archivos para descarga
            srt_obj = (srt, Path(srt).name) if os.path.exists(srt) else None
            json_obj = (json_file, Path(json_file).name) if os.path.exists(json_file) else None
            
            return srt_obj, json_obj, msg
        
        except Exception as e:
            return "", "", f"❌ Error: {str(e)}"
    
    def handle_subtitle_rendering(video_input, srt_path):
        """Renderizar video con subtítulos."""
        if not video_input:
            return "", "❌ Selecciona un video primero"
        
        if not srt_path:
            return "", "❌ Carga un archivo SRT primero"
        
        try:
            if ui_render_subtitles is None:
                return "", "❌ Módulo de subtítulos no disponible"
            
            # Determinar si es un archivo o una ruta
            if isinstance(video_input, str) and os.path.exists(video_input):
                video_path = video_input
            else:
                return "", "❌ Video no encontrado"

            # Render requiere fuente de video. Si es audio, sugerir usar un video.
            if Path(video_path).suffix.lower() in {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}:
                return "", "❌ Para renderizar subtítulos sobre imagen, selecciona un archivo de video. El audio sí sirve para generar SRT/JSON."
            
            # Ejecutar en background
            def update_log(msg):
                ui_log.info(msg)
            
            output_video, msg = ui_render_subtitles(video_path, srt_path, 'srt', update_log)
            
            return output_video, msg
        
        except Exception as e:
            return "", f"❌ Error: {str(e)}"
    
    def update_subs_video_dropdown():
        media_files = _get_user_videos() + _get_user_audios()
        return gr.update(choices=media_files)
    
    def handle_subs_video_upload(filepath):
        if not filepath:
            return gr.update()
        _ensure_dirs()
        filename = os.path.basename(filepath)
        dest = os.path.join(NETWORK_USER_INPUT, filename)
        shutil.copy2(filepath, dest)
        ui_log.info(f"Video de subtítulos subido: {dest}")
        media_files = _get_user_videos() + _get_user_audios()
        new_val = next((v for v in media_files if v.startswith(filename)), None)
        return gr.update(choices=media_files, value=new_val)

    def handle_extract_frame(video_input, second):
        if extract_frame is None:
            return None, None, "❌ Módulo frame_editor no disponible"
        if not video_input:
            return None, None, "❌ Selecciona un video primero"
        try:
            video_path = _parse_dropdown_choice(video_input, NETWORK_USER_INPUT)
            if not video_path or not os.path.isfile(video_path):
                return None, None, f"❌ Video no encontrado: {video_path}"

            frame_path = extract_frame(video_path, LOCAL_TEMP, float(second))
            adjusted_path = apply_frame_adjustments(
                frame_path,
                brightness=1.0,
                contrast=1.0,
                color=1.0,
                sharpness=1.0,
                output_dir=LOCAL_TEMP,
            )
            return frame_path, adjusted_path, f"✅ Frame extraído en t={int(second)}s"
        except Exception as e:
            return None, None, f"❌ Error al extraer frame: {e}"

    def handle_apply_frame(frame_input, brightness, contrast, color, sharpness):
        if apply_frame_adjustments is None:
            return None, "❌ Módulo frame_editor no disponible"
        if not frame_input:
            return None, "❌ Extrae un frame primero"
        try:
            out = apply_frame_adjustments(
                frame_input,
                brightness=brightness,
                contrast=contrast,
                color=color,
                sharpness=sharpness,
                output_dir=LOCAL_TEMP,
            )
            return out, "✅ Preview actualizado"
        except Exception as e:
            return None, f"❌ Error aplicando ajustes: {e}"
    
    # Conectar events
    def update_file_dropdowns():
        videos = _get_user_videos()
        ui_log.info(
            f"Refresh {NETWORK_USER_INPUT}: {len(videos)} video(s) encontrados"
        )
        return gr.update(choices=videos)

    def update_json_dropdown():
        jsons = _get_local_jsons()
        return gr.update(choices=jsons)

    def update_reels_video_dropdown():
        return gr.update(choices=_get_output_videos())
        
    def handle_video_upload(filepath):
        if not filepath:
            return gr.update()
        _ensure_dirs()
        filename = os.path.basename(filepath)
        dest = os.path.join(NETWORK_USER_INPUT, filename)
        shutil.copy2(filepath, dest)
        ui_log.info(f"Archivo subido y copiado a red: {dest}")
        
        videos = _get_user_videos()
        new_val = next((v for v in videos if v.startswith(filename)), None)
        return gr.update(choices=videos, value=new_val)

    def handle_reel_video_upload(filepath):
        if not filepath:
            return "", "❌ No se subió video"
        if not os.path.exists(filepath):
            return "", "❌ Video no encontrado"
        return filepath, f"✅ Video para reels cargado: {Path(filepath).name}"

    up_video.upload(handle_video_upload, inputs=[up_video], outputs=[dd_video])

    btn_refresh_files.click(update_file_dropdowns, inputs=None, outputs=[dd_video])
    btn_refresh_json_reels.click(update_json_dropdown, inputs=None, outputs=[dd_json_reels])
    btn_refresh_video_reels.click(update_reels_video_dropdown, inputs=None, outputs=[dd_video_reels])

    btn_extract_frame.click(
        handle_extract_frame,
        inputs=[dd_video, frame_second],
        outputs=[frame_original, frame_adjusted, ui_frame_log],
    )

    btn_apply_frame.click(
        handle_apply_frame,
        inputs=[frame_original, slider_brightness, slider_contrast, slider_color, slider_sharpness],
        outputs=[frame_adjusted, ui_frame_log],
    )

    # ── Phase 1: Analizar speakers ────────────────────────────────────────────
    # Outputs: phase_text, log, spk_section, 6×(group, id, label, dd), btn_phase2, state
    _p1_outputs = (
        [ui_active_phase, ui_terminal, spk_section]
        + [comp for i in range(N_SPK_MAX)
           for comp in [spk_groups[i], spk_id_boxes[i], spk_label_boxes[i], spk_voice_dds[i]]]
        + [btn_phase2, state_speakers, btn_phase1]
    )
    btn_phase1.click(
        run_phase1,
        inputs=[dd_video, cb_test_mode],
        outputs=_p1_outputs,
    )

    # ── Phase 2: Doblar con voces asignadas ──────────────────────────────────
    _p2_outputs = [ui_active_phase, ui_terminal, out_master_audio, out_master_video,
                   file_master_json, file_master_srt, file_master_audio, btn_phase2]
    btn_phase2.click(
        run_phase2,
        inputs=[
            dd_video, cb_test_mode, state_speakers,
            *spk_voice_dds,   # 6 voice dropdowns
            *spk_label_boxes, # 6 label textboxes
            cb_apply_video_fx, slider_brightness, slider_contrast, slider_color, slider_sharpness,
        ],
        outputs=_p2_outputs,
    )

    # Tab 2: Subtítulos
    up_video_subs.upload(handle_subs_video_upload, inputs=[up_video_subs], outputs=[dd_video_subs])
    btn_refresh_subs.click(update_subs_video_dropdown, inputs=None, outputs=[dd_video_subs])
    
    btn_gen_subs.click(
        handle_subtitle_generation,
        inputs=[dd_video_subs, dd_subs_lang],
        outputs=[file_srt, file_json, ui_subs_log]
    )
    
    btn_render_subs.click(
        handle_subtitle_rendering,
        inputs=[dd_video_subs, file_srt],
        outputs=[out_video_subs, ui_subs_log]
    )
    
    # Tab 3: Viral Shorts - Reel Generation
    def handle_json_upload(file_obj):
        """Handle JSON file upload for reels analysis."""
        if not file_obj:
            return "", "❌ No file uploaded"
        
        try:
            # file_obj is a NamedTemporaryFile
            uploaded_path = file_obj.name
            ui_log.info(f"JSON uploaded: {uploaded_path}")
            
            # Store path for later use
            return uploaded_path, f"✅ JSON cargado: {Path(file_obj.name).name}"
        except Exception as e:
            return "", f"❌ Error al cargar JSON: {str(e)}"

    def _resolve_reels_video_path(uploaded_video_path, selected_video_choice):
        if uploaded_video_path and os.path.exists(uploaded_video_path):
            return uploaded_video_path
        if selected_video_choice:
            parsed = _parse_dropdown_choice(selected_video_choice, NETWORK_OUTPUT)
            if parsed and os.path.exists(parsed):
                return parsed
        return None

    def _find_subtitle_for_video(video_path):
        stem = Path(video_path).stem
        candidates = [
            os.path.join(NETWORK_OUTPUT, f"{stem}_subtitles.srt"),
            os.path.join(NETWORK_OUTPUT, f"{stem}.srt"),
            os.path.join(NETWORK_OUTPUT, f"{stem.replace('_dubbed', '')}_subtitles.srt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _build_reel_labels(reels):
        labels = []
        for reel in reels:
            labels.append(
                f"Reel {reel['reel_num']}: {reel.get('title', 'Sin titulo')} "
                f"({reel.get('duration_seconds', 0):.0f}s)"
            )
        return labels
    
    def handle_analyze_reels(json_choice, uploaded_json_path):
        """Analyze timestamps JSON or load edited reels JSON."""
        final_json = uploaded_json_path
        if not final_json and json_choice:
            final_json = _parse_dropdown_choice(json_choice, LOCAL_TEMP)

        if not final_json:
            return "❌ Selecciona o sube un JSON", gr.update(choices=[], value=[], visible=False), None, None, gr.update(visible=False)
        if not os.path.exists(final_json):
            return "❌ Archivo JSON no existe", gr.update(choices=[], value=[], visible=False), None, None, gr.update(visible=False)

        try:
            data = json.loads(Path(final_json).read_text(encoding='utf-8'))
            if isinstance(data, dict) and 'reels' in data:
                reels = data.get('reels', [])
                labels = _build_reel_labels(reels)
                summary = f"✅ JSON de reels cargado ({len(labels)} reels)"
                return summary, gr.update(choices=labels, value=labels, visible=True), final_json, final_json, gr.update(visible=True)

            ui_log.info(f"Analizando con Gemini: {final_json}")

            def progress_callback(msg):
                ui_log.info(msg)

            status, specs_path, reel_list = ui_analyze_reels(final_json, progress_callback)
            if status == "Error":
                return f"❌ {specs_path}", gr.update(choices=[], value=[], visible=False), None, None, gr.update(visible=False)

            specs_data = json.loads(Path(specs_path).read_text(encoding='utf-8'))
            labels = _build_reel_labels(specs_data.get('reels', []))
            summary = f"✅ {len(labels)} reels detectados\n\n{specs_data['metadata'].get('gemini_analysis', '')}"
            return summary, gr.update(choices=labels, value=labels, visible=True), specs_path, specs_path, gr.update(visible=True)
        except Exception as e:
            ui_log.error(f"Error en análisis/carga de reels: {e}")
            return f"❌ {str(e)}", gr.update(choices=[], value=[], visible=False), None, None, gr.update(visible=False)

    def handle_render_selected_reels(specs_json_path, selected_reels, uploaded_video_path, selected_video_choice):
        if not specs_json_path or not os.path.exists(specs_json_path):
            return "❌ Carga o genera primero un JSON de reels válido", None, []
        if not selected_reels:
            return "❌ Marca al menos un reel para generar", None, []

        video_path = _resolve_reels_video_path(uploaded_video_path, selected_video_choice)
        if not video_path:
            return "❌ Selecciona o sube el video doblado final", None, []

        subtitle_path = _find_subtitle_for_video(video_path)
        if not subtitle_path:
            return "❌ No se encontró archivo SRT para ese video", None, []

        reels_output_dir = os.path.join(NETWORK_OUTPUT, "reels")
        os.makedirs(reels_output_dir, exist_ok=True)

        rendered = []
        logs = []
        for reel_label in selected_reels:
            status, output_video = ui_render_reel(
                specs_json_path,
                video_path,
                video_path,
                subtitle_path,
                reel_label,
                reels_output_dir,
                lambda msg: ui_log.info(msg),
            )
            if status == "Error":
                logs.append(f"❌ {reel_label}: {output_video}")
                continue
            logs.append(f"✅ {reel_label}: {Path(output_video).name}")
            rendered.append(output_video)

        if not rendered:
            return "\n".join(logs) if logs else "❌ No se pudo generar ningún reel", None, []
        return "\n".join(logs), rendered[0], rendered
    
    # Tab 3 Event Handlers
    up_video_reels.upload(
        handle_reel_video_upload,
        inputs=[up_video_reels],
        outputs=[state_reel_video, ui_reels_terminal],
    )

    up_json_reels.upload(
        handle_json_upload,
        inputs=[up_json_reels],
        outputs=[state_reel_specs, ui_reels_terminal]
    )
    
    btn_analyze_reels.click(
        handle_analyze_reels,
        inputs=[dd_json_reels, state_reel_specs],
        outputs=[ui_reels_terminal, cg_reels_select, state_reel_specs, file_reel_json, btn_render_reels]
    )

    btn_render_reels.click(
        handle_render_selected_reels,
        inputs=[state_reel_specs, cg_reels_select, state_reel_video, dd_video_reels],
        outputs=[ui_reels_terminal, out_reel_video, files_reels_output],
    )
    
if __name__ == "__main__":
    _ensure_dirs()
    ui_log.info("Booting Quantum Dubbing Pipeline UI - Apple Style")
    _start_background_warmups()
    
    
    allowed = [os.path.abspath(d) for d in [NETWORK_DIR, LOCAL_DIR]]
    preferred_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    try:
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=preferred_port,
            share=True,
            allowed_paths=allowed,
        )
    except OSError as e:
        # If preferred port is busy, retry with an adjacent fallback port.
        fallback_port = preferred_port + 1
        ui_log.warning(f"Puerto {preferred_port} ocupado ({e}). Reintentando en {fallback_port}...")
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=fallback_port,
            share=True,
            allowed_paths=allowed,
        )
