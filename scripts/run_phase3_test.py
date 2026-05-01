"""
Phase 3 test script — 30-second dubbed video using cached Phase 1+2 artifacts.

Usage:
    .venv_clean\\Scripts\\python.exe scripts\\run_phase3_test.py

Assumptions:
- full_video_en_first30s.mp4 in input/ (already 30 s, so test_mode=False keeps the
  pyvideotrans output dir name matching the cached Phase 1/2 artifacts:
  pyvideotrans/output/full_video_en_first30s-mp4/)
- Phase 1+2 artifacts already cached in that directory (es.srt, translated_segments.json, etc.)
- Voice: Professor Jiang (Hombre) → speaker A
- Background audio volume: 0.01
"""

import os
import sys

# ── locate repo root ──────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ── load .env before anything else ───────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"), override=True)

# Ensure the pyvideotrans engine uses .venv-py310
os.environ.setdefault(
    "PYVIDEOTRANS_PYTHON",
    os.path.join(REPO_ROOT, ".venv-py310", "Scripts", "python.exe"),
)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ── import main_ui ────────────────────────────────────────────────────────────
import main_ui  # noqa: E402

# Inject N_SPK_MAX into main_ui's global scope (it's normally set in __main__)
N_SPK_MAX = 50
main_ui.N_SPK_MAX = N_SPK_MAX

# ── resolve the 30-second test video ─────────────────────────────────────────
from src.paths import NETWORK_USER_INPUT, NETWORK_OUTPUT
VIDEO_NAME = "full_video_en_first30s.mp4"
video_file = os.path.join(NETWORK_USER_INPUT, VIDEO_NAME)

if not os.path.isfile(video_file):
    sys.exit(f"ERROR: Video not found: {video_file}")

# ── build speaker_and_render_args exactly as the Gradio UI would pass them ───
# Layout: voice_inputs[N_SPK_MAX] + label_inputs[N_SPK_MAX] + tail_args[6]
VOICE_JIANG = "Professor Jiang (Hombre)"
voice_inputs = [VOICE_JIANG] + [None] * (N_SPK_MAX - 1)   # first speaker = Jiang
label_inputs = [""] * N_SPK_MAX                             # no custom labels
tail_args = [
    False,   # apply_video_fx
    0.01,    # backaudio_volume   ← requested
    1.0,     # brightness
    1.0,     # contrast
    1.0,     # color
    1.0,     # sharpness
]
speaker_and_render_args = voice_inputs + label_inputs + tail_args

# ── state_speakers: speaker A is the only detected speaker ───────────────────
state_speakers = [{"speaker_id": "A", "label": "Speaker A"}]

# ── run Phase 3 ───────────────────────────────────────────────────────────────
print("=" * 72)
print("PHASE 3 TEST — 30-second dubbed video")
print(f"  Video   : {video_file}")
print(f"  Voice   : {VOICE_JIANG}")
print(f"  Backaudio volume: 0.01")
print(f"  test_mode=False (video is already 30 s → cache dir name preserved)")
print("=" * 72)

last_result = None
for result in main_ui.run_phase3(
    video_file,       # video_file
    False,            # test_mode=False (already 30 s; keeps cache dir name intact)
    state_speakers,   # state_speakers
    *speaker_and_render_args,
):
    phase, log, audio, video, json_ts, srt, dl_audio, ts_translate, btn_update = result
    last_result = result
    print(f"[PHASE3] {phase}")

print()
print("=" * 72)
print("FINAL RESULT")
print("=" * 72)
if last_result:
    phase, log, audio, video, json_ts, srt, dl_audio, ts_translate, btn_update = last_result
    print(f"  Status     : {phase}")
    print(f"  Video out  : {video}")
    print(f"  Audio out  : {audio}")
    print(f"  SRT out    : {srt}")
    print(f"  JSON ts    : {json_ts}")
    if video and os.path.isfile(video):
        size_mb = os.path.getsize(video) / 1024 / 1024
        print(f"  Video size : {size_mb:.2f} MB")
        print("  ✓ SUCCESS — dubbed video generated")
    else:
        print("  ✗ FAILED — no output video")
        sys.exit(1)
