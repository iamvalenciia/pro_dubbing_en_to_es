#!/usr/bin/env python3
"""
Benchmark de tiempos del pipeline EN -> ES
Mide cada paso por separado para calcular costos de servidor.
"""

import os, sys, time, json, shutil, subprocess, tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Configuración ─────────────────────────────────────────────────
INPUT_VIDEO   = "audio_prueba_1minuto/prueba.mp4"
REF_AUDIO     = "voice_reference/audio_reference_natural.wav"
REF_TEXT_FILE = "voice_reference/audio_reference_natural.txt"
OUTPUT_DIR    = "output"
WHISPER_MODEL = "base"
TTS_MODEL     = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TGT_LANG      = "Spanish"

# ── Helpers ───────────────────────────────────────────────────────

@dataclass
class Timer:
    label: str
    _start: float = field(default=0.0, init=False, repr=False)
    elapsed: float = field(default=0.0, init=False)

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


def run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-300:])


def gpu_info():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        return r.stdout.strip()
    except:
        return "N/A"


def vram_used_mb():
    try:
        import torch
        return torch.cuda.memory_allocated() / 1024**2
    except:
        return 0

# ── Pasos individuales ────────────────────────────────────────────

def step_extract(input_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, "input.wav")
    run(["ffmpeg", "-y", "-i", input_path,
         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav])
    return wav


def step_transcribe(wav_path, model_size):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segs, info = model.transcribe(wav_path, language="en", beam_size=5,
                                   vad_filter=True,
                                   vad_parameters={"min_silence_duration_ms": 300})
    return [{"start": round(s.start, 3), "end": round(s.end, 3),
             "text": s.text.strip()} for s in segs if s.text.strip()]


def step_translate(segments):
    from deep_translator import GoogleTranslator
    tr = GoogleTranslator(source="en", target="es")
    for seg in segments:
        seg["text_es"] = tr.translate(seg["text"]) if seg["text"] else ""
        time.sleep(0.08)
    return segments


def step_load_tts(model_id):
    import torch
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(
        model_id, device_map="cuda:0", dtype=torch.bfloat16)


def step_voice_prompt(model, ref_audio, ref_text):
    return model.create_voice_clone_prompt(
        ref_audio=str(ref_audio), ref_text=ref_text, x_vector_only_mode=False)


def step_generate_tts(model, voice_prompt, segments, language):
    texts = [s["text_es"] for s in segments if s.get("text_es", "").strip()]
    langs = [language] * len(texts)
    wavs, sr = model.generate_voice_clone(
        text=texts, language=langs, voice_clone_prompt=voice_prompt)
    return wavs, sr


# ── Main benchmark ────────────────────────────────────────────────

def main():
    base = Path(__file__).parent
    out_dir = str(base / OUTPUT_DIR)
    wav_path = os.path.join(out_dir, "input.wav")
    segs_json = os.path.join(out_dir, "segments_bench.json")

    print("=" * 62)
    print("  BENCHMARK: Pipeline EN -> ES  (1 min de audio)")
    print("=" * 62)
    print(f"  GPU : {gpu_info()}")
    print(f"  Whisper: {WHISPER_MODEL}  |  TTS: {TTS_MODEL.split('/')[-1]}")
    print()

    ref_text = (base / REF_TEXT_FILE).read_text(encoding="utf-8").strip()
    timers = {}

    # ── 1. Extracción ─────────────────────────────────────────────
    print("[1] Extracción de audio...", end=" ", flush=True)
    with Timer("extract") as t:
        step_extract(str(base / INPUT_VIDEO), out_dir)
    timers["1_extract"] = t.elapsed
    print(f"{t.elapsed:.2f}s")

    # ── 2. Transcripción ──────────────────────────────────────────
    print("[2] Transcripción Whisper...", end=" ", flush=True)
    with Timer("transcribe") as t:
        segments = step_transcribe(wav_path, WHISPER_MODEL)
    timers["2_transcribe"] = t.elapsed
    print(f"{t.elapsed:.2f}s  ({len(segments)} segmentos)")
    with open(segs_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # ── 3. Traducción ─────────────────────────────────────────────
    print("[3] Traducción EN->ES...", end=" ", flush=True)
    with Timer("translate") as t:
        segments = step_translate(segments)
    timers["3_translate"] = t.elapsed
    print(f"{t.elapsed:.2f}s")

    # ── 4a. Carga del modelo TTS ──────────────────────────────────
    print("[4a] Carga modelo TTS (ya en cache)...", end=" ", flush=True)
    with Timer("load_tts") as t:
        tts_model = step_load_tts(TTS_MODEL)
    timers["4a_load_tts"] = t.elapsed
    vram = vram_used_mb()
    print(f"{t.elapsed:.2f}s  (VRAM usada: {vram:.0f} MB)")

    # ── 4b. Extracción de voz de referencia ───────────────────────
    print("[4b] Extracción de voz de referencia...", end=" ", flush=True)
    with Timer("voice_prompt") as t:
        voice_prompt = step_voice_prompt(tts_model, base / REF_AUDIO, ref_text)
    timers["4b_voice_prompt"] = t.elapsed
    print(f"{t.elapsed:.2f}s")

    # ── 4c. Generación TTS ────────────────────────────────────────
    audio_duration = sum(s["end"] - s["start"] for s in segments)
    print(f"[4c] Generación TTS ({len(segments)} segmentos, {audio_duration:.1f}s de speech)...",
          end=" ", flush=True)
    with Timer("generate_tts") as t:
        wavs, sr = step_generate_tts(tts_model, voice_prompt, segments, TGT_LANG)
    timers["4c_generate_tts"] = t.elapsed
    generated_sec = sum(len(w) / sr for w in wavs)
    print(f"{t.elapsed:.2f}s  (generó {generated_sec:.1f}s de audio)")

    # ── 5. Ajuste + ensamblado (estimado con ffmpeg) ───────────────
    print("[5] Ajuste de duraciones...", end=" ", flush=True)
    import soundfile as sf
    tts_dir = os.path.join(out_dir, "tts_bench")
    os.makedirs(tts_dir, exist_ok=True)
    with Timer("adjust") as t:
        for i, (seg, wav) in enumerate(zip(segments, wavs)):
            raw = os.path.join(tts_dir, f"seg_{i:04d}_raw.wav")
            sf.write(raw, wav, sr)
            # time-stretch con ffmpeg (atempo)
            cur_dur = len(wav) / sr
            tgt_dur = seg["end"] - seg["start"]
            ratio = cur_dur / tgt_dur
            ratio = max(0.8, min(1.5, ratio))
            adj = os.path.join(tts_dir, f"seg_{i:04d}_adj.wav")
            run(["ffmpeg", "-y", "-i", raw,
                 "-filter:a", f"atempo={ratio:.4f}", adj])
    timers["5_adjust"] = t.elapsed
    print(f"{t.elapsed:.2f}s")

    # ── RESUMEN ───────────────────────────────────────────────────
    shutil.rmtree(tts_dir, ignore_errors=True)

    total_pipeline = sum(timers.values())
    total_no_load = total_pipeline - timers["4a_load_tts"]
    input_duration = 60.0

    print()
    print("=" * 62)
    print("  RESULTADOS  (audio de entrada: 60 segundos)")
    print("=" * 62)

    labels = {
        "1_extract":      "Extraccion de audio   (ffmpeg)",
        "2_transcribe":   "Transcripcion Whisper (GPU)",
        "3_translate":    "Traduccion EN->ES     (Google)",
        "4a_load_tts":    "Carga modelo TTS      (GPU, cache)",
        "4b_voice_prompt":"Extraccion voz ref.   (GPU)",
        "4c_generate_tts":"Generacion TTS        (GPU, batch)",
        "5_adjust":       "Ajuste duraciones     (ffmpeg x8)",
    }

    for key, label in labels.items():
        elapsed = timers[key]
        pct = elapsed / total_pipeline * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:38s} {elapsed:6.2f}s  {pct:5.1f}%  |{bar}")

    print(f"  {'─'*55}")
    print(f"  {'TOTAL (con carga modelo)':38s} {total_pipeline:6.2f}s")
    print(f"  {'TOTAL (produccion, sin carga modelo)':38s} {total_no_load:6.2f}s")
    print()

    # ── Proyecciones ──────────────────────────────────────────────
    rtf = total_no_load / input_duration   # Real-Time Factor
    rtf_tts = timers["4c_generate_tts"] / audio_duration

    print("  VELOCIDAD")
    print(f"  RTF total (sin carga modelo)   : {rtf:.2f}x  "
          f"({'MAS rapido' if rtf < 1 else 'MAS lento'} que tiempo real)")
    print(f"  RTF solo TTS                   : {rtf_tts:.2f}x")
    print()

    # Proyeccion a 1 hora
    factor_60 = 3600 / input_duration
    total_1h = total_no_load * factor_60
    tts_1h   = timers["4c_generate_tts"] * factor_60
    trans_1h = timers["3_translate"] * factor_60

    print("  PROYECCION para 1 HORA de video")
    print(f"  Tiempo total estimado          : {total_1h/60:.1f} min  ({total_1h:.0f}s)")
    print(f"  - del cual TTS GPU             : {tts_1h/60:.1f} min")
    print(f"  - del cual Traduccion (red)    : {trans_1h/60:.1f} min")
    print()

    # ── Costos de servidor (referencia) ──────────────────────────
    # horas de GPU necesarias para procesar 1h de video
    gpu_hours_per_video_hour = total_1h / 3600

    servers = [
        # (nombre,              $/hr,   GPU_equiv)
        ("RTX 3060 12GB (tuya)",     0.00,  1.0),
        ("Vast.ai RTX 3060 12GB",    0.25,  1.0),
        ("Vast.ai RTX 3090 24GB",    0.40,  2.5),  # ~2.5x mas rapido
        ("Lambda RTX A6000 48GB",    1.10,  4.0),
        ("RunPod RTX 4090 24GB",     0.69,  3.0),
        ("AWS g4dn.xlarge (T4 16G)", 0.526, 1.5),
    ]

    print("  COSTO PARA PROCESAR 1 HORA DE VIDEO (estimado)")
    print(f"  {'Servidor':30s}  {'$/hr':>6}  {'Tiempo':>8}  {'Costo':>7}")
    print(f"  {'─'*55}")
    for name, price_hr, speedup in servers:
        gpu_h = gpu_hours_per_video_hour / speedup
        cost = gpu_h * price_hr
        minutes = gpu_h * 60
        price_str = "GRATIS" if price_hr == 0 else f"${cost:.3f}"
        print(f"  {name:30s}  ${price_hr:.3f}/hr  {minutes:6.1f} min  {price_str:>7}")

    print()
    print("  NOTAS:")
    print(f"  * VRAM utilizada por TTS: {vram:.0f} MB / 6144 MB (RTX 3060 Laptop)")
    print(f"  * GPU de mayor VRAM permite lotes mas grandes (mas rapido)")
    print(f"  * Traduccion depende de red, escala linealmente con # segmentos")
    print(f"  * Los tiempos de nube son estimados (speedup relativo a RTX 3060)")
    print("=" * 62)

    # Guardar JSON con resultados
    results = {
        "input_duration_sec": input_duration,
        "timings_sec": timers,
        "total_with_model_load_sec": total_pipeline,
        "total_production_sec": total_no_load,
        "rtf_total": rtf,
        "rtf_tts_only": rtf_tts,
        "projection_1h_total_min": total_1h / 60,
        "vram_used_mb": vram,
    }
    results_path = os.path.join(out_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en: {results_path}")


if __name__ == "__main__":
    main()
