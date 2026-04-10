#!/usr/bin/env python3
"""
EN -> ES Voice Dubbing Pipeline
--------------------------------
1. Extrae audio de prueba.mp4  ->  WAV (Whisper) + MP3
2. Transcribe en inglés con faster-whisper (timestamps por segmento)
3. Traduce cada segmento EN -> ES con deep-translator
4. Genera TTS en voz clonada (Qwen3-TTS Voice Clone 1.7B)
5. Ajusta duración de cada segmento TTS para cuadrar con el original
6. Concatena y exporta final_es.wav + final_es.mp3
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

INPUT_VIDEO    = "audio_prueba_1minuto/prueba.mp4"
REF_AUDIO      = "voice_reference/audio_reference_natural.wav"
REF_TEXT_FILE  = "voice_reference/audio_reference_natural.txt"
OUTPUT_DIR     = "output"

# Token de HuggingFace (opcional — solo necesario si el modelo requiere acceso privado)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

WHISPER_MODEL  = "medium"  # opciones: tiny, base, small, medium, large-v2
TGT_LANGUAGE   = "Spanish"
TTS_MODEL_SIZE = "1.7B"

# Límites de velocidad para el ajuste de duración (fuera de rango -> se recorta)
# Ahora configurados en 1.0 para mantener SIEMPRE el tono y velocidad natural
MIN_SPEED = 1.0
MAX_SPEED = 1.0

# Doblaje inteligente: ventana dinámica + pre-borrow
MAX_PRE_BORROW = 2.0   # segundos máximos a adelantar el inicio de un segmento
                       # aprovechando el silencio ANTES del segmento original
MIN_GAP        = 0.12  # silencio mínimo garantizado entre segmentos

# Modo libre: el doblado fluye a su propio ritmo, sin sincronizarse al inglés
FREE_FLOWING   = False  # True = ritmo libre; False = sincronizado al original EN
MAX_FREE_GAP    = 0.50   # segundos máximos de pausa entre segmentos en modo libre
MAX_SEGMENT_GAP = 4.0    # silencio máximo entre segmentos — ahora aplica al gap original (cap para interludios)

# Pausas de puntuación: preprocesa el texto TTS para que el modelo genere pausas
# más naturales en oraciones y cláusulas.  True = activado (recomendado).
PUNCTUATION_PAUSES = True

# ─── UTILIDADES ───────────────────────────────────────────────────────────────

import re as _re


def _add_pause_markers(text: str) -> str:
    """Preprocesa texto para que Qwen3-TTS genere pausas naturales de puntuación.

    Estándares de doblaje profesional:
      - Punto / ? / !  →  pausa ~500-1000 ms  (separamos oraciones con newline)
      - Punto y coma / dos puntos  →  pausa ~200-300 ms  (reforzamos con coma)
      - Coma  →  pausa ~200-500 ms  (el modelo ya la maneja bien, no tocamos)
      - Elipsis (...)  →  pausa ~500-700 ms  (el modelo ya la interpreta)

    El modelo Qwen3-TTS interpreta saltos de línea como fronteras de párrafo,
    generando pausas más claras que un simple espacio entre oraciones.
    """
    if not text or not text.strip():
        return text

    # 1. Normalizar múltiples espacios
    t = _re.sub(r' {2,}', ' ', text.strip())

    # 2. Separar oraciones: después de . ? ! seguido de espacio+mayúscula,
    #    insertar un newline para que el modelo haga una pausa de oración.
    #    Preservamos abreviaturas: solo cortamos si el punto va precedido de
    #    al menos 2 letras minúsculas (evita Dr. Sr. Sra. Jr. vs. EE.UU. etc.)
    t = _re.sub(
        r'(?<=[a-záéíóúñ]{2})\.\s+(?=[A-ZÁÉÍÓÚÑ¿¡])',
        '.\n',
        t,
    )
    t = _re.sub(
        r'\?\s+(?=[A-ZÁÉÍÓÚÑ¿¡])',  # interrogación + espacio + mayúscula
        '?\n',
        t,
    )
    t = _re.sub(
        r'!\s+(?=[A-ZÁÉÍÓÚÑ¿¡])',  # exclamación + espacio + mayúscula
        '!\n',
        t,
    )

    # 3. Punto y coma → agregar coma después para reforzar la micro-pausa
    #    "idea A; idea B" → "idea A;, idea B"  (el modelo lee la coma como respiración)
    t = _re.sub(r';\s+', ';, ', t)

    # 4. Dos puntos seguidos de texto → reforzar con coma
    t = _re.sub(r':\s+(?=[a-záéíóúñA-ZÁÉÍÓÚÑ])', ':, ', t)

    return t


def run_cmd(cmd: list, desc: str = "") -> str:
    """Ejecuta un comando y retorna stdout. Lanza excepción si falla."""
    if desc:
        print(f"  -> {desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Comando fallido: {' '.join(str(c) for c in cmd)}\n"
            f"stderr: {result.stderr[-500:]}"
        )
    return result.stdout


def get_duration(path: str) -> float:
    """Retorna duración en segundos de un archivo de audio/video."""
    out = run_cmd([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", str(path)
    ])
    return float(json.loads(out)["format"]["duration"])


def _log(msg: str) -> None:
    """Imprime con timestamp HH:MM:SS y flush inmediato — nunca silencioso."""
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _Heartbeat:
    """
    Context manager que imprime un ping cada `interval` segundos en un hilo
    daemon mientras el bloque con está activo (p.ej. durante inferencia GPU
    bloqueante).  Uso:

        with _Heartbeat("Generando TTS lote 1/3", interval=15):
            wavs, sr = model.generate_voice_clone(...)
    """
    def __init__(self, label: str, interval: int = 15):
        self._label    = label
        self._interval = interval
        self._stop     = threading.Event()
        self._t0       = None
        self._thread   = None

    def _run(self):
        while not self._stop.wait(self._interval):
            elapsed = time.time() - self._t0
            _log(f"  ⏳ {self._label} — {elapsed:.0f}s transcurridos, aún procesando...")

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)


def _build_chunk_text(segs: list, field: str = "text") -> str:
    """
    Une los textos de una lista de segmentos en un párrafo coherente para el TTS.

    Preserva límites de oraciones añadiendo puntuación según el gap entre segmentos:
      gap > 0.5s  → punto final  (pausa larga → nueva oración)
      gap > 0.2s  → coma         (pausa media → continuación de frase)
      gap ≤ 0.2s  → join directo (sin pausa → misma oración)

    Esto permite al modelo TTS generar silencios y prosodia natural en lugar
    de recibir texto sin señales de ritmo.
    """
    parts = []
    for i, s in enumerate(segs):
        t = s.get(field, "").strip()
        if not t:
            continue
        if i < len(segs) - 1:
            gap = segs[i + 1]["start"] - s["end"]
            if t[-1] not in ".!?…;":
                if gap > 0.5:
                    t = t + "."
                elif gap > 0.2:
                    t = t + ","
        parts.append(t)
    return " ".join(parts)


# ─── PASO 1: EXTRAER AUDIO ────────────────────────────────────────────────────

def extract_audio(input_path: str, out_dir: str) -> tuple:
    """Extrae WAV (16kHz mono, para Whisper) y MP3 del video de entrada."""
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, "input.wav")
    mp3 = os.path.join(out_dir, "input.mp3")

    run_cmd([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        wav
    ], "Extrayendo WAV 16kHz mono (para Whisper)")

    run_cmd([
        "ffmpeg", "-y", "-i", input_path,
        "-q:a", "2", "-ar", "44100",
        mp3
    ], "Extrayendo MP3 (copia de referencia)")

    return wav, mp3


# ─── PASO 1.5: AUDITORÍA Y DIARIZACIÓN ────────────────────────────────────────

def audit_audio(wav_path: str, out_dir: str) -> str:
    """
    Diarización de locutores.
    Método principal: Pyannote.audio 3.1 (requiere HF_TOKEN en variable de entorno).
    Fallback: SpeechBrain ECAPA-TDNN + clustering (sin tokens, 100% local).

    Genera diarization.json con los segmentos etiquetados por locutor.
    """
    out_json = os.path.join(out_dir, "diarization.json")
    if os.path.exists(out_json):
        _log("Diarización ya cacheada, reutilizando...")
        return out_json

    # ── Intentar con Pyannote primero ─────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        try:
            return _audit_pyannote(wav_path, out_dir, out_json, hf_token)
        except Exception as e:
            _log(f"  WARN Pyannote falló: {e}")
            _log("  Usando SpeechBrain como fallback...")

    # ── Fallback: SpeechBrain ─────────────────────────────────────────────
    return _audit_speechbrain(wav_path, out_dir, out_json)


def _audit_pyannote(wav_path: str, out_dir: str, out_json: str, hf_token: str) -> str:
    """Diarización con Pyannote.audio 3.1."""
    import torch
    from pyannote.audio import Pipeline

    _log("Ejecutando auditoría de audio (Pyannote speaker-diarization-3.1)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    diarization = pipeline(wav_path)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    n_spk = len({r["speaker"] for r in results})
    _log(f"Auditoría completada (Pyannote). {len(results)} turnos, {n_spk} locutores.")
    return out_json


def _audit_speechbrain(wav_path: str, out_dir: str, out_json: str) -> str:
    """Diarización con SpeechBrain ECAPA-TDNN + clustering (sin tokens)."""
    import torch
    import numpy as np
    import torchaudio
    from sklearn.cluster import AgglomerativeClustering

    _log("Ejecutando auditoría de audio (SpeechBrain ECAPA-TDNN)...")

    # ── Paso 1: VAD con Whisper ───────────────────────────────────────────
    _log("  [1/3] Detectando segmentos con habla (Whisper VAD)...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel("base", device="cuda", compute_type="float16")
    segments_iter, info = whisper.transcribe(
        wav_path, language="en", beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    vad_segments = []
    for seg in segments_iter:
        if seg.text.strip():
            vad_segments.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            })
    _log(f"  [1/3] {len(vad_segments)} segmentos con habla detectados")

    if not vad_segments:
        _log("  WARN: No se detectó habla. Generando JSON vacío.")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"results": []}, f)
        return out_json

    del whisper
    torch.cuda.empty_cache()

    # ── Paso 2: Embeddings ────────────────────────────────────────────────
    _log("  [2/3] Extrayendo embeddings de voz (ECAPA-TDNN)...")
    from speechbrain.inference.speaker import EncoderClassifier

    sb_cache = os.path.join(out_dir, "speechbrain_cache")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=sb_cache,
        run_opts={"device": "cuda"} if torch.cuda.is_available() else {},
    )

    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    embeddings = []
    valid_indices = []
    for i, seg in enumerate(vad_segments):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = waveform[:, start_sample:end_sample]
        if chunk.shape[1] < int(sr * 0.5):
            continue
        with torch.no_grad():
            emb = classifier.encode_batch(chunk)
        embeddings.append(emb.squeeze().cpu().numpy())
        valid_indices.append(i)

    _log(f"  [2/3] {len(embeddings)} embeddings extraídos")
    del classifier
    torch.cuda.empty_cache()

    # ── Paso 3: Clustering ────────────────────────────────────────────────
    _log("  [3/3] Agrupando por locutor (Agglomerative Clustering)...")
    speaker_labels = ["SPEAKER_00"] * len(vad_segments)

    if len(embeddings) >= 2:
        X = np.array(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(X)
        for idx, label in zip(valid_indices, labels):
            speaker_labels[idx] = f"SPEAKER_{label:02d}"
        n_speakers = len(set(labels))
        _log(f"  [3/3] {n_speakers} locutor(es) detectado(s)")
    else:
        _log("  [3/3] Pocos segmentos — asignando locutor único")

    results = []
    for seg, spk in zip(vad_segments, speaker_labels):
        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": spk,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    n_spk = len({r["speaker"] for r in results})
    _log(f"Auditoría completada (SpeechBrain). {len(results)} turnos, {n_spk} locutores.")
    return out_json


# ─── PASO 2: TRANSCRIPCIÓN ────────────────────────────────────────────────────

def transcribe(wav_path: str, model_size: str = "base", diarization_json_path: str = None) -> list:
    """
    Transcribe con faster-whisper.
    Retorna lista de dicts: {start, end, text}
    """
    from faster_whisper import WhisperModel

    print(f"  -> Cargando Whisper '{model_size}' en GPU...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print("  -> Transcribiendo...")
    segments_iter, info = model.transcribe(
        wav_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    diarization_data = []
    if diarization_json_path and os.path.exists(diarization_json_path):
        try:
            with open(diarization_json_path, "r", encoding="utf-8") as f:
                diarization_data = json.load(f).get("results", [])
        except Exception:
            pass

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
            
        start = round(seg.start, 3)
        end = round(seg.end, 3)
        
        # Asignar speaker por mayor superposición si hay diarization_data
        speaker = "SPEAKER_UNKNOWN"
        if diarization_data:
            max_overlap = 0.0
            for d in diarization_data:
                overlap = max(0, min(end, d["end"]) - max(start, d["start"]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = d["speaker"]
                    
        segments.append({
            "start": start,
            "end":   end,
            "text":  text,
            "speaker": speaker
        })
        print(f"    [{start:6.1f}s -> {end:6.1f}s] [{speaker}] {text[:60]}")

    print(f"  OK {len(segments)} segmentos | idioma detectado: {info.language} ({info.language_probability:.0%})")
    return segments


# ─── PASO 3: TRADUCCIÓN ───────────────────────────────────────────────────────

def translate_segments(segments: list) -> list:
    """Traduce cada segmento EN -> ES usando Google Translate (deep-translator)."""
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="en", target="es")

    print(f"  -> Traduciendo {len(segments)} segmentos...")
    for i, seg in enumerate(segments):
        raw = seg["text"]
        if not raw:
            seg["text_es"] = ""
            continue
        if seg.get("text_es"):          # ya traducido (p.ej. bloque cargado del cache)
            continue
        try:
            seg["text_es"] = translator.translate(raw)
        except Exception as e:
            print(f"    WARN segmento {i}: {e}. Reintentando en 2s...")
            time.sleep(2)
            seg["text_es"] = translator.translate(raw)

        print(f"    [{i+1:3d}/{len(segments)}] {raw[:45]:45s} -> {seg['text_es'][:45]}")
        time.sleep(0.08)   # evitar rate limiting

    return segments


# ─── CHUNKING POR PAUSA NATURAL ──────────────────────────────────────────────

def chunk_by_natural_pause(
    segments: list,
    min_dur: float = 15.0,    # duración mínima antes de buscar un corte
    max_dur: float = 40.0,    # techo duro: cortar aunque no haya pausa ideal
    pause_threshold: float = 2.0,  # cortar en la PRIMERA pausa ≥ este valor (s)
) -> list:
    """
    Agrupa segmentos de Whisper en chunks cortando en pausas naturales de ≥2s.

    Estrategia:
      1. Una vez transcurrido min_dur, cortar en la PRIMERA pausa ≥ pause_threshold.
         Esto garantiza que el corte cae en un silencio real de 2-3 s → imperceptible.
      2. Si no hay pausa larga antes de max_dur (40s), cortar igualmente en la pausa
         más grande que se haya encontrado (fallback).
      3. Si no hay ninguna pausa en absoluto, cortar en max_dur.

    El texto inglés se une con _build_chunk_text (preserva puntuación y ritmo).
    text_es se deja vacío: se traduce DESPUÉS para que el traductor reciba el
    párrafo completo y produzca una traducción contextualmente coherente.
    """
    _DROP = {"text_es", "tts_adjusted_path", "tts_adjusted_dur"}

    if not segments:
        return []

    chunks = []
    i = 0

    while i < len(segments):
        chunk_t0  = segments[i]["start"]
        best_gap  = -1.0
        best_cut  = None   # índice del primer segmento del SIGUIENTE chunk

        for j in range(i + 1, len(segments)):
            elapsed = segments[j]["start"] - chunk_t0
            gap     = segments[j]["start"] - segments[j - 1]["end"]

            # Techo duro: no superar max_dur en ningún caso
            if elapsed > max_dur:
                if best_cut is None:
                    best_cut = j          # cortar aquí aunque no sea ideal
                break

            # Cortar obligatoriamente si el speaker cambia (y no es el mismo segmento)
            if segments[j].get("speaker") != segments[j - 1].get("speaker"):
                best_cut = j
                break

            # Solo buscar cortes una vez superado min_dur
            if elapsed >= min_dur:
                # Guardar siempre el mejor gap visto hasta ahora (fallback)
                if gap > best_gap:
                    best_gap = gap
                    best_cut = j

                # Cortar en la PRIMERA pausa que alcanza el umbral natural
                if gap >= pause_threshold:
                    best_cut = j
                    break     # corte ideal encontrado → no seguir buscando

        if best_cut is None:
            best_cut = len(segments)

        chunk_segs = segments[i:best_cut]
        chunk = {k: v for k, v in chunk_segs[0].items() if k not in _DROP}
        chunk["end"]      = chunk_segs[-1]["end"]
        chunk["text"]     = _build_chunk_text(chunk_segs, "text")  # EN con puntuación
        chunk["_chunked"] = True   # marcador para detección en checkpoint

        chunks.append(chunk)
        i = best_cut

    return chunks


# ─── FUSIÓN DE SEGMENTOS (legado) ─────────────────────────────────────────────

def merge_nearby_segments(
    segments: list,
    max_gap: float = 0.8,
    max_merged_dur: float = 12.0,
    max_merged_chars: int = 200,
) -> list:
    """
    Fusiona segmentos de Whisper consecutivos con gaps pequeños en bloques más grandes.

    Por qué: cuando se genera TTS para cada segmento por separado, cada clip empieza
    y termina de forma independiente, produciendo un audio entrecortado con "reinicios"
    de voz en cada frontera. Al fusionar los segmentos en frases o párrafos completos,
    el TTS genera un audio natural y continuo.

    Parámetros:
        max_gap         — distancia máxima (s) entre segmentos para fusionarlos (default 0.8s)
        max_merged_dur  — duración máxima del bloque fusionado (default 12s)
        max_merged_chars— caracteres máximos del texto fusionado EN (default 200)

    Al fusionar, se eliminan text_es / tts_adjusted_path / tts_adjusted_dur para
    que se regeneren con el texto completo del bloque.
    """
    _DROP = {"text_es", "tts_adjusted_path", "tts_adjusted_dur"}

    if len(segments) < 2:
        return segments

    merged = []
    cur = {k: v for k, v in segments[0].items() if k not in _DROP}

    for nxt in segments[1:]:
        gap         = nxt["start"] - cur["end"]
        merged_dur  = nxt["end"]   - cur["start"]
        merged_text = (cur["text"] + " " + nxt["text"]).strip()

        if gap <= max_gap and merged_dur <= max_merged_dur and len(merged_text) <= max_merged_chars:
            # Fusionar: extender el bloque actual
            cur["end"]  = nxt["end"]
            cur["text"] = merged_text
        else:
            merged.append(cur)
            cur = {k: v for k, v in nxt.items() if k not in _DROP}

    merged.append(cur)
    return merged


# ─── PASO 4: GENERACIÓN TTS (LOCAL, GPU) ─────────────────────────────────────

# ID del modelo local. Opciones:
#   "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  (~4.5 GB, mejor calidad)
#   "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  (~2.5 GB, más rápido)
TTS_LOCAL_MODEL = os.environ.get("TTS_LOCAL_MODEL", f"Qwen/Qwen3-TTS-12Hz-{TTS_MODEL_SIZE}-Base")

# Generamos 1 chunk a la vez — Qwen3-TTS es autoregresivo y se cuelga con
# textos largos.  batch_size=1 permite guardar checkpoint tras cada chunk.
TTS_BATCH_SIZE = int(os.environ.get("TTS_BATCH_SIZE", "1"))


def load_tts_model():
    """Carga el modelo Qwen3-TTS localmente en GPU (se descarga la primera vez)."""
    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"  -> Cargando {TTS_LOCAL_MODEL} en GPU...")
    print("     (Primera vez: descarga ~4.5 GB, espera un momento...)")
    model = Qwen3TTSModel.from_pretrained(
        TTS_LOCAL_MODEL,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("  OK Modelo cargado en GPU")
    return model


def generate_tts_local(
    model,
    segments: list,
    ref_audio_path: str,
    ref_text: str,
    tts_dir: str,
    language: str = "Spanish",
    batch_size: int = TTS_BATCH_SIZE,
    progress_cb=None,          # callback(done, total) para actualizar UI
    checkpoint_path: str = "", # si se da, guarda segments.json tras cada chunk
    speaker_references: dict = None, # mapeo speaker -> {'audio': ruta, 'text': texto}
) -> list:
    """
    Genera TTS para todos los segmentos usando el modelo local.
    Usa create_voice_clone_prompt una sola vez para mayor eficiencia.
    """
    import soundfile as sf

    # Extraer features de la voz de referencia (una sola vez)
    _log("Extrayendo features de las voces de referencia...")
    t0_feat = time.time()
    voice_prompts = {}
    
    if speaker_references:
        for spk, ref in speaker_references.items():
            if ref.get("audio") and ref.get("text"):
                try:
                    voice_prompts[spk] = model.create_voice_clone_prompt(
                        ref_audio=str(ref["audio"]),
                        ref_text=ref["text"],
                        x_vector_only_mode=False
                    )
                    _log(f"  OK Prompt generado para {spk}")
                except Exception as e:
                    _log(f"  ERROR generando prompt para {spk}: {e}")
    
    if not voice_prompts:
        # Fallback si no hay referencias o fallaron
        _log("  Usando prompt de referencia DEFAULT")
        voice_prompts["DEFAULT"] = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    _log(f"Features extraídos en {time.time()-t0_feat:.1f}s")

    # Recopilar segmentos pendientes (sin cache)
    pending = []
    for i, seg in enumerate(segments):
        text_es = seg.get("text_es", "").strip()
        raw_path_check = os.path.join(tts_dir, f"seg_{i:04d}_raw.wav")
        adj_path = os.path.join(tts_dir, f"seg_{i:04d}_adj.wav")
        seg["tts_adjusted_path"] = adj_path
        if not text_es:
            continue
        # En modo libre, si el raw ya existe lo copiamos directamente a adj
        # (evita reusar adj comprimidos de una corrida sincronizada anterior)
        if FREE_FLOWING and os.path.exists(raw_path_check):
            shutil.copy(raw_path_check, adj_path)
            seg["tts_adjusted_dur"] = get_duration(adj_path)
            continue
        if os.path.exists(adj_path):
            if not seg.get("tts_adjusted_dur"):
                seg["tts_adjusted_dur"] = get_duration(adj_path)
            continue
        pending.append((i, seg, text_es))

    if not pending:
        _log("Todos los segmentos TTS ya están cacheados")
        return segments

    _log(f"Generando {len(pending)} segmentos en lotes de {batch_size}...")
    t0_total = time.time()

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        texts = [_add_pause_markers(item[2]) if PUNCTUATION_PAUSES else item[2]
                 for item in batch]
        langs  = [language] * len(batch)
        indices = [item[0] for item in batch]

        batch_num = batch_start // batch_size + 1
        total_batches = (len(pending) + batch_size - 1) // batch_size
        _log(f"Lote {batch_num}/{total_batches} — {len(batch)} seg(s) "
             f"[chunks {[i+1 for i in indices]}] iniciando inferencia GPU...")

        t0_batch = time.time()
        hb_label = f"inferencia GPU lote {batch_num}/{total_batches} ({len(batch)} seg)"
        try:
            spk_set = set(item[1].get("speaker", "SPEAKER_UNKNOWN") for item in batch)
            if len(spk_set) > 1:
                raise ValueError("Múltiples locutores en el mismo lote, procesando 1 a 1...")
                
            spk = list(spk_set)[0]
            prompt_to_use = voice_prompts.get(spk) or voice_prompts.get("DEFAULT") or list(voice_prompts.values())[0]
            
            with _Heartbeat(hb_label, interval=15):
                wavs, sr = model.generate_voice_clone(
                    text=texts,
                    language=langs,
                    voice_clone_prompt=prompt_to_use,
                    max_new_tokens=600,
                )
        except Exception as e:
            _log(f"Reintentando uno a uno (razón: {e})...")
            wavs = []
            for k, (txt, lang) in enumerate(zip(texts, langs)):
                spk = batch[k][1].get("speaker", "SPEAKER_UNKNOWN")
                prompt_to_use = voice_prompts.get(spk) or voice_prompts.get("DEFAULT") or list(voice_prompts.values())[0]
                
                _log(f"  Reintento individual {k+1}/{len(texts)} [{spk}]...")
                try:
                    with _Heartbeat(f"reintento individual {k+1}/{len(texts)}", interval=15):
                        w, sr = model.generate_voice_clone(
                            text=[txt],
                            language=[lang],
                            voice_clone_prompt=prompt_to_use,
                            max_new_tokens=600,
                        )
                    wavs.append(w[0])
                except Exception as e2:
                    _log(f"  ERROR individual: {e2}")
                    wavs.append(None)

        elapsed_batch = time.time() - t0_batch
        _log(f"Lote {batch_num}/{total_batches} completado en {elapsed_batch:.1f}s")

        for (i, seg, _), wav in zip(batch, wavs):
            raw_path = os.path.join(tts_dir, f"seg_{i:04d}_raw.wav")
            adj_path = seg["tts_adjusted_path"]
            target_dur  = seg["end"] - seg["start"]

            if wav is None:
                _log(f"WARN segmento {i+1} — omitido (sin audio generado)")
                continue

            sf.write(raw_path, wav, sr)
            tts_dur = len(wav) / sr

            if FREE_FLOWING:
                shutil.copy(raw_path, adj_path)
                action = "NATURAL"
                adj_dur = tts_dur
                seg["tts_adjusted_dur"] = adj_dur
                _log(f"  seg {i+1:3d} | {tts_dur:.2f}s | {action} | "
                     f"{seg.get('text_es','')[:55]}")
            else:
                if i + 1 < len(segments):
                    available   = segments[i + 1]["start"] - seg["start"]
                    prev_en_end = segments[i - 1]["end"] if i > 0 else 0.0
                    pre_gap     = max(0.0, seg["start"] - prev_en_end - MIN_GAP)
                    pre_borrow  = min(pre_gap, MAX_PRE_BORROW)
                    window      = available - MIN_GAP + pre_borrow
                else:
                    window = tts_dur  # último segmento: sin restricción de ventana

                # Objetivo: estirar el TTS para llenar la ventana EN (limitado por MIN_SPEED),
                # o comprimir si el TTS desborda la ventana.
                # adj_target puede ser MAYOR que tts_dur (ESTIRADO) o MENOR (COMPRIMIDO).
                adj_target = min(tts_dur / MIN_SPEED, window)

                if tts_dur > window:
                    # Caso compresión: mantener piso mínimo para no cortar demasiado
                    adj_target = max(adj_target, target_dur * 0.80)

                if abs(adj_target - tts_dur) / max(tts_dur, 0.001) < 0.03:
                    shutil.copy(raw_path, adj_path)
                    action = "NATURAL"
                elif adj_target > tts_dur:
                    adjust_duration(raw_path, adj_path, adj_target)
                    action = "ESTIRADO"
                else:
                    adjust_duration(raw_path, adj_path, adj_target)
                    action = "COMPRIMIDO"

                adj_dur = get_duration(adj_path)
                seg["tts_adjusted_dur"] = adj_dur
                _log(f"  seg {i+1:3d} | {tts_dur:.2f}s→{adj_dur:.2f}s "
                     f"(ventana:{window:.2f}s) | {action} | "
                     f"{seg.get('text_es','')[:45]}")

        # Actualizar progress bar de Gradio si se proporcionó callback
        if progress_cb:
            progress_cb(batch_start + len(batch), len(pending))

        # Checkpoint tras cada chunk — si el proceso se interrumpe, los chunks
        # ya generados no se pierden y la siguiente ejecución los saltará.
        if checkpoint_path:
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as _f:
                    json.dump(segments, _f, ensure_ascii=False, indent=2)
                _log(f"  Checkpoint guardado ({batch_start + len(batch)}/{len(pending)} chunks)")
            except Exception as _e:
                _log(f"  WARN checkpoint no guardado: {_e}")

    _log(f"TTS total: {time.time()-t0_total:.1f}s para {len(pending)} segmentos")
    return segments


# ─── PASO 5: AJUSTE DE DURACIÓN ───────────────────────────────────────────────

def adjust_duration(
    src: str,
    dst: str,
    target_sec: float,
    min_speed: float = MIN_SPEED,
    max_speed: float = MAX_SPEED,
):
    """
    Estira o comprime el audio para que dure exactamente target_sec segundos.
    Si la diferencia es demasiado grande (fuera de rango), recorta o rellena con silencio.
    """
    cur = get_duration(src)
    if cur <= 0:
        _make_silence(dst, target_sec)
        return

    ratio = cur / target_sec   # > 1 = habló más lento de lo necesario (hay que acelerar)

    # Clamping: si el ajuste necesario está fuera del rango permitido,
    # aceptamos que no quepa perfecto y usamos el límite más cercano.
    ratio_clamped = max(min_speed, min(max_speed, ratio))

    if abs(ratio_clamped - 1.0) < 0.015:
        shutil.copy(src, dst)
        return

    # Construir cadena de filtros atempo (cada eslabón: 0.5 ≤ x ≤ 2.0)
    filters = []
    r = ratio_clamped
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r *= 2.0
    filters.append(f"atempo={r:.5f}")

    run_cmd([
        "ffmpeg", "-y", "-i", src,
        "-filter:a", ",".join(filters),
        dst
    ])

    # Si después del ajuste quedó más corto que target, rellena con silencio al final
    new_dur = get_duration(dst)
    if new_dur < target_sec - 0.05:
        _pad_with_silence(dst, target_sec)

    # Si sigue siendo más largo que target (velocidad máxima limitada), truncar duro
    new_dur = get_duration(dst)
    if new_dur > target_sec + 0.05:
        tmp = dst + ".trunc.wav"
        run_cmd(["ffmpeg", "-y", "-t", str(target_sec), "-i", dst, tmp])
        shutil.move(tmp, dst)


def _make_silence(path: str, duration: float, sr: int = 22050):
    run_cmd([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r={sr}:cl=mono",
        "-t", str(duration), path
    ])


def _pad_with_silence(path: str, target_sec: float, sr: int = 22050):
    """Añade silencio al final de un archivo hasta alcanzar target_sec."""
    cur = get_duration(path)
    pad = target_sec - cur
    if pad <= 0:
        return
    tmp = path + ".tmp.wav"
    silence = path + ".sil.wav"
    _make_silence(silence, pad, sr)
    concat_list = path + ".concat.txt"
    with open(concat_list, "w") as f:
        f.write(f"file '{os.path.abspath(path)}'\n")
        f.write(f"file '{os.path.abspath(silence)}'\n")
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, tmp])
    shutil.move(tmp, path)
    os.remove(silence)
    os.remove(concat_list)


# ─── PASO 6: MONTAJE FINAL ────────────────────────────────────────────────────

def build_final_audio(
    segments: list,
    total_duration: float,
    output_wav: str,
    sr: int = 22050,
    free_flowing: bool = FREE_FLOWING,
):
    """
    Construye el audio final ensamblando los segmentos TTS en orden.

    Modo libre (free_flowing=True):
        Concatena los segmentos con pausas naturales entre ellos (cappadas a
        MAX_FREE_GAP). No hay sincronización con los timestamps originales.
        El audio resultante tiene su propia duración natural.

    Modo sincronizado (free_flowing=False):
        Inserta silencios para alinear cada segmento con su timestamp original
        en inglés. Usa compensación de drift y pre-borrow.
    """
    tmp_dir = tempfile.mkdtemp(prefix="qwen_dub_")
    parts = []

    try:
        if free_flowing:
            # ── Modo libre: concatenar en orden con pausas naturales cappadas ──
            _log(f"Ensamblando audio libre ({len(segments)} segmentos)...")
            for i, seg in enumerate(segments):
                orig_start = seg["start"]

                # Pausa antes de este segmento
                if i == 0:
                    leading = min(orig_start, MAX_FREE_GAP)
                else:
                    natural_gap = max(0.0, orig_start - segments[i - 1]["end"])
                    leading = min(natural_gap, MAX_FREE_GAP)

                if leading > 0.02:
                    sil = os.path.join(tmp_dir, f"sil_{i:04d}.wav")
                    _make_silence(sil, leading, sr)
                    parts.append(sil)

                adj = seg.get("tts_adjusted_path", "")
                if adj and os.path.exists(adj):
                    norm = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
                    run_cmd([
                        "ffmpeg", "-y", "-i", adj,
                        "-ar", str(sr), "-ac", "1",
                        norm
                    ])
                    adj_dur = get_duration(norm) if os.path.exists(norm) else 0.0
                    _log(f"  [{i+1:3d}/{len(segments)}] pausa={leading:.2f}s + audio={adj_dur:.2f}s  «{seg.get('text_es','')[:55]}»")
                    parts.append(norm)
                else:
                    # Fallback: silencio por la duración original
                    fallback_dur = seg["end"] - orig_start
                    sil = os.path.join(tmp_dir, f"seg_sil_{i:04d}.wav")
                    _make_silence(sil, fallback_dur, sr)
                    _log(f"  [{i+1:3d}/{len(segments)}] pausa={leading:.2f}s + FALLBACK silencio={fallback_dur:.2f}s (sin TTS)")
                    parts.append(sil)

            concat_list = os.path.join(tmp_dir, "concat.txt")
            with open(concat_list, "w") as f:
                for p in parts:
                    f.write(f"file '{p}'\n")

            _log(f"Concatenando {len(parts)} partes → {output_wav}")
            run_cmd([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-ar", str(sr), "-ac", "1",
                output_wav
            ], "Ensamblando audio libre (free-flowing)")
            final_dur = get_duration(output_wav) if os.path.exists(output_wav) else 0.0
            _log(f"Audio libre ensamblado: {len(parts)} partes → {final_dur:.1f}s totales → {output_wav}")
            return

        # ── Modo sincronizado ────────────────────────────────────────────────
        actual_pos = 0.0   # posición real acumulada en el audio doblado
        for i, seg in enumerate(segments):
            orig_start = seg["start"]
            end        = seg["end"]

            # Duración real del TTS ajustado (pre-computada en la generación TTS)
            adj_path_for_dur = seg.get("tts_adjusted_path", "")
            adj_dur = seg.get("tts_adjusted_dur") or (
                get_duration(adj_path_for_dur)
                if (adj_path_for_dur and os.path.exists(adj_path_for_dur))
                else (end - orig_start)
            )

            # ── Gap-relative: insertar el silencio EXACTO del original entre segmentos ──
            # El primer segmento se ancla al silencio de apertura del original.
            # Los siguientes usan el gap real original (end[i-1] → start[i]).
            # Así se preservan las pausas naturales sin generar silencios artificiales
            # donde el original tenía voz continua.
            if i == 0:
                silence_needed = max(0.0, orig_start)
            else:
                prev_orig_end  = segments[i - 1]["end"]
                orig_gap       = max(0.0, orig_start - prev_orig_end)
                silence_needed = min(orig_gap, MAX_SEGMENT_GAP)

            if silence_needed > 0.02:
                sil = os.path.join(tmp_dir, f"sil_{i:04d}.wav")
                _make_silence(sil, silence_needed, sr)
                parts.append(sil)
                actual_pos += silence_needed

            # Audio TTS ajustado (o silencio si falló)
            adj = seg.get("tts_adjusted_path", "")
            if adj and os.path.exists(adj):
                norm = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
                run_cmd([
                    "ffmpeg", "-y", "-i", adj,
                    "-ar", str(sr), "-ac", "1",
                    norm
                ])
                parts.append(norm)
                actual_pos += get_duration(norm)   # rastrear duración REAL
            else:
                seg_dur = end - orig_start
                sil = os.path.join(tmp_dir, f"seg_sil_{i:04d}.wav")
                _make_silence(sil, seg_dur, sr)
                parts.append(sil)
                actual_pos += seg_dur

        # Archivo de lista para concat
        concat_list = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")

        run_cmd([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-ar", str(sr), "-ac", "1",
            output_wav
        ], "Ensamblando audio final")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── PASO 7: FUSIÓN VIDEO + AUDIO DOBLADO ────────────────────────────────────

def merge_video_with_audio(video_path: str, audio_path: str, output_path: str):
    """
    Fusiona el video original con el audio doblado.

    - Audio más corto que el video  → recorta el video al punto donde termina el audio.
    - Audio más largo que el video  → extiende el video con pantalla negra al final
                                      hasta que el audio termine.
    """
    video_dur = get_duration(video_path)
    audio_dur = get_duration(audio_path)
    diff = audio_dur - video_dur
    _log(f"Fusión video+audio: video={video_dur:.1f}s  audio={audio_dur:.1f}s  diff={diff:+.1f}s")

    if diff > 0.1:
        # Audio más largo: extender video con pantalla negra
        _log(f"Audio más largo que el video — extendiendo con pantalla negra ({diff:.2f}s)...")
        probe_out = run_cmd([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-select_streams", "v:0",
            video_path
        ])
        vstream = json.loads(probe_out)["streams"][0]
        width  = vstream["width"]
        height = vstream["height"]
        fps_raw = vstream.get("r_frame_rate", "25/1")
        num, den = fps_raw.split("/")
        fps = float(num) / float(den)
        _log(f"Resolución: {width}x{height} @ {fps:.2f}fps")

        tmp_dir = tempfile.mkdtemp(prefix="qwen_merge_")
        try:
            black_clip = os.path.join(tmp_dir, "black.mp4")
            _log(f"[1/3] Generando pantalla negra {width}x{height} ({diff:.2f}s)...")
            run_cmd([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s={width}x{height}:r={fps:.4f}",
                "-t", str(diff + 0.1),
                "-an",
                black_clip
            ], f"Generando pantalla negra {width}x{height} ({diff:.2f}s)")

            concat_list = os.path.join(tmp_dir, "concat.txt")
            with open(concat_list, "w") as f:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
                f.write(f"file '{os.path.abspath(black_clip)}'\n")

            extended = os.path.join(tmp_dir, "extended.mp4")
            _log(f"[2/3] Concatenando video original + pantalla negra...")
            run_cmd([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c:v", "libx264", "-preset", "fast",
                "-an",
                extended
            ], "Concatenando video + pantalla negra")

            _log(f"[3/3] Fusionando video extendido + audio doblado → {output_path}")
            run_cmd([
                "ffmpeg", "-y",
                "-i", extended,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path
            ], "Fusionando video extendido + audio doblado")
            _log(f"Fusión completa (ruta: pantalla negra) → {output_path}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    else:
        # Audio más corto o igual: recortar video al final del audio
        _log(f"Audio más corto que el video — recortando video a {audio_dur:.1f}s...")
        _log(f"[1/1] Fusionando video (recortado a {audio_dur:.2f}s) + audio doblado → {output_path}")
        run_cmd([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-t", str(audio_dur),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ], f"Fusionando video (recortado a {audio_dur:.2f}s) + audio doblado")
        _log(f"Fusión completa (ruta: recorte) → {output_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  EN -> ES Voice Dubbing Pipeline  |  Qwen3-TTS + Whisper")
    print("=" * 65)

    base = Path(__file__).parent
    input_video   = base / INPUT_VIDEO
    ref_audio     = base / REF_AUDIO
    ref_text_file = base / REF_TEXT_FILE
    out_dir       = base / OUTPUT_DIR

    # Validar entradas
    for p in [input_video, ref_audio, ref_text_file]:
        if not p.exists():
            print(f"ERROR: No encontrado: {p}")
            sys.exit(1)

    ref_text = ref_text_file.read_text(encoding="utf-8").strip()
    os.makedirs(out_dir, exist_ok=True)

    segments_json = out_dir / "segments.json"
    tts_dir       = out_dir / "tts_segments"
    tts_dir.mkdir(exist_ok=True)

    steps = 7 if FREE_FLOWING else 6

    # ── 1. Extraer audio ──────────────────────────────────────────────
    print(f"\n[1/{steps}] Extrayendo audio...")
    wav_path, mp3_path = extract_audio(str(input_video), str(out_dir))
    total_dur = get_duration(wav_path)
    print(f"  OK Duración: {total_dur:.1f}s  |  WAV: {wav_path}  |  MP3: {mp3_path}")

    # ── 2. Transcribir ────────────────────────────────────────────────
    print(f"\n[2/{steps}] Transcribiendo con Whisper '{WHISPER_MODEL}'...")
    if segments_json.exists():
        with open(segments_json, encoding="utf-8") as f:
            segments = json.load(f)
        print(f"  -> Cargando {len(segments)} segmentos cacheados de {segments_json.name}")
    else:
        segments = transcribe(wav_path, WHISPER_MODEL)
        with open(segments_json, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"  OK {len(segments)} segmentos")

    # ── 3. Chunk + Traducir (orden: chunk primero para traducción contextual) ──
    # Si el checkpoint ya tiene chunks traducidos, saltar ambos pasos.
    already_done = all(s.get("_chunked") and s.get("text_es") for s in segments)
    if already_done:
        print(f"\n[3/{steps}] Chunks y traducción cacheados ({len(segments)} chunks)")
    else:
        # 3a. Agrupar segmentos en chunks con texto EN bien puntuado
        old_count = len(segments)
        segments  = chunk_by_natural_pause(segments)
        print(f"\n[3/{steps}] Chunking: {old_count} segmentos → {len(segments)} chunks")
        for stale in list(tts_dir.glob("seg_*_adj.wav")) + list(tts_dir.glob("seg_*_raw.wav")):
            stale.unlink(missing_ok=True)

        # 3b. Traducir cada chunk completo (párrafo coherente → mejor traducción)
        print(f"  -> Traduciendo {len(segments)} chunks EN->ES (texto completo por chunk)...")
        segments = translate_segments(segments)
        with open(segments_json, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    print("  OK")

    # ── 4. Generar TTS (local GPU) ────────────────────────────────────
    print(f"\n[4/{steps}] Generando TTS con Qwen3-TTS LOCAL...")
    print(f"       Modelo: {TTS_LOCAL_MODEL}  |  Voz: {ref_audio.name}")

    # Verificar qwen-tts
    try:
        import qwen_tts  # noqa: F401
    except ImportError:
        print("\n  ERROR: qwen-tts no instalado. Ejecuta: pip install -U qwen-tts")
        sys.exit(1)

    # Eliminar cache TTS previo de la API remota (archivos de 0 bytes o inválidos)
    for f in tts_dir.glob("seg_*_adj.wav"):
        if f.stat().st_size < 1000:   # menos de 1KB = archivo vacío/inválido
            f.unlink()
            raw = tts_dir / f.name.replace("_adj", "_raw")
            if raw.exists():
                raw.unlink()

    tts_model = load_tts_model()

    segments = generate_tts_local(
        model=tts_model,
        segments=segments,
        ref_audio_path=str(ref_audio),
        ref_text=ref_text,
        tts_dir=str(tts_dir),
        language=TGT_LANGUAGE,
    )

    # Guardar paths actualizados
    with open(segments_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print("  OK TTS completo")

    # ── 5. Ensamblar ─────────────────────────────────────────────────
    print(f"\n[5/{steps}] Ensamblando audio final...")
    final_wav = str(out_dir / "final_es.wav")
    build_final_audio(segments, total_dur, final_wav)
    print(f"  OK WAV: {final_wav}")

    # ── 6. Exportar MP3 ───────────────────────────────────────────────
    print(f"\n[6/{steps}] Exportando MP3...")
    final_mp3 = str(out_dir / "final_es.mp3")
    run_cmd([
        "ffmpeg", "-y", "-i", final_wav,
        "-q:a", "2", "-ar", "44100",
        final_mp3
    ], "Convirtiendo a MP3")
    print(f"  OK MP3: {final_mp3}")

    # ── 7. Fusionar video + audio doblado ─────────────────────────────
    if FREE_FLOWING:
        print(f"\n[7/{steps}] Fusionando video + audio doblado...")
        final_dubbed_mp4 = str(out_dir / "final_dubbed.mp4")
        merge_video_with_audio(str(input_video), final_wav, final_dubbed_mp4)
        print(f"  OK MP4 doblado: {final_dubbed_mp4}")

    print("\n" + "=" * 65)
    print("  LISTO")
    print(f"  -> {final_mp3}")
    if FREE_FLOWING:
        print(f"  -> {final_dubbed_mp4}")
    print("=" * 65)


if __name__ == "__main__":
    main()
