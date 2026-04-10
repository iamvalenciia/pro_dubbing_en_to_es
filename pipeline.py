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


# ─── PASO 2: TRANSCRIPCIÓN (WhisperX — Forced Alignment) ─────────────────────

def _group_words_into_segments(
    words: list,
    min_dur: float = 2.0,
    max_dur: float = 12.0,
    gap_threshold: float = 0.4,
    diarization_data: list = None,
) -> list:
    """
    Agrupa palabras alineadas por WhisperX en frases de 2-12 segundos.

    Estrategia de corte (por prioridad):
      1. Respetar max_dur como techo duro (nunca superar 12s).
      2. Después de alcanzar min_dur (2s), buscar el PRIMER punto de corte
         natural: fin de oración (.!?) o un silencio entre palabras ≥ gap_threshold.
      3. Si no hay punto de corte natural antes de max_dur, cortar en la mayor
         pausa encontrada.  Si no hay ninguna, cortar en max_dur.

    Parámetros:
        words           — lista de dicts con {start, end, word} de WhisperX
        min_dur         — duración mínima de un segmento antes de buscar un corte
        max_dur         — techo duro: cortar aunque no haya pausa ideal
        gap_threshold   — silencio mínimo (s) entre palabras para considerar corte
        diarization_data— resultados de diarización para asignar speaker
    """
    if not words:
        return []

    # Filtrar palabras sin timestamps válidos
    valid_words = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    if not valid_words:
        return []

    def _assign_speaker(start, end):
        """Asigna el locutor más probable basándose en la diarización."""
        if not diarization_data:
            return "SPEAKER_UNKNOWN"
        best_spk = "SPEAKER_UNKNOWN"
        best_overlap = 0.0
        for d in diarization_data:
            overlap = max(0.0, min(end, d["end"]) - max(start, d["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = d["speaker"]
        return best_spk

    segments = []
    i = 0

    while i < len(valid_words):
        seg_start = valid_words[i]["start"]
        seg_words = [valid_words[i]]
        best_cut = None
        best_gap = -1.0

        for j in range(i + 1, len(valid_words)):
            elapsed = valid_words[j]["end"] - seg_start

            # Techo duro: no superar max_dur
            if elapsed > max_dur:
                if best_cut is None:
                    best_cut = j
                break

            seg_words_candidate = seg_words + [valid_words[j]]

            # Gap entre la palabra actual y la anterior
            gap = valid_words[j]["start"] - valid_words[j - 1]["end"]

            # Solo buscar cortes una vez alcanzado min_dur
            current_dur = valid_words[j - 1]["end"] - seg_start
            if current_dur >= min_dur:
                # Corte por fin de oración
                prev_word_text = valid_words[j - 1].get("word", "").strip()
                if prev_word_text and prev_word_text[-1] in ".!?":
                    best_cut = j
                    break

                # Corte por silencio natural
                if gap >= gap_threshold:
                    best_cut = j
                    break

                # Guardar mejor gap como fallback
                if gap > best_gap:
                    best_gap = gap
                    best_cut = j

            seg_words = seg_words_candidate

        if best_cut is None:
            best_cut = len(valid_words)

        chunk_words = valid_words[i:best_cut]
        seg_start = chunk_words[0]["start"]
        seg_end = chunk_words[-1]["end"]
        seg_text = " ".join(w.get("word", "").strip() for w in chunk_words).strip()

        if seg_text:
            speaker = _assign_speaker(seg_start, seg_end)
            segments.append({
                "start": round(seg_start, 3),
                "end":   round(seg_end, 3),
                "text":  seg_text,
                "speaker": speaker,
            })

        i = best_cut

    return segments


def detect_emotions_batch(segments: list, audio_path: str, device: str = "cuda"):
    """
    Analiza la emoción de cada segmento original en inglés.
    Utiliza un modelo Wav2Vec2 especializado en reconocimiento de emociones en habla.
    """
    from transformers import pipeline
    import torch
    import librosa
    import os

    _log("Cargando modelo de reconocimiento de emociones (Prosodia)...")
    try:
        # Usamos un modelo robusto de clasificación de emociones
        classifier = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            device=0 if device == "cuda" else -1
        )
    except Exception as e:
        _log(f"ERROR cargando modelo de emoción: {e}. Saltando prosodia.")
        return segments

    # Cargar audio completo una vez para extraer trozos rápido
    y, sr = librosa.load(audio_path, sr=16000)

    _log(f"Analizando emociones de {len(segments)} segmentos...")
    for i, seg in enumerate(segments):
        start_s = seg["start"]
        end_s = seg["end"]
        
        # Extraer clip
        start_sample = int(start_s * 16000)
        end_sample = int(end_s * 16000)
        clip = y[start_sample:end_sample]
        
        if len(clip) < 1600: # Ignorar clips < 0.1s
            seg["emotion"] = "neutral"
            continue
            
        try:
            # Predicción
            preds = classifier(clip)
            # El modelo devuelve etiquetas como 'angry', 'happy', etc.
            top_emotion = preds[0]['label']
            seg["emotion"] = top_emotion
        except:
            seg["emotion"] = "neutral"
            
    # Liberar memoria
    del classifier
    if device == "cuda":
        torch.cuda.empty_cache()
        
    _log("Detección de emociones completada.")
    return segments


def extract_speaker_references(
    wav_path: str,
    segments: list,
    output_dir: str,
    min_continuous_dur: float = 5.0,
    buffer_sec: float = 1.0,
    target_total_dur: float = 10.0,
) -> dict:
    """
    Cazador de Referencias Pro:
    1. Agrupa segmentos por SPEAKER.
    2. Filtra segmentos de >5s que NO tengan solapamiento (buffer 1s) con otros locutores.
    3. Concatena hasta llegar a 10s.
    4. Guarda en output_dir/ref_SPEAKER_XX.wav y retorna mapeo.
    """
    import pydub
    from collections import defaultdict
    
    os.makedirs(output_dir, exist_ok=True)
    audio = pydub.AudioSegment.from_wav(wav_path)
    
    # 1. Agrupar por locutor
    spk_groups = defaultdict(list)
    for s in segments:
        spk_groups[s["speaker"]].append(s)
        
    speaker_refs = {}
    
    for spk, segs in spk_groups.items():
        if spk == "SPEAKER_UNKNOWN": continue
        
        _log(f"Analizando locutor {spk} ({len(segs)} segmentos)...")
        
        # 2. Buscar candidatos "limpios"
        candidatos = []
        for i, s in enumerate(segs):
            dur = s["end"] - s["start"]
            if dur < min_continuous_dur:
                continue
            
            # Validar buffer vs otros locutores
            # Buscamos en TODOS los segmentos (no solo de este speaker)
            is_clean = True
            for other in segments:
                if other["speaker"] == spk: continue
                
                # Si algún otro habla a menos de buffer_sec del inicio o fin de este
                dist_start = abs(s["start"] - other["end"])
                dist_end   = abs(other["start"] - s["end"])
                
                # Si hay solapamiento real o distancia < buffer
                overlaps = max(0.0, min(s["end"], other["end"]) - max(s["start"], other["start"]))
                if overlaps > 0 or dist_start < buffer_sec or dist_end < buffer_sec:
                    is_clean = False
                    break
            
            if is_clean:
                candidatos.append(s)
        
        # 3. Extraer y concatenar
        if not candidatos:
            _log(f"  WARN: No se encontraron monólogos limpios >{min_continuous_dur}s para {spk}.")
            # Fallback: tomar lo más largo disponible aunque sea < 5s (mejor que nada para clonar)
            candidatos = sorted(segs, key=lambda x: x["end"] - x["start"], reverse=True)[:3]
            
        combined = pydub.AudioSegment.empty()
        current_dur = 0.0
        
        for cand in candidatos:
            start_ms = int(cand["start"] * 1000)
            end_ms   = int(cand["end"] * 1000)
            combined += audio[start_ms:end_ms]
            current_dur += (cand["end"] - cand["start"])
            if current_dur >= target_total_dur:
                break
                
        if len(combined) > 0:
            out_wav = os.path.join(output_dir, f"ref_{spk}.wav")
            out_txt = os.path.join(output_dir, f"ref_{spk}.txt")
            
            combined.export(out_wav, format="wav")
            # El texto de referencia es clave para Qwen-TTS: tomamos el primer segmento largo
            ref_texts = []
            for cand in candidatos:
                ref_texts.append(cand["text"])
            
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(" ".join(ref_texts)[:500])
                
            speaker_refs[spk] = {
                "audio": out_wav,
                "text": " ".join(ref_texts)[:500]
            }
            _log(f"  OK Referencia generada: {out_wav} ({current_dur:.1f}s)")
            
    return speaker_refs


def transcribe_whisperx(
    wav_path: str,
    model_size: str = "base",
    diarization_json_path: str = None,
    device: str = "cuda",
    min_seg_dur: float = 2.0,
    max_seg_dur: float = 12.0,
    hf_token: str = None,
) -> list:
    """
    Transcribe con WhisperX (Forced Alignment a nivel de palabra).

    Pipeline:
      1. Transcripción con whisperx.load_model (basado en faster-whisper internamente).
      2. Alineación forzada (forced alignment) con modelo de alineación fonética
         → timestamps precisos a nivel de palabra (milisegundos).
      3. Agrupación de palabras en frases de 2-12 segundos, cortando en
         finales de oración o en silencios naturales.

    Retorna lista de dicts: {start, end, text, speaker}
    """
    # ── MONKEY PATCH: Soluciona el error de "unexpected keyword 'token'" ──
    try:
        from pyannote.audio.core.inference import Inference
        if not getattr(Inference.__init__, "_is_patched", False):
            _original_init = Inference.__init__
            def _patched_init(self, *args, **kwargs):
                kwargs.pop("token", None)  # Eliminamos el token problemático aquí
                _original_init(self, *args, **kwargs)
            _patched_init._is_patched = True
            Inference.__init__ = _patched_init
    except Exception:
        pass

    import whisperx
    import torch
    import gc

    _log(f"WhisperX: Cargando modelo '{model_size}' en {device}...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    # ── 1. Transcripción ──────────────────────────────────────────────────
    _log("WhisperX: Transcribiendo audio...")
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, language="en", batch_size=16)

    n_segs = len(result.get("segments", []))
    _log(f"WhisperX: Transcripción bruta → {n_segs} segmentos")

    # Liberar modelo de transcripción de VRAM antes de alignment
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── 2. Forced Alignment (word-level timestamps) ───────────────────────
    _log("WhisperX: Cargando modelo de alineación forzada...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en", device=device,
    )

    _log("WhisperX: Alineando palabras (forced alignment)...")
    result = whisperx.align(
        result["segments"], align_model, align_metadata,
        audio, device,
        return_char_alignments=False,
    )

    # Liberar modelo de alineación
    del align_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── 3. Extraer todas las palabras alineadas ───────────────────────────
    all_words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None:
                all_words.append({
                    "start": round(w["start"], 3),
                    "end":   round(w["end"], 3),
                    "word":  w.get("word", "").strip(),
                })

    _log(f"WhisperX: {len(all_words)} palabras alineadas con timestamps exactos")

    if not all_words:
        _log("WARN: No se obtuvieron palabras alineadas")
        return []

    # ── 4. Diarización Integrada (opcional) ──────────────────────────────
    diarization_data = []
    
    # ── 4. Diarización Integrada (opcional) ──────────────────────────────
    diarization_data = []
    
    # Si se pide diarización y hay token, la corremos aquí mismo
    if hf_token:
        _log("WhisperX: Iniciando diarización integrada (Pyannote directa)...")
        
        # --- BYPASS: Usamos Pyannote directo y creamos el DataFrame manual ---
        from whisperx.diarize import assign_word_speakers
        from pyannote.audio import Pipeline
        import pandas as pd
        import torch
        
        # Cargamos el modelo real saltándonos el wrapper roto de WhisperX
        pyannote_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        if device == "cuda":
            pyannote_model.to(torch.device("cuda"))
            
        # Formateamos el audio en memoria como a Pyannote le gusta: [1, samples]
        audio_data = {
            'waveform': torch.from_numpy(audio).unsqueeze(0),
            'sample_rate': 16000
        }
        
        # Ejecutamos la diarización
        diarization_raw = pyannote_model(audio_data)
        
        # Convertimos el resultado a un DataFrame de Pandas (el formato estricto que exige WhisperX)
        diarize_df = pd.DataFrame(diarization_raw.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        
        # Alinear locutores con los segmentos de texto de Whisper
        result = assign_word_speakers(diarize_df, result)
        
        # Extraer los bloques para tu Cazador de Referencias
        for _, row in diarize_df.iterrows():
            diarization_data.append({
                "start": round(row["start"], 3),
                "end": round(row["end"], 3),
                "speaker": row["speaker"]
            })
            
        # Actualizar all_words con la información de speaker ya alineada
        all_words = []
        for seg in result.get("segments", []):
            spk = seg.get("speaker", "SPEAKER_UNKNOWN")
            for w in seg.get("words", []):
                if w.get("start") is not None and w.get("end") is not None:
                    all_words.append({
                        "start": round(w["start"], 3),
                        "end":   round(w["end"], 3),
                        "word":  w.get("word", "").strip(),
                        "speaker": spk
                    })
    else:
        # Modo compatible: Cargar diarización de archivo si no hay token HF
        if diarization_json_path and os.path.exists(diarization_json_path):
            try:
                with open(diarization_json_path, "r", encoding="utf-8") as f:
                    diarization_data = json.load(f).get("results", [])
            except Exception:
                pass

    # ── 5. Agrupar palabras en segmentos de 2-12s ─────────────────────────
    _log(f"WhisperX: Agrupando palabras en frases ({min_seg_dur}-{max_seg_dur}s)...")
    segments = _group_words_into_segments(
        all_words,
        min_dur=min_seg_dur,
        max_dur=max_seg_dur,
        gap_threshold=0.4,
        diarization_data=diarization_data,
    )

    for seg in segments:
        _log(f"  [{seg['start']:6.1f}s → {seg['end']:5.1f}s] "
             f"[{seg['speaker']}] {seg['text'][:60]}")

    _log(f"WhisperX: OK {len(segments)} segmentos finales "
         f"({min_seg_dur}-{max_seg_dur}s por segmento)")
    return segments


def transcribe(wav_path: str, model_size: str = "base", diarization_json_path: str = None) -> list:
    """
    Transcribe con faster-whisper (legacy, sin forced alignment).
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


# ─── PASO 3: TRADUCCIÓN Y ADAPTACIÓN (Qwen3.5-9B) ────────────────────────────

# Ruta al modelo local Qwen3.5-9B (relativa al proyecto o absoluta)
QWEN_TRANSLATE_MODEL = os.environ.get(
    "QWEN_TRANSLATE_MODEL",
    str(Path(__file__).parent / "Qwen3.5-9B"),
)


def _count_syllables_en(text: str) -> int:
    """Estimación rápida de sílabas en inglés (heurística de vocales)."""
    import re
    text = text.lower().strip()
    if not text:
        return 0
    # Contar grupos de vocales consecutivas
    count = len(re.findall(r'[aeiouy]+', text))
    # Ajuste: "e" final silenciosa
    if text.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)


def translate_segments_qwen(
    segments: list,
    model_path: str = None,
    device: str = "cuda",
    progress_cb=None,
) -> list:
    """
    Traduce/adapta cada segmento EN→ES usando Qwen3.5-9B como LLM local.

    No es una traducción literal: es una ADAPTACIÓN DE GUION para doblaje.
    El prompt le indica al modelo la duración original y el conteo de sílabas
    para que genere una traducción concisa que quepa en el tiempo disponible.

    Libera la VRAM del modelo al terminar (para dejar espacio al TTS).

    Parámetros:
        segments    — lista de dicts con {start, end, text, ...}
        model_path  — ruta al directorio del modelo Qwen3.5-9B
        device      — "cuda" o "cpu"
        progress_cb — callback(done, total) para UI

    Retorna la misma lista con text_es rellenado.
    """
    import torch
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = model_path or QWEN_TRANSLATE_MODEL

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Modelo Qwen3.5-9B no encontrado en {model_dir}. "
            f"Descárgalo y colócalo en esa ruta antes de ejecutar el pipeline."
        )

    # Contar pendientes
    pending = [(i, seg) for i, seg in enumerate(segments)
               if seg.get("text", "").strip() and not seg.get("text_es")]
    if not pending:
        _log("Todos los segmentos ya tienen traducción")
        return segments

    _log(f"Cargando Qwen3.5-9B desde {model_dir} en {device}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    _log(f"Qwen3.5-9B cargado en {time.time()-t0:.1f}s")

    SYSTEM_PROMPT = (
        "Eres un director de doblaje experto. Tu objetivo es traducir guiones del inglés al español "
        "manteniendo una coherencia narrativa perfecta, respetando el género de los hablantes y "
        "manteniendo el tuteo (tú) o formalidad (usted) consistente.\n\n"
        "Recibirás un JSON con frases. Debes devolver el MISMO formato JSON exacto en una lista, "
        "alterando únicamente el valor de 'texto_espanol', adaptando la longitud al 'target_duration_ms'."
    )

    BATCH_SIZE = 5
    CONTEXT_SIZE = 2
    
    # Trabajamos sobre todos los segmentos para poder extraer contexto
    for start_idx in range(0, len(segments), BATCH_SIZE):
        batch_indices = range(start_idx, min(start_idx + BATCH_SIZE, len(segments)))
        batch_segments = [segments[i] for i in batch_indices]
        
        # Filtrar si ya están traducidos (opcional, pero mejor procesar lote completo para coherencia)
        # Si todos en el lote ya tienen text_es, saltamos.
        if all(s.get("text_es") for s in batch_segments):
            continue

        # 1. Obtener contexto previo (2 frases en inglés)
        context_texts = []
        if start_idx > 0:
            for c_idx in range(max(0, start_idx - CONTEXT_SIZE), start_idx):
                context_texts.append(segments[c_idx].get("text", ""))

        # 2. Preparar bloque a traducir
        bloque_input = []
        for i in batch_indices:
            seg = segments[i]
            bloque_input.append({
                "id": i,
                "speaker": seg.get("speaker", "UNKNOWN"),
                "texto_original": seg.get("text", "").strip(),
                "target_duration_ms": int((seg.get("end", 0) - seg.get("start", 0)) * 1000)
            })

        user_input_json = {
            "contexto_previo_no_traducir": context_texts,
            "bloque_a_traducir": bloque_input
        }

        user_prompt = (
            f"Traduce el siguiente JSON siguiendo las reglas de doblaje:\n\n"
            f"```json\n{json.dumps(user_input_json, ensure_ascii=False, indent=2)}\n```"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024, # Aumentado por JSON batch
                    temperature=0.2,    # Más bajo para JSON estable
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                )

            # Decodificar y extraer JSON
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Limpiar posibles bloques de código triple backtick
            if "```json" in response_text:
                json_str = response_text.split("```json")[-1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[-1].split("```")[0].strip()
            else:
                json_str = response_text

            translated_data = json.loads(json_str)
            
            # Mapear de vuelta a segments
            if isinstance(translated_data, list):
                for item in translated_data:
                    ver_id = item.get("id")
                    txt_es = item.get("texto_espanol")
                    if ver_id is not None and txt_es:
                        segments[ver_id]["text_es"] = txt_es
                        _log(f"  [{ver_id:3d}] {segments[ver_id]['text'][:30]}... → {txt_es[:30]}...")
            else:
                _log(f"  ERROR: El modelo no devolvió una lista JSON en lote {start_idx}")

        except Exception as e:
            _log(f"  ERROR en lote {start_idx}: {e}. Reintentando individuales o saltando...")
            # Fallback opcional: si el lote falla, podrías intentar uno a uno aquí
            # Por ahora marcamos vacíos para no bloquear
            for i in batch_indices:
                if not segments[i].get("text_es"):
                    segments[i]["text_es"] = segments[i].get("text", "")

        if progress_cb:
            progress_cb(min(start_idx + BATCH_SIZE, len(segments)), len(segments))

    # Liberar modelo de VRAM
    _log("Liberando Qwen3.5-9B de VRAM...")
    del model
    del tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    _log(f"Adaptación completada: {len(pending)} segmentos en {time.time()-t0:.1f}s")
    return segments


def resummarize_text(text_en: str, target_dur_ms: int, model_path: str = None, device: str = "cuda") -> str:
    """
    LLM Fallback: Cuando un segmento es demasiado largo (>25%), se vuelve a llamar a Qwen
    con instrucciones de resumen agresivo para que el texto traducido sea más corto.
    """
    import torch
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_dir = model_path or QWEN_TRANSLATE_MODEL
    _log(f"Resummarize: Cargando LLM para acortar texto (objetivo: {target_dur_ms}ms)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    
    PROMPT = (
        f"Eres un experto en doblaje. El siguiente texto en español es demasiado largo para el tiempo disponible ({target_dur_ms}ms).\n"
        f"TAREA: Resume o parafrasea el texto agresivamente para que sea MUCHO MÁS CORTO pero mantenga el sentido esencial.\n"
        f"Responde ÚNICAMENTE con el nuevo texto en español.\n\n"
        f"Texto original: {text_en}"
    )
    
    messages = [{"role": "user", "content": PROMPT}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=128, temperature=0.2)
        new_tokens = ids[0][inputs["input_ids"].shape[1]:]
        short_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
    _log(f"Resummarize: '{text_en[:30]}...' -> '{short_text[:30]}...'")
    
    # Liberar memoria inmediatamente
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return short_text


def translate_segments(segments: list, progress_cb=None) -> list:
    """
    Punto de entrada de traducción — siempre usa Qwen3.5-9B.
    """
    _log("Usando Qwen3.5-9B para adaptación de guion...")
    return translate_segments_qwen(segments, progress_cb=progress_cb)



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
            base_prompt = voice_prompts.get(spk) or voice_prompts.get("DEFAULT") or list(voice_prompts.values())[0]
            
            # --- INYECCIÓN DE PROSODIA (EMOCIÓN) ---
            emotion = batch[0][1].get("emotion", "neutral").lower()
            style_map = {
                "happy": "Habla con tono alegre y entusiasta.",
                "angry": "Habla con tono firme, molesto y autoritario.",
                "sad": "Habla con tono pausado y melancólico.",
                "fear": "Habla con tono ansioso y temeroso.",
                "surprise": "Habla con tono de asombro y sorpresa.",
                "disgust": "Habla con tono de desprecio y desagrado.",
                "neutral": "Habla con tono natural y equilibrado."
            }
            style_instruction = style_map.get(emotion, style_map["neutral"])
            prompt_to_use = f"{style_instruction} {base_prompt}"
            
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
            target_dur = seg["end"] - seg["start"]
            target_dur_ms = int(target_dur * 1000)

            if wav is None:
                _log(f"WARN segmento {i+1} — omitido")
                continue

            sf.write(raw_path, wav, sr)
            tts_dur = len(wav) / sr
            ratio = tts_dur / max(target_dur, 0.001)

            # --- ARQUITECTURA HÍBRIDA DE DECISIONES ---
            
            # CASO C: > 25% Excedido -> Re-traducción / Resúmen
            if ratio > 1.25:
                _log(f"  seg {i+1:3d} | ratio {ratio:.2f} EXCEDE 1.25. Re-traduciendo...")
                new_text = resummarize_text(seg.get("text_es", ""), target_dur_ms)
                seg["text_es"] = new_text
                # Re-generar este segmento específico (fuera de lote para simplicidad)
                try:
                    spk = seg.get("speaker", "SPEAKER_UNKNOWN")
                    prompt_to_use = voice_prompts.get(spk) or voice_prompts.get("DEFAULT")
                    w_retry, _ = model.generate_voice_clone(
                        text=[new_text], language=[language], voice_clone_prompt=prompt_to_use
                    )
                    wav = w_retry[0]
                    sf.write(raw_path, wav, sr)
                    tts_dur = len(wav) / sr
                    ratio = tts_dur / max(target_dur, 0.001)
                    _log(f"  seg {i+1:3d} | Nuevo ratio tras resumen: {ratio:.2f}")
                except Exception as ex:
                    _log(f"  ERROR en re-traducción seg {i+1}: {ex}")

            # CASO B: 1.11 - 1.25 -> Video Freeze (Pads at end)
            if 1.10 < ratio <= 1.25:
                _log(f"  seg {i+1:3d} | ratio {ratio:.2f} -> Requiere VIDEO FREEZE")
                seg["needs_video_freeze"] = True
                seg["freeze_duration"] = tts_dur - target_dur
                # En audio, lo dejamos natural (sin comprimir) para que el video lo espere
                shutil.copy(raw_path, adj_path)
                action = "FREEZE"
                adj_dur = tts_dur
            
            # CASO A: <= 1.10 -> Audio Stretch (Time-stretch imperceptible)
            elif ratio <= 1.10:
                if FREE_FLOWING:
                    shutil.copy(raw_path, adj_path)
                    action = "NATURAL"
                else:
                    # Aplicar atempo para encajar exacto en target_dur (o ventana)
                    adjust_duration(raw_path, adj_path, target_dur)
                    action = "STRETCH"
                adj_dur = get_duration(adj_path)
            
            else:
                # Si falló el resumen o ratio sigue alto, comprimimos al límite (1.25x)
                # para que el freeze no sea eterno
                adjust_duration(raw_path, adj_path, target_dur * 1.25)
                seg["needs_video_freeze"] = True
                seg["freeze_duration"] = get_duration(adj_path) - target_dur
                action = "COMPRIMIDO+FREEZE"
                adj_dur = get_duration(adj_path)


            seg["tts_adjusted_dur"] = adj_dur
            _log(f"  seg {i+1:3d} | {tts_dur:.2f}s → {adj_dur:.2f}s | {action} | {seg.get('text_es','')[:40]}")

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
    Ensamblado No-Lineal (Paralelo): Superpone los segmentos TTS en una línea 
    de tiempo dinámica que permite Crosstalk (interrupciones).
    """
    from pydub import AudioSegment
    import io

    _log(f"Ensamblando audio PARALELO ({len(segments)} segmentos)...")
    
    # 1. Calcular duración total estimada del 'canvas'
    extra_time = sum(s.get("freeze_duration", 0) for s in segments)
    estimated_total_ms = int((total_duration + extra_time + 10.0) * 1000)
    
    canvas = AudioSegment.silent(duration=estimated_total_ms, frame_rate=sr)
    canvas = canvas.set_channels(1)
    
    timeline_shift_ms = 0
    MAX_FREE_GAP_MS = int(MAX_FREE_GAP * 1000)

    for i, seg in enumerate(segments):
        start_en_ms = int(seg["start"] * 1000)
        
        # Modo libre: reducir silencios pero preservar overlaps
        if free_flowing and i > 0:
            prev_end_en_ms = int(segments[i-1]["end"] * 1000)
            gap_en_ms = start_en_ms - prev_end_en_ms
            if gap_en_ms > MAX_FREE_GAP_MS:
                timeline_shift_ms -= (gap_en_ms - MAX_FREE_GAP_MS)
            
        pos_es_ms = start_en_ms + timeline_shift_ms
        if pos_es_ms < 0: pos_es_ms = 0

        adj_path = seg.get("tts_adjusted_path", "")
        if adj_path and os.path.exists(adj_path):
            seg_audio = AudioSegment.from_file(adj_path).set_frame_rate(sr).set_channels(1)
            canvas = canvas.overlay(seg_audio, position=pos_es_ms)
            
            # Freeze shift
            freeze_dur_ms = int(seg.get("freeze_duration", 0) * 1000)
            if freeze_dur_ms > 0:
                timeline_shift_ms += freeze_dur_ms

    canvas.export(output_wav, format="wav")
    _log(f"Audio paralelo ensamblado: {output_wav}")


def mix_voice_with_background(voice_path, bg_path, output_path, voice_vol=1.5, bg_vol=1.0, ducking=True):
    """
    Mezcla voz con fondo con Auto-Ducking tipo Netflix.
    - El fondo baja al ~20% (-14dB) cuando hay voz.
    - El fondo vuelve al 100% durante los silencios.
    """
    _log(f"Mezclando audio PRO: Voz={voice_vol} Fondo={bg_vol} Ducking={ducking}")
    
    if not ducking:
        filter_str = f"[0:a]volume={voice_vol}[v];[1:a]volume={bg_vol}[bg];[v][bg]amix=inputs=2:duration=longest"
    else:
        # Sidechain compress optimizado para sensación documental (caída profunda, retorno gradual)
        filter_str = (
            f"[0:a]volume={voice_vol}[v];"
            f"[1:a]volume={bg_vol},asidelp=threshold=0.03:ratio=12:attack=5:release=1000:sidechain=0[bg];"
            f"[v][bg]amix=inputs=2:duration=longest"
        )

    run_cmd([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", bg_path,
        "-filter_complex", filter_str,
        "-ac", "2",
        output_path
    ])


def merge_video_with_audio_hybrid(video_path: str, audio_path: str, segments: list, output_path: str):
    """
    ARQUITECTURA HÍBRIDA 2.0: Reconstruye el video manejando SOLAPAMIENTOS (Crosstalk)
    agrupando segmentos que se pisan para evitar tartamudeo visual.
    """
    import tempfile
    _log("Iniciando Fusión Híbrida PRO (Manejo de Crosstalk)...")
    tmp_dir = tempfile.mkdtemp(prefix="hybrid_merge_")
    chunk_list = os.path.join(tmp_dir, "chunks.txt")
    
    try:
        # 1. Agrupar segmentos que se solapan o están muy cerca
        groups = []
        if not segments: return
        
        current_group = [segments[0]]
        for i in range(1, len(segments)):
            prev_end = current_group[-1]["end"]
            curr_start = segments[i]["start"]
            
            # Si el siguiente empieza antes de que el anterior termine (o < 0.1s de gap)
            if curr_start < prev_end + 0.1:
                current_group.append(segments[i])
            else:
                groups.append(current_group)
                current_group = [segments[i]]
        groups.append(current_group)

        # 2. Procesar cada grupo como una unidad de video
        processed_chunks = []
        for i, group in enumerate(groups):
            g_start = min(s["start"] for s in group)
            g_end = max(s["end"] for s in group)
            g_freeze = sum(s.get("freeze_duration", 0) for s in group)
            
            chunk_raw = os.path.join(tmp_dir, f"group_{i:04d}_raw.mp4")
            chunk_final = os.path.join(tmp_dir, f"group_{i:04d}_final.mp4")
            
            # Recortar el bloque original del video
            run_cmd([
                "ffmpeg", "-y", "-ss", str(g_start), "-to", str(g_end),
                "-i", video_path, "-c", "copy", chunk_raw
            ])
            
            if g_freeze > 0.05:
                # Aplicar freeze acumulado al final de la interacción
                _log(f"  Grupo {i+1}: Crosstalk detectado ({len(group)} segs). Aplicando Freeze acumulado de {g_freeze:.2f}s")
                run_cmd([
                    "ffmpeg", "-y", "-i", chunk_raw,
                    "-vf", f"tpad=stop_duration={g_freeze}:stop_mode=clone",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-an", chunk_final
                ])
            else:
                run_cmd([
                    "ffmpeg", "-y", "-i", chunk_raw,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-an", chunk_final
                ])
            processed_chunks.append(chunk_final)

        # 3. Concatenar y fusionar con audio paralelo
        with open(chunk_list, "w") as f:
            for c in processed_chunks:
                f.write(f"file '{os.path.abspath(c)}'\n")
        
        merged_video = os.path.join(tmp_dir, "merged_video.mp4")
        run_cmd([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", chunk_list,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", merged_video
        ])
        
        run_cmd([
            "ffmpeg", "-y", "-i", merged_video, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            output_path
        ], "Fusión Final Híbrida 2.0")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    """CLI Entry Point (Modo básico para pruebas locales)"""
    print("=" * 65)
    print("  EN -> ES Voice Dubbing Pipeline  |  QAV PRO")
    print("=" * 65)
    print("Use app.py para la experiencia completa con UI.")

if __name__ == "__main__":
    main()
