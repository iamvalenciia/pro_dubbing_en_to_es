import json
import os
import sys
import time
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper no está instalado. Ejecuta: pip install faster-whisper")
    sys.exit(1)

def extraer_segmentos(audio_path, output_json, model_size="medium", device="cuda"):
    """
    Transcribe un audio (por defecto en español) y extrae segmentos con timestamps.
    """
    if not os.path.exists(audio_path):
        print(f"Error: No se encuentra el archivo de audio: {audio_path}")
        return

    t0 = time.time()
    print(f"\n--- Iniciando Extracción de Segmentos ---")
    print(f"Audio: {audio_path}")
    print(f"Modelo: {model_size} ({device})")
    
    # 1. Cargar modelo
    try:
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        if device == "cuda":
            print("Reintentando en CPU...")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
            return

    # 2. Transcribir
    print(f"Procesando transcripción en español (ES)...")
    segments_iter, info = model.transcribe(
        audio_path,
        language="es",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )

    results = []
    print(f"\nSegmentos detectados:")
    print(f"{'Inicio':>8} | {'Fin':>8} | {'Texto'}")
    print("-" * 60)

    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
            
        res = {
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
            "text":  text
        }
        results.append(res)
        print(f"{res['start']:8.2f}s | {res['end']:8.2f}s | {text}")

    # 3. Guardar JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    duracion = time.time() - t0
    print("-" * 60)
    print(f"¡Extracción completada!")
    print(f"Segmentos guardados en: {output_path.absolute()}")
    print(f"Total segmentos: {len(results)}")
    print(f"Tiempo procesado: {duracion:.1f}s")
    print(f"----------------------------------------\n")

if __name__ == "__main__":
    # Valores por defecto configurados para tu caso específico
    DEFAULT_AUDIO = r"c:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\cache\asdasd\final_es.mp3"
    DEFAULT_OUT   = r"c:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\cache\asdasd\segments_final_es.json"

    audio_in = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO
    json_out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    model_sz = sys.argv[3] if len(sys.argv) > 3 else "medium"

    extraer_segmentos(audio_in, json_out, model_size=model_sz)
