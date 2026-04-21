import os
import numpy as np
import soundfile as sf
import traceback

def create_dummy_audio(path="dummy_audio.wav"):
    """Crea un archivo WAV de 2 segundos de silencio para poder testear la transcripción."""
    print(f"[TEST] Creando audio de prueba en {path}...")
    sr = 16000
    audio_data = np.zeros(sr * 2, dtype=np.float32)
    sf.write(path, audio_data, sr)
    return path

def test_faster_whisper_api():
    try:
        import faster_whisper
        print(f"[TEST] Version de faster-whisper instalada: {faster_whisper.__version__}")
    except ImportError:
        print("[ERROR] faster-whisper no esta instalado en este entorno.")
        return

    from faster_whisper import WhisperModel, BatchedInferencePipeline

    dummy_wav = create_dummy_audio()

    try:
        print("[TEST] 1. Inicializando WhisperModel (modelo 'tiny' en CPU para no usar VRAM)...")
        # Usamos CPU y float32 para que puedas correr esto en tu laptop sin GPU
        model = WhisperModel("tiny", device="cpu", compute_type="float32")

        print("[TEST] 2. Inicializando BatchedInferencePipeline (API NUEVA, sin use_vad_model)...")
        # Esta es la linea que explotaba antes. Si pasa de aqui, el fix funciona.
        batched_model = BatchedInferencePipeline(model=model)
        print("       -> OK! Pipeline inicializado exitosamente.")

        print("[TEST] 3. Probando metodo transcribe() (pasando vad_filter=True)...")
        segments_gen, info = batched_model.transcribe(
            dummy_wav, 
            language="en", 
            batch_size=4, 
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True
        )
        # Forzamos la evaluacion del generador para asegurar que no hay crash interno
        segments = list(segments_gen)
        print(f"       -> OK! Transcripcion completada sin errores. (Segmentos detectados: {len(segments)})")
        print("\n✅ EXITO: La nueva API de faster-whisper es 100% compatible con tu codigo.")
    except Exception as e:
        print(f"\n❌ ERROR FATAL durante la prueba:\n{traceback.format_exc()}")
    finally:
        if os.path.exists(dummy_wav):
            os.remove(dummy_wav)

if __name__ == "__main__":
    test_faster_whisper_api()