import os
import json
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline

from src.logger import get_logger, phase_timer, step_timer, log_gpu_snapshot, log_file_info
from src.paths import NETWORK_MODELS

log = get_logger("phase2")

SINGLE_SPEAKER_ID = "SPEAKER_00"


def run_phase2_diarization_and_asr(
    vocals_path: str,
    output_json: str,
    temp_workspace: str = None,
):
    """Fase 2: ASR con Faster-Whisper (Large-V3).
    Agrupa lógicamente basándose en VAD y puntuación natural del modelo.
    Todos los segmentos se asignan a SPEAKER_00 para cumplir con la SPEC.
    """
    with phase_timer(log, "FASE 2 — ASR Lógico con Faster-Whisper"):
        log.info(f"Input vocals: {vocals_path}")
        log_file_info(log, vocals_path, "vocals_input")

        log_gpu_snapshot(log, "pre-asr")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA no disponible dentro del contenedor. "
                "Verifica que el pod tenga GPU asignada (nvidia-smi) y runtime NVIDIA activo."
            )
        torch.cuda.empty_cache()

        # Configurar y cargar modelo Faster-Whisper
        model_size = "large-v3"
        download_dir = os.path.join(NETWORK_MODELS, "faster-whisper")
        os.makedirs(download_dir, exist_ok=True)
        
        with step_timer(log, f"Cargando Faster-Whisper ({model_size}) en bfloat16"):
            model = WhisperModel(
                model_size, 
                device="cuda", 
                compute_type="bfloat16", 
                download_root=download_dir
            )
            # Pipeline para inferencia en lotes (exprime la VRAM)
            batched_model = BatchedInferencePipeline(model=model)
        
        log_gpu_snapshot(log, "post-asr-load")

        log.info("Transcribiendo audio completo en Lotes (Batched Inference)...")
        with step_timer(log, "WhisperModel.transcribe"):
            # batch_size=32 satura la GPU reduciendo el tiempo radicalmente.
            # condition_on_previous_text=False evita bloqueos de paralelismo y alucinaciones.
            segments_gen, info = batched_model.transcribe(
                vocals_path, 
                language="en", 
                batch_size=32,
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True
            )
            
            master_timeline = []
            for segment in segments_gen:
                text = segment.text.strip()
                if not text:
                    continue
                    
                start = float(segment.start)
                end = float(segment.end)
                
                master_timeline.append({
                    "segment_id": len(master_timeline),
                    "speaker": SINGLE_SPEAKER_ID,
                    "start": start,
                    "end": end,
                    "original_duration": end - start,
                    "text_en": text,
                    "text_es": "",
                    "cloned_audio_path": None,
                    "dubbed_duration": 0.0
                })
                
                preview = text[:80] + ("..." if len(text) > 80 else "")
                log.info(f"  seg[{len(master_timeline)-1}] [{start:.2f}s-{end:.2f}s] "
                         f"dur={end-start:.2f}s text='{preview}'")

        log.info(f"Total segmentos lógicos generados: {len(master_timeline)}")

        # Liberación de Memoria Crítica (Requisito SPEC 6)
        del model
        torch.cuda.empty_cache()
        log_gpu_snapshot(log, "post-asr-free")

        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(master_timeline, f, indent=2, ensure_ascii=False)
        log_file_info(log, output_json, "timeline_json")

    return output_json
