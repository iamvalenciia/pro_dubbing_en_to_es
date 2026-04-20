import os
import time
import torch
import torchaudio

from src.logger import get_logger, phase_timer, step_timer, log_gpu_snapshot, log_file_info

try:
    from src.hw_autotune import autotune, hw_summary
    _HAS_AUTOTUNE = True
except Exception:  # pragma: no cover
    _HAS_AUTOTUNE = False

log = get_logger("phase1")


# Batch size de inferencia: numero de chunks de audio procesados en paralelo por
# forward pass del modelo. Calibrado: 96 funciona bien en H100 94GB VRAM
# (~15GB uso, margen ~5x antes de OOM). autotune lo escala linealmente por VRAM:
#   - 94GB (H100)    -> 96
#   - 48GB (A6000)   -> ~49
#   - 24GB (RTX 3090) -> ~24
#   - 12GB (RTX 3060) -> ~12
# Min 4 para no romper con GPUs muy chicas; max 128 porque el sweet-spot htdemucs
# termina alli (mas batch no acelera, solo quema VRAM).
if _HAS_AUTOTUNE:
    DEMUCS_BATCH_SIZE = autotune(
        "DEMUCS_BATCH_SIZE", baseline=96, scale_with="vram",
        baseline_ref=94, min_v=4, max_v=128,
    )
else:
    DEMUCS_BATCH_SIZE = int(os.environ.get("DEMUCS_BATCH_SIZE", "24"))

# Modelo demucs. "htdemucs" es bag of 1 (rapido + suficiente para lipsync).
# "htdemucs_ft" es bag of 4 (4x mas lento, no lo soportamos en batcheado custom).
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")

# Segmento (seg por chunk). 0 = usa el default del modelo (htdemucs=7.8s).
# Subir a 15-30s ahorra overhead de chunking pero puede perder quality en los bordes.
DEMUCS_SEGMENT = float(os.environ.get("DEMUCS_SEGMENT", "0"))

# Overlap entre chunks (fraccion). 0.25 es el default demucs. 0.10 acelera ~17%
# (menos chunks totales) con perdida de quality imperceptible en voz: la ventana
# triangular sigue suavizando bordes, y el pipeline downstream solo usa vocals.wav
# para ASR+TTS, no para mastering.
DEMUCS_OVERLAP = float(os.environ.get("DEMUCS_OVERLAP", "0.10"))


def _batched_separate(model, wav, sr, segment_s, overlap, batch_size, log_):
    """Inferencia batcheada manual de demucs para saturar GPU.

    A diferencia del CLI demucs (que procesa chunks batch=1 secuencialmente),
    aca stackeamos B chunks en un solo tensor (B, channels, segment_samples)
    y pasamos ese batch completo al forward del modelo. Con B=32-64 en A100,
    el transformer/conv stack finalmente aprovecha los cores.

    wav: tensor (n_channels, n_samples) en GPU, fp32
    returns: dict {source_name: tensor(n_channels, n_samples)} en CPU, fp32
    """
    device = wav.device
    n_channels, n_samples = wav.shape
    segment_length = int(segment_s * sr)
    overlap_length = int(segment_length * overlap)
    stride = segment_length - overlap_length
    if stride <= 0:
        raise RuntimeError(f"overlap={overlap} invalido: stride={stride} <= 0")

    offsets = list(range(0, n_samples, stride))
    n_chunks = len(offsets)

    # Normalizacion demucs-style: (wav - mean) / std, sobre el audio entero.
    ref = wav.mean(0)
    mean = ref.mean()
    std = ref.std().clamp(min=1e-8)

    sources = list(model.sources)
    n_sources = len(sources)

    # Ventana triangular demucs-style: peso mas alto en el centro del chunk,
    # decae linealmente hacia los bordes. Combinado con overlap-add, los
    # samples de cada chunk contribuyen proporcionalmente a su confianza.
    L = segment_length
    weight = torch.cat([
        torch.arange(1, L // 2 + 1, device=device, dtype=torch.float32),
        torch.arange(L - L // 2, 0, -1, device=device, dtype=torch.float32),
    ])

    # Buffers de reconstruccion (fp32, en GPU para no saltar device-memory boundary
    # en el hot loop). Para un wav de 94 min stereo 44.1kHz, esto son ~8 GB para
    # 4 stems en A100 80GB — margen de sobra.
    reconstructed = torch.zeros(
        n_sources, n_channels, n_samples, device=device, dtype=torch.float32
    )
    weights_sum = torch.zeros(n_samples, device=device, dtype=torch.float32)

    num_batches = (n_chunks + batch_size - 1) // batch_size
    log_.info(
        f"Batched inference: n_chunks={n_chunks} batch_size={batch_size} "
        f"num_batches={num_batches} segment={segment_length/sr:.2f}s "
        f"overlap={overlap*100:.0f}% stride={stride/sr:.2f}s"
    )

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch_start in enumerate(range(0, n_chunks, batch_size)):
            batch_offsets = offsets[batch_start:batch_start + batch_size]
            b = len(batch_offsets)

            # Arma el batch slicing directo del wav en GPU (sin copia extra).
            batch = torch.zeros(
                b, n_channels, segment_length, device=device, dtype=torch.float32
            )
            for j, off in enumerate(batch_offsets):
                end = min(off + segment_length, n_samples)
                batch[j, :, :end - off] = wav[:, off:end]

            batch_norm = (batch - mean) / std

            # Forward con bfloat16 autocast: A100 tiene tensor cores bf16 nativos.
            # Linear/Conv downcast a bf16, STFT/LayerNorm se quedan fp32 (auto).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch_norm)  # (b, n_sources, n_channels, segment_length)

            out = out.float() * std + mean  # denormalize back to fp32

            # Overlap-add en el hot loop (sin buffer intermedio de outputs).
            for j, off in enumerate(batch_offsets):
                end = min(off + segment_length, n_samples)
                length = end - off
                w = weight[:length]
                reconstructed[:, :, off:end] += out[j, :, :, :length] * w
                weights_sum[off:end] += w

            # Log periodico de progreso
            if batch_idx == 0 or (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
                done_chunks = min(batch_start + b, n_chunks)
                pct = 100.0 * done_chunks / n_chunks
                elapsed = time.time() - t0
                rate_chunks = done_chunks / max(elapsed, 1e-6)
                log_.info(
                    f"  batch {batch_idx+1}/{num_batches} | "
                    f"{done_chunks}/{n_chunks} chunks ({pct:.1f}%) | "
                    f"{elapsed:.1f}s ({rate_chunks:.1f} chunks/s)"
                )

    # Normalize by accumulated weights
    reconstructed /= weights_sum.clamp(min=1e-8)

    # Return stems to CPU for I/O
    result = {src: reconstructed[j].cpu() for j, src in enumerate(sources)}
    del reconstructed, weights_sum
    return result


def run_phase1_audio_separation(audio_input_path: str, output_dir: str):
    """Fase 1: Desacoplamiento Acustico con Demucs htdemucs batcheado en GPU.

    GPU-only por diseno. No hay fallback CPU. Si CUDA no esta, RuntimeError.

    Diferencias vs la version anterior basada en CLI subprocess:
      - Inference INLINE (mismo proceso): la VRAM se ve en los snapshots.
      - Batching manual (DEMUCS_BATCH_SIZE chunks simultaneos): satura el A100.
      - bf16 autocast: usa tensor cores bf16 de Ampere.
      - TF32 + cudnn.benchmark: acelera conv stack de htdemucs.
      - Carga audio una sola vez (fp32 en GPU), procesa en-place.

    Env vars:
      DEMUCS_BATCH_SIZE (default 32): chunks por forward pass.
      DEMUCS_MODEL (default htdemucs): solo bag-of-1 soportado.
      DEMUCS_SEGMENT (default 0=model): duracion del chunk en segundos.
      DEMUCS_OVERLAP (default 0.25): fraccion de overlap entre chunks.
    """
    with phase_timer(log, "FASE 1 — Separacion Acustica (Demucs)"):
        log.info(f"Input audio: {audio_input_path}")
        log_file_info(log, audio_input_path, "input_audio")
        log.info(f"Output dir: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # GPU-only hard-check.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA no disponible para Demucs. Fase 1 es GPU-only por diseno — "
                "no hay fallback a CPU. Verifica que el pod tenga GPU asignada."
            )

        # Ampere-friendly perf knobs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        torch.cuda.empty_cache()
        log_gpu_snapshot(log, "pre-demucs")

        dev_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        log.info(
            f"GPU forzada dev={dev_idx}: {props.name} "
            f"sm={props.major}.{props.minor} "
            f"vram={props.total_memory/1e9:.2f}GB | "
            f"tf32=ON cudnn.benchmark=ON bf16_autocast=ON | "
            f"model={DEMUCS_MODEL} batch_size={DEMUCS_BATCH_SIZE} "
            f"overlap={DEMUCS_OVERLAP}"
        )

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels
            from demucs.audio import convert_audio
        except ImportError as e:
            raise RuntimeError(
                f"demucs no disponible: {e}. Instalar: pip install demucs==4.0.1"
            ) from e

        with step_timer(log, f"Cargando modelo {DEMUCS_MODEL} a CUDA"):
            bag = get_model(DEMUCS_MODEL)
            if isinstance(bag, BagOfModels):
                if len(bag.models) == 1:
                    model = bag.models[0]
                    log.info(f"Modelo {DEMUCS_MODEL} es BagOfModels con 1 sub-modelo — unwrap OK")
                else:
                    raise RuntimeError(
                        f"Modelo {DEMUCS_MODEL} es bag de {len(bag.models)} sub-modelos. "
                        f"El batcheado custom solo soporta bag-of-1. "
                        f"Usa DEMUCS_MODEL=htdemucs (default)."
                    )
            else:
                model = bag
            model = model.cuda().eval()
        log_gpu_snapshot(log, "post-demucs-load")

        sr_model = model.samplerate
        n_audio_channels = model.audio_channels
        segment_s = DEMUCS_SEGMENT if DEMUCS_SEGMENT > 0 else float(model.segment)

        log.info(
            f"Modelo listo: sources={list(model.sources)} "
            f"sr={sr_model} channels={n_audio_channels} segment={segment_s:.2f}s"
        )

        with step_timer(log, "Cargando audio + resample/channels al modelo"):
            wav, in_sr = torchaudio.load(audio_input_path)
            wav = convert_audio(wav, in_sr, sr_model, n_audio_channels)
            wav = wav.to(device="cuda", dtype=torch.float32)
        log.info(
            f"Audio en GPU: shape={list(wav.shape)} "
            f"({wav.shape[1]/sr_model:.2f}s @ {sr_model}Hz)"
        )
        log_gpu_snapshot(log, "audio-loaded")

        with step_timer(log, f"Batched GPU inference (batch_size={DEMUCS_BATCH_SIZE})"):
            stems = _batched_separate(
                model, wav, sr_model,
                segment_s=segment_s,
                overlap=DEMUCS_OVERLAP,
                batch_size=DEMUCS_BATCH_SIZE,
                log_=log,
            )
        log_gpu_snapshot(log, "post-demucs-inference")

        if "vocals" not in stems:
            raise RuntimeError(f"demucs no retorno stem 'vocals'. Stems: {list(stems.keys())}")

        # Merge non-vocal stems into "no_vocals" (estilo --two-stems vocals del CLI).
        non_vocal_keys = [k for k in stems.keys() if k != "vocals"]
        if not non_vocal_keys:
            raise RuntimeError(
                f"demucs no retorno stems de fondo. Stems: {list(stems.keys())}"
            )
        vocals_tensor = stems["vocals"]
        no_vocals_tensor = sum(stems[k] for k in non_vocal_keys)
        log.info(f"Stems: vocals + merge({'+'.join(non_vocal_keys)}) -> no_vocals")

        vocals_out = os.path.join(output_dir, "voces_crudas.wav")
        ambient_out = os.path.join(output_dir, "ambiente.wav")

        with step_timer(log, "Guardando stems wav"):
            torchaudio.save(vocals_out, vocals_tensor, sr_model)
            torchaudio.save(ambient_out, no_vocals_tensor, sr_model)

        # Libera VRAM (modelo + audio + stems) antes de que arranque Fase 2.
        del model, bag, wav, stems, vocals_tensor, no_vocals_tensor
        torch.cuda.empty_cache()
        log_gpu_snapshot(log, "post-demucs-free")

        log_file_info(log, vocals_out, "vocals_out")
        log_file_info(log, ambient_out, "ambient_out")

    return vocals_out, ambient_out
