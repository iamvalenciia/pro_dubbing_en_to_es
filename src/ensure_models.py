import json
import os
import time

from src.logger import get_logger, phase_timer, step_timer

log = get_logger("ensure_models")


QWEN_MODELS = [
    ("QWEN_ASR_PATH", "Qwen/Qwen3-ASR-1.7B", "/runpod-volume/models/Qwen/Qwen3-ASR-1.7B"),
    ("QWEN_ALIGNER_PATH", "Qwen/Qwen3-ForcedAligner-0.6B", "/runpod-volume/models/Qwen/Qwen3-ForcedAligner-0.6B"),
    ("QWEN_TTS_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "/runpod-volume/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    # MarianMT EN->ES para Fase 3 (reemplaza Gemini). ~300MB, sin API key.
    ("MARIAN_MODEL_PATH", "Helsinki-NLP/opus-mt-en-es", "/runpod-volume/models/Helsinki-NLP/opus-mt-en-es"),
]


def _is_complete(local_dir: str) -> bool:
    """Valida que el snapshot este completo: config.json + todos los shards del index.

    Solo considera completo si hay model.safetensors (no pytorch_model.bin suelto)
    — los .bin son bloqueados por transformers con torch < 2.6 (CVE-2025-32434),
    asi que requerimos conversion a safetensors antes de marcar OK.
    """
    if not os.path.isdir(local_dir):
        return False
    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        return False

    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            required_shards = set(idx.get("weight_map", {}).values())
        except (OSError, json.JSONDecodeError):
            return False
        for shard in required_shards:
            shard_path = os.path.join(local_dir, shard)
            if not os.path.isfile(shard_path):
                return False
            if os.path.getsize(shard_path) < 1024 * 1024:
                return False
        return bool(required_shards)

    single = os.path.join(local_dir, "model.safetensors")
    if os.path.isfile(single) and os.path.getsize(single) > 50 * 1024 * 1024:
        return True
    return False


def _convert_bin_to_safetensors(local_dir: str, log_) -> bool:
    """Si hay pytorch_model.bin y falta model.safetensors, convierte.

    Necesario para modelos legacy (ej. Helsinki-NLP/opus-mt-en-es) porque
    transformers >=4.46 bloquea torch.load con torch <2.6 por CVE-2025-32434.
    Devuelve True si al terminar hay un .safetensors utilizable.
    """
    st_path = os.path.join(local_dir, "model.safetensors")
    bin_path = os.path.join(local_dir, "pytorch_model.bin")
    if os.path.isfile(st_path):
        return True
    if not os.path.isfile(bin_path):
        return False
    try:
        import torch
        from safetensors.torch import save_file
        log_.info(f"  Convirtiendo pytorch_model.bin -> model.safetensors en {local_dir}")
        sd = torch.load(bin_path, map_location="cpu", weights_only=False)
        # Marian y otros encoder-decoders comparten embeddings; safetensors
        # requiere tensores independientes, asi que forzamos copia contigua.
        sd = {k: (v.detach().contiguous().clone() if hasattr(v, "detach") else v)
              for k, v in sd.items()}
        save_file(sd, st_path)
        log_.info(f"  OK: safetensors generado ({os.path.getsize(st_path)/1024**2:.1f} MB)")
        try:
            os.remove(bin_path)
            log_.info("  pytorch_model.bin eliminado (ya no se usa)")
        except OSError:
            pass
        return True
    except Exception as e:
        log_.error(f"  Conversion bin->safetensors fallo: {e}")
        return False


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024 ** 2


def ensure_qwen_models():
    """Descarga los modelos Qwen3 al volumen si faltan o estan incompletos."""
    with phase_timer(log, "FASE -1 — Verificacion/Descarga de Modelos Qwen3"):
        from huggingface_hub import snapshot_download

        hf_token = os.environ.get("HF_TOKEN")
        log.info(f"HF_TOKEN presente: {bool(hf_token)}")

        for env_var, repo_id, default_path in QWEN_MODELS:
            local_dir = os.environ.get(env_var, default_path)
            log.info(f"Modelo: {repo_id}  ->  {local_dir}")

            if _is_complete(local_dir):
                size_mb = _dir_size_mb(local_dir)
                log.info(f"  OK: {repo_id} ya presente ({size_mb:.1f} MB)")
                continue

            # Caso legacy: bajado antes con solo pytorch_model.bin. Convertir in-place
            # sin re-descargar.
            if os.path.isdir(local_dir) and os.path.isfile(os.path.join(local_dir, "pytorch_model.bin")) \
               and not os.path.isfile(os.path.join(local_dir, "model.safetensors")):
                log.info(f"  Convirtiendo legacy bin->safetensors para {repo_id}")
                if _convert_bin_to_safetensors(local_dir, log) and _is_complete(local_dir):
                    size_mb = _dir_size_mb(local_dir)
                    log.info(f"  OK tras conversion: {repo_id} ({size_mb:.1f} MB)")
                    continue

            log.warning(f"  INCOMPLETO: {repo_id} falta o esta roto. Descargando...")
            os.makedirs(local_dir, exist_ok=True)
            with step_timer(log, f"snapshot_download {repo_id}"):
                t0 = time.time()
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    token=hf_token,
                    resume_download=True,
                    max_workers=8,
                )
                dt = time.time() - t0
                size_mb = _dir_size_mb(local_dir)
                log.info(f"  Descargado {size_mb:.1f} MB en {dt:.1f}s")

            # Post-download: si el modelo solo trae pytorch_model.bin, convertir
            # a safetensors para evitar bloqueo por torch < 2.6.
            _convert_bin_to_safetensors(local_dir, log)
