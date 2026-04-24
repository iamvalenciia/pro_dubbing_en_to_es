import os
import time
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
from packaging.version import Version

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from videotrans.configure.config import ROOT_DIR, logger, tr
from videotrans.translator._base import BaseTrans
from videotrans.util import tools


_LANGUAGE_CODE_MAP = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "th": "tha_Thai",
    "it": "ita_Latn",
    "el": "ell_Grek",
    "pt": "por_Latn",
    "vi": "vie_Latn",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "hi": "hin_Deva",
    "hu": "hun_Latn",
    "uk": "ukr_Cyrl",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "kk": "kaz_Cyrl",
    "cs": "ces_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "he": "heb_Hebr",
    "bn": "ben_Beng",
    "fa": "pes_Arab",
    "fi": "tgl_Latn",
    "ur": "urd_Arab",
    "yue": "zho_Hant",
    "nb": "nob_Latn",
}


@dataclass
class NLLB200Trans(BaseTrans):
    def __post_init__(self):
        super().__post_init__()
        self.aisendsrt = False
        self.model_name = "facebook/nllb-200-distilled-600M"
        src_code = (self.source_code or "").strip().lower()
        tgt_code = (self.target_code or "").strip().lower()

        if not src_code or src_code == "auto":
            # NLLB needs a concrete source language; default to English for current EN->ES workflow.
            self.from_lang = "eng_Latn"
        else:
            self.from_lang = _LANGUAGE_CODE_MAP.get(src_code[:2], "eng_Latn")

        self.to_lang = _LANGUAGE_CODE_MAP.get(tgt_code[:2], "spa_Latn")
        self.trans_thread = self._resolve_trans_thread()

    def _resolve_trans_thread(self) -> int:
        # Allow explicit override from env to simplify production tuning.
        forced = os.environ.get("PYVIDEOTRANS_NLLB_BATCH_LINES", "").strip()
        if forced:
            try:
                return max(1, min(int(forced), 1024))
            except Exception:
                pass

        try:
            current = max(int(getattr(self, "trans_thread", 10)), 1)
        except Exception:
            current = 10

        if not torch.cuda.is_available():
            return current

        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
        except Exception:
            return max(current, 32)

        # Optional target VRAM knob for operators.
        # Example: PYVIDEOTRANS_NLLB_TARGET_VRAM_GB=60 on A100-80GB.
        target_vram_env = (
            os.environ.get("PYVIDEOTRANS_NLLB_TARGET_VRAM_GB", "").strip()
            or os.environ.get("PYVIDEOTRANS_GPU_TARGET_VRAM_GB", "").strip()
        )
        if target_vram_env:
            try:
                target_vram_gb = max(float(target_vram_env), 1.0)
                # Empirical mapping from current profile: ~512 lines around ~70 GB available.
                target_from_vram = int(round((target_vram_gb / 70.0) * 512.0))
                # Hard safety cap and floor.
                target_from_vram = max(32, min(target_from_vram, 1024))
                # Never choose a batch that is absurd relative to currently free VRAM.
                free_bound = max(32, min(int(round((free_gb / 70.0) * 640.0)), 1024))
                return max(current, min(target_from_vram, free_bound))
            except Exception:
                pass

        if free_gb >= 70:
            target = 512
        elif free_gb >= 40:
            target = 160
        elif free_gb >= 24:
            target = 96
        elif free_gb >= 16:
            target = 64
        elif free_gb >= 10:
            target = 48
        else:
            target = 32

        return max(current, target)

    def _download(self):
        model_cache_dir = self._resolve_cache_dir()
        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=model_cache_dir,
            src_lang=self.from_lang,
        )
        target_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Transformers en versiones recientes exige torch>=2.6 cuando internamente usa torch.load.
        # Forzamos safetensors para evitar la ruta vulnerable y mantener compatibilidad.
        if Version(torch.__version__.split("+")[0]) < Version("2.6.0"):
            raise RuntimeError(
                f"Torch {torch.__version__} es incompatible para cargar NLLB-200 de forma segura. "
                "Actualiza a torch>=2.6 o reconstruye la imagen con las versiones recomendadas."
            )

        def _has_meta_params(model) -> bool:
            try:
                return any(getattr(p, "is_meta", False) for p in model.parameters())
            except Exception:
                return False

        def _load_model(*, use_safetensors: bool, low_cpu_mem_usage: bool):
            model_kwargs = {
                "cache_dir": model_cache_dir,
                "use_safetensors": use_safetensors,
                "low_cpu_mem_usage": low_cpu_mem_usage,
            }

            return AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

        # Estrategia robusta: probar rutas de carga sin forzar dtype durante from_pretrained.
        # Forzar dtype durante load puede activar rutas internas con tensores meta.
        last_err = None
        load_attempts = [
            # Prioridad: safetensors (ruta segura y recomendada)
            (True, False),
            (True, True),
            # Fallback de formato de checkpoint, solo permitido con torch>=2.6 (guard arriba)
            (False, False),
            (False, True),
        ]
        self.model = None
        for use_safetensors, low_cpu_mem_usage in load_attempts:
            try:
                model = _load_model(use_safetensors=use_safetensors, low_cpu_mem_usage=low_cpu_mem_usage)
                if _has_meta_params(model):
                    raise RuntimeError(
                        f"NLLB loaded with meta tensors (use_safetensors={use_safetensors}, "
                        f"low_cpu_mem_usage={low_cpu_mem_usage})"
                    )
                self.model = model
                break
            except Exception as e:
                last_err = e
                try:
                    del model
                except Exception:
                    pass

        if self.model is None:
            raise RuntimeError(f"NLLB model load failed after all strategies: {last_err}")

        # Mover al dispositivo final y ajustar precision despues de que el modelo existe en memoria real.
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.to(dtype=target_dtype)
        self.model.eval()
        return True

    def _resolve_cache_dir(self):
        cache_root = (
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or (f"{os.environ.get('QDP_LOCAL_DIR')}/torch_cache" if os.environ.get("QDP_LOCAL_DIR") else "")
            or f"{ROOT_DIR}/models"
        )
        return f"{cache_root}/nllb200_distilled_600m"

    def _unload(self):
        try:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @torch.inference_mode()
    def _item_task(self, data: Union[List[str], str]):
        queries = data if isinstance(data, list) else [data]
        if self._exit():
            return ""

        self.tokenizer.src_lang = self.from_lang
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        try:
            seq_len = int(inputs["input_ids"].shape[-1])
            batch_lines = len(queries)
            if self.device == "cuda" and torch.cuda.is_available():
                dev = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
                free_now, total_now = torch.cuda.mem_get_info(dev)
                free_now_gb = free_now / (1024 ** 3)
                total_now_gb = total_now / (1024 ** 3)
                logger.info(
                    f"[nllb200] device=cuda:{dev} batch_lines={batch_lines} seq_len={seq_len} "
                    f"trans_thread={self.trans_thread} alloc_gb={allocated:.2f} reserved_gb={reserved:.2f} "
                    f"free_gb={free_now_gb:.2f} total_gb={total_now_gb:.2f}"
                )
            else:
                logger.info(
                    f"[nllb200] device={self.device} batch_lines={batch_lines} seq_len={seq_len} "
                    f"trans_thread={self.trans_thread}"
                )
        except Exception:
            pass
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.to_lang),
            pad_token_id=eos_id,
            num_beams=4,
            max_new_tokens=512,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return "\n".join([it.strip() for it in translated])

    def _tokenize_batch(self, queries: List[str]) -> dict:
        """CPU-only tokenization; safe to call from a background thread."""
        self.tokenizer.src_lang = self.from_lang
        return self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

    @torch.inference_mode()
    def _generate_from_pretokenized(self, inputs: dict, queries: List[str]) -> str:
        """GPU generation from already-tokenized inputs; CPU tokenization already done in background."""
        try:
            seq_len = int(inputs["input_ids"].shape[-1])
            batch_lines = len(queries)
            if self.device == "cuda" and torch.cuda.is_available():
                dev = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
                free_now, total_now = torch.cuda.mem_get_info(dev)
                free_now_gb = free_now / (1024 ** 3)
                total_now_gb = total_now / (1024 ** 3)
                logger.info(
                    f"[nllb200] device=cuda:{dev} batch_lines={batch_lines} seq_len={seq_len} "
                    f"trans_thread={self.trans_thread} alloc_gb={allocated:.2f} reserved_gb={reserved:.2f} "
                    f"free_gb={free_now_gb:.2f} total_gb={total_now_gb:.2f}"
                )
            else:
                logger.info(
                    f"[nllb200] device={self.device} batch_lines={batch_lines} seq_len={seq_len} "
                    f"trans_thread={self.trans_thread}"
                )
        except Exception:
            pass
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.to_lang),
            pad_token_id=eos_id,
            num_beams=4,
            max_new_tokens=512,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return "\n".join([it.strip() for it in translated])

    def _run_text(self, split_source_text):
        """Override base _run_text to pipeline CPU tokenization with GPU generation.

        A single background thread tokenizes batch i+1 while the GPU processes
        batch i, eliminating the tokenizer latency from the GPU critical path.
        """
        target_list = []
        total_batches = len(split_source_text)
        total_lines = len(self.text_list)
        done_lines = 0
        run_started_at = time.time()

        if not split_source_text:
            return self.text_list

        tok_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nllb_tok")
        try:
            # Tokenize the first batch immediately so it’s ready before the loop body runs.
            pending_inputs_future = tok_executor.submit(self._tokenize_batch, split_source_text[0])

            for i, it in enumerate(split_source_text):
                if self._exit():
                    return
                batch_no = i + 1
                elapsed = max(time.time() - run_started_at, 0.001)
                avg_batch_sec = elapsed / batch_no
                remaining_batches = max(total_batches - batch_no, 0)
                eta_sec = max(int(math.ceil(avg_batch_sec * remaining_batches)), 0)
                eta_min, eta_rem_sec = divmod(eta_sec, 60)
                eta_text = f"{eta_min:02d}:{eta_rem_sec:02d}"
                self._signal(
                    text=(
                        f"{tr('starttrans')} {batch_no}/{total_batches} "
                        f"lines={done_lines}/{total_lines} eta={eta_text}"
                    )
                )

                # Retrieve pre-tokenized inputs (may already be done while prev batch was on GPU).
                inputs = pending_inputs_future.result()

                # Immediately kick off tokenization of the NEXT batch — overlaps GPU generation below.
                if i + 1 < len(split_source_text):
                    pending_inputs_future = tok_executor.submit(self._tokenize_batch, split_source_text[i + 1])
                else:
                    pending_inputs_future = None

                result = self._get_cache(it)
                if not result:
                    result = tools.cleartext(self._generate_from_pretokenized(inputs, it))
                    self._set_cache(it, result)

                sep_res = result.split("\n")
                for x, result_item in enumerate(sep_res):
                    if x < len(it):
                        target_list.append(result_item.strip())
                        self._signal(text=result_item + "\n", type='subtitle')
                if len(sep_res) < len(it):
                    print(f'\u884c\u6570\u4e0d\u5339\u914d\uff0c\u539f\u59cb\uff1a{len(it)}, \u7ed3\u679c\uff1a{len(sep_res)}\n{it=}\n{sep_res=}')
                    target_list += ["" for _ in range(len(it) - len(sep_res))]

                done_lines += len(it)
                elapsed_done = max(time.time() - run_started_at, 0.001)
                avg_line_sec = elapsed_done / max(done_lines, 1)
                line_per_sec = done_lines / elapsed_done
                remaining_lines = max(total_lines - done_lines, 0)
                remaining_batches_after = max(total_batches - batch_no, 0)
                eta_done_sec = max(int(math.ceil(avg_line_sec * remaining_lines)), 0)
                eta_done_min, eta_done_rem_sec = divmod(eta_done_sec, 60)
                eta_finish = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_done_sec))
                self._signal(
                    text=(
                        f"[translate-progress] {done_lines}/{total_lines} lines "
                        f"({(done_lines / max(total_lines, 1)) * 100:.1f}%) "
                        f"remaining_lines={remaining_lines} remaining_batches={remaining_batches_after} "
                        f"speed={line_per_sec:.2f} lines/s eta={eta_done_min:02d}:{eta_done_rem_sec:02d} "
                        f"eta_finish={eta_finish}"
                    )
                )
                time.sleep(self.wait_sec)
        finally:
            tok_executor.shutdown(wait=False)

        max_i = len(target_list)
        logger.debug(f'\u4ee5\u666e\u901a\u6587\u672c\u884c\u6309\u884c\u7ffb\u8bd1\uff1a\u539f\u59cb\u884c\u6570:{len(self.text_list)},\u7ffb\u8bd1\u540e\u884c\u6570:{max_i}')
        for i, it in enumerate(self.text_list):
            if i < max_i:
                self.text_list[i]['text'] = target_list[i]
            else:
                self.text_list[i]['text'] = ""
        return self.text_list
