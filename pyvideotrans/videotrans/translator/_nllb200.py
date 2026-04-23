import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from videotrans.configure.config import ROOT_DIR
from videotrans.translator._base import BaseTrans


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

    def _download(self):
        model_cache_dir = self._resolve_cache_dir()
        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=model_cache_dir,
            src_lang=self.from_lang,
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=model_cache_dir,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.to_lang),
            num_beams=4,
            max_new_tokens=512,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return "\n".join([it.strip() for it in translated])
