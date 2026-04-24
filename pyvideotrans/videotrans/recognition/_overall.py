import time, json, shutil, os
from pathlib import Path
from dataclasses import dataclass
from videotrans.configure.config import settings, ROOT_DIR, TEMP_DIR, logger
from videotrans.recognition._base import BaseRecogn

from videotrans.util import tools
from pydub import AudioSegment
from videotrans.process import openai_whisper, faster_whisper
from videotrans.util.contants import FASTER_MODELS_DICT


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _resolve_a100_stt_model(*, requested_model: str, is_cuda: bool, recogn_type: int) -> str:
    # Only affect Faster-Whisper local mode.
    if recogn_type != 0:
        return requested_model
    if not is_cuda:
        return requested_model

    enabled = str(os.environ.get("PYVIDEOTRANS_A100_STT_MODE", "1")).lower() in ("1", "true", "yes", "on")
    if not enabled:
        return requested_model

    prefer_40 = os.environ.get("PYVIDEOTRANS_A100_STT_MODEL_40GB", "large-v3-turbo").strip() or "large-v3-turbo"
    prefer_16 = os.environ.get("PYVIDEOTRANS_A100_STT_MODEL_16GB", "medium").strip() or "medium"
    if prefer_40 not in FASTER_MODELS_DICT:
        prefer_40 = "large-v3-turbo"
    if prefer_16 not in FASTER_MODELS_DICT:
        prefer_16 = "medium"

    threshold_40 = _safe_int_env("PYVIDEOTRANS_A100_STT_THRESHOLD_40GB", 40)
    threshold_16 = _safe_int_env("PYVIDEOTRANS_A100_STT_THRESHOLD_16GB", 16)

    try:
        import torch
        if not torch.cuda.is_available():
            return requested_model
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_gb = total_bytes / float(1024 ** 3)
        free_gb = free_bytes / float(1024 ** 3)
    except Exception:
        return requested_model

    # Apply only on high-memory cards by default (A100 class and above).
    if total_gb < threshold_40:
        return requested_model

    selected = prefer_40
    logger.info(f"[a100_stt_mode] auto model selected: requested={requested_model} selected={selected} free_gb={free_gb:.1f} total_gb={total_gb:.1f}")
    return selected

@dataclass
class FasterAll(BaseRecogn):
    def __post_init__(self):
        super().__post_init__()
        self.model_name = _resolve_a100_stt_model(
            requested_model=self.model_name,
            is_cuda=self.is_cuda,
            recogn_type=self.recogn_type,
        )
        local_dir = f'{ROOT_DIR}/models/models--'
        if self.model_name in FASTER_MODELS_DICT:
            local_dir += FASTER_MODELS_DICT[self.model_name].replace('/', '--')
        else:
            local_dir += self.model_name.replace('/', '--')
        self.local_dir = local_dir
        self.audio_duration=len(AudioSegment.from_wav(self.audio_file))
        self.speech_timestamps_file=None

    def _exec(self):
        if self._exit():
            return
        self.error = ''
        self._signal(text="STT starting, hold on...")
        if self.recogn_type == 1:  # openai-whisper
            raws = self._openai()
        else:
            raws = self._faster()
        return raws

    def _download(self):
        if self.recogn_type == 0:
            if self.model_name in FASTER_MODELS_DICT:
                repo_id = FASTER_MODELS_DICT[self.model_name]
            else:
                repo_id = self.model_name
            try:
                tools.check_and_down_hf(self.model_name,repo_id,self.local_dir,callback=self._process_callback)
            except Exception as e:
                raise RuntimeError(
                    f"[a100_stt_mode] model download failed for {self.model_name}: {e}"
                ) from e
        # 批量时预先vad切分
        # 否则后断句处理
        if settings.get('whisper_prepare'):
            self._vad_split()
            self.speech_timestamps_file=f'{self.cache_folder}/speech_timestamps_{time.time()}.json'
            Path(self.speech_timestamps_file).write_text(json.dumps(self.speech_timestamps),encoding='utf-8')


    def _openai(self):
        title=f'STT use {self.model_name}'
        self._signal(text=title)
        # 起一个进程
        logs_file = f'{TEMP_DIR}/{self.uuid}/openai-{self.detect_language}-{time.time()}.log'
        kwargs = {
            "prompt": settings.get(
                f'initial_prompt_{self.detect_language}') if self.detect_language != 'auto' else None,
            "detect_language": self.detect_language,
            "model_name": self.model_name,
            "logs_file": logs_file,
            "is_cuda": self.is_cuda,
            "no_speech_threshold": float(settings.get('no_speech_threshold', 0.5)),
            "condition_on_previous_text": settings.get('condition_on_previous_text', False),
            "speech_timestamps": self.speech_timestamps_file,
            "audio_file": self.audio_file,
            "jianfan": self.jianfan,
            "audio_duration":self.audio_duration,
            "temperature":settings.get('temperature'),
            "compression_ratio_threshold":float(settings.get('compression_ratio_threshold',2.2)),
            "max_speech_ms":int(float(settings.get('max_speech_duration_s', 6)) * 1000)
        }
        raws=self._new_process(callback=openai_whisper,title=title,is_cuda=self.is_cuda,kwargs=kwargs)
        return raws


    def _faster(self):
        title=f"STT use {self.model_name}"
        self._signal(text=title)
        logs_file = f'{TEMP_DIR}/{self.uuid}/faster-{self.detect_language}-{time.time()}.log'

        kwargs = {
            "detect_language": self.detect_language,
            "model_name": self.model_name,
            "logs_file": logs_file,
            "is_cuda": self.is_cuda,
            "no_speech_threshold": float(settings.get('no_speech_threshold', 0.5)),
            "condition_on_previous_text": settings.get('condition_on_previous_text', False),
            "speech_timestamps": self.speech_timestamps_file,
            "audio_file": self.audio_file,
            "local_dir": self.local_dir,
            "compute_type": settings.get('cuda_com_type', 'default'),
            "batch_size": int(settings.get('stt_batch_size', 0)),
            "jianfan": self.jianfan,
            "audio_duration":self.audio_duration,
            "hotwords":settings.get('hotwords'),
            "prompt": settings.get(f'initial_prompt_{self.detect_language}') if self.detect_language != 'auto' else None,
            "beam_size": int(settings.get('beam_size', 5)),
            "best_of": int(settings.get('best_of', 5)),
            "temperature":settings.get('temperature'),
            "repetition_penalty":float(settings.get('repetition_penalty',1.0)),
            "compression_ratio_threshold":float(settings.get('compression_ratio_threshold',2.2)),
            "max_speech_ms":int(float(settings.get('max_speech_duration_s', 6)) * 1000)
        }

        raws=self._new_process(callback=faster_whisper,title=title,is_cuda=self.is_cuda,kwargs=kwargs)
        return raws


