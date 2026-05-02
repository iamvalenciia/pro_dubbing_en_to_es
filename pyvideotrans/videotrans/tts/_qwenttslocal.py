import os
import time
from dataclasses import dataclass
from pathlib import Path

import json

from videotrans import translator
from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang
from videotrans.tts._base import BaseTTS
from videotrans.util import tools
from videotrans.process import qwen3tts_fun

@dataclass
class QwenttsLocal(BaseTTS):
    def __post_init__(self):
        super().__post_init__()
        self.model_name="1.7B"
        _langnames = translator.LANG_CODE.get(self.language, [])
        if _langnames and len(_langnames) >= 10:
            self.target_language = _langnames[9]
        else:
            self.target_language = 'Auto'
        self.target_language=self.target_language.capitalize()

    
    def _download(self):
        def _model_ready(local_dir: str) -> bool:
            p = Path(local_dir)
            if not p.exists():
                return False
            has_cfg = (p / "config.json").exists() and (p / "tokenizer.json").exists()
            has_weights = any(p.glob("*.safetensors")) or (p / "model.bin").exists() or (p / "pytorch_model.bin").exists()
            return has_cfg and has_weights

        base_dir = f'{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{self.model_name}-Base'
        custom_dir = f'{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{self.model_name}-CustomVoice'

        self._signal(text=f"[qwen3tts-patch] download_check model={self.model_name}")

        if _model_ready(base_dir) and _model_ready(custom_dir):
            self._signal(text="[qwen3tts-patch] cache_hit Base+CustomVoice, skip download")
            return

        if defaulelang == 'zh':
            tools.check_and_down_ms(f'Qwen/Qwen3-TTS-12Hz-{self.model_name}-Base',callback=self._process_callback,local_dir=base_dir)
            tools.check_and_down_ms(f'Qwen/Qwen3-TTS-12Hz-{self.model_name}-CustomVoice',callback=self._process_callback,local_dir=custom_dir)
        else:
            tools.check_and_down_hf(model_id=f'Qwen3-TTS-12Hz-{self.model_name}-Base',repo_id=f'Qwen/Qwen3-TTS-12Hz-{self.model_name}-Base',local_dir=base_dir,callback=self._process_callback)
            tools.check_and_down_hf(model_id=f'Qwen3-TTS-12Hz-{self.model_name}-CustomVoice',repo_id=f'Qwen/Qwen3-TTS-12Hz-{self.model_name}-CustomVoice',local_dir=custom_dir,callback=self._process_callback)


    def _exec(self):
        Path(f'{TEMP_DIR}/{self.uuid}').mkdir(parents=True,exist_ok=True)
        logs_file = f'{TEMP_DIR}/{self.uuid}/qwen3tts-{time.time()}.log'
        
        queue_tts_file = f'{TEMP_DIR}/{self.uuid}/queuetts-{time.time()}.json'
        Path(queue_tts_file).write_text(json.dumps(self.queue_tts),encoding='utf-8')
        title="Qwen3-TTS"
        kwargs = {            
            "queue_tts_file":queue_tts_file,
            "language": self.target_language,
            "logs_file": logs_file,
            "defaulelang": defaulelang,
            "is_cuda": self.is_cuda,
            "model_name":self.model_name,
            "roledict":tools.get_qwenttslocal_rolelist(),
            "prompt":params.get('qwenttslocal_prompt', '')
        }
        self._new_process(callback=qwen3tts_fun,title=title,is_cuda=self.is_cuda,kwargs=kwargs)
    
        self._signal(text=f'convert wav')
        all_task = []
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        env_workers = os.environ.get("PYVIDEOTRANS_CONVERT_WAV_WORKERS", "").strip()
        try:
            max_workers = int(env_workers) if env_workers else min(4, len(self.queue_tts), os.cpu_count() or 1)
        except Exception:
            max_workers = min(4, len(self.queue_tts), os.cpu_count() or 1)
        max_workers = max(1, max_workers)
        total = 0
        done = 0
        self._signal(text=f"[qwen3tts] convert_wav_start workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for item in self.queue_tts:
                filename=item.get('filename','')+"-qwen3tts.wav"
                if tools.vail_file(filename):
                    all_task.append(pool.submit(self.convert_to_wav, filename,item['filename']))
            total = len(all_task)
            if total > 0:
                for fut in as_completed(all_task):
                    fut.result()
                    done += 1
                    if done == 1 or done == total or done % 10 == 0:
                        self._signal(text=f"[qwen3tts] convert_wav {done}/{total}")
            else:
                self._signal(text="[qwen3tts] convert_wav 0/0")
        self._signal(text=f"[qwen3tts] convert_wav_done {done}/{total}")


