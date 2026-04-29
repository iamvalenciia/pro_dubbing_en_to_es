# 语音识别前进行各项处理，单独进程实现
# 返回元组
# 失败：第一个值为 False，第二个值存储失败原因
# 成功，第一个返回数据，不需要数据时返回True，第二个值为None

import traceback
from videotrans.configure.config import ROOT_DIR, logger, TEMP_ROOT


def _write_log(file=None, msg=None, type='logs'):
    if not file or not msg:
        return
    from pathlib import Path
    import json
    try:
        Path(file).write_text(json.dumps({"text": msg, "type": type}), encoding='utf-8')
    except Exception:
        logger.exception('写入新进程日志时出错', exc_info=True)


# 1. 分离背景声和人声 https://k2-fsa.github.io/sherpa/onnx/source-separation/models.html#uvr
# 仅使用cpu，不使用gpu
def vocal_bgm(*, input_file, vocal_file, instr_file, logs_file=None, is_cuda=False, uvr_models="UVR-MDX-NET-Inst_HQ_4"):
    """UVR for source separation."""
    import time
    import torch
    from pathlib import Path
    import numpy as np
    import sherpa_onnx
    import soundfile as sf

    def create_offline_source_separation():
        model = f"{ROOT_DIR}/models/onnx/{uvr_models}.onnx"

        if not Path(model).is_file():
            raise ValueError(f"{model} does not exist.")

        config = sherpa_onnx.OfflineSourceSeparationConfig(
            model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=model),
                num_threads=4,
                debug=False,
                provider="cpu",
            )
        )
        if not config.validate():
            raise ValueError("Please check your config.")

        return sherpa_onnx.OfflineSourceSeparation(config)

    def load_audio(wav_file):
        samples, sample_rate = sf.read(wav_file, dtype="float32", always_2d=True)
        samples = np.transpose(samples)
        assert samples.shape[1] > samples.shape[0], f"You should use (num_channels, num_samples). {samples.shape}"
        assert samples.dtype == np.float32, f"Expect np.float32 as dtype. Given: {samples.dtype}"
        return samples, sample_rate

    separator = None
    try:
        separator = create_offline_source_separation()
        samples, sample_rate = load_audio(input_file)
        samples = np.ascontiguousarray(samples)
        _write_log(logs_file, "vocals non_vocals...")
        start = time.time()
        output = separator.process(sample_rate=sample_rate, samples=samples)
        elapsed_seconds = time.time() - start

        vocals = np.transpose(output.stems[1].data)
        non_vocals = np.transpose(output.stems[0].data)
        sf.write(vocal_file, vocals, samplerate=output.sample_rate)
        sf.write(instr_file, non_vocals, samplerate=output.sample_rate)

        _write_log(logs_file, f" use time:{elapsed_seconds}s")
        return True, None
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f"人声背景声分离失败:{msg}", exc_info=True)
        return False, msg
    finally:
        try:
            del separator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass


# 2. 降噪 https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k
def remove_noise(*, input_file, output_file, is_cuda=False, logs_file=None, device_index=0):
    import time
    import torch
    from pathlib import Path
    from videotrans.util import tools
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import platform

    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = f"cuda:{device_index}" if is_cuda else "cpu"

    start = time.time()
    logger.info('开始语音降噪')
    pipeline_obj = None
    result = None
    tmp_name = Path(output_file).parent.as_posix() + f'/noise-{time.time()}.wav'
    try:
        pipeline_obj = pipeline(
            Tasks.acoustic_noise_suppression,
            model='iic/speech_frcrn_ans_cirm_16k',
            disable_update=True,
            disable_progress_bar=True,
            disable_log=True,
            device=device,
        )
        result = pipeline_obj(input_file, output_path=tmp_name, disable_pbar=True)
        tools.runffmpeg(['-y', '-i', tmp_name, '-af', 'volume=1.5', output_file])
        logger.info(f'降噪成功完成，耗时:{int(time.time() - start)}s')
        return output_file, None
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f'降噪失败:{msg}', exc_info=True)
        return False, msg
    finally:
        try:
            del pipeline_obj
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass


# 3. 恢复标点 https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large
def fix_punc(*, text_dict, is_cuda=False, logs_file=None, device_index=0):
    import time
    import torch
    from pathlib import Path
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import platform

    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = f"cuda:{device_index}" if is_cuda else "cpu"

    result = None
    pipeline_obj = None
    try:
        start = time.time()
        logger.debug('开始标点恢复')
        pipeline_obj = pipeline(
            task=Tasks.punctuation,
            model='iic/punc_ct-transformer_cn-en-common-vocab471067-large',
            model_revision='v2.0.4',
            disable_update=True,
            disable_progress_bar=True,
            disable_log=True,
            device=device,
        )
        payload = "\n".join([f'{line}\t{text}' for line, text in text_dict.items()])
        tmp_name = f'{TEMP_ROOT}/fix_flag-{time.time()}.txt'
        Path(tmp_name).write_text(payload, encoding='utf-8')
        result = pipeline_obj(tmp_name, disable_pbar=True)
        for item in result:
            text_dict[item['key']] = item['text']
        logger.debug(f'标点恢复完成，耗时:{int(time.time() - start)}s')
        return text_dict, None
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f'恢复标点失败:{msg}', exc_info=True)
        return False, msg
    finally:
        try:
            del pipeline_obj
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass


# Legacy local diarization backends were removed from the active workflow.
