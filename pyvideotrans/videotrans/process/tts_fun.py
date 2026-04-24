# 语音识别，新进程执行
# 返回元组
# 失败：第一个值为False，则为失败，第二个值存储失败原因
# 成功，第一个值存在需要的返回值，不需要时返回True，第二个值为None
from videotrans.configure.config import logger,ROOT_DIR

# ========== SINGLETON CACHE FOR QWEN3-TTS MODELS ==========
# Avoid re-downloading/re-loading models during duplicate initialization
_QWEN3_MODEL_CACHE = {}

def _get_or_load_qwen3_model(model_key: str, model_path: str, device_map: str, dtype, attn_implementation):
    """
    Retrieve model from cache or load it once.
    model_key: 'base' or 'custom' (identifies which variant)
    model_path: full path to model directory
    Returns: Qwen3TTSModel instance
    """
    cache_key = (model_key, device_map, str(dtype))
    if cache_key in _QWEN3_MODEL_CACHE:
        return _QWEN3_MODEL_CACHE[cache_key]
    
    # Not in cache, load it
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation
    )
    _QWEN3_MODEL_CACHE[cache_key] = model
    return model


def _resolve_qwen_tts_batch_lines(is_cuda: bool):
    import os
    import torch
    env_val = os.environ.get("PYVIDEOTRANS_QWEN_TTS_BATCH_LINES")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except Exception:
            pass

    if not is_cuda or not torch.cuda.is_available():
        return 2

    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gib = free_bytes / float(1024 ** 3)
    except Exception:
        free_gib = 0.0

    if free_gib >= 60:
        return 12
    if free_gib >= 40:
        return 10
    if free_gib >= 24:
        return 12
    if free_gib >= 12:
        return 8
    if free_gib >= 6:
        return 4
    return 2


def _batched(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def _write_log(file, msg):
    from pathlib import Path
    try:
        Path(file).write_text(msg, encoding='utf-8')
    except Exception as e:
        logger.exception(f'写入新进程日志时出错', exc_info=True)


def _prepare_ref_wav(wavfile: str, max_ms: int = 60_000) -> str:
    """Return wavfile unchanged or a trimmed copy capped at max_ms milliseconds."""
    import subprocess
    from pathlib import Path
    try:
        from videotrans.util import tools
        duration_ms = int(tools.get_audio_time(wavfile) or 0)
    except Exception:
        duration_ms = 0
    if duration_ms == 0 or duration_ms <= max_ms:
        return wavfile
    # Build a trimmed copy alongside the original file.
    p = Path(wavfile)
    trimmed = str(p.parent / (p.stem + f"_ref_trim{max_ms}ms" + p.suffix))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", wavfile,
        "-t", f"{max_ms / 1000:.3f}",
        "-c", "copy",
        trimmed,
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return wavfile  # use original if trim fails
    return trimmed


def qwen3tts_fun(
        queue_tts_file=None,# 配音数据存在 json文件下，根据文件路径获取
        language='Auto',#语言
        logs_file=None,
        defaulelang="en",
        is_cuda=False,
        prompt=None,
        model_name='1.7B',
        roledict=None,
        device_index=0 # gpu索引
):
    import re, os, traceback, json, time
    import shutil
    from pathlib import Path
    from videotrans.util import tools

    import torch
    # Avoid hard-coding a single CPU thread on large servers.
    # Tokenization/audio post-processing still use CPU and can bottleneck GPU.
    try:
        env_threads = os.environ.get("PYVIDEOTRANS_QWEN_TTS_TORCH_THREADS", "").strip()
        if env_threads:
            torch_threads = max(1, int(env_threads))
        else:
            cpu_count = os.cpu_count() or 4
            # Keep a conservative default to avoid oversubscription.
            torch_threads = min(8, max(2, cpu_count // 2)) if is_cuda else min(8, max(1, cpu_count // 2))
        torch.set_num_threads(torch_threads)
    except Exception:
        torch_threads = 1
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
    import soundfile as sf

    
    CUSTOM_VOICE= {"Vivian", "Serena", "Uncle_fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_anna", "Sohee"}

    
    queue_tts=json.loads(Path(queue_tts_file).read_text(encoding='utf-8'))
    # Clone policy: always clone using whatever reference is available.
    # Cap reference audio at 60 s — longer refs are trimmed automatically.
    MAX_REF_MS = int(os.environ.get("PYVIDEOTRANS_QWEN_MAX_REF_MS", "60000") or "60000")
    
    # Fallback: allow forcing CUDA for Qwen-TTS even when upstream is_cuda is False.
    force_qwen_cuda = str(os.environ.get("PYVIDEOTRANS_FORCE_QWEN_TTS_CUDA", "1")).lower() in ("1", "true", "yes", "on")
    if not is_cuda and force_qwen_cuda and torch.cuda.is_available():
        is_cuda = True

    atten=None
    if is_cuda:
        device_map = f'cuda:{device_index}'
        dtype=torch.float16
        try:
            import flash_attn
        except ImportError:
            pass
        else:
            atten='flash_attention_2'
    else:
        device_map = 'cpu'
        dtype=torch.float32
    
    BASE_OBJ=None
    CUSTOM_OBJ=None
    

    all_roles={ r.get('role') for r in queue_tts}
    if all_roles & CUSTOM_VOICE:
        # 存在自定义音色
        # Use cached model if available to avoid re-downloading on duplicate initialization
        CUSTOM_OBJ=_get_or_load_qwen3_model(
            model_key="custom",
            model_path=f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-CustomVoice",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten
        )
    if "clone" in all_roles or all_roles-CUSTOM_VOICE:
        # 存在克隆音色
        # Use cached model if available to avoid re-downloading on duplicate initialization
        BASE_OBJ=_get_or_load_qwen3_model(
            model_key="base",
            model_path=f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-Base",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten
        )

    try:
        batch_lines = _resolve_qwen_tts_batch_lines(is_cuda=is_cuda)
        if is_cuda and torch.cuda.is_available():
            try:
                alloc_gb = torch.cuda.memory_allocated(device_index) / float(1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(device_index) / float(1024 ** 3)
            except Exception:
                alloc_gb = reserved_gb = 0.0
            _write_log(
                logs_file,
                json.dumps({
                    "type": "logs",
                    "text": f"[qwen3tts] device=cuda:{device_index} dtype={dtype} batch_lines={batch_lines} torch_threads={torch_threads} alloc_gb={alloc_gb:.2f} reserved_gb={reserved_gb:.2f}"
                })
            )
        else:
            _write_log(
                logs_file,
                json.dumps({
                    "type": "logs",
                    "text": f"[qwen3tts] device=cpu dtype={dtype} batch_lines={batch_lines} torch_threads={torch_threads}"
                })
            )

        _write_log(
            logs_file,
            json.dumps({
                "type": "logs",
                "text": f"[qwen3tts] clone_ref_policy=always_clone max_ref_ms={MAX_REF_MS}"
            })
        )

        # Reuse clone prompts by reference tuple to avoid recomputing features each sentence.
        clone_prompt_cache = {}
        ref_trim_cache = {}

        _len=len(queue_tts)
        requested_count = sum(1 for it in queue_tts if it.get('text'))
        generated_count = 0
        missing_ref_count = 0
        min_success_ratio = float(os.environ.get("PYVIDEOTRANS_QWEN_MIN_SUCCESS_RATIO", "0.51") or "0.51")
        min_success_ratio = max(0.0, min(1.0, min_success_ratio))
        # Build valid job list first so we can run batched generation safely.
        valid_jobs = []
        for i, it in enumerate(queue_tts):
            text = it.get('text')
            if not text:
                continue
            role = it.get('role')
            filename = it.get('filename', '') + "-qwen3tts.wav"
            _write_log(logs_file, json.dumps({"type": "logs", "text": f'{i+1}/{_len} {role}'}))

            if role in CUSTOM_VOICE and CUSTOM_OBJ:
                valid_jobs.append({
                    "kind": "custom",
                    "role": role,
                    "text": text,
                    "filename": filename,
                })
                continue

            if not BASE_OBJ:
                continue
            if role == 'clone':
                wavfile = it.get('ref_wav', '')
                ref_text = it.get('ref_text', '')
            else:
                wavfile = f'{ROOT_DIR}/f5-tts/{role}'
                ref_text = roledict.get(role) if roledict else None

            if not wavfile or not Path(wavfile).is_file():
                missing_ref_count += 1
                _write_log(logs_file, json.dumps({
                    "type": "logs",
                    "text": f"[qwen3tts] ref_wav missing, skipping line {i+1}: role={role} wav={wavfile}"
                }))
                continue

            if role == 'clone':
                # Cap reference to MAX_REF_MS; use all available audio otherwise.
                _orig_wav = wavfile
                effective_wav = ref_trim_cache.get(_orig_wav)
                if effective_wav is None:
                    effective_wav = _prepare_ref_wav(_orig_wav, MAX_REF_MS)
                    ref_trim_cache[_orig_wav] = effective_wav
                    try:
                        ref_ms = int(tools.get_audio_time(effective_wav) or 0)
                    except Exception:
                        ref_ms = 0
                    if ref_ms == 0:
                        try:
                            _fsz = Path(effective_wav).stat().st_size if Path(effective_wav).exists() else -1
                        except Exception:
                            _fsz = -1
                        _write_log(logs_file, json.dumps({
                            "type": "logs",
                            "text": f"[qwen3tts] ref_ms=0 diagnostic: wav={effective_wav} file_size={_fsz}"
                        }))
                    else:
                        _trimmed = "(trimmed)" if effective_wav != _orig_wav else ""
                        _write_log(logs_file, json.dumps({
                            "type": "logs",
                            "text": f"[qwen3tts] clone ref: role={role} ref_ms={ref_ms} {_trimmed} wav={effective_wav}"
                        }))
                wavfile = effective_wav

            with_emotion = it.get('with_emotion', False)
            temp = 0.9 if with_emotion else 0.7
            top_p = 0.92 if with_emotion else 0.85

            valid_jobs.append({
                "kind": "clone",
                "role": role,
                "text": text,
                "filename": filename,
                "wavfile": wavfile,
                "ref_text": ref_text,
                "with_emotion": with_emotion,
                "temperature": temp,
                "top_p": top_p,
            })

        # Group jobs by compatible generation parameters so each batch is safe and deterministic.
        grouped = {}
        for job in valid_jobs:
            if job["kind"] == "custom":
                group_key = ("custom", job["role"])
            else:
                group_key = (
                    "clone",
                    job["wavfile"],
                    job.get("ref_text") or "",
                    bool(job["with_emotion"]),
                )
            grouped.setdefault(group_key, []).append(job)

        total_jobs = len(valid_jobs)
        total_chunks = 0
        for jobs in grouped.values():
            total_chunks += (len(jobs) + batch_lines - 1) // batch_lines

        _write_log(logs_file, json.dumps({
            "type": "logs",
            "text": f"[qwen3tts] generation_start jobs={total_jobs} groups={len(grouped)} chunks={total_chunks}"
        }))

        chunk_idx = 0
        for group_key, jobs in grouped.items():
            for chunk in _batched(jobs, batch_lines):
                chunk_idx += 1
                chunk_started_at = time.time()
                _write_log(logs_file, json.dumps({
                    "type": "logs",
                    "text": f"[qwen3tts] batch_start {chunk_idx}/{total_chunks} size={len(chunk)} mode={group_key[0]}"
                }))
                try:
                    if group_key[0] == "custom":
                        texts = [j["text"] for j in chunk]
                        langs = [language] * len(texts)
                        wavs, sr = CUSTOM_OBJ.generate_custom_voice(
                            text=texts,
                            language=langs,
                            speaker=chunk[0]["role"],
                            instruct=prompt
                        )
                    else:
                        texts = [j["text"] for j in chunk]
                        langs = [language] * len(texts)
                        ref_text = chunk[0].get("ref_text")
                        wavfile = chunk[0]["wavfile"]
                        prompt_key = (wavfile, ref_text or "", bool(chunk[0]["with_emotion"]))
                        voice_clone_prompt = clone_prompt_cache.get(prompt_key)
                        if voice_clone_prompt is None:
                            voice_clone_prompt = BASE_OBJ.create_voice_clone_prompt(
                                ref_audio=wavfile,
                                ref_text=ref_text or None,
                                x_vector_only_mode=not bool(ref_text)
                            )
                            clone_prompt_cache[prompt_key] = voice_clone_prompt

                        wavs, sr = BASE_OBJ.generate_voice_clone(
                            text=texts,
                            language=langs,
                            voice_clone_prompt=voice_clone_prompt,
                            temperature=chunk[0]["temperature"],
                            top_p=chunk[0]["top_p"]
                        )

                    for j, wav in zip(chunk, wavs):
                        sf.write(j["filename"], wav, sr)
                        generated_count += 1

                    _write_log(logs_file, json.dumps({
                        "type": "logs",
                        "text": f"[qwen3tts] batch_done {chunk_idx}/{total_chunks} elapsed_s={time.time() - chunk_started_at:.1f} generated={generated_count}/{total_jobs}"
                    }))

                except Exception as batch_exc:
                    _write_log(logs_file, json.dumps({
                        "type": "logs",
                        "text": f"[qwen3tts] batch fallback to single: {batch_exc}"
                    }))

                    # Safety fallback: if a batch fails, process each line independently.
                    for j in chunk:
                        try:
                            if j["kind"] == "custom":
                                wavs, sr = CUSTOM_OBJ.generate_custom_voice(
                                    text=j["text"],
                                    language=language,
                                    speaker=j["role"],
                                    instruct=prompt
                                )
                            else:
                                kw = {
                                    "text": j["text"],
                                    "language": language,
                                    "ref_audio": j["wavfile"],
                                    "temperature": j["temperature"],
                                    "top_p": j["top_p"],
                                }
                                if j.get("ref_text"):
                                    kw["ref_text"] = j["ref_text"]
                                else:
                                    kw["x_vector_only_mode"] = True
                                wavs, sr = BASE_OBJ.generate_voice_clone(**kw)

                            sf.write(j["filename"], wavs[0], sr)
                            generated_count += 1
                        except Exception as single_exc:
                            _write_log(logs_file, json.dumps({
                                "type": "logs",
                                "text": f"[qwen3tts] single line skipped: role={j.get('role')} err={single_exc}"
                            }))

        failed_count = max(0, requested_count - generated_count)
        success_ratio = (generated_count / requested_count) if requested_count > 0 else 0.0
        _write_log(logs_file, json.dumps({
            "type": "logs",
            "text": (
                f"[qwen3tts] generation_summary requested={requested_count} "
                f"generated={generated_count} failed={failed_count} "
                f"missing_ref={missing_ref_count} success_ratio={success_ratio:.3f}"
            )
        }))

        # Root guardrail: do not treat run as success when failed lines are majority.
        if requested_count > 0 and success_ratio < min_success_ratio:
            return False, (
                "Qwen3-TTS aborted: success ratio below threshold "
                f"(generated={generated_count}, failed={failed_count}, "
                f"ratio={success_ratio:.3f}, min={min_success_ratio:.3f})."
            )

        if generated_count < 1:
            if missing_ref_count > 0:
                return False, f"No TTS audio generated: {missing_ref_count} line(s) had missing reference audio."
            return False, "No TTS audio generated."
        return True,None
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f'Qwen3-TTS 配音失败:{msg}', exc_info=True)
        return False, msg
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if CUSTOM_OBJ:
                del CUSTOM_OBJ
            if BASE_OBJ:
                del BASE_OBJ
            import gc
            gc.collect()
        except Exception:
            pass
