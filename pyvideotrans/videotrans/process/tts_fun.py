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
    details = "mode=cpu-default"
    env_val = os.environ.get("PYVIDEOTRANS_QWEN_TTS_BATCH_LINES")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n, f"mode=env-fixed batch_lines={n}"
        except Exception:
            pass

    if not is_cuda or not torch.cuda.is_available():
        return 2, details

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / float(1024 ** 3)
        total_gib = total_bytes / float(1024 ** 3)
    except Exception:
        free_gib = 0.0
        total_gib = 0.0

    target_vram_env = (
        os.environ.get("PYVIDEOTRANS_QWEN_TTS_TARGET_VRAM_GB", "").strip()
        or os.environ.get("PYVIDEOTRANS_GPU_TARGET_VRAM_GB", "").strip()
    )
    target_vram_gb = 0.0
    if target_vram_env:
        try:
            target_vram_gb = float(target_vram_env)
        except Exception:
            target_vram_gb = 0.0

    # Dynamic default: use ~92% of currently free VRAM for TTS work if target is unset/<=0.
    # Mapping keeps historical calibration (24 GiB ~= 12 lines) but scales up for larger GPUs.
    if target_vram_gb <= 0:
        effective_target_gb = max(1.0, free_gib * 0.92)
        target_mode = "auto"
    else:
        effective_target_gb = min(target_vram_gb, max(1.0, free_gib * 0.95))
        target_mode = "env-target"

    lines_per_gib = 12.0 / 24.0
    batch_lines = int(round(effective_target_gb * lines_per_gib))
    batch_lines = max(2, min(batch_lines, 128))

    details = (
        f"mode={target_mode} free_gb={free_gib:.2f} total_gb={total_gib:.2f} "
        f"target_gb={effective_target_gb:.2f} lines_per_gib={lines_per_gib:.3f}"
    )
    return batch_lines, details


def _batched(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def _write_log(file, msg):
    from pathlib import Path
    try:
        Path(file).write_text(msg, encoding='utf-8')
    except Exception as e:
        logger.exception(f'写入新进程日志时出错', exc_info=True)


def _emit_progress(logs_file, text: str):
    """Emit progress both to inter-process log file and stdout for CLI/UI visibility."""
    payload = json.dumps({"type": "progress", "text": text}, ensure_ascii=False)
    _write_log(logs_file, payload)
    try:
        print(text, flush=True)
    except Exception:
        pass


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

    # By default, clone jobs are grouped by shared reference audio to maximize GPU batching.
    # Set PYVIDEOTRANS_QWEN_CLONE_GROUP_BY_REF_TEXT=1 to preserve strict ref_text isolation.
    clone_group_by_ref_text = str(
        os.environ.get("PYVIDEOTRANS_QWEN_CLONE_GROUP_BY_REF_TEXT", "0")
    ).strip().lower() in ("1", "true", "yes", "on")
    
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
        batch_lines, batch_policy = _resolve_qwen_tts_batch_lines(is_cuda=is_cuda)
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
                    "text": (
                        f"[qwen3tts] device=cuda:{device_index} dtype={dtype} batch_lines={batch_lines} "
                        f"torch_threads={torch_threads} alloc_gb={alloc_gb:.2f} reserved_gb={reserved_gb:.2f} "
                        f"batch_policy=({batch_policy})"
                    )
                })
            )
        else:
            _write_log(
                logs_file,
                json.dumps({
                    "type": "logs",
                    "text": f"[qwen3tts] device=cpu dtype={dtype} batch_lines={batch_lines} torch_threads={torch_threads} batch_policy=({batch_policy})"
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
        min_success_ratio = 1.0
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
                raise RuntimeError("Qwen3-TTS clone engine is not initialized.")
            if role == 'clone':
                wavfile = it.get('ref_wav', '')
                ref_text = it.get('ref_text', '')
            else:
                wavfile = f'{ROOT_DIR}/f5-tts/{role}'
                ref_text = roledict.get(role) if roledict else None

            if not wavfile or not Path(wavfile).is_file():
                missing_ref_count += 1
                raise RuntimeError(
                    f"[qwen3tts] ref_wav missing for line {i+1}: role={role} wav={wavfile}"
                )

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
                # For clone voice, avoid splitting batches by per-line subtitle ref_text.
                group_ref_text = (
                    job.get("ref_text") or ""
                ) if (clone_group_by_ref_text or job.get("role") != "clone") else ""
                group_key = (
                    "clone",
                    job["wavfile"],
                    group_ref_text,
                    bool(job["with_emotion"]),
                )
            grouped.setdefault(group_key, []).append(job)

        total_jobs = len(valid_jobs)
        total_chunks = 0
        for jobs in grouped.values():
            total_chunks += (len(jobs) + batch_lines - 1) // batch_lines

        _emit_progress(
            logs_file,
            (
                f"[qwen3tts] generation_start jobs={total_jobs} groups={len(grouped)} "
                f"chunks={total_chunks} clone_group_by_ref_text={clone_group_by_ref_text}"
            ),
        )

        # ---- GPU PREFETCH: pre-build all clone prompts in parallel threads ----
        # create_voice_clone_prompt() is CPU-bound (wav feature extraction).
        # Computing all prompts upfront in a thread pool means the generation loop
        # never stalls on CPU work at the start of a new group — GPU stays fed.
        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed
        if BASE_OBJ:
            def _build_clone_prompt_pre(group_key, jobs):
                if group_key[0] != "clone":
                    return None, None
                j0 = jobs[0]
                wavfile_pre = j0["wavfile"]
                ref_text_pre = j0.get("ref_text") if (clone_group_by_ref_text or j0.get("role") != "clone") else ""
                pk = (wavfile_pre, ref_text_pre or "", bool(j0["with_emotion"]))
                if pk in clone_prompt_cache:
                    return pk, None  # already cached
                try:
                    vcp = BASE_OBJ.create_voice_clone_prompt(
                        ref_audio=wavfile_pre,
                        ref_text=ref_text_pre or None,
                        x_vector_only_mode=not bool(ref_text_pre)
                    )
                    return pk, vcp
                except Exception as _pre_exc:
                    _write_log(logs_file, json.dumps({
                        "type": "logs",
                        "text": f"[qwen3tts] prefetch_prompt failed (will retry inline): {_pre_exc}"
                    }))
                    return None, None

            _clone_groups_pre = [(gk, jlist) for gk, jlist in grouped.items() if gk[0] != "custom"]
            if _clone_groups_pre:
                _write_log(logs_file, json.dumps({
                    "type": "logs",
                    "text": f"[qwen3tts] prefetch_clone_prompts start groups={len(_clone_groups_pre)}"
                }))
                with _TPE(max_workers=min(4, len(_clone_groups_pre)), thread_name_prefix="qwen_prompt_pre") as _pre_exec:
                    _pre_futures = {
                        _pre_exec.submit(_build_clone_prompt_pre, gk, jlist): gk
                        for gk, jlist in _clone_groups_pre
                    }
                    for _pfut in _as_completed(_pre_futures):
                        try:
                            _pk, _vcp = _pfut.result()
                            if _pk is not None and _vcp is not None:
                                clone_prompt_cache[_pk] = _vcp
                        except Exception:
                            pass
                _write_log(logs_file, json.dumps({
                    "type": "logs",
                    "text": f"[qwen3tts] prefetch_clone_prompts done cache_size={len(clone_prompt_cache)}"
                }))
        # ---- end prefetch ----

        # Background ThreadPoolExecutor for sf.write — disk I/O runs while GPU generates next batch.
        _sfwrite_exec = _TPE(max_workers=4, thread_name_prefix="sfwrite")
        _sfwrite_futures = []

        chunk_idx = 0
        gen_started_at = time.time()
        for group_key, jobs in grouped.items():
            for chunk in _batched(jobs, batch_lines):
                chunk_idx += 1
                chunk_started_at = time.time()
                elapsed = max(time.time() - gen_started_at, 0.001)
                avg_chunk_sec = elapsed / max(chunk_idx, 1)
                remaining_chunks = max(total_chunks - chunk_idx, 0)
                eta_sec = max(int(avg_chunk_sec * remaining_chunks), 0)
                eta_min, eta_rem_sec = divmod(eta_sec, 60)
                pct = (generated_count / max(total_jobs, 1)) * 100.0
                _emit_progress(
                    logs_file,
                    (
                        f"[qwen3tts] batch_start {chunk_idx}/{total_chunks} size={len(chunk)} mode={group_key[0]} "
                        f"progress={pct:.1f}% generated={generated_count}/{total_jobs} "
                        f"remaining_chunks={remaining_chunks} eta={eta_min:02d}:{eta_rem_sec:02d}"
                    ),
                )
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
                        if clone_group_by_ref_text or chunk[0].get("role") != "clone":
                            ref_text = chunk[0].get("ref_text")
                        else:
                            ref_text = ""
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
                        # Copy array so GPU buffer can be freed before the write completes.
                        _wav_copy = wav.copy() if hasattr(wav, 'copy') else wav
                        _sfwrite_futures.append(_sfwrite_exec.submit(sf.write, j["filename"], _wav_copy, sr))
                        generated_count += 1

                    elapsed_done = max(time.time() - gen_started_at, 0.001)
                    seg_per_sec = generated_count / elapsed_done
                    remaining_segments = max(total_jobs - generated_count, 0)
                    eta_done_sec = max(int(remaining_segments / max(seg_per_sec, 0.0001)), 0)
                    eta_done_min, eta_done_rem_sec = divmod(eta_done_sec, 60)
                    eta_finish = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_done_sec))

                    pct_done = (generated_count / max(total_jobs, 1)) * 100.0
                    _emit_progress(
                        logs_file,
                        (
                            f"[qwen3tts] batch_done {chunk_idx}/{total_chunks} elapsed_s={time.time() - chunk_started_at:.1f} "
                            f"progress={pct_done:.1f}% generated={generated_count}/{total_jobs} "
                            f"remaining_segments={remaining_segments} speed={seg_per_sec:.2f} seg/s "
                            f"eta={eta_done_min:02d}:{eta_done_rem_sec:02d} eta_finish={eta_finish}"
                        ),
                    )

                except Exception as batch_exc:
                    raise RuntimeError(
                        f"[qwen3tts] batch generation failed at chunk {chunk_idx}/{total_chunks}: {batch_exc}"
                    ) from batch_exc

        # Drain background write futures; must complete before success-ratio check.
        for _wfut in _sfwrite_futures:
            try:
                _wfut.result()
            except Exception as _wfut_exc:
                _write_log(logs_file, json.dumps({
                    "type": "logs",
                    "text": f"[qwen3tts] background sf.write error: {_wfut_exc}"
                }))
        _sfwrite_exec.shutdown(wait=False)

        failed_count = max(0, requested_count - generated_count)
        success_ratio = (generated_count / requested_count) if requested_count > 0 else 0.0
        _emit_progress(
            logs_file,
            (
                f"[qwen3tts] generation_summary requested={requested_count} "
                f"generated={generated_count} failed={failed_count} "
                f"missing_ref={missing_ref_count} success_ratio={success_ratio:.3f}"
            ),
        )

        # Strict guardrail: all requested lines must be generated.
        if requested_count > 0 and success_ratio < min_success_ratio:
            return False, (
                "Qwen3-TTS aborted: success ratio below strict threshold "
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
