import os
import time
import traceback
import json
from pathlib import Path
from typing import Any

import requests

from videotrans.configure.config import logger


def _resolve_speech_models() -> list[str]:
    allowed = {"universal-3-pro", "universal-2"}
    raw = str(
        os.environ.get("ASSEMBLYAI_SPEECH_MODELS")
        or os.environ.get("ASSEMBLYAI_SPEECH_MODEL")
        or ""
    ).strip()
    if not raw:
        return ["universal-3-pro", "universal-2"]

    models: list[str] = []
    for item in raw.split(","):
        m = item.strip().lower()
        if m in allowed and m not in models:
            models.append(m)

    if not models:
        return ["universal-3-pro", "universal-2"]
    return models


def _normalize_ms(value: Any) -> int:
    try:
        v = float(value)
    except Exception:
        return -1
    if v < 0:
        return -1
    return int(round(v))


def _normalize_speaker_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    return raw


def _build_diarizations_from_utterances(utterances: list[dict[str, Any]]) -> list[list[Any]]:
    """Convert AssemblyAI utterances into [[start_ms,end_ms],speaker_id] format."""
    diarizations: list[list[Any]] = []

    for item in utterances:
        if not isinstance(item, dict):
            continue
        start_ms = _normalize_ms(item.get("start"))
        end_ms = _normalize_ms(item.get("end"))
        if start_ms < 0 or end_ms <= start_ms:
            continue

        raw_spk = _normalize_speaker_label(item.get("speaker"))
        diarizations.append([[start_ms, end_ms], raw_spk])

    return diarizations


def _assign_subtitles_to_speakers(subtitles: list[list[int]], diarizations: list[list[Any]]) -> list[str]:
    output: list[str] = []
    fallback = "unknown"

    for sub in subtitles:
        if len(sub) != 2 or sub[0] >= sub[1]:
            output.append(fallback)
            continue

        s_start, s_end = int(sub[0]), int(sub[1])
        s_duration = s_end - s_start
        if s_duration <= 0:
            output.append(fallback)
            continue

        overlaps: dict[str, int] = {}
        for dia in diarizations:
            if len(dia) != 2 or len(dia[0]) != 2:
                continue
            d_start, d_end = int(dia[0][0]), int(dia[0][1])
            if d_end <= d_start:
                continue
            speaker = str(dia[1]).strip() or fallback

            overlap_start = max(s_start, d_start)
            overlap_end = min(s_end, d_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                overlaps[speaker] = overlaps.get(speaker, 0) + overlap

        if not overlaps:
            output.append(fallback)
            continue

        num_unique_speakers = len(overlaps)
        max_overlap = max(overlaps.values())
        best_speaker = max(overlaps, key=overlaps.get)

        if num_unique_speakers > 1:
            output.append(best_speaker)
        elif max_overlap > 0.2 * s_duration:
            output.append(best_speaker)
        else:
            output.append(fallback)

    return output


def _write_json_file(path: str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text_file(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(text or ""), encoding="utf-8")


def _ms_to_seconds(ms: Any) -> float:
    try:
        val = float(ms)
    except Exception:
        return 0.0
    if val < 0:
        return 0.0
    return val / 1000.0


def _build_timestamps_payload_from_utterances(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    segments = []
    for row in utterances or []:
        if not isinstance(row, dict):
            continue
        start_ms = _normalize_ms(row.get("start"))
        end_ms = _normalize_ms(row.get("end"))
        if start_ms < 0 or end_ms <= start_ms:
            continue
        segments.append(
            {
                "start": _ms_to_seconds(start_ms),
                "end": _ms_to_seconds(end_ms),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text_es": str(row.get("text") or "").strip(),
                "speaker": _normalize_speaker_label(row.get("speaker")),
                "speaker_id": _normalize_speaker_label(row.get("speaker")),
            }
        )
    return {"segments": segments}


def _fetch_transcript_export(
    *,
    base_url: str,
    transcript_id: str,
    api_key: str,
    kind: str,
    timeout: float,
) -> tuple[bool, Any]:
    url = f"{base_url}/transcript/{transcript_id}/{kind}"
    resp = requests.get(url, headers={"authorization": api_key}, timeout=timeout)
    if resp.status_code >= 400:
        return False, f"HTTP {resp.status_code}"
    content_type = str(resp.headers.get("content-type") or "").lower()
    if "json" in content_type:
        try:
            return True, resp.json()
        except Exception:
            return False, "invalid-json"
    return True, resp.text


def _persist_assemblyai_exports(
    *,
    base_url: str,
    transcript_id: str,
    api_key: str,
    raw_body: dict[str, Any],
    cache_folder: str | None,
    target_dir: str | None,
    request_timeout: float,
) -> None:
    output_roots = []
    if cache_folder:
        output_roots.append(str(cache_folder))
    if target_dir and str(target_dir) not in output_roots:
        output_roots.append(str(target_dir))

    utterances = raw_body.get("utterances") or []
    timestamps_payload = _build_timestamps_payload_from_utterances(utterances)

    export_map: dict[str, tuple[bool, Any]] = {
        "sentences.json": _fetch_transcript_export(
            base_url=base_url,
            transcript_id=transcript_id,
            api_key=api_key,
            kind="sentences",
            timeout=request_timeout,
        ),
        "paragraphs.json": _fetch_transcript_export(
            base_url=base_url,
            transcript_id=transcript_id,
            api_key=api_key,
            kind="paragraphs",
            timeout=request_timeout,
        ),
        "transcript.srt": _fetch_transcript_export(
            base_url=base_url,
            transcript_id=transcript_id,
            api_key=api_key,
            kind="srt",
            timeout=request_timeout,
        ),
        "transcript.vtt": _fetch_transcript_export(
            base_url=base_url,
            transcript_id=transcript_id,
            api_key=api_key,
            kind="vtt",
            timeout=request_timeout,
        ),
    }

    for root in output_roots:
        _write_json_file(os.path.join(root, "timestamps.json"), timestamps_payload)
        _write_text_file(os.path.join(root, "transcript.txt"), str(raw_body.get("text") or ""))

        for filename, (ok, payload) in export_map.items():
            if not ok:
                logger.warning(f"[AAI] export not available {filename}: {payload}")
                continue
            abs_path = os.path.join(root, filename)
            if filename.endswith(".json"):
                _write_json_file(abs_path, payload)
            else:
                _write_text_file(abs_path, str(payload or ""))


def _persist_assemblyai_artifacts(
    *,
    raw_body: dict[str, Any],
    payload: dict[str, Any],
    diarizations: list[list[Any]],
    assigned: list[str],
    subtitles: list[list[int]],
    transcript_id: str,
    elapsed_seconds: int,
    cache_folder: str | None,
    target_dir: str | None,
) -> None:
    utterances = raw_body.get("utterances") or []
    derived = {
        "transcript_id": transcript_id,
        "status": raw_body.get("status"),
        "language_code": raw_body.get("language_code"),
        "audio_duration": raw_body.get("audio_duration"),
        "utterances_count": len(utterances),
        "raw_speakers": sorted(
            {str(u.get("speaker")) for u in utterances if isinstance(u, dict) and u.get("speaker") is not None}
        ),
        "normalized_speakers": sorted(
            {str(d[1]) for d in diarizations if isinstance(d, list) and len(d) == 2}
        ),
        "subtitles_count": len(subtitles),
        "assigned_count": len(assigned),
        "assigned_unique": sorted(set(assigned)),
        "first_assigned": assigned[:20],
        "elapsed_seconds": elapsed_seconds,
        "request_payload": payload,
    }

    output_roots = []
    if cache_folder:
        output_roots.append(str(cache_folder))
    if target_dir and str(target_dir) not in output_roots:
        output_roots.append(str(target_dir))

    for root in output_roots:
        _write_json_file(os.path.join(root, "phase1_assemblyai_raw.json"), raw_body)
        _write_json_file(os.path.join(root, "phase1_assemblyai_derived.json"), derived)


def _build_headers(api_key: str) -> dict[str, str]:
    return {
        "authorization": api_key,
        "content-type": "application/json",
    }


def assemblyai_speakers(
    *,
    input_file,
    subtitles,
    num_speakers=-1,
    is_cuda=False,
    device_index=0,
    logs_file=None,
    api_key=None,
    cache_folder=None,
    target_dir=None,
    **_ignored_kwargs,
):
    """Diarize with AssemblyAI and return a speaker label list aligned to subtitles."""
    del is_cuda, device_index, logs_file, _ignored_kwargs

    key = str(api_key or os.environ.get("ASSEMBLY_AI_KEY") or os.environ.get("ASSEMBLYAI_API_KEY") or "").strip()
    if not key:
        return False, "AssemblyAI diarization requires ASSEMBLY_AI_KEY in environment."

    if not input_file or not os.path.isfile(input_file):
        return False, f"AssemblyAI diarization input audio not found: {input_file}"

    if not subtitles or not isinstance(subtitles, list):
        return False, "AssemblyAI diarization requires subtitle time ranges for alignment."

    base_url = os.environ.get("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com/v2").rstrip("/")
    timeout_seconds = int(os.environ.get("ASSEMBLYAI_TIMEOUT_SECONDS", "1800"))
    poll_interval = float(os.environ.get("ASSEMBLYAI_POLL_SECONDS", "2.0"))
    request_timeout = float(os.environ.get("ASSEMBLYAI_REQUEST_TIMEOUT", "60"))

    try:
        started = time.time()
        logger.info("[AAI] Uploading audio for diarization")
        with open(input_file, "rb") as f:
            upload_resp = requests.post(
                f"{base_url}/upload",
                headers={"authorization": key},
                data=f,
                timeout=request_timeout,
            )
        upload_resp.raise_for_status()
        upload_data = upload_resp.json()
        audio_url = str(upload_data.get("upload_url") or "").strip()
        if not audio_url:
            return False, "AssemblyAI upload failed: missing upload_url in response."

        payload = {
            "audio_url": audio_url,
            "speech_models": _resolve_speech_models(),
            "speaker_labels": True,
            "language_detection": True,
            "format_text": True,
        }
        if isinstance(num_speakers, int) and num_speakers > 1:
            payload["speakers_expected"] = int(num_speakers)

        logger.info("[AAI] Creating transcript with speaker labels")
        create_resp = requests.post(
            f"{base_url}/transcript",
            headers=_build_headers(key),
            json=payload,
            timeout=request_timeout,
        )
        create_resp.raise_for_status()
        transcript_id = str((create_resp.json() or {}).get("id") or "").strip()
        if not transcript_id:
            return False, "AssemblyAI transcript creation failed: missing transcript id."

        status_url = f"{base_url}/transcript/{transcript_id}"
        terminal_error = None
        while True:
            elapsed = time.time() - started
            if elapsed > timeout_seconds:
                return False, f"AssemblyAI diarization timeout after {int(elapsed)}s."

            status_resp = requests.get(
                status_url,
                headers={"authorization": key},
                timeout=request_timeout,
            )
            status_resp.raise_for_status()
            body = status_resp.json() or {}
            status = str(body.get("status") or "").strip().lower()

            if status == "completed":
                utterances = body.get("utterances") or []
                if not utterances:
                    return False, "AssemblyAI completed without utterances/diarization data."

                diarizations = _build_diarizations_from_utterances(utterances)
                if not diarizations:
                    return False, "AssemblyAI diarization utterances could not be normalized."

                output = _assign_subtitles_to_speakers(subtitles, diarizations)
                if not output:
                    return False, "AssemblyAI diarization produced an empty speaker map."

                _persist_assemblyai_artifacts(
                    raw_body=body,
                    payload=payload,
                    diarizations=diarizations,
                    assigned=output,
                    subtitles=subtitles,
                    transcript_id=transcript_id,
                    elapsed_seconds=int(time.time() - started),
                    cache_folder=cache_folder,
                    target_dir=target_dir,
                )

                _persist_assemblyai_exports(
                    base_url=base_url,
                    transcript_id=transcript_id,
                    api_key=key,
                    raw_body=body,
                    cache_folder=cache_folder,
                    target_dir=target_dir,
                    request_timeout=request_timeout,
                )

                logger.info(
                    f"[AAI] Diarization completed: speakers={len(set(output))} "
                    f"segments={len(output)} elapsed={int(time.time() - started)}s"
                )
                return output, None

            if status == "error":
                terminal_error = str(body.get("error") or "AssemblyAI transcript status=error")
                break

            time.sleep(max(0.5, poll_interval))

        return False, f"AssemblyAI diarization failed: {terminal_error}"

    except Exception:
        msg = traceback.format_exc()
        logger.exception(f"AssemblyAI diarization failed: {msg}", exc_info=True)
        return False, msg
