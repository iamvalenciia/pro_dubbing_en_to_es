import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import requests

from videotrans.configure._except import StopRetry
from videotrans.configure.config import logger
from videotrans.recognition._base import BaseRecogn
from videotrans.util import tools


def _resolve_models() -> list[str]:
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
    return models or ["universal-3-pro", "universal-2"]


def _speaker_from_utterance(row: dict[str, Any], idx: int) -> str:
    raw = str((row or {}).get("speaker") or "").strip()
    if raw:
        return raw
    return f"SPK_{idx:02d}"


def _to_raw_subtitles(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raws: list[dict[str, Any]] = []
    for idx, row in enumerate(utterances or [], start=1):
        if not isinstance(row, dict):
            continue
        try:
            start_ms = int(float(row.get("start", 0)))
            end_ms = int(float(row.get("end", 0)))
        except Exception:
            continue
        if end_ms <= start_ms:
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        startraw = tools.ms_to_time_string(ms=start_ms)
        endraw = tools.ms_to_time_string(ms=end_ms)
        raws.append(
            {
                "line": idx,
                "start_time": start_ms,
                "end_time": end_ms,
                "startraw": startraw,
                "endraw": endraw,
                "time": f"{startraw} --> {endraw}",
                "text": text,
            }
        )
    return raws


def _write_json(path: str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text or "", encoding="utf-8")


def _persist_exports(
    *,
    base_url: str,
    transcript_id: str,
    api_key: str,
    body: dict[str, Any],
    cache_folder: str,
    target_dir: str,
    timeout: float,
) -> None:
    roots = [cache_folder]
    if target_dir and target_dir != cache_folder:
        roots.append(target_dir)

    utterances = body.get("utterances") or []
    timestamps = {
        "segments": [
            {
                "start": float(int(float(it.get("start", 0))) / 1000.0),
                "end": float(int(float(it.get("end", 0))) / 1000.0),
                "start_ms": int(float(it.get("start", 0))),
                "end_ms": int(float(it.get("end", 0))),
                "text_es": str(it.get("text") or "").strip(),
                "speaker": _speaker_from_utterance(it, i),
                "speaker_id": _speaker_from_utterance(it, i),
            }
            for i, it in enumerate(utterances)
            if isinstance(it, dict)
        ]
    }

    export_kinds = {
        "sentences.json": "sentences",
        "paragraphs.json": "paragraphs",
        "transcript.srt": "srt",
        "transcript.vtt": "vtt",
    }

    export_payloads: dict[str, Any] = {}
    headers = {"authorization": api_key}
    for name, kind in export_kinds.items():
        url = f"{base_url}/transcript/{transcript_id}/{kind}"
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                logger.warning(f"[AAI-ASR] export unavailable {name}: HTTP {resp.status_code}")
                continue
            ctype = str(resp.headers.get("content-type") or "").lower()
            export_payloads[name] = resp.json() if "json" in ctype else (resp.text or "")
        except Exception as exc:
            logger.warning(f"[AAI-ASR] export error {name}: {exc}")

    for root in roots:
        _write_json(os.path.join(root, "timestamps.json"), timestamps)
        _write_text(os.path.join(root, "transcript.txt"), str(body.get("text") or ""))
        for name, payload in export_payloads.items():
            out = os.path.join(root, name)
            if name.endswith(".json"):
                _write_json(out, payload)
            else:
                _write_text(out, str(payload or ""))


@dataclass
class AssemblyAIRecogn(BaseRecogn):
    def __post_init__(self):
        super().__post_init__()

    def _exec(self) -> Union[List[Dict], None]:
        if self._exit():
            return

        api_key = str(
            os.environ.get("ASSEMBLY_AI_KEY")
            or os.environ.get("ASSEMBLYAI_API_KEY")
            or ""
        ).strip()
        if not api_key:
            raise StopRetry("AssemblyAI requires ASSEMBLY_AI_KEY in environment.")

        base_url = os.environ.get("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com/v2").rstrip("/")
        timeout_seconds = int(os.environ.get("ASSEMBLYAI_TIMEOUT_SECONDS", "1800"))
        poll_interval = float(os.environ.get("ASSEMBLYAI_POLL_SECONDS", "2.0"))
        request_timeout = float(os.environ.get("ASSEMBLYAI_REQUEST_TIMEOUT", "60"))

        self._signal(text="AssemblyAI ASR: uploading audio...")
        started = time.time()

        with open(self.audio_file, "rb") as f:
            upload_resp = requests.post(
                f"{base_url}/upload",
                headers={"authorization": api_key},
                data=f,
                timeout=request_timeout,
            )
        upload_resp.raise_for_status()
        upload_url = str((upload_resp.json() or {}).get("upload_url") or "").strip()
        if not upload_url:
            raise StopRetry("AssemblyAI upload failed: missing upload_url")

        payload = {
            "audio_url": upload_url,
            "speech_models": _resolve_models(),
            "speaker_labels": True,
            "language_detection": True,
            "format_text": True,
        }
        if isinstance(self.max_speakers, int) and self.max_speakers > 1:
            payload["speakers_expected"] = int(self.max_speakers)

        self._signal(text="AssemblyAI ASR: transcribing...")
        create_resp = requests.post(
            f"{base_url}/transcript",
            headers={"authorization": api_key, "content-type": "application/json"},
            json=payload,
            timeout=request_timeout,
        )
        create_resp.raise_for_status()
        transcript_id = str((create_resp.json() or {}).get("id") or "").strip()
        if not transcript_id:
            raise StopRetry("AssemblyAI transcript creation failed: missing transcript id")

        status_url = f"{base_url}/transcript/{transcript_id}"
        body: dict[str, Any] = {}
        while True:
            if (time.time() - started) > timeout_seconds:
                raise StopRetry(f"AssemblyAI ASR timeout after {timeout_seconds}s")
            status_resp = requests.get(status_url, headers={"authorization": api_key}, timeout=request_timeout)
            status_resp.raise_for_status()
            body = status_resp.json() or {}
            status = str(body.get("status") or "").strip().lower()
            if status == "completed":
                break
            if status == "error":
                raise StopRetry(f"AssemblyAI ASR failed: {body.get('error') or 'status=error'}")
            time.sleep(max(0.5, poll_interval))

        utterances = body.get("utterances") or []
        raws = _to_raw_subtitles(utterances)
        if not raws:
            raise StopRetry("AssemblyAI ASR returned no usable utterances")

        speakers = [_speaker_from_utterance(it, idx) for idx, it in enumerate(utterances, start=1) if isinstance(it, dict)]
        if speakers:
            Path(f"{self.cache_folder}/speaker.json").write_text(json.dumps(speakers), encoding="utf-8")

        try:
            _persist_exports(
                base_url=base_url,
                transcript_id=transcript_id,
                api_key=api_key,
                body=body,
                cache_folder=self.cache_folder,
                target_dir=self.cache_folder,
                timeout=request_timeout,
            )
        except Exception as exc:
            logger.warning(f"[AAI-ASR] export persist warning: {exc}")

        return raws