import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import requests

from videotrans.configure._except import StopRetry
from videotrans.configure.config import logger
from videotrans.recognition._base import BaseRecogn
from videotrans.util import tools


def _default_prompt() -> str:
    """Verbatim + entity-accuracy prompt for Universal-3 Pro dubbing pipelines."""
    return (
        "Required: Preserve the original language(s) and script as spoken, "
        "including code-switching and mixed-language phrases.\n\n"
        "Mandatory: Preserve linguistic speech patterns including disfluencies, "
        "filler words (um, uh, mhm, hmm), hesitations, repetitions, stutters, "
        "false starts, and colloquialisms exactly as spoken.\n\n"
        "Always: Transcribe speech with your best guess based on context in "
        "all possible scenarios where speech is present in the audio.\n\n"
        "Accuracy: Accurately transcribe proper nouns, person names, place names, "
        "organization names, brands, and technical terminology exactly as spoken, "
        "preserving correct spelling and capitalization.\n\n"
        "Formatting: Use standard punctuation. Capitalize proper nouns and the "
        "first word of each sentence. Write numbers as numerals for measurements, "
        "statistics, and counts. Format percentages, dates, and currencies in "
        "standard written form."
    )


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


def _to_raw_subtitles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AAI rows (utterances or sentences) into pyvideotrans raw subtitle entries.

    Both utterances and sentences expose `start`, `end` (ms) and `text` fields, so
    the same conversion logic works. Sentences yield much finer-grained segments,
    which is critical for accurate TTS dubbing alignment.
    """
    raws: list[dict[str, Any]] = []
    line_no = 0
    for row in rows or []:
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

        line_no += 1
        startraw = tools.ms_to_time_string(ms=start_ms)
        endraw = tools.ms_to_time_string(ms=end_ms)
        raws.append(
            {
                "line": line_no,
                "start_time": start_ms,
                "end_time": end_ms,
                "startraw": startraw,
                "endraw": endraw,
                "time": f"{startraw} --> {endraw}",
                "text": text,
            }
        )
    return raws


def _fetch_sentences(
    *,
    base_url: str,
    transcript_id: str,
    api_key: str,
    timeout: float,
) -> list[dict[str, Any]]:
    """Fetch sentence-level breakdown from AAI for a completed transcript.

    Returns an empty list on failure; the caller should fall back to utterances.
    """
    try:
        resp = requests.get(
            f"{base_url}/transcript/{transcript_id}/sentences",
            headers={"authorization": api_key},
            timeout=timeout,
        )
        if resp.status_code >= 400:
            logger.warning(f"[AAI-ASR] sentences fetch HTTP {resp.status_code}")
            return []
        payload = resp.json() or {}
        sentences = payload.get("sentences") if isinstance(payload, dict) else None
        if isinstance(sentences, list):
            return [s for s in sentences if isinstance(s, dict)]
    except Exception as exc:
        logger.warning(f"[AAI-ASR] sentences fetch error: {exc}")
    return []


def _propagate_speaker_from_utterances(
    sentences: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure each sentence has a speaker label by overlap with utterances.

    The AAI /sentences endpoint already includes a `speaker` field, but for
    safety we fill it from the overlapping utterance when missing.
    """
    if not sentences:
        return sentences
    utt_ranges: list[tuple[int, int, str]] = []
    for u in utterances or []:
        try:
            us = int(float(u.get("start", 0)))
            ue = int(float(u.get("end", 0)))
            sp = str(u.get("speaker") or "").strip()
        except Exception:
            continue
        if ue > us and sp:
            utt_ranges.append((us, ue, sp))

    out: list[dict[str, Any]] = []
    for s in sentences:
        sp = str(s.get("speaker") or "").strip()
        if not sp and utt_ranges:
            try:
                ss = int(float(s.get("start", 0)))
                se = int(float(s.get("end", 0)))
                mid = (ss + se) // 2
            except Exception:
                mid = -1
            for us, ue, usp in utt_ranges:
                if us <= mid <= ue:
                    sp = usp
                    break
        merged = dict(s)
        if sp:
            merged["speaker"] = sp
        out.append(merged)
    return out


def _write_json(path: str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text or "", encoding="utf-8")


def _raws_to_srt(raws: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, row in enumerate(raws or [], start=1):
        start = str(row.get("startraw") or "00:00:00,000")
        end = str(row.get("endraw") or "00:00:00,000")
        text = str(row.get("text") or "").strip()
        blocks.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks).strip() + "\n"


def _resolve_fw_runtime() -> tuple[str, str]:
    device = str(os.environ.get("QDP_FW_DEVICE") or "cuda").strip().lower() or "cuda"
    compute = str(os.environ.get("QDP_FW_COMPUTE_TYPE") or "float16").strip().lower() or "float16"
    return device, compute


def _faster_whisper_sentences(audio_file: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise StopRetry(
            "faster-whisper is required for Phase 1 transcription. Install faster-whisper in runtime env."
        ) from exc

    model_name = str(os.environ.get("QDP_FW_MODEL") or "large-v3").strip() or "large-v3"
    beam_size = int(os.environ.get("QDP_FW_BEAM_SIZE", "5"))
    use_vad = str(os.environ.get("QDP_FW_VAD_FILTER", "1")).strip().lower() not in {"0", "false", "no"}
    gap_ms = int(os.environ.get("QDP_FW_SENTENCE_GAP_MS", "900"))

    device, compute = _resolve_fw_runtime()
    runtime = {"device": device, "compute_type": compute}
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute)
    except Exception:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        runtime = {"device": "cpu", "compute_type": "int8"}

    segments, info = model.transcribe(
        audio_file,
        language="en",
        vad_filter=use_vad,
        word_timestamps=True,
        beam_size=beam_size,
        temperature=0.0,
    )

    words: list[dict[str, Any]] = []
    for seg in segments:
        for w in (seg.words or []):
            if w.start is None or w.end is None:
                continue
            txt = str(w.word or "").strip()
            if not txt:
                continue
            words.append(
                {
                    "text": txt,
                    "start": int(float(w.start) * 1000),
                    "end": int(float(w.end) * 1000),
                    "confidence": float(getattr(w, "probability", 0.0) or 0.0),
                    "speaker": "UNK",
                }
            )

    sentences: list[dict[str, Any]] = []
    buf: list[dict[str, Any]] = []
    for idx, row in enumerate(words):
        buf.append(row)
        token = str(row.get("text") or "")
        has_terminal = token.endswith(".") or token.endswith("!") or token.endswith("?")
        next_gap = 0
        if idx + 1 < len(words):
            next_gap = int(words[idx + 1]["start"]) - int(row["end"])

        if has_terminal or next_gap > gap_ms:
            text = " ".join(str(x.get("text") or "").strip() for x in buf).strip()
            text = re.sub(r"\s+([,.;:!?])", r"\1", text)
            conf = sum(float(x.get("confidence") or 0.0) for x in buf) / max(1, len(buf))
            sentences.append(
                {
                    "text": text,
                    "start": int(buf[0]["start"]),
                    "end": int(buf[-1]["end"]),
                    "words": list(buf),
                    "confidence": conf,
                    "speaker": "UNK",
                }
            )
            buf = []

    if buf:
        text = " ".join(str(x.get("text") or "").strip() for x in buf).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        conf = sum(float(x.get("confidence") or 0.0) for x in buf) / max(1, len(buf))
        sentences.append(
            {
                "text": text,
                "start": int(buf[0]["start"]),
                "end": int(buf[-1]["end"]),
                "words": list(buf),
                "confidence": conf,
                "speaker": "UNK",
            }
        )

    meta = {
        "engine": "faster-whisper",
        "model": model_name,
        "runtime": runtime,
        "language": getattr(info, "language", None),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
    }
    return sentences, meta


def _persist_faster_whisper_exports(
    *,
    raws: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    meta: dict[str, Any],
    cache_folder: str,
    target_dir: str,
) -> None:
    roots = [cache_folder]
    if target_dir and target_dir != cache_folder:
        roots.append(target_dir)

    transcript_text = "\n".join(str(it.get("text") or "").strip() for it in sentences if isinstance(it, dict))
    fw_sentences_payload = {
        "id": None,
        "confidence": None,
        "audio_duration": None,
        "meta": meta,
        "sentences": sentences,
    }
    # Placeholder timestamps; real speaker mapping is produced by diarization step.
    timestamps = {
        "segments": [
            {
                "start": round(int(it.get("start_time", 0)) / 1000.0, 3),
                "end": round(int(it.get("end_time", 0)) / 1000.0, 3),
                "start_ms": int(it.get("start_time", 0)),
                "end_ms": int(it.get("end_time", 0)),
                "text_es": str(it.get("text") or "").strip(),
                "speaker": "UNK",
                "speaker_id": "UNK",
            }
            for it in raws
            if isinstance(it, dict)
        ]
    }

    for root in roots:
        _write_json(os.path.join(root, "sentences.json"), fw_sentences_payload)
        _write_text(os.path.join(root, "transcript.srt"), _raws_to_srt(raws))
        _write_text(os.path.join(root, "transcript.txt"), transcript_text)
        _write_json(os.path.join(root, "timestamps.json"), timestamps)


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

        self._signal(text="Faster-Whisper ASR: transcribing local audio...")
        started = time.time()
        fw_sentences, fw_meta = _faster_whisper_sentences(self.audio_file)
        raws = _to_raw_subtitles(fw_sentences)
        if not raws:
            raise StopRetry("Faster-Whisper returned no usable sentence segments")

        try:
            _persist_faster_whisper_exports(
                raws=raws,
                sentences=fw_sentences,
                meta=fw_meta,
                cache_folder=self.cache_folder,
                target_dir=self.target_dir,
            )
        except Exception as exc:
            logger.warning(f"[FW-ASR] export persist warning: {exc}")

        logger.info(
            "[FW-ASR] completed segments=%s elapsed=%ss model=%s runtime=%s/%s",
            len(raws),
            int(time.time() - started),
            fw_meta.get("model"),
            (fw_meta.get("runtime") or {}).get("device"),
            (fw_meta.get("runtime") or {}).get("compute_type"),
        )

        return raws