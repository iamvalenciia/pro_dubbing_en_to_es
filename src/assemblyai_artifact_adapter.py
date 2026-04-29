import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ASSEMBLYAI_IMPORT_PRIORITY = [
    "timestamps.json",
    "sentences.json",
    "paragraphs.json",
    "transcript.srt",
    "transcript.vtt",
]

ASSEMBLYAI_PERSISTED_EXPORTS = [
    "timestamps.json",
    "sentences.json",
    "paragraphs.json",
    "transcript.srt",
    "transcript.vtt",
    "transcript.txt",
]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: str = "unknown"


def _safe_json_read(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _to_seconds(raw: Any) -> float:
    try:
        if raw is None:
            return 0.0
        val = float(raw)
    except Exception:
        return 0.0
    if val < 0:
        return 0.0
    # Heuristic: values larger than a day are very likely ms.
    if val > 86_400:
        return val / 1000.0
    return val


def _srt_time_to_seconds(value: str) -> float:
    token = str(value or "").strip()
    token = token.replace(".", ",")
    h, m, sec_ms = token.split(":")
    s, ms = sec_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.0)


def _seconds_to_srt_time(value: float) -> str:
    total_ms = max(0, int(round(float(value) * 1000.0)))
    h = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    m = rem // 60_000
    rem = rem % 60_000
    s = rem // 1000
    ms = rem % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _sanitize_speaker(raw: Any) -> str:
    token = str(raw or "").strip()
    return token or "unknown"


def _parse_srt_or_vtt(path: str, kind: str) -> list[Segment]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    out: list[Segment] = []
    ts_re = re.compile(
        r"(\d{2}:\d{2}:\d{2}[\.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{3})"
    )

    for block in blocks:
        lines = [ln.rstrip("\r") for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        # SRT may start with numeric index, VTT may have WEBVTT header or cue id.
        ts_line_index = -1
        for idx, ln in enumerate(lines):
            if "-->" in ln and ts_re.search(ln):
                ts_line_index = idx
                break
        if ts_line_index < 0:
            continue

        m = ts_re.search(lines[ts_line_index])
        if not m:
            continue

        try:
            start = _srt_time_to_seconds(m.group(1))
            end = _srt_time_to_seconds(m.group(2))
        except Exception:
            continue

        if end <= start:
            continue

        cue_lines = lines[ts_line_index + 1 :]
        cue_lines = [ln for ln in cue_lines if not ln.strip().lower().startswith("note")]
        text_value = " ".join(cue_lines).strip()
        out.append(Segment(start=start, end=end, text=text_value, speaker_id="unknown"))

    return out


def _from_timestamps_payload(payload: Any) -> list[Segment]:
    rows = payload
    if isinstance(payload, dict):
        rows = payload.get("segments") or payload.get("timestamps") or []

    out: list[Segment] = []
    if not isinstance(rows, list):
        return out

    for item in rows:
        if not isinstance(item, dict):
            continue
        start = _to_seconds(item.get("start") if item.get("start") is not None else item.get("start_ms"))
        end = _to_seconds(item.get("end") if item.get("end") is not None else item.get("end_ms"))
        if end <= start:
            continue
        text = str(item.get("text_es") or item.get("text") or item.get("text_en") or "").strip()
        spk = _sanitize_speaker(item.get("speaker") or item.get("speaker_id"))
        out.append(Segment(start=start, end=end, text=text, speaker_id=spk))
    return out


def _from_sentences_or_paragraphs_payload(payload: Any, field_name: str) -> list[Segment]:
    rows: Any = []
    if isinstance(payload, dict):
        rows = payload.get(field_name) or payload.get("results") or []
    elif isinstance(payload, list):
        rows = payload

    out: list[Segment] = []
    if not isinstance(rows, list):
        return out

    for item in rows:
        if not isinstance(item, dict):
            continue
        start = _to_seconds(item.get("start"))
        end = _to_seconds(item.get("end"))
        if end <= start:
            continue
        text = str(item.get("text") or item.get("text_es") or "").strip()
        spk = _sanitize_speaker(item.get("speaker") or item.get("speaker_id"))
        out.append(Segment(start=start, end=end, text=text, speaker_id=spk))
    return out


def load_segments_from_assemblyai_artifacts(artifact_paths: dict[str, str]) -> tuple[list[Segment], str | None]:
    """Return normalized segments and the source filename selected by canonical priority."""
    for filename in ASSEMBLYAI_IMPORT_PRIORITY:
        path = str((artifact_paths or {}).get(filename) or "").strip()
        if not path or not os.path.isfile(path):
            continue

        low = filename.lower()
        if low == "timestamps.json":
            segments = _from_timestamps_payload(_safe_json_read(path, {}))
        elif low == "sentences.json":
            segments = _from_sentences_or_paragraphs_payload(_safe_json_read(path, {}), "sentences")
        elif low == "paragraphs.json":
            segments = _from_sentences_or_paragraphs_payload(_safe_json_read(path, {}), "paragraphs")
        elif low.endswith(".srt"):
            segments = _parse_srt_or_vtt(path, "srt")
        elif low.endswith(".vtt"):
            segments = _parse_srt_or_vtt(path, "vtt")
        else:
            segments = []

        if segments:
            return segments, filename

    return [], None


def copy_assemblyai_artifacts_to_dir(artifact_paths: dict[str, str], target_dir: str) -> dict[str, str]:
    os.makedirs(target_dir, exist_ok=True)
    persisted: dict[str, str] = {}

    for filename in ASSEMBLYAI_PERSISTED_EXPORTS:
        src = str((artifact_paths or {}).get(filename) or "").strip()
        if not src or not os.path.isfile(src):
            continue
        dst = os.path.join(target_dir, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        persisted[filename] = dst

    return persisted


def build_timestamps_payload(segments: list[Segment]) -> dict[str, Any]:
    timeline = []
    for seg in sorted(segments, key=lambda x: (x.start, x.end)):
        timeline.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "start_ms": int(round(seg.start * 1000.0)),
                "end_ms": int(round(seg.end * 1000.0)),
                "text_es": seg.text,
                "speaker": seg.speaker_id,
                "speaker_id": seg.speaker_id,
            }
        )
    return {"segments": timeline}


def write_timestamps_json(segments: list[Segment], output_path: str) -> str:
    payload = build_timestamps_payload(segments)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def build_speaker_payload(
    *,
    source_video: str,
    target_dir: str,
    audit_audio: str | None,
    segments: list[Segment],
    assemblyai_exports: dict[str, str] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[Segment]] = defaultdict(list)
    for seg in sorted(segments, key=lambda x: (x.start, x.end)):
        grouped[_sanitize_speaker(seg.speaker_id)].append(seg)

    speakers = []
    for speaker_id, spk_segments in grouped.items():
        time_ranges = []
        sample_sentences = []
        for seg in spk_segments:
            time_ranges.append({
                "start": _seconds_to_srt_time(seg.start),
                "end": _seconds_to_srt_time(seg.end),
                "text": seg.text,
            })
            if seg.text:
                sample_sentences.append(seg.text)

        speakers.append(
            {
                "speaker_id": speaker_id,
                "ai_label": speaker_id,
                "sample": sample_sentences[0] if sample_sentences else "",
                "sample_sentences_head": sample_sentences[:3],
                "time_ranges": time_ranges,
                "segments_text": "",
                "segments_html": "",
                "target_dir": target_dir,
            }
        )

    exports = dict(assemblyai_exports or {})
    return {
        "phase": 1,
        "source_video": source_video,
        "target_dir": target_dir,
        "audit_audio": audit_audio or "",
        "generated_at": int(Path(target_dir).stat().st_mtime) if os.path.isdir(target_dir) else 0,
        "assemblyai_exports": exports,
        "speakers": speakers,
    }


def build_translated_payload(
    *,
    source_video: str,
    phase1_artifact: str,
    target_dir: str,
    segments: list[Segment],
    translation_engine: str = "assemblyai_import",
) -> dict[str, Any]:
    flat_segments = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, seg in enumerate(sorted(segments, key=lambda x: (x.start, x.end)), start=1):
        row = {
            "line": idx,
            "speaker_id": _sanitize_speaker(seg.speaker_id),
            "label": _sanitize_speaker(seg.speaker_id),
            "voice_ref_id": None,
            "voice_ref": None,
            "start": float(seg.start),
            "end": float(seg.end),
            "text_es": str(seg.text or "").strip(),
            "text_en": "",
            "source": "assemblyai_import",
        }
        flat_segments.append(row)
        grouped[row["speaker_id"]].append(row)

    speakers = []
    for speaker_id, rows in grouped.items():
        speakers.append(
            {
                "speaker_id": speaker_id,
                "label": speaker_id,
                "voice_ref_id": None,
                "voice_ref": None,
                "segments": [
                    {
                        "start": r["start"],
                        "end": r["end"],
                        "text_es": r["text_es"],
                        "source": r["source"],
                        "line": r["line"],
                    }
                    for r in rows
                ],
            }
        )

    return {
        "phase": 2,
        "generated_at": int(Path(target_dir).stat().st_mtime) if os.path.isdir(target_dir) else 0,
        "source_video": source_video,
        "phase1_artifact": phase1_artifact,
        "translation_engine": translation_engine,
        "target_dir": target_dir,
        "source_srt": os.path.join(target_dir, "en.srt"),
        "target_srt": os.path.join(target_dir, "es.srt"),
        "segments": flat_segments,
        "speakers": speakers,
    }


def summarize_artifact_presence(artifact_paths: dict[str, str]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for filename in ASSEMBLYAI_PERSISTED_EXPORTS:
        path = str((artifact_paths or {}).get(filename) or "").strip()
        status[filename] = bool(path and os.path.isfile(path))
    return status
