import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYVIDEOTRANS_ROOT = ROOT / "pyvideotrans"
if str(PYVIDEOTRANS_ROOT) not in sys.path:
    sys.path.insert(0, str(PYVIDEOTRANS_ROOT))

from pyvideotrans.videotrans.process.assemblyai_diarization import (
    _assign_subtitles_to_speakers,
    _build_diarizations_from_utterances,
)


def _read_key() -> str:
    key = os.environ.get("ASSEMBLY_AI_KEY") or os.environ.get("ASSEMBLYAI_API_KEY")
    if key:
        return key.strip().strip('"')

    env_path = Path(".env")
    if not env_path.is_file():
        return ""

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "ASSEMBLY_AI_KEY":
            return v.strip().strip('"').strip("'")
    return ""


def _load_subtitles_ms(path: Path) -> list[list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in raw:
        start = int(round(float(item.get("start", 0.0)) * 1000.0))
        end = int(round(float(item.get("end", 0.0)) * 1000.0))
        if end > start:
            out.append([start, end])
    return out


def _resolve_speech_models() -> list[str]:
    raw = str(
        os.environ.get("ASSEMBLYAI_SPEECH_MODELS")
        or os.environ.get("ASSEMBLYAI_SPEECH_MODEL")
        or ""
    ).strip()
    allowed = {"universal-3-pro", "universal-2"}
    if not raw:
        return ["universal-3-pro", "universal-2"]
    models = []
    for item in raw.split(","):
        m = item.strip().lower()
        if m in allowed and m not in models:
            models.append(m)
    return models or ["universal-3-pro", "universal-2"]


def main() -> int:
    base_url = os.environ.get("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com/v2").rstrip("/")
    request_timeout = float(os.environ.get("ASSEMBLYAI_REQUEST_TIMEOUT", "60"))
    poll_seconds = float(os.environ.get("ASSEMBLYAI_POLL_SECONDS", "2.0"))
    max_seconds = int(os.environ.get("ASSEMBLYAI_TIMEOUT_SECONDS", "1800"))

    input_media = Path(os.environ.get("INPUT_MEDIA", "input/full_video_en.mp4"))
    ts_path = Path(os.environ.get("TIMESTAMPS_JSON", "output/full_video_en_timestamps.json"))
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_media.is_file():
        raise FileNotFoundError(f"Missing input media: {input_media}")
    if not ts_path.is_file():
        raise FileNotFoundError(f"Missing timestamps file: {ts_path}")

    key = _read_key()
    if not key:
        raise RuntimeError("ASSEMBLY_AI_KEY missing in env/.env")

    subtitles = _load_subtitles_ms(ts_path)
    started = time.time()

    with input_media.open("rb") as f:
        upload_resp = requests.post(
            f"{base_url}/upload",
            headers={"authorization": key},
            data=f,
            timeout=request_timeout,
        )
    upload_resp.raise_for_status()
    upload_url = (upload_resp.json() or {}).get("upload_url")
    if not upload_url:
        raise RuntimeError("Upload response without upload_url")

    payload = {
        "audio_url": upload_url,
        "speech_models": _resolve_speech_models(),
        "speaker_labels": True,
        "language_detection": True,
        "format_text": True,
    }
    create_resp = requests.post(
        f"{base_url}/transcript",
        headers={"authorization": key, "content-type": "application/json"},
        json=payload,
        timeout=request_timeout,
    )
    if create_resp.status_code >= 400:
        try:
            body = create_resp.json()
        except Exception:
            body = create_resp.text
        raise RuntimeError(
            f"Create transcript failed HTTP {create_resp.status_code}: {body} | payload={payload}"
        )
    transcript_id = (create_resp.json() or {}).get("id")
    if not transcript_id:
        raise RuntimeError("Create transcript response without id")

    final_body = {}
    while True:
        if (time.time() - started) > max_seconds:
            raise TimeoutError(f"Timed out waiting transcript after {max_seconds}s")

        r = requests.get(
            f"{base_url}/transcript/{transcript_id}",
            headers={"authorization": key},
            timeout=request_timeout,
        )
        r.raise_for_status()
        body = r.json() or {}
        status = str(body.get("status", "")).lower().strip()

        if status == "completed":
            final_body = body
            break
        if status == "error":
            raise RuntimeError(f"AssemblyAI status=error: {body.get('error')}")

        time.sleep(max(0.5, poll_seconds))

    utterances = final_body.get("utterances") or []
    diarizations = _build_diarizations_from_utterances(utterances)
    assigned = _assign_subtitles_to_speakers(subtitles, diarizations)

    raw_speakers = sorted({str(u.get("speaker")) for u in utterances if isinstance(u, dict) and u.get("speaker") is not None})
    norm_speakers = sorted({str(d[1]) for d in diarizations if isinstance(d, list) and len(d) == 2})

    summary = {
        "transcript_id": transcript_id,
        "status": final_body.get("status"),
        "language_code": final_body.get("language_code"),
        "audio_duration": final_body.get("audio_duration"),
        "utterances_count": len(utterances),
        "raw_speakers": raw_speakers,
        "normalized_speakers": norm_speakers,
        "subtitles_count": len(subtitles),
        "assigned_count": len(assigned),
        "assigned_unique": sorted(set(assigned)),
        "first_assigned": assigned[:20],
        "sample_utterances": utterances[:5],
        "elapsed_seconds": int(time.time() - started),
        "request_payload": payload,
    }

    raw_path = out_dir / "aai_probe_response.json"
    summary_path = out_dir / "aai_probe_summary.json"
    raw_path.write_text(json.dumps(final_body, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved raw response to: {raw_path}")
    print(f"Saved summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
