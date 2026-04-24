from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS_ROOT = ROOT / "pyvideotrans"
if str(PYVIDEOTRANS_ROOT) not in sys.path:
    sys.path.insert(0, str(PYVIDEOTRANS_ROOT))


def _build_fake_torch(*, free_gib: float, cuda_available: bool = True):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def mem_get_info():
            free_bytes = int(free_gib * (1024 ** 3))
            total_bytes = int(80 * (1024 ** 3))
            return free_bytes, total_bytes

        @staticmethod
        def memory_allocated(_idx=0):
            return 0

        @staticmethod
        def memory_reserved(_idx=0):
            return 0

        @staticmethod
        def empty_cache():
            return None

    fake_torch = types.SimpleNamespace(
        cuda=_FakeCuda,
        float16="float16",
        float32="float32",
        set_num_threads=lambda *_args, **_kwargs: None,
    )
    return fake_torch


def _install_fake_qwen_modules(monkeypatch, *, fail_batch: bool, free_gib: float = 80):
    fake_torch = _build_fake_torch(free_gib=free_gib, cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def _sf_write(filename, _wav, _sr):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"wav")

    fake_sf = types.SimpleNamespace(write=_sf_write)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    class _DummyModel:
        def __init__(self, model_path: str):
            self.model_path = model_path

        @classmethod
        def from_pretrained(cls, model_path, **_kwargs):
            return cls(model_path)

        def generate_custom_voice(self, text, language=None, speaker=None, instruct=None):
            _ = language, speaker, instruct
            if isinstance(text, list):
                if fail_batch and len(text) > 1:
                    raise RuntimeError("forced batch error")
                return [np.zeros(2400, dtype=np.float32) for _ in text], 24000
            return [np.zeros(2400, dtype=np.float32)], 24000

        def create_voice_clone_prompt(self, **_kwargs):
            return "prompt"

        def generate_voice_clone(self, text, **_kwargs):
            _ = _kwargs
            fail_tokens = [t for t in os.environ.get("PYTEST_QWEN_FAIL_TEXT_TOKENS", "").split(",") if t]

            def _should_fail(s: str) -> bool:
                return any(tok in s for tok in fail_tokens)

            if isinstance(text, list):
                if fail_batch and len(text) > 1:
                    raise RuntimeError("forced batch error")
                if any(_should_fail(t) for t in text):
                    raise RuntimeError("forced line error")
                return [np.zeros(2400, dtype=np.float32) for _ in text], 24000
            if _should_fail(text):
                raise RuntimeError("forced line error")
            return [np.zeros(2400, dtype=np.float32)], 24000

    fake_qwen = types.SimpleNamespace(Qwen3TTSModel=_DummyModel)
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen)


def _load_tts_module(monkeypatch, *, fail_batch: bool = False, free_gib: float = 80):
    _install_fake_qwen_modules(monkeypatch, fail_batch=fail_batch, free_gib=free_gib)
    mod = importlib.import_module("videotrans.process.tts_fun")
    return importlib.reload(mod)


def _build_queue_json(tmp_path: Path, n: int = 5) -> tuple[str, list[Path]]:
    rows = []
    expected_out = []
    for i in range(n):
        base = tmp_path / f"line_{i}"
        expected_out.append(Path(str(base) + "-qwen3tts.wav"))
        rows.append(
            {
                "line": i + 1,
                "text": f"texto {i}",
                "role": "Serena",
                "filename": str(base),
            }
        )

    queue_file = tmp_path / "queue.json"
    queue_file.write_text(json.dumps(rows), encoding="utf-8")
    return str(queue_file), expected_out


def test_resolve_batch_lines_a100_default(monkeypatch):
    mod = _load_tts_module(monkeypatch, fail_batch=False, free_gib=72)
    n = mod._resolve_qwen_tts_batch_lines(is_cuda=True)
    assert n == 12


def test_qwen3tts_emits_batch_progress_logs(monkeypatch, tmp_path):
    mod = _load_tts_module(monkeypatch, fail_batch=False, free_gib=72)
    queue_file, out_files = _build_queue_json(tmp_path, n=5)

    logs = []

    def _capture(_file, msg):
        payload = json.loads(msg)
        logs.append(payload.get("text", ""))

    monkeypatch.setattr(mod, "_write_log", _capture)
    monkeypatch.setenv("PYVIDEOTRANS_QWEN_TTS_BATCH_LINES", "3")

    ok, err = mod.qwen3tts_fun(
        queue_tts_file=queue_file,
        language="Spanish",
        logs_file=str(tmp_path / "tts.log"),
        defaulelang="en",
        is_cuda=True,
        prompt="",
        model_name="1.7B",
        roledict={},
        device_index=0,
    )

    assert ok is True
    assert err is None
    assert any("generation_start" in s for s in logs)
    assert any("batch_start" in s for s in logs)
    assert any("batch_done" in s for s in logs)
    assert all(p.exists() for p in out_files)


def test_qwen3tts_batch_error_falls_back_to_single(monkeypatch, tmp_path):
    mod = _load_tts_module(monkeypatch, fail_batch=True, free_gib=72)
    queue_file, out_files = _build_queue_json(tmp_path, n=4)

    logs = []

    def _capture(_file, msg):
        payload = json.loads(msg)
        logs.append(payload.get("text", ""))

    monkeypatch.setattr(mod, "_write_log", _capture)
    monkeypatch.setenv("PYVIDEOTRANS_QWEN_TTS_BATCH_LINES", "4")

    ok, err = mod.qwen3tts_fun(
        queue_tts_file=queue_file,
        language="Spanish",
        logs_file=str(tmp_path / "tts.log"),
        defaulelang="en",
        is_cuda=True,
        prompt="",
        model_name="1.7B",
        roledict={},
        device_index=0,
    )

    assert ok is True
    assert err is None
    assert any("batch fallback to single" in s for s in logs)
    assert all(p.exists() for p in out_files)


def test_qwen3tts_short_ref_always_clones(monkeypatch, tmp_path):
    """Short reference (15 s) must always be used for cloning — no fallback by duration."""
    mod = _load_tts_module(monkeypatch, fail_batch=False, free_gib=72)

    ref_wav = tmp_path / "spk_ref.wav"
    ref_wav.write_bytes(b"wav")
    out_base = tmp_path / "line_clone"
    out_file = Path(str(out_base) + "-qwen3tts.wav")

    queue_file = tmp_path / "queue_clone.json"
    queue_file.write_text(
        json.dumps(
            [
                {
                    "line": 1,
                    "text": "texto con clone",
                    "role": "clone",
                    "filename": str(out_base),
                    "ref_wav": str(ref_wav),
                    "ref_text": "",
                    "with_emotion": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    logs = []

    def _capture(_file, msg):
        payload = json.loads(msg)
        logs.append(payload.get("text", ""))

    monkeypatch.setattr(mod, "_write_log", _capture)

    import videotrans.util.tools as tools_mod

    monkeypatch.setattr(tools_mod, "get_audio_time", lambda _p: 15000)

    ok, err = mod.qwen3tts_fun(
        queue_tts_file=str(queue_file),
        language="Spanish",
        logs_file=str(tmp_path / "tts.log"),
        defaulelang="en",
        is_cuda=True,
        prompt="",
        model_name="1.7B",
        roledict={},
        device_index=0,
    )

    assert ok is True
    assert err is None
    # No fallback log — clone was used directly.
    assert not any("fallback" in s for s in logs)
    # Clone ref log should appear (ref_ms > 0, no trimming needed for 15 s).
    assert any("clone ref:" in s for s in logs)
    assert out_file.exists()


def test_qwen3tts_aborts_when_majority_lines_fail(monkeypatch, tmp_path):
    mod = _load_tts_module(monkeypatch, fail_batch=False, free_gib=72)

    ref_wav = tmp_path / "spk_ref.wav"
    ref_wav.write_bytes(b"wav")

    rows = []
    for i in range(5):
        base = tmp_path / f"line_{i}"
        txt = f"ok_{i}"
        if i in (1, 2, 3):
            txt = f"fail_{i}"
        rows.append(
            {
                "line": i + 1,
                "text": txt,
                "role": "clone",
                "filename": str(base),
                "ref_wav": str(ref_wav),
                "ref_text": "",
                "with_emotion": False,
            }
        )

    queue_file = tmp_path / "queue_clone_fail_majority.json"
    queue_file.write_text(json.dumps(rows), encoding="utf-8")

    monkeypatch.setenv("PYTEST_QWEN_FAIL_TEXT_TOKENS", "fail_")
    monkeypatch.setenv("PYVIDEOTRANS_QWEN_MIN_SUCCESS_RATIO", "0.51")

    import videotrans.util.tools as tools_mod

    monkeypatch.setattr(tools_mod, "get_audio_time", lambda _p: 15000)

    ok, err = mod.qwen3tts_fun(
        queue_tts_file=str(queue_file),
        language="Spanish",
        logs_file=str(tmp_path / "tts.log"),
        defaulelang="en",
        is_cuda=True,
        prompt="",
        model_name="1.7B",
        roledict={},
        device_index=0,
    )

    assert ok is False
    assert err is not None
    assert "success ratio below threshold" in err
