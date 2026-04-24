from huggingface_hub import snapshot_download
from pathlib import Path
import os

ROOT = Path(r"C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es")
TARGET_ROOT = ROOT / "pyvideotrans" / "models"
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

REPOS = [
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice"),
]

print(f"Target root: {TARGET_ROOT}")

for repo_id, folder_name in REPOS:
    print(f"\n=== Downloading {repo_id} ===")
    local_dir = TARGET_ROOT / folder_name
    local_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        resume_download=True,
    )
    print(f"Downloaded to: {path}")

print("\n=== Verification ===")
for _, folder_name in REPOS:
    p = TARGET_ROOT / folder_name
    files = [x for x in p.rglob("*") if x.is_file()]
    total_bytes = sum(x.stat().st_size for x in files)
    print(f"{p.name}: files={len(files)}, size_gb={total_bytes / (1024**3):.2f}")
