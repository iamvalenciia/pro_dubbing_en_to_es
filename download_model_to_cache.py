#!/usr/bin/env python3
"""
Download faster-whisper-tiny model to standard HuggingFace cache
"""
from huggingface_hub import snapshot_download
import os

model_repo = 'Systran/faster-whisper-tiny'
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')

print(f"Downloading {model_repo} to standard HF cache...")
print(f"Cache dir: {cache_dir}\n")

try:
    model_dir = snapshot_download(
        repo_id=model_repo,
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True
    )
    print(f"\n✓ SUCCESS: Model downloaded to")
    print(f"  {model_dir}")
    
    # Verificar archivos
    files = os.listdir(model_dir)
    print(f"\nFiles in model directory ({len(files)}): {files}")
    
    # Verificar model.bin específicamente
    model_bin = os.path.join(model_dir, 'model.bin')
    if os.path.exists(model_bin):
        size_mb = os.path.getsize(model_bin) / (1024*1024)
        print(f"✓ model.bin found: {size_mb:.1f} MB")
    else:
        print("✗ model.bin NOT found in root directory")
        
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
