import os
import subprocess
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageEnhance


def extract_frame(video_path: str, output_dir: str, second: float = 1.0) -> str:
    """Extract a single frame from video at `second` and return image path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    frame_path = os.path.join(output_dir, "frame_reference.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(max(0.0, float(second))),
        "-i",
        video_path,
        "-frames:v",
        "1",
        frame_path,
    ]
    subprocess.run(cmd, check=True)

    if not os.path.exists(frame_path):
        raise RuntimeError("No se pudo extraer el frame de referencia")
    return frame_path


def apply_frame_adjustments(
    frame_path: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    color: float = 1.0,
    sharpness: float = 1.0,
    output_dir: str = ".",
) -> str:
    """Apply adjustments and save processed preview image."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "frame_preview_adjusted.jpg")

    with Image.open(frame_path).convert("RGB") as img:
        img = ImageEnhance.Brightness(img).enhance(float(brightness))
        img = ImageEnhance.Contrast(img).enhance(float(contrast))
        img = ImageEnhance.Color(img).enhance(float(color))
        img = ImageEnhance.Sharpness(img).enhance(float(sharpness))
        img.save(output_path, format="JPEG", quality=95)

    return output_path


def extract_and_adjust(
    video_path: str,
    temp_dir: str,
    second: float,
    brightness: float,
    contrast: float,
    color: float,
    sharpness: float,
) -> Tuple[str, str]:
    """Convenience workflow: extract frame then apply adjustments."""
    frame_path = extract_frame(video_path=video_path, output_dir=temp_dir, second=second)
    adjusted_path = apply_frame_adjustments(
        frame_path=frame_path,
        brightness=brightness,
        contrast=contrast,
        color=color,
        sharpness=sharpness,
        output_dir=temp_dir,
    )
    return frame_path, adjusted_path
