import torchaudio
torchaudio.set_audio_backend("soundfile")

import torch
import soundfile as sf
import numpy as np
from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import load_audio, save_audio

INPUT  = r"C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\reference_voices\dw_documentales-mujer\dw_documental_DeepFilterNet3.wav"
OUTPUT = r"C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\output\dw_documental_DeepFilterNet3_clean.wav"

model, df_state, _ = init_df()

audio, _ = load_audio(INPUT, sr=df_state.sr())
enhanced = enhance(model, df_state, audio)
save_audio(OUTPUT, enhanced, df_state.sr())

print(f"Audio limpio guardado en: {OUTPUT}")
