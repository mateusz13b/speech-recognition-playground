from __future__ import annotations

from pathlib import Path

import sounddevice as sd
import soundfile as sf


def record_audio(output_path: str | Path, seconds: float = 1.0, sample_rate: int = 16_000) -> Path:
    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return output_path
