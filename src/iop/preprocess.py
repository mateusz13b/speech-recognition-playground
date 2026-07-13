from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from .config import AudioConfig


@dataclass(slots=True)
class AudioSample:
    waveform: np.ndarray
    sample_rate: int


def _pcm16_to_float32(raw: bytes) -> np.ndarray:
    audio = np.frombuffer(raw, dtype=np.int16)
    return (audio.astype(np.float32) / 32768.0).copy()


def load_audio(path: str | Path, config: AudioConfig) -> AudioSample:
    audio_path = Path(path)
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM wav file, got sample width {sample_width} bytes: {audio_path}")

    waveform = _pcm16_to_float32(frames)
    if num_channels > 1:
        waveform = waveform.reshape(-1, num_channels).mean(axis=1)

    if sample_rate != config.sample_rate:
        raise ValueError(
            f"Unexpected sample rate {sample_rate} Hz for {audio_path}. "
            f"Expected {config.sample_rate} Hz."
        )

    return AudioSample(waveform=waveform, sample_rate=sample_rate)


def pad_or_trim(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    if len(waveform) > num_samples:
        return waveform[:num_samples]
    if len(waveform) < num_samples:
        pad_width = num_samples - len(waveform)
        return np.pad(waveform, (0, pad_width), mode="constant")
    return waveform


def spectrogram(waveform: np.ndarray, config: AudioConfig) -> np.ndarray:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch") from exc

    tensor = torch.from_numpy(waveform).float()
    window = torch.hann_window(config.n_fft)
    spec = torch.stft(
        tensor,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spec.abs().pow(2.0)
    log_spec = torch.log1p(power)
    return log_spec.numpy().astype(np.float32)


def mel_spectrogram(waveform: np.ndarray, config: AudioConfig) -> np.ndarray:
    return spectrogram(waveform, config)