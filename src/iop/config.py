from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16_000
    clip_duration: float = 1.0
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration)


@dataclass(slots=True)
class ProjectPaths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def dataset_root(self) -> Path:
        return self.data_raw / "speech_commands_v0.02"

    @property
    def dataset_archive(self) -> Path:
        return self.data_raw / "speech_commands_v0.02.tar.gz"

    @property
    def results(self) -> Path:
        return self.root / "results"
