from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - lets data prep work without torch.
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from .config import AudioConfig
from .preprocess import load_audio, pad_or_trim, spectrogram


@dataclass(slots=True)
class DatasetBundle:
    frame: pd.DataFrame
    label_to_id: dict[str, int]


def load_manifest(processed_dir: Path) -> pd.DataFrame:
    manifest_path = processed_dir / "dataset_index.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return pd.read_csv(manifest_path)


def load_label_map(processed_dir: Path) -> dict[str, int]:
    summary_path = processed_dir / "summary.json"
    if summary_path.exists():
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        labels = summary.get("labels", [])
        if labels:
            return {label: idx for idx, label in enumerate(labels)}

    frame = load_manifest(processed_dir)
    labels = sorted(frame["label"].dropna().unique().tolist())
    return {label: idx for idx, label in enumerate(labels)}


def load_bundle(processed_dir: Path) -> DatasetBundle:
    frame = load_manifest(processed_dir)
    label_to_id = load_label_map(processed_dir)
    return DatasetBundle(frame=frame, label_to_id=label_to_id)


def make_split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = frame[frame["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"Split '{split}' is empty")
    return subset.reset_index(drop=True)


class SpeechCommandsWaveformDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_id: dict[str, int],
        config: AudioConfig | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.config = config or AudioConfig()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        if torch is None:
            raise ModuleNotFoundError("torch")
        row = self.frame.iloc[index]
        sample = load_audio(row["path"], self.config)
        waveform = pad_or_trim(sample.waveform, self.config.num_samples)
        x = torch.from_numpy(waveform).float()
        y = torch.tensor(self.label_to_id[str(row["label"])], dtype=torch.long)
        return x, y


class SpeechCommandsSpectrogramDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_id: dict[str, int],
        config: AudioConfig | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.config = config or AudioConfig()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        if torch is None:
            raise ModuleNotFoundError("torch")
        row = self.frame.iloc[index]
        sample = load_audio(row["path"], self.config)
        waveform = pad_or_trim(sample.waveform, self.config.num_samples)
        spec = spectrogram(waveform, self.config)
        x = torch.from_numpy(spec).float()
        y = torch.tensor(self.label_to_id[str(row["label"])], dtype=torch.long)
        return x, y