from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random

import pandas as pd


DEFAULT_CLASSES = ["go", "stop", "left", "right", "on", "off", "up", "down"]
BACKGROUND_LABEL = "background"
UNKNOWN_LABEL = "unknown"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
SPLIT_COLUMNS = ["path", "relpath", "label", "split", "speaker", "is_target"]


@dataclass(slots=True)
class DatasetIndex:
    frame: pd.DataFrame

    @property
    def labels(self) -> list[str]:
        return sorted(self.frame["label"].dropna().unique().tolist())

    @property
    def splits(self) -> list[str]:
        return sorted(self.frame["split"].dropna().unique().tolist())


def load_split_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().replace("\\", "/").removeprefix("./")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def label_from_path(path: Path, target_classes: Iterable[str]) -> str:
    label = path.parent.name
    if label == "_background_noise_":
        return BACKGROUND_LABEL
    if label in target_classes:
        return label
    return UNKNOWN_LABEL


def speaker_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "_nohash_" in stem:
        return stem.split("_nohash_", 1)[0]
    return stem


def _speaker_split_map(speakers: list[str], seed: int = 42) -> dict[str, str]:
    speakers = sorted(set(speakers))
    rng = random.Random(seed)
    rng.shuffle(speakers)

    total = len(speakers)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)

    if total >= 3:
        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, total - 1)

    split_map: dict[str, str] = {}
    for idx, speaker in enumerate(speakers):
        if idx < train_end:
            split_map[speaker] = TRAIN_SPLIT
        elif idx < val_end:
            split_map[speaker] = VAL_SPLIT
        else:
            split_map[speaker] = TEST_SPLIT
    return split_map


def _background_split_map(relpaths: list[str]) -> dict[str, str]:
    relpaths = sorted(relpaths)
    total = len(relpaths)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)

    if total >= 3:
        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, total - 1)

    split_map: dict[str, str] = {}
    for idx, relpath in enumerate(relpaths):
        if idx < train_end:
            split_map[relpath] = TRAIN_SPLIT
        elif idx < val_end:
            split_map[relpath] = VAL_SPLIT
        else:
            split_map[relpath] = TEST_SPLIT
    return split_map


def build_index(
    root: Path,
    target_classes: Iterable[str] = DEFAULT_CLASSES,
    validation_files: set[str] | None = None,
    testing_files: set[str] | None = None,
    split_mode: str = "speaker",
    seed: int = 42,
) -> DatasetIndex:
    del validation_files, testing_files
    target_classes = tuple(target_classes)
    target_set = set(target_classes)

    base_rows: list[dict[str, object]] = []
    for wav_path in root.rglob("*.wav"):
        relpath = wav_path.relative_to(root).as_posix()
        label = label_from_path(wav_path, target_set)
        base_rows.append(
            {
                "path": str(wav_path),
                "relpath": relpath,
                "label": label,
                "speaker": speaker_from_filename(wav_path.name),
                "is_target": label in target_set,
            }
        )

    frame = pd.DataFrame(base_rows, columns=["path", "relpath", "label", "speaker", "is_target"])
    if frame.empty:
        return DatasetIndex(frame=pd.DataFrame(columns=SPLIT_COLUMNS))

    if split_mode != "speaker":
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    non_background = frame[frame["label"] != BACKGROUND_LABEL].copy()
    background = frame[frame["label"] == BACKGROUND_LABEL].copy()

    speaker_map = _speaker_split_map(non_background["speaker"].dropna().tolist(), seed=seed)
    non_background["split"] = non_background["speaker"].map(speaker_map)

    if not background.empty:
        bg_map = _background_split_map(background["relpath"].tolist())
        background["split"] = background["relpath"].map(bg_map)

    frame = pd.concat([non_background, background], ignore_index=True)
    frame = frame[SPLIT_COLUMNS]
    frame = frame.sort_values(["split", "label", "relpath"]).reset_index(drop=True)
    return DatasetIndex(frame=frame)


def summarize_index(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "num_files": 0,
            "labels": [],
            "splits": {},
            "label_counts": {},
            "speaker_counts": {},
        }

    speaker_counts = (
        frame[frame["label"] != BACKGROUND_LABEL]
        .groupby("split")["speaker"]
        .nunique()
        .sort_index()
        .to_dict()
    )

    return {
        "num_files": int(len(frame)),
        "labels": sorted(frame["label"].unique().tolist()),
        "splits": frame["split"].value_counts().sort_index().to_dict(),
        "label_counts": frame["label"].value_counts().sort_index().to_dict(),
        "speaker_counts": speaker_counts,
    }


def save_index(index: DatasetIndex, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_index.csv"
    index.frame.to_csv(csv_path, index=False)
    return csv_path