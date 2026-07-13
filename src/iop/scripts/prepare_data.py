from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from ..config import ProjectPaths
from ..data import (
    BACKGROUND_LABEL,
    DEFAULT_CLASSES,
    DatasetIndex,
    build_index,
    save_index,
    summarize_index,
)


DATASET_DIRNAME = "speech_commands_v0.02"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
SPLIT_MODE = "speaker"
SPLIT_SEED = 42


def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return target.is_relative_to(directory)


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = target_dir / member.name
            if not _is_within_directory(target_dir, member_path):
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(target_dir)


def _find_dataset_root(raw_root: Path) -> Path:
    if (raw_root / "validation_list.txt").exists() and (raw_root / "testing_list.txt").exists():
        return raw_root

    extracted = raw_root / DATASET_DIRNAME
    if extracted.exists():
        return extracted

    archive = raw_root / ARCHIVE_NAME
    if archive.exists():
        _safe_extract_tar(archive, raw_root)
        if extracted.exists():
            return extracted

    raise FileNotFoundError(
        "Could not find extracted dataset or archive. "
        f"Expected {raw_root}, {extracted} or {archive}"
    )


def _write_summary(index: DatasetIndex, output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary = summarize_index(index.frame)
    summary["background_label"] = BACKGROUND_LABEL
    summary["target_classes"] = DEFAULT_CLASSES
    summary["split_mode"] = SPLIT_MODE
    summary["split_seed"] = SPLIT_SEED
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def prepare_data(raw_root: Path, output_dir: Path) -> DatasetIndex:
    dataset_root = _find_dataset_root(raw_root)
    index = build_index(
        dataset_root,
        target_classes=DEFAULT_CLASSES,
        split_mode=SPLIT_MODE,
        seed=SPLIT_SEED,
    )
    save_index(index, output_dir)
    _write_summary(index, output_dir)
    return index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the speech commands dataset.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Directory with the dataset archive or extracted speech_commands_v0.02 folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where manifest files will be written.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    paths = ProjectPaths(root=repo_root)
    raw_root = args.raw_root or paths.data_raw
    output_dir = args.output_dir or paths.data_processed

    index = prepare_data(raw_root=raw_root, output_dir=output_dir)
    summary = summarize_index(index.frame)
    print(f"Prepared {summary['num_files']} audio files in {output_dir}")
    print(f"Labels: {', '.join(summary['labels'])}")
    print(f"Splits: {summary['splits']}")
    print(f"Speaker counts: {summary['speaker_counts']}")


if __name__ == "__main__":
    main()