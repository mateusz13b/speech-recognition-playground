from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ProjectPaths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a voice-command baseline model.")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--model-name", choices=["raw-cnn", "raw-cnn-lstm", "spec-cnn"], default="raw-cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        from ..engine import run_from_cli
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is not available in this Python environment. "
                "Install torch, then run the training script again."
            ) from exc
        raise

    repo_root = Path(__file__).resolve().parents[3]
    paths = ProjectPaths(root=repo_root)
    processed_dir = args.processed_dir or paths.data_processed
    results_dir = args.results_dir or paths.results

    run_from_cli(
        {
            "processed_dir": processed_dir,
            "results_dir": results_dir,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "device": args.device,
        }
    )


if __name__ == "__main__":
    main()