from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ProjectPaths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--model-name", choices=["raw-cnn", "raw-cnn-lstm", "spec-cnn"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        from ..evaluate import evaluate_model
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("PyTorch is not available in this Python environment.") from exc
        raise

    repo_root = Path(__file__).resolve().parents[3]
    paths = ProjectPaths(root=repo_root)
    processed_dir = args.processed_dir or paths.data_processed
    results_dir = args.results_dir or paths.results

    metrics = evaluate_model(
        processed_dir=processed_dir,
        results_dir=results_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device_name=args.device,
        max_test_samples=args.max_test_samples,
    )
    print(metrics)


if __name__ == "__main__":
    main()