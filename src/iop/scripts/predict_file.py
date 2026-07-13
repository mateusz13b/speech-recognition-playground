from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import AudioConfig, ProjectPaths
from ..dataset import load_label_map
from ..engine import build_model, select_device
from ..infer import load_checkpoint, predict
from ..preprocess import load_audio, pad_or_trim, spectrogram


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference for a single wav file.")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model-name", choices=["raw-cnn", "raw-cnn-lstm", "spec-cnn"], default="raw-cnn")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def _resolve_model_path(results_dir: Path, model_name: str) -> Path:
    weights_path = results_dir / "weights" / f"{model_name}_weights.pt"
    checkpoint_path = results_dir / "checkpoints" / f"{model_name}.pt"
    if weights_path.exists():
        return weights_path
    if checkpoint_path.exists():
        return checkpoint_path
    raise FileNotFoundError(f"Missing model weights: {weights_path} or {checkpoint_path}")


def _prepare_tensor(wav_path: Path, model_name: str, audio_config: AudioConfig) -> torch.Tensor:
    sample = load_audio(wav_path, audio_config)
    waveform = pad_or_trim(sample.waveform, audio_config.num_samples)

    if model_name == "spec-cnn":
        x = torch.from_numpy(spectrogram(waveform, audio_config)).float().unsqueeze(0)
    else:
        x = torch.from_numpy(waveform).float().unsqueeze(0)
    return x


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    paths = ProjectPaths(root=repo_root)
    processed_dir = args.processed_dir or paths.data_processed
    results_dir = args.results_dir or paths.results

    label_to_id = load_label_map(processed_dir)
    labels = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]

    device = select_device(args.device)
    model = build_model(args.model_name, num_classes=len(labels))
    model_path = _resolve_model_path(results_dir, args.model_name)
    load_checkpoint(model_path, model)
    model = model.to(device)

    batch = _prepare_tensor(args.wav, args.model_name, AudioConfig()).to(device)
    probabilities = predict(model, batch)[0].detach().cpu()

    top_k = max(1, min(int(args.top_k), len(labels)))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    predicted_index = int(top_indices[0].item())
    predicted_label = labels[predicted_index]
    confidence = float(top_probs[0].item())

    print(f"wav: {args.wav}")
    print(f"model: {args.model_name}")
    print(f"weights: {model_path}")
    print(f"predicted_label: {predicted_label}")
    print(f"confidence: {confidence:.4f}")
    print("top_predictions:")
    for rank, (score, index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        print(f"  {rank}. {labels[int(index)]}: {float(score):.4f}")


if __name__ == "__main__":
    main()
