from __future__ import annotations

from pathlib import Path
from typing import Literal
import json
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from .config import AudioConfig, ProjectPaths
from .data import DEFAULT_CLASSES
from .dataset import (
    SpeechCommandsSpectrogramDataset,
    SpeechCommandsWaveformDataset,
    load_bundle,
    make_split_frame,
)
from .models import Conv1DClassifier, Conv1DLSTMClassifier, SpectrogramCNN


ModelName = Literal["raw-cnn", "raw-cnn-lstm", "spec-cnn"]


class FitConfig:
    def __init__(
        self,
        model_name: ModelName = "raw-cnn",
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        num_workers: int = 0,
        seed: int = 42,
        max_train_samples: int = 0,
        max_val_samples: int = 0,
    ) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.seed = seed
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples


class EvalResult:
    def __init__(
        self,
        loss: float,
        accuracy: float,
        precision: float,
        macro_f1: float,
        recall: float,
        y_true: list[int],
        y_pred: list[int],
    ) -> None:
        self.loss = loss
        self.accuracy = accuracy
        self.precision = precision
        self.macro_f1 = macro_f1
        self.recall = recall
        self.y_true = y_true
        self.y_pred = y_pred


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_frame(frame, limit: int):
    if limit and limit > 0:
        return frame.head(limit).reset_index(drop=True)
    return frame


def build_model(model_name: ModelName, num_classes: int) -> nn.Module:
    if model_name == "raw-cnn-lstm":
        return Conv1DLSTMClassifier(num_classes=num_classes)
    if model_name == "spec-cnn":
        return SpectrogramCNN(num_classes=num_classes)
    return Conv1DClassifier(num_classes=num_classes)


def build_datasets(bundle, config: FitConfig, audio_config: AudioConfig, split: str = "val"):
    train_frame = limit_frame(make_split_frame(bundle.frame, "train"), config.max_train_samples)
    eval_frame = limit_frame(make_split_frame(bundle.frame, split), config.max_val_samples if split == "val" else 0)

    if config.model_name == "spec-cnn":
        train_ds = SpeechCommandsSpectrogramDataset(train_frame, bundle.label_to_id, audio_config)
        eval_ds = SpeechCommandsSpectrogramDataset(eval_frame, bundle.label_to_id, audio_config)
    else:
        train_ds = SpeechCommandsWaveformDataset(train_frame, bundle.label_to_id, audio_config)
        eval_ds = SpeechCommandsWaveformDataset(eval_frame, bundle.label_to_id, audio_config)

    return train_ds, eval_ds


def build_eval_dataset(bundle, model_name: ModelName, audio_config: AudioConfig, split: str):
    eval_frame = make_split_frame(bundle.frame, split)
    if model_name == "spec-cnn":
        return SpeechCommandsSpectrogramDataset(eval_frame, bundle.label_to_id, audio_config)
    return SpeechCommandsWaveformDataset(eval_frame, bundle.label_to_id, audio_config)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        batch_size = len(batch_x)
        total_loss += float(loss.item()) * batch_size
        correct += int((preds == batch_y).sum().item())
        total += batch_size
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        preds = logits.argmax(dim=1)
        batch_size = len(batch_x)
        total_loss += float(loss.item()) * batch_size
        total += batch_size
        y_true.extend(batch_y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return EvalResult(avg_loss, accuracy, precision, macro_f1, recall, y_true, y_pred)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def save_weights(path: str | Path, model: nn.Module) -> None:
    weights_path = Path(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)


def save_history(path: Path, history: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def fit_model(
    processed_dir: Path,
    results_dir: Path,
    config: FitConfig,
    device: torch.device,
) -> dict[str, float | int | str]:
    bundle = load_bundle(processed_dir)
    audio_config = AudioConfig()
    train_ds, val_ds = build_datasets(bundle, config, audio_config, split="val")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_model(config.model_name, num_classes=len(bundle.label_to_id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    checkpoint_path = results_dir / "checkpoints" / f"{config.model_name}.pt"
    weights_path = results_dir / "weights" / f"{config.model_name}_weights.pt"
    history_path = results_dir / "metrics" / f"{config.model_name}_history.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = -1.0
    best_metrics: dict[str, float | int | str] = {}
    history: list[dict[str, float | int | float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_result.loss, 6),
            "val_accuracy": round(val_result.accuracy, 6),
            "val_macro_f1": round(val_result.macro_f1, 6),
        }
        history.append(row)

        if val_result.accuracy >= best_val_accuracy:
            best_val_accuracy = val_result.accuracy
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
                "val_precision": val_result.precision,
                "val_macro_f1": val_result.macro_f1,
                "val_recall": val_result.recall,
                "model_name": config.model_name,
                "num_classes": len(bundle.label_to_id),
            }
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_metrics)
            save_weights(weights_path, model)

        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_result.loss:.4f} | val_acc={val_result.accuracy:.4f} | "
            f"val_macro_f1={val_result.macro_f1:.4f}"
        )

    save_history(history_path, history)
    return best_metrics


def run_from_cli(args: dict[str, object]) -> dict[str, float | int | str]:
    repo_root = Path(__file__).resolve().parents[3]
    paths = ProjectPaths(root=repo_root)
    processed_dir = Path(args["processed_dir"]) if args.get("processed_dir") else paths.data_processed
    results_dir = Path(args["results_dir"]) if args.get("results_dir") else paths.results
    fit_config = FitConfig(
        model_name=args["model_name"],
        epochs=int(args["epochs"]),
        batch_size=int(args["batch_size"]),
        learning_rate=float(args["learning_rate"]),
        weight_decay=float(args["weight_decay"]),
        num_workers=int(args["num_workers"]),
        seed=int(args["seed"]),
        max_train_samples=int(args["max_train_samples"]),
        max_val_samples=int(args["max_val_samples"]),
    )

    set_seed(fit_config.seed)
    device = select_device(str(args.get("device") or "auto"))
    print(f"Using device: {device}")
    print(f"Target classes: {', '.join(DEFAULT_CLASSES)}")
    metrics = fit_model(processed_dir, results_dir, fit_config, device)
    print(f"Best metrics: {metrics}")
    return metrics
