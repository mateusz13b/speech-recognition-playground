from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .config import AudioConfig
from .dataset import load_bundle
from .engine import build_eval_dataset, build_model, evaluate, select_device


def load_checkpoint(path: str | Path, model: nn.Module) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        return state
    model.load_state_dict(state)
    return {"model_state_dict": state}


def save_json(path: str | Path, payload: dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_confusion_outputs(
    matrix: np.ndarray,
    labels: list[str],
    csv_path: Path,
    png_path: Path,
    title: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    header = ",".join(["label", *labels])
    rows = [header]
    for label, row in zip(labels, matrix.tolist()):
        rows.append(",".join([label, *[str(int(value)) for value in row]]))
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def save_history_plots(history_path: Path, plots_dir: Path, model_name: str) -> None:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    val_accuracy = [row["val_accuracy"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train loss")
    ax.plot(epochs, val_loss, label="val loss")
    ax.set_title(f"Loss curves: {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{model_name}_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, val_accuracy, label="val accuracy")
    ax.set_title(f"Validation accuracy: {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{model_name}_val_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_accuracy, label="train accuracy")
    ax.plot(epochs, val_accuracy, label="val accuracy")
    ax.set_title(f"Accuracy curves: {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{model_name}_accuracy.png", dpi=150)
    plt.close(fig)


def evaluate_model(
    processed_dir: Path,
    results_dir: Path,
    model_name: str,
    batch_size: int,
    device_name: str,
    max_test_samples: int = 0,
) -> dict[str, object]:
    bundle = load_bundle(processed_dir)
    labels = [label for label, _ in sorted(bundle.label_to_id.items(), key=lambda item: item[1])]
    dataset = build_eval_dataset(bundle, model_name, AudioConfig(), split="test")
    if max_test_samples and max_test_samples > 0:
        dataset.frame = dataset.frame.head(max_test_samples).reset_index(drop=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name, num_classes=len(labels))
    checkpoint_path = results_dir / "checkpoints" / f"{model_name}.pt"
    weights_path = results_dir / "weights" / f"{model_name}_weights.pt"
    if weights_path.exists():
        load_checkpoint(weights_path, model)
    elif checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model)
    else:
        raise FileNotFoundError(f"Missing model weights: {weights_path} or {checkpoint_path}")

    device = select_device(device_name)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    result = evaluate(model, loader, criterion, device)

    metrics = {
        "model_name": model_name,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "macro_f1": result.macro_f1,
        "loss": result.loss,
        "num_test_samples": len(dataset),
    }

    report = classification_report(
        result.y_true,
        result.y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(result.y_true, result.y_pred, labels=list(range(len(labels))))

    metrics_dir = results_dir / "metrics"
    plots_dir = results_dir / "plots"
    save_json(metrics_dir / f"{model_name}_test_metrics.json", metrics)
    save_json(metrics_dir / f"{model_name}_classification_report.json", report)
    save_confusion_outputs(
        matrix,
        labels,
        metrics_dir / f"{model_name}_confusion_matrix.csv",
        plots_dir / f"{model_name}_confusion_matrix.png",
        title=f"Confusion matrix: {model_name}",
    )

    history_path = metrics_dir / f"{model_name}_history.json"
    if history_path.exists():
        save_history_plots(history_path, plots_dir, model_name)

    return metrics


def main() -> None:
    raise SystemExit("Use iop.scripts.evaluate:main")
