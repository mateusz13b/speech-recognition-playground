from __future__ import annotations

from pathlib import Path

import torch


def predict(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        return torch.softmax(logits, dim=-1)


def load_checkpoint(path: str | Path, model: torch.nn.Module) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model
