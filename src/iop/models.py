from __future__ import annotations

import torch
from torch import nn


class Conv1DClassifier(nn.Module):
    def __init__(self, num_classes: int, input_length: int = 16_000, base_channels: int = 32, stride: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=80, stride=stride),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(base_channels, base_channels, kernel_size=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(base_channels, 2 * base_channels, kernel_size=3),
            nn.BatchNorm1d(2 * base_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * base_channels, 2 * base_channels, kernel_size=3),
            nn.BatchNorm1d(2 * base_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * base_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self.input_length = input_length
        self.base_channels = base_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x)


class Conv1DLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        stride: int = 16,
    ) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=80, stride=stride),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(base_channels, base_channels, kernel_size=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(
            input_size=base_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.25 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        x = self.frontend(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.head(x[:, -1, :])


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.net(x)
