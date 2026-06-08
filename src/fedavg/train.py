from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def train_local(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("lr", 0.05)), momentum=0.9)
    epochs = int(config.get("local_epochs", 1))

    total_loss = 0.0
    total_seen = 0
    started = time.perf_counter()
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size

    elapsed = time.perf_counter() - started
    return {
        "train_loss": total_loss / max(total_seen, 1),
        "train_time": elapsed,
        "samples": float(total_seen),
    }
