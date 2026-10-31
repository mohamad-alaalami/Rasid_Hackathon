"""Validation-only evaluation. This module never loads the test split."""

from __future__ import annotations


def evaluate_classifier(model, loader, criterion, device) -> dict[str, float]:
    import torch

    model.eval()
    loss_total = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            loss_total += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += batch_size
    if count == 0:
        raise ValueError("Validation loader is empty.")
    return {"loss": loss_total / count, "accuracy": correct / count}

