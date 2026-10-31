"""Validation metrics for classification and semantic segmentation."""

from __future__ import annotations

from .models import model_logits


def evaluate_model(model, loader, criterion, device, task_type, num_classes, ignore_index=None):
    import torch

    model.eval()
    loss_total = 0.0
    weight_total = 0
    correct = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model_logits(model(images), task_type)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            if task_type == "classification":
                weight = targets.size(0)
                correct += int((predictions == targets).sum().item())
            else:
                valid = torch.ones_like(targets, dtype=torch.bool)
                if ignore_index is not None:
                    valid &= targets != ignore_index
                target_valid = targets[valid].detach().cpu()
                pred_valid = predictions[valid].detach().cpu()
                weight = int(valid.sum().item())
                if weight:
                    bins = torch.bincount(
                        target_valid * num_classes + pred_valid,
                        minlength=num_classes * num_classes,
                    )
                    confusion += bins.reshape(num_classes, num_classes)
            loss_total += float(loss.item()) * weight
            weight_total += weight
    if weight_total == 0:
        raise ValueError("Validation loader contains no valid targets.")
    if task_type == "classification":
        return {"loss": loss_total / weight_total, "accuracy": correct / weight_total}
    intersection = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    valid_classes = union > 0
    per_class = torch.where(valid_classes, intersection / union.clamp_min(1), torch.nan)
    return {
        "loss": loss_total / weight_total,
        "mean_iou": float(per_class[valid_classes].mean().item()),
        "pixel_accuracy": float(intersection.sum().item() / confusion.sum().item()),
        "per_class_iou": [None if torch.isnan(x) else float(x) for x in per_class],
        "absent_class_policy": "exclude classes with zero union from mean_iou",
    }


def evaluate_classifier(model, loader, criterion, device):
    """Backward-compatible wrapper."""
    return evaluate_model(model, loader, criterion, device, "classification", 1)
