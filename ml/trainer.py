"""Deterministic training engine exposed to the future Strands tool."""

from __future__ import annotations

import json
import random
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import TrainingConfig


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _error_type(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text:
        return "CUDA_OUT_OF_MEMORY"
    if isinstance(exc, FileNotFoundError):
        return "DATASET_NOT_FOUND"
    if isinstance(exc, (TypeError, ValueError)):
        return "INVALID_CONFIGURATION_OR_DATA"
    if isinstance(exc, ImportError) or "not installed" in text or "dependencies" in text:
        return "DEPENDENCY_ERROR"
    return "TRAINING_ERROR"


def train_model(dataset_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Train one controlled classification experiment and return JSON-safe metadata."""
    cfg = TrainingConfig.from_mapping(config)
    if dataset_id != "m-eurosat":
        raise ValueError("Initial engine supports only dataset_id='m-eurosat'.")

    experiment_id = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    experiment_dir = Path(cfg.output_dir).expanduser().resolve() / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    config_path = experiment_dir / "config.json"
    metrics_path = experiment_dir / "metrics.json"
    checkpoint_path = experiment_dir / "best.pt"
    log_path = experiment_dir / "error.log"
    _json_write(config_path, {"dataset_id": dataset_id, **cfg.to_dict()})
    started = time.perf_counter()

    try:
        import torch

        from .dataset import create_dataloaders
        from .evaluate import evaluate_classifier
        from .models import create_model

        _seed_everything(cfg.seed)
        device = _select_device(cfg.device)
        data = create_dataloaders(dataset_id, cfg)
        model = create_model(cfg.model, data.num_classes, cfg.pretrained).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        amp_enabled = bool(cfg.mixed_precision and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        history: list[dict[str, Any]] = []
        best_accuracy = -1.0
        best_epoch = 0
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            loss_total = 0.0
            seen = 0
            for images, labels in data.train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_size = labels.size(0)
                loss_total += float(loss.item()) * batch_size
                seen += batch_size
            scheduler.step()
            validation = evaluate_classifier(model, data.val_loader, criterion, device)
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": loss_total / seen,
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_metrics)
            _json_write(metrics_path, history)
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": cfg.model,
                        "dataset_id": dataset_id,
                        "num_classes": data.num_classes,
                        "class_names": data.class_names,
                        "preprocessing": data.preprocessing,
                        "config": cfg.to_dict(),
                        "seed": cfg.seed,
                        "best_epoch": best_epoch,
                        "best_val_accuracy": best_accuracy,
                    },
                    checkpoint_path,
                )

        result = {
            "status": "completed",
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "model": cfg.model,
            "input_mode": cfg.input_mode,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_accuracy,
            "training_seconds": round(time.perf_counter() - started, 3),
            "device": str(device),
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "config_path": str(config_path),
        }
        _json_write(experiment_dir / "result.json", result)
        return result
    except Exception as exc:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        result = {
            "status": "failed",
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "error_type": _error_type(exc),
            "message": str(exc),
            "training_seconds": round(time.perf_counter() - started, 3),
            "log_path": str(log_path),
            "config_path": str(config_path),
        }
        _json_write(experiment_dir / "result.json", result)
        return result

