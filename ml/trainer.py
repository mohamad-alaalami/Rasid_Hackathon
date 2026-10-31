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
    """Train one registered classification or segmentation experiment."""
    experiment_id = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    try:
        cfg = TrainingConfig.from_mapping(config)
        selected_model = cfg.validate_for_dataset(dataset_id)
    except Exception as exc:
        return {
            "status": "failed",
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "error_type": _error_type(exc),
            "message": str(exc),
        }

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
        from .evaluate import evaluate_model
        from .model_registry import get_model_spec, model_runtime_status
        from .models import create_model, model_logits

        _seed_everything(cfg.seed)
        device = _select_device(cfg.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        data = create_dataloaders(dataset_id, cfg)
        model_spec = get_model_spec(selected_model)
        model_environment = model_runtime_status(selected_model, cfg.checkpoint_dir)
        pretrained_checkpoint = cfg.pretrained_checkpoint
        if pretrained_checkpoint is None and model_spec.checkpoint_filename:
            pretrained_checkpoint = str(
                Path(cfg.checkpoint_dir).expanduser() / model_spec.checkpoint_filename
            )
        model = create_model(
            selected_model,
            data.task_type,
            data.num_classes,
            data.input_channels,
            cfg.pretrained,
            pretrained_checkpoint,
        ).to(device)
        criterion_kwargs = {}
        if data.ignore_index is not None:
            criterion_kwargs["ignore_index"] = data.ignore_index
        criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
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
                    logits = model_logits(model(images), data.task_type)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                item_count = labels.size(0)
                loss_total += float(loss.item()) * item_count
                seen += item_count
            scheduler.step()
            validation = evaluate_model(
                model,
                data.val_loader,
                criterion,
                device,
                data.task_type,
                data.num_classes,
                data.ignore_index,
            )
            score = validation[data.primary_metric]
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": loss_total / seen,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"val_{key}": value for key, value in validation.items()},
            }
            history.append(epoch_metrics)
            _json_write(metrics_path, history)
            if score > best_accuracy:
                best_accuracy = score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": selected_model,
                        "dataset_id": dataset_id,
                        "task_type": data.task_type,
                        "num_classes": data.num_classes,
                        "class_names": data.class_names,
                        "preprocessing": data.preprocessing,
                        "pretraining": {
                            "source": model_spec.pretraining_source,
                            "checkpoint_id": model_spec.checkpoint_id,
                            "license": model_spec.license,
                            "installed_package_version": model_environment["installed_version"],
                        },
                        "config": cfg.to_dict(),
                        "seed": cfg.seed,
                        "best_epoch": best_epoch,
                        "primary_metric": data.primary_metric,
                        "best_val_score": best_accuracy,
                    },
                    checkpoint_path,
                )

        result = {
            "status": "completed",
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "model": selected_model,
            "task_type": data.task_type,
            "input_mode": cfg.input_mode,
            "best_epoch": best_epoch,
            "primary_metric": data.primary_metric,
            "best_val_score": best_accuracy,
            "training_seconds": round(time.perf_counter() - started, 3),
            "device": str(device),
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "config_path": str(config_path),
            "pretraining_source": model_spec.pretraining_source,
            "pretrained_checkpoint_id": model_spec.checkpoint_id,
            "preprocessing": data.preprocessing,
            "model_environment": model_environment,
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_gpu_memory_mb"] = round(
                torch.cuda.max_memory_allocated(device) / (1024**2), 2
            )
        else:
            result["peak_gpu_memory_mb"] = None
        result[f"best_val_{data.primary_metric}"] = best_accuracy
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

