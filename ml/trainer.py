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


def _update_early_stopping(score, reference_score, epochs_without_improvement, min_delta):
    """Update patience using a meaningful validation-score improvement."""
    if reference_score is None or score > reference_score + min_delta:
        return score, 0
    return reference_score, epochs_without_improvement + 1


def _publish_model_version(result: dict[str, Any], experiment_dir: Path) -> dict[str, Any]:
    """Version the run in $GOTRAIN_STORE when configured; returns {} otherwise."""
    from storage.model_versions import publish_experiment

    return publish_experiment(result, experiment_dir)


def _record_solution_run(solution_id, run: dict[str, Any]) -> None:
    from .solution import environment_info, record_run

    record_run(solution_id, {**run, "environment": environment_info()})


def train_model(
    dataset_id: str,
    config: Mapping[str, Any],
    solution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one registered classification or segmentation experiment.

    Every run is recorded against an immutable SolutionSpec (ml/solution.py).
    ``solution_context`` carries optional lineage and campaign metadata.
    """
    experiment_id = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    try:
        from .solution import build_solution_spec, propose_solution

        cfg = TrainingConfig.from_mapping(config)
        selected_model = cfg.validate_for_dataset(dataset_id)
        solution_id = propose_solution(build_solution_spec(dataset_id, cfg), solution_context)
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
    _json_write(config_path, {"dataset_id": dataset_id, "solution_id": solution_id, **cfg.to_dict()})
    started = time.perf_counter()
    data = None
    criterion = None
    custom_components: list[dict[str, Any]] = []

    try:
        import torch

        from .dataset import create_dataloaders
        from .evaluate import evaluate_model
        from .model_registry import get_model_spec, model_runtime_status
        from .components import build_training_components, class_weights
        from .models import (
            apply_head_dropout,
            create_model,
            model_logits,
            validate_model_output_shape,
        )

        _seed_everything(cfg.seed)
        device = _select_device(cfg.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        data = create_dataloaders(dataset_id, cfg)
        _json_write(
            config_path,
            {
                "dataset_id": dataset_id,
                "solution_id": solution_id,
                **cfg.to_dict(),
                "resolved_pipeline": data.pipeline,
                "native_resolution": list(data.native_resolution),
                "effective_resolution": list(data.effective_resolution),
            },
        )
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
            rgb_positions=data.rgb_positions,
        ).to(device)
        apply_head_dropout(model, data.pipeline["dropout"])
        criterion, optimizer, scheduler = build_training_components(
            data.pipeline,
            model,
            num_classes=data.num_classes,
            ignore_index=data.ignore_index,
            class_counts=data.class_counts,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            device=device,
            task_type=data.task_type,
        )
        if data.pipeline["loss"]["name"] == "custom":
            from .custom_components import archive_component

            # The exact generated code travels with the checkpoint.
            custom_components.append(archive_component(
                data.pipeline["loss"]["component_id"], experiment_dir / "custom_components"))
        weights = class_weights(
            data.class_counts, data.pipeline["loss"].get("class_weighting", "none"), data.ignore_index
        )
        weights = None if weights is None else [round(float(w), 6) for w in weights]
        amp_enabled = bool(cfg.mixed_precision and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        history: list[dict[str, Any]] = []
        best_accuracy = -1.0
        best_epoch = 0
        early_stopping_reference = None
        epochs_without_improvement = 0
        stopped_early = False
        stop_reason = None
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
                    validate_model_output_shape(logits, labels, data.task_type)
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
                "native_resolution": list(data.native_resolution),
                "effective_resolution": list(data.effective_resolution),
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
                        "input_padding": getattr(model, "gotrain_input_padding", None),
                        "solution_id": solution_id,
                        "custom_components": custom_components,
                        "pipeline": data.pipeline,
                        "class_counts": data.class_counts,
                        "class_weights": weights,
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
            early_stopping_reference, epochs_without_improvement = (
                _update_early_stopping(
                    score,
                    early_stopping_reference,
                    epochs_without_improvement,
                    cfg.early_stopping_min_delta,
                )
            )
            if (
                cfg.early_stopping
                and epochs_without_improvement >= cfg.early_stopping_patience
            ):
                stopped_early = True
                stop_reason = (
                    f"{data.primary_metric} did not improve by more than "
                    f"{cfg.early_stopping_min_delta} for "
                    f"{cfg.early_stopping_patience} consecutive epochs"
                )
                break

        result = {
            "status": "completed",
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "model": selected_model,
            "task_type": data.task_type,
            "input_mode": cfg.input_mode,
            "native_resolution": list(data.native_resolution),
            "effective_resolution": list(data.effective_resolution),
            "best_epoch": best_epoch,
            "requested_epochs": cfg.epochs,
            "completed_epochs": len(history),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
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
            "solution_id": solution_id,
            "pipeline": data.pipeline,
            "class_weights": weights,
            "custom_components": custom_components,
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_gpu_memory_mb"] = round(
                torch.cuda.max_memory_allocated(device) / (1024**2), 2
            )
        else:
            result["peak_gpu_memory_mb"] = None
        result[f"best_val_{data.primary_metric}"] = best_accuracy
        result.update(_publish_model_version(result, experiment_dir))
        _json_write(experiment_dir / "result.json", result)
        from .solution import file_sha256

        _record_solution_run(solution_id, {
            "run_id": experiment_id,
            "status": "completed",
            "campaign_id": (solution_context or {}).get("campaign_id"),
            "validation": {
                "split": "validation",
                "primary_metric": data.primary_metric,
                "best_val_score": best_accuracy,
                "best_epoch": best_epoch,
                "final_epoch": history[-1],
            },
            "resources": {
                "training_seconds": result["training_seconds"],
                "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                "device": result["device"],
                "requested_epochs": cfg.epochs,
                "completed_epochs": len(history),
                "stopped_early": stopped_early,
            },
            "artifacts": {
                "experiment_dir": str(experiment_dir),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "metrics_path": str(metrics_path),
                "config_path": str(config_path),
                "optimizer_state_path": None,  # not saved in Phase 1A
                "custom_components": custom_components,
            },
            "class_weights": weights,
            "dataset_fingerprint": data.dataset_fingerprint,
        })
        return result
    except Exception as exc:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        message = str(exc)
        if _error_type(exc) == "CUDA_OUT_OF_MEMORY":
            message += (
                " The engine kept the requested resolution unchanged. Retry with a smaller "
                "batch_size, or set image_size explicitly for a deliberate resize experiment."
            )
        result = {
            "status": "failed",
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "error_type": _error_type(exc),
            "message": message,
            "training_seconds": round(time.perf_counter() - started, 3),
            "log_path": str(log_path),
            "config_path": str(config_path),
        }
        if data is not None:
            result["native_resolution"] = list(data.native_resolution)
            result["effective_resolution"] = list(data.effective_resolution)
        result["solution_id"] = solution_id
        _json_write(experiment_dir / "result.json", result)
        try:
            _record_solution_run(solution_id, {
                "run_id": experiment_id,
                "status": "failed",
                "campaign_id": (solution_context or {}).get("campaign_id"),
                "error_type": result["error_type"],
                "error": message,
                "resources": {"training_seconds": result["training_seconds"]},
            })
        except Exception:  # the training failure is the error worth reporting
            pass
        return result
    finally:
        from .components import close_criterion

        close_criterion(criterion)  # stops a custom loss's sandbox worker

