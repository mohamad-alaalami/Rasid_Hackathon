"""Deterministic training engine exposed to the future Strands tool.

Continuation (Phase 1D): ``cfg.epochs`` is the candidate's training horizon
(it is part of the solution identity and sets the learning-rate schedule
length). One call trains epochs ``start..until_epoch`` of that horizon. With
``save_training_state`` the call also writes ``training_state.pt``: model,
optimizer, scheduler, AMP scaler, RNG states, epoch history, best-so-far, and
early-stopping state at the end of the call. A later call with ``resume_from``
(that file's path and SHA-256) continues the same training run into a new
experiment directory; earlier directories, including the best checkpoint, are
never written again. Continuation is not initialization: the solution identity
is unchanged and the run is one continuous training trajectory.
"""

from __future__ import annotations

import json
import os
import random
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import TrainingConfig

STATE_FORMAT = "gotrain.training_state/1"
STATE_FILENAME = "training_state.pt"


class ResumeError(ValueError):
    """The training state cannot be resumed (missing, modified, or mismatched)."""


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rng_state(loader_generator) -> dict[str, Any]:
    import numpy as np
    import torch

    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [name, torch.tensor(keys.astype(np.int64)), int(position), int(has_gauss),
                  float(cached)],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_loader": None if loader_generator is None else loader_generator.get_state(),
    }


def _restore_rng(state: Mapping[str, Any], loader_generator) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
    if loader_generator is not None and state["train_loader"] is not None:
        loader_generator.set_state(state["train_loader"])


def load_training_state(path, expected_sha256: str) -> dict[str, Any]:
    """Verify the file's SHA-256 before deserializing it (tensors and plain data only)."""
    import torch

    from .solution import file_sha256

    path = Path(path)
    if not path.is_file():
        raise ResumeError(f"training state {path} does not exist")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ResumeError(f"training state {path} was modified or corrupted "
                          f"(sha256 {actual[:16]}... != recorded {expected_sha256[:16]}...)")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != STATE_FORMAT:
        raise ResumeError(f"{path} is not a {STATE_FORMAT} file")
    return state


def _atomic_save(value, path: Path) -> None:
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


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
    *,
    until_epoch: int | None = None,
    resume_from: Mapping[str, Any] | None = None,
    save_training_state: bool = False,
    max_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Train one registered classification or segmentation experiment.

    Every run is recorded against an immutable SolutionSpec (ml/solution.py).
    ``solution_context`` carries optional lineage and campaign metadata.
    ``until_epoch`` stops after that epoch of the ``config["epochs"]`` horizon;
    ``resume_from`` ({"path", "sha256"}) continues a saved training state;
    ``max_runtime_seconds`` stops at an epoch boundary when the next epoch would
    exceed it (measured epoch time; the first epoch of a fresh run cannot be
    predicted and always runs). Defaults reproduce the historical behavior.
    """
    experiment_id = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    resume_state = None
    try:
        from .solution import build_solution_spec, propose_solution, solution_exists
        from .solution import solution_id as spec_solution_id

        cfg = TrainingConfig.from_mapping(config)
        selected_model = cfg.validate_for_dataset(dataset_id)
        target_epoch = cfg.epochs if until_epoch is None else until_epoch
        if type(target_epoch) is not int or not 1 <= target_epoch <= cfg.epochs:
            raise ValueError(f"until_epoch must be an integer in 1..{cfg.epochs} (the horizon).")
        spec = build_solution_spec(dataset_id, cfg)
        if resume_from is None:
            solution_id = propose_solution(spec, solution_context)
        else:
            # Same candidate, same training run: no new proposal, no new identity.
            solution_id = spec_solution_id(spec)
            if not solution_exists(solution_id):
                raise ResumeError(f"unknown solution {solution_id}")
            resume_state = load_training_state(resume_from["path"], resume_from["sha256"])
            mismatch = [
                name for name, expected in (("solution_id", solution_id),
                                            ("dataset_id", dataset_id),
                                            ("horizon_epochs", cfg.epochs))
                if resume_state.get(name) != expected
            ]
            if mismatch:
                raise ResumeError(f"training state does not belong to this candidate: {mismatch}")
            if resume_state["finished"]:
                raise ResumeError(f"training run already finished ({resume_state['finish_reason']})")
            if target_epoch <= resume_state["completed_epochs"]:
                raise ResumeError(f"until_epoch {target_epoch} must exceed the "
                                  f"{resume_state['completed_epochs']} completed epochs")
    except Exception as exc:
        return {
            "status": "failed",
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "error_type": "RESUME_STATE_INVALID" if isinstance(exc, ResumeError) else _error_type(exc),
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
        epoch_seconds: list[float] = []
        best_accuracy = -1.0
        best_epoch = 0
        best_checkpoint = None  # {"path", "sha256"} once a best epoch exists
        early_stopping_reference = None
        epochs_without_improvement = 0
        stopped_early = False
        stop_reason = None
        stopped_for_time = False
        start_epoch = 1
        resumed = None
        loader_generator = getattr(data.train_loader, "generator", None)
        if resume_state is not None:
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            scaler.load_state_dict(resume_state["scaler_state_dict"])
            history = list(resume_state["history"])
            epoch_seconds = list(resume_state["epoch_seconds"])
            best_accuracy = resume_state["best"]["score"]
            best_epoch = resume_state["best"]["epoch"]
            best_checkpoint = resume_state["best"]["checkpoint"]
            early_stopping_reference = resume_state["early_stopping"]["reference"]
            epochs_without_improvement = resume_state["early_stopping"]["epochs_without_improvement"]
            start_epoch = resume_state["completed_epochs"] + 1
            resumed = {"path": str(resume_from["path"]), "sha256": resume_from["sha256"],
                       "from_experiment_id": resume_state["experiment_id"],
                       "completed_epochs": resume_state["completed_epochs"]}
            _restore_rng(resume_state["rng"], loader_generator)
            del resume_state
        segment_started = time.perf_counter()
        for epoch in range(start_epoch, target_epoch + 1):
            if max_runtime_seconds is not None and epoch_seconds and (
                time.perf_counter() - segment_started + max(epoch_seconds) > max_runtime_seconds
            ):
                stopped_for_time = True
                stop_reason = (f"time allocation: another epoch (measured up to "
                               f"{max(epoch_seconds):.1f}s) would exceed {max_runtime_seconds:.1f}s")
                break
            epoch_started = time.perf_counter()
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
            epoch_seconds.append(round(time.perf_counter() - epoch_started, 3))
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
                best_checkpoint = {"path": str(checkpoint_path), "sha256": None}
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

        from .solution import file_sha256

        completed_through = len(history)
        segment_epochs = completed_through - start_epoch + 1
        finished = stopped_early or completed_through >= cfg.epochs
        finish_reason = ("early_stopping" if stopped_early
                         else "horizon_reached" if completed_through >= cfg.epochs else None)
        if not stopped_early and not stopped_for_time and completed_through < cfg.epochs:
            stop_reason = f"allocation reached: epoch {completed_through} of {cfg.epochs}"
        if best_checkpoint is not None and best_checkpoint["path"] == str(checkpoint_path):
            best_checkpoint["sha256"] = file_sha256(checkpoint_path)
        best_integrity = None
        if best_checkpoint is not None:
            # A checkpoint from an earlier segment is referenced, never rewritten; verify it.
            best_file = Path(best_checkpoint["path"])
            best_integrity = ("verified" if best_file.is_file()
                              and file_sha256(best_file) == best_checkpoint["sha256"] else "mismatch")
        new_best_this_segment = checkpoint_path.is_file()
        training_state = None
        if save_training_state:
            state_path = experiment_dir / STATE_FILENAME
            _atomic_save({
                "format": STATE_FORMAT,
                "solution_id": solution_id,
                "dataset_id": dataset_id,
                "experiment_id": experiment_id,
                "horizon_epochs": cfg.epochs,
                "completed_epochs": completed_through,
                "finished": finished,
                "finish_reason": finish_reason,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng": _rng_state(loader_generator),
                "history": history,
                "epoch_seconds": epoch_seconds,
                "best": {"score": best_accuracy, "epoch": best_epoch, "checkpoint": best_checkpoint},
                "early_stopping": {"reference": early_stopping_reference,
                                   "epochs_without_improvement": epochs_without_improvement},
                "resumed_from": resumed,
                "config": cfg.to_dict(),
            }, state_path)
            training_state = {"path": str(state_path), "sha256": file_sha256(state_path)}

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
            # Epochs this call was asked to train / actually trained; for a fresh,
            # unallocated run both equal the historical values.
            "requested_epochs": target_epoch - start_epoch + 1,
            "completed_epochs": segment_epochs,
            "horizon_epochs": cfg.epochs,
            "start_epoch": start_epoch,
            "completed_through_epoch": completed_through,
            "finished": finished,
            "finish_reason": finish_reason,
            "stopped_early": stopped_early,
            "stopped_for_time": stopped_for_time,
            "stop_reason": stop_reason,
            "primary_metric": data.primary_metric,
            "best_val_score": best_accuracy,
            "training_seconds": round(time.perf_counter() - started, 3),
            "epoch_seconds": epoch_seconds,
            "device": str(device),
            "checkpoint_path": best_checkpoint["path"] if best_checkpoint else str(checkpoint_path),
            "checkpoint_sha256": best_checkpoint["sha256"] if best_checkpoint else None,
            "checkpoint_integrity": best_integrity,
            "new_best_this_segment": new_best_this_segment,
            "training_state": training_state,
            "resumed_from": resumed,
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
        if new_best_this_segment:
            result.update(_publish_model_version(result, experiment_dir))
        else:  # the best checkpoint lives in an earlier segment and was versioned there
            result["model_version_skipped"] = "no new best checkpoint in this continuation"
        _json_write(experiment_dir / "result.json", result)

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
                "requested_epochs": result["requested_epochs"],
                "completed_epochs": segment_epochs,
                "stopped_early": stopped_early,
            },
            "continuation": {
                "horizon_epochs": cfg.epochs,
                "start_epoch": start_epoch,
                "completed_through_epoch": completed_through,
                "finished": finished,
                "resumed_from": resumed,
            },
            "artifacts": {
                "experiment_dir": str(experiment_dir),
                "checkpoint_path": result["checkpoint_path"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "metrics_path": str(metrics_path),
                "config_path": str(config_path),
                "optimizer_state_path": training_state["path"] if training_state else None,
                "training_state_sha256": training_state["sha256"] if training_state else None,
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
            # A failed continuation leaves its source state and every earlier checkpoint as-is.
            "resumed_from": dict(resume_from) if resume_from else None,
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

