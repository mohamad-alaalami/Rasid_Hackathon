"""One-batch runtime verification for a registered dataset/model pair."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from typing import Any

from .config import TrainingConfig


def run_smoke_test(
    dataset_id: str,
    *,
    model_id: str,
    batch_size: int = 2,
    benchmark_dir: str | None = None,
    checkpoint_dir: str = "checkpoints",
    pretrained_checkpoint: str | None = None,
    pretrained: bool = True,
    image_size: int | None = None,
    pipeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one train batch and one validation batch without touching test data."""
    import torch

    from .dataset import create_dataloaders
    from .evaluate import evaluate_model
    from .components import build_training_components, close_criterion
    from .models import apply_head_dropout, create_model, model_logits, validate_model_output_shape
    from .model_registry import get_model_spec

    config = TrainingConfig.from_mapping(
        {
            "model": model_id,
            "pretrained": pretrained,
            "input_mode": "registry",
            "epochs": 1,
            "batch_size": batch_size,
            "num_workers": 0,
            "mixed_precision": True,
            "device": "auto",
            "benchmark_dir": benchmark_dir,
            "checkpoint_dir": checkpoint_dir,
            "pretrained_checkpoint": pretrained_checkpoint,
            "image_size": image_size,
            "pipeline": pipeline,
        }
    )
    model_name = config.validate_for_dataset(dataset_id)
    model_spec = get_model_spec(model_name)
    checkpoint_path = config.pretrained_checkpoint
    if checkpoint_path is None and model_spec.checkpoint_filename:
        checkpoint_path = f"{config.checkpoint_dir}/{model_spec.checkpoint_filename}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    data = create_dataloaders(dataset_id, config)
    model = create_model(
        model_name,
        data.task_type,
        data.num_classes,
        data.input_channels,
        config.pretrained,
        checkpoint_path,
        rgb_positions=data.rgb_positions,
    ).to(device)
    apply_head_dropout(model, data.pipeline["dropout"])
    criterion, optimizer, _ = build_training_components(
        data.pipeline, model, num_classes=data.num_classes, ignore_index=data.ignore_index,
        class_counts=data.class_counts, learning_rate=config.learning_rate,
        weight_decay=config.weight_decay, epochs=1, device=device, task_type=data.task_type,
    )

    train_images, train_targets = next(iter(data.train_loader))
    train_images = train_images.to(device)
    train_targets = train_targets.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_logits = model_logits(model(train_images), data.task_type)
    validate_model_output_shape(train_logits, train_targets, data.task_type)
    train_loss = criterion(train_logits, train_targets)
    train_loss.backward()
    optimizer.step()

    validation_batch = next(iter(data.val_loader))
    validation = evaluate_model(
        model,
        [validation_batch],
        criterion,
        device,
        data.task_type,
        data.num_classes,
        data.ignore_index,
    )
    peak_memory_mb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)

    with tempfile.TemporaryDirectory(prefix="gotrain-smoke-") as temporary:
        checkpoint_path = f"{temporary}/model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        reloaded = create_model(
            model_name,
            data.task_type,
            data.num_classes,
            data.input_channels,
            pretrained=False,
        ).to(device)
        reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device))

    result = {
        "status": "completed",
        "dataset_id": dataset_id,
        "task_type": data.task_type,
        "model": model_name,
        "device": str(device),
        "input_channels": data.input_channels,
        "band_names": data.preprocessing["band_names"],
        "excluded_channels": data.preprocessing["excluded_channels"],
        "rgb_positions": data.preprocessing["rgb_positions"],
        "normalization": (
            data.preprocessing["normalization"]
            if isinstance(data.preprocessing["normalization"], str)
            else {key: data.preprocessing["normalization"][key]
                  for key in ("type", "samples", "mean", "std", "constant_channels")}
        ),
        "input_padding": getattr(model, "gotrain_input_padding", None),
        "class_names": data.class_names,
        "pipeline": data.pipeline,
        "criterion": type(criterion).__name__,
        "optimizer": type(optimizer).__name__,
        "num_classes": data.num_classes,
        "native_resolution": list(data.native_resolution),
        "effective_resolution": list(data.effective_resolution),
        "resolution_override": image_size,
        "train_image_shape": list(train_images.shape),
        "train_target_shape": list(train_targets.shape),
        "output_shape": list(train_logits.shape),
        "train_loss": float(train_loss.item()),
        "validation": validation,
        "primary_metric": data.primary_metric,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_gpu_memory_mb": peak_memory_mb,
        "checkpoint_reload_verified": True,
        "test_split_accessed": False,
        "custom_loss": getattr(criterion, "component_id", None),
        "custom_loss_source_sha256": getattr(criterion, "source_sha256", None),
    }
    close_criterion(criterion)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--benchmark-dir")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument(
        "--image-size",
        type=int,
        help="Explicit square resize for a deliberate memory experiment; native is default.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--pipeline", type=json.loads,
                        help="Pipeline components as JSON (see ml/components.py).")
    args = parser.parse_args()
    try:
        result = run_smoke_test(
            args.dataset_id,
            model_id=args.model,
            batch_size=args.batch_size,
            benchmark_dir=args.benchmark_dir,
            checkpoint_dir=args.checkpoint_dir,
            pretrained_checkpoint=args.pretrained_checkpoint,
            pretrained=not args.no_pretrained,
            image_size=args.image_size,
            pipeline=args.pipeline,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "dataset_id": args.dataset_id,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "test_split_accessed": False,
        }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
