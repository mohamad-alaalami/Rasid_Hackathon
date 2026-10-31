"""One-batch runtime verification for a registered dataset/model pair."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from .config import TrainingConfig


def run_smoke_test(
    dataset_id: str,
    *,
    batch_size: int = 2,
    benchmark_dir: str | None = None,
    pretrained: bool = True,
) -> dict[str, Any]:
    """Run one train batch and one validation batch without touching test data."""
    import torch

    from .dataset import create_dataloaders
    from .evaluate import evaluate_model
    from .models import create_model, model_logits

    config = TrainingConfig.from_mapping(
        {
            "model": "auto",
            "pretrained": pretrained,
            "input_mode": "registry",
            "epochs": 1,
            "batch_size": batch_size,
            "num_workers": 0,
            "mixed_precision": True,
            "device": "auto",
            "benchmark_dir": benchmark_dir,
        }
    )
    model_name = config.validate_for_dataset(dataset_id)
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
    ).to(device)
    criterion_kwargs = {}
    if data.ignore_index is not None:
        criterion_kwargs["ignore_index"] = data.ignore_index
    criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_images, train_targets = next(iter(data.train_loader))
    train_images = train_images.to(device)
    train_targets = train_targets.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_logits = model_logits(model(train_images), data.task_type)
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

    result = {
        "status": "completed",
        "dataset_id": dataset_id,
        "task_type": data.task_type,
        "model": model_name,
        "device": str(device),
        "input_channels": data.input_channels,
        "num_classes": data.num_classes,
        "train_image_shape": list(train_images.shape),
        "train_target_shape": list(train_targets.shape),
        "output_shape": list(train_logits.shape),
        "train_loss": float(train_loss.item()),
        "validation": validation,
        "primary_metric": data.primary_metric,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_gpu_memory_mb": peak_memory_mb,
        "test_split_accessed": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--benchmark-dir")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    try:
        result = run_smoke_test(
            args.dataset_id,
            batch_size=args.batch_size,
            benchmark_dir=args.benchmark_dir,
            pretrained=not args.no_pretrained,
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
