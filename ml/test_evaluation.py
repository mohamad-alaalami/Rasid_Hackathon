"""One-time final test-split evaluation of a saved GoTrain checkpoint.

Campaign tools never import this module. It runs only through the
human-authorized ``python -m storage final-test`` command, which allows one
test evaluation per dataset and only for the promoted champion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_checkpoint_on_test(
    checkpoint_path: str | Path,
    *,
    benchmark_dir: str | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    device: str = "auto",
) -> dict[str, Any]:
    import torch

    from .dataset import GeoBenchDataset, find_task
    from .evaluate import evaluate_model
    from .models import create_model
    from .registry import get_dataset_spec
    from .trainer import _select_device

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dataset_id = checkpoint["dataset_id"]
    spec = get_dataset_spec(dataset_id)
    preprocessing = checkpoint["preprocessing"]
    band_indices = tuple(preprocessing["band_indices"])
    # The checkpoint already holds every trained weight, so no pretrained
    # download is needed; the architecture is rebuilt identically.
    model = create_model(
        checkpoint["model_name"],
        checkpoint["task_type"],
        checkpoint["num_classes"],
        len(band_indices),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_device = _select_device(device)
    model.to(selected_device)

    task = find_task(dataset_id, benchmark_dir)
    # Same preprocessing the model was trained and validated with.
    test_dataset = GeoBenchDataset(
        task.get_dataset(split="test", band_names=None),
        spec,
        band_indices,
        preprocessing["normalization"],
        tuple(preprocessing["effective_resolution"]),
        augment=False,
    )
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    criterion_kwargs = {}
    if spec.ignore_index is not None:
        criterion_kwargs["ignore_index"] = spec.ignore_index
    metrics = evaluate_model(
        model,
        loader,
        torch.nn.CrossEntropyLoss(**criterion_kwargs),
        selected_device,
        checkpoint["task_type"],
        checkpoint["num_classes"],
        spec.ignore_index,
    )
    return {
        "split": "test",
        "dataset_id": dataset_id,
        "model": checkpoint["model_name"],
        "num_samples": len(test_dataset),
        "primary_metric": checkpoint["primary_metric"],
        "test_score": metrics[checkpoint["primary_metric"]],
        "metrics": metrics,
        "best_val_score": checkpoint["best_val_score"],
        "best_epoch": checkpoint["best_epoch"],
    }
