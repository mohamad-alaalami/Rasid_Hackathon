"""Verified dataset contracts used by the generic GEO-Bench pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    task_type: str
    benchmark_name: str
    num_classes: int
    band_indices: tuple[int, ...]
    scale_divisor: float
    default_model: str
    primary_metric: str
    ignore_index: int | None = None

    @property
    def input_channels(self) -> int:
        return len(self.band_indices)


# Adding another compatible GEO-Bench classification or semantic-segmentation
# dataset requires one verified entry here; the rest of the pipeline is generic.
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "m-eurosat": DatasetSpec(
        dataset_id="m-eurosat",
        task_type="classification",
        benchmark_name="classification_v1.0",
        num_classes=10,
        # All 13 channels, reordered so the pretrained RGB stem receives
        # verified B04/B03/B02 first; remaining channels follow afterward.
        band_indices=(3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        scale_divisor=10000.0,
        default_model="resnet18",
        primary_metric="accuracy",
    ),
    "m-pv4ger": DatasetSpec(
        dataset_id="m-pv4ger",
        task_type="classification",
        benchmark_name="classification_v1.0",
        num_classes=2,
        band_indices=(0, 1, 2),  # native aerial RGB order
        scale_divisor=255.0,
        default_model="resnet18",
        primary_metric="accuracy",
    ),
    "m-pv4ger-seg": DatasetSpec(
        dataset_id="m-pv4ger-seg",
        task_type="segmentation",
        benchmark_name="segmentation_v1.0",
        num_classes=2,
        band_indices=(0, 1, 2),
        scale_divisor=255.0,
        default_model="fcn_resnet50",
        primary_metric="mean_iou",
    ),
    "m-chesapeake": DatasetSpec(
        dataset_id="m-chesapeake",
        task_type="segmentation",
        benchmark_name="segmentation_v1.0",
        num_classes=7,
        band_indices=(0, 1, 2, 3),  # native aerial RGB+NIR order
        scale_divisor=255.0,
        default_model="fcn_resnet50",
        primary_metric="mean_iou",
    ),
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[dataset_id]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unsupported dataset_id={dataset_id!r}; registered datasets: {supported}."
        ) from exc


def list_dataset_specs() -> list[dict]:
    return [
        {
            "dataset_id": spec.dataset_id,
            "task_type": spec.task_type,
            "benchmark_name": spec.benchmark_name,
            "num_classes": spec.num_classes,
            "input_channels": spec.input_channels,
            "default_model": spec.default_model,
            "primary_metric": spec.primary_metric,
        }
        for spec in DATASET_REGISTRY.values()
    ]
