"""Verified dataset contracts used by the generic GEO-Bench pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    task_type: str
    benchmark_name: str
    num_classes: int
    available_channels: int
    band_indices: tuple[int, ...]
    scale_divisor: float
    primary_metric: str
    native_resolution: tuple[int, int]
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
        available_channels=13,
        # Preserve the verified ImageNet-pretrained RGB baseline. The dataset
        # contains 13 bands, but this experiment profile selects B04/B03/B02.
        band_indices=(3, 2, 1),
        scale_divisor=10000.0,
        primary_metric="accuracy",
        native_resolution=(64, 64),
    ),
    "m-pv4ger": DatasetSpec(
        dataset_id="m-pv4ger",
        task_type="classification",
        benchmark_name="classification_v1.0",
        num_classes=2,
        available_channels=3,
        band_indices=(0, 1, 2),  # native aerial RGB order
        scale_divisor=255.0,
        primary_metric="accuracy",
        native_resolution=(320, 320),
    ),
    "m-pv4ger-seg": DatasetSpec(
        dataset_id="m-pv4ger-seg",
        task_type="segmentation",
        benchmark_name="segmentation_v1.0",
        num_classes=2,
        available_channels=3,
        band_indices=(0, 1, 2),
        scale_divisor=255.0,
        primary_metric="mean_iou",
        native_resolution=(320, 320),
    ),
    "m-chesapeake": DatasetSpec(
        dataset_id="m-chesapeake",
        task_type="segmentation",
        benchmark_name="segmentation_v1.0",
        num_classes=7,
        available_channels=4,
        band_indices=(0, 1, 2, 3),  # native aerial RGB+NIR order
        scale_divisor=255.0,
        primary_metric="mean_iou",
        native_resolution=(256, 256),
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
            "available_channels": spec.available_channels,
            "input_channels": spec.input_channels,
            "primary_metric": spec.primary_metric,
            "native_resolution": list(spec.native_resolution),
            "supported_resolution_options": {
                "default": "native",
                "override": "explicit square image_size >= 32",
            },
        }
        for spec in DATASET_REGISTRY.values()
    ]
