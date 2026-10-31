"""Registry-driven GEO-Bench classification and segmentation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import DatasetSpec, get_dataset_spec

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Verified RGB view retained for compatibility and optional visualisation.
M_EUROSAT_RGB_INDICES = (3, 2, 1)


def _require_ml_dependencies():
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("ML dependencies are missing. Install torch and numpy.") from exc
    return np, torch


def find_task(dataset_id: str, benchmark_dir: str | None = None):
    spec = get_dataset_spec(dataset_id)
    try:
        import geobench
    except ImportError as exc:
        raise RuntimeError("geobench is not installed.") from exc
    kwargs = (
        {"benchmark_dir": str(Path(benchmark_dir).expanduser())}
        if benchmark_dir
        else {"benchmark_name": spec.benchmark_name}
    )
    try:
        for task in geobench.task_iterator(**kwargs):
            if task.dataset_name == dataset_id:
                return task
    except FileNotFoundError as exc:
        location = benchmark_dir or f"$GEO_BENCH_DIR/{spec.benchmark_name}"
        raise FileNotFoundError(
            f"GEO-Bench data was not found at {location}. Pass the benchmark directory."
        ) from exc
    raise ValueError(f"Dataset {dataset_id!r} was not found in {spec.benchmark_name!r}.")


def _band_array(band: Any):
    data = getattr(band, "data", None)
    if data is None:
        raise TypeError(f"{type(band).__name__}.data is None; expected a loaded array.")
    return data


def _classification_label(sample: Any, spec: DatasetSpec) -> int:
    value = getattr(sample, "label", None)
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, int):
        raise TypeError(f"Expected integer classification label, got {type(value).__name__}.")
    if not 0 <= value < spec.num_classes:
        raise ValueError(f"Label {value} is outside [0, {spec.num_classes - 1}].")
    return value


def _extract_label(sample: Any) -> int:
    """Compatibility helper for the original m-eurosat inspector."""
    return _classification_label(sample, get_dataset_spec("m-eurosat"))


class GeoBenchDataset:
    """PyTorch view shared by registered classification and segmentation tasks."""

    def __init__(
        self,
        dataset: Any,
        spec: DatasetSpec,
        band_indices: tuple[int, ...],
        normalization: str,
        image_size: int,
        augment: bool,
    ):
        self.dataset = dataset
        self.spec = spec
        self.band_indices = band_indices
        self.normalization = normalization
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        np, torch = _require_ml_dependencies()
        sample = self.dataset[index]
        bands = getattr(sample, "bands", None)
        if bands is None or max(self.band_indices) >= len(bands):
            raise TypeError(
                f"{self.spec.dataset_id} requires band indices {self.band_indices}; "
                f"sample has {0 if bands is None else len(bands)} bands."
            )
        arrays = [np.asarray(_band_array(bands[i])).squeeze() for i in self.band_indices]
        image_np = np.stack(arrays, axis=0).astype(np.float32)
        if image_np.ndim != 3 or not np.isfinite(image_np).all():
            raise ValueError(f"Invalid image tensor shape or values: {image_np.shape}.")
        image_np = np.clip(image_np / self.spec.scale_divisor, 0.0, 1.0)
        image = torch.from_numpy(image_np)

        if self.spec.task_type == "classification":
            target = torch.tensor(_classification_label(sample, self.spec), dtype=torch.long)
        else:
            label_band = getattr(sample, "label", None)
            mask_np = np.asarray(_band_array(label_band)).squeeze()
            if mask_np.ndim != 2:
                raise ValueError(f"Expected segmentation mask [H,W], got {mask_np.shape}.")
            target = torch.from_numpy(mask_np.astype(np.int64, copy=False))
            valid = target if self.spec.ignore_index is None else target[target != self.spec.ignore_index]
            if valid.numel() and (int(valid.min()) < 0 or int(valid.max()) >= self.spec.num_classes):
                raise ValueError("Segmentation mask contains a class outside the registry contract.")

        image, target = self._resize_and_augment(image, target)
        if self.normalization == "imagenet_rgb":
            mean_values, std_values = IMAGENET_MEAN, IMAGENET_STD
        elif self.normalization == "imagenet_rgb_plus_nir":
            mean_values = IMAGENET_MEAN + (0.5,)
            std_values = IMAGENET_STD + (0.25,)
        elif self.normalization == "none":
            return image, target
        else:
            raise ValueError(f"Unknown normalization contract: {self.normalization}")
        mean = torch.tensor(mean_values, dtype=image.dtype)[:, None, None]
        std = torch.tensor(std_values, dtype=image.dtype)[:, None, None]
        return (image - mean) / std, target

    def _resize_and_augment(self, image, target):
        import torch
        import torch.nn.functional as functional

        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            image = functional.interpolate(
                image[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )[0]
            if self.spec.task_type == "segmentation":
                target = functional.interpolate(
                    target[None, None].float(), size=(self.image_size, self.image_size), mode="nearest"
                )[0, 0].long()
        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
            if self.spec.task_type == "segmentation":
                target = torch.flip(target, dims=(-1,))
        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-2,))
            if self.spec.task_type == "segmentation":
                target = torch.flip(target, dims=(-2,))
        return image, target


@dataclass(frozen=True)
class DataBundle:
    train_loader: Any
    val_loader: Any
    task_type: str
    num_classes: int
    input_channels: int
    class_names: list[str]
    ignore_index: int | None
    primary_metric: str
    preprocessing: dict[str, Any]


def create_dataloaders(dataset_id: str, config) -> DataBundle:
    _, torch = _require_ml_dependencies()
    spec = get_dataset_spec(dataset_id)
    from .model_registry import get_model_spec

    model_spec = get_model_spec(config.model)
    if model_spec.input_profile == "sentinel2_13":
        if dataset_id != "m-eurosat":
            raise ValueError("sentinel2_13 input is currently verified only for m-eurosat.")
        # Reorder GEO-Bench's known physical storage order to SSL4EO-S12's
        # canonical S2 order: B01..B08, B8A, B09..B12.
        band_indices = (0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 9, 10, 11)
        normalization = "none"
    elif model_spec.input_profile == "rgb":
        if len(spec.band_indices) < 3:
            raise ValueError(f"{dataset_id} has no verified RGB profile.")
        band_indices = spec.band_indices[:3]
        normalization = "imagenet_rgb"
    elif model_spec.input_profile == "dataset_channels":
        band_indices = spec.band_indices
        normalization = "imagenet_rgb" if len(band_indices) == 3 else "imagenet_rgb_plus_nir"
    else:
        raise ValueError(f"Unknown input profile {model_spec.input_profile!r}.")
    task = find_task(dataset_id, config.benchmark_dir)
    # Load all bands, then apply verified physical indices from the registry.
    # This is essential for datasets such as m-eurosat whose stored band names
    # are known to be wrong even though the pixel order is valid.
    train_source = task.get_dataset(split="train", band_names=None)
    val_source = task.get_dataset(split="valid", band_names=None)
    train_dataset = GeoBenchDataset(
        train_source, spec, band_indices, normalization, config.image_size, augment=True
    )
    val_dataset = GeoBenchDataset(
        val_source, spec, band_indices, normalization, config.image_size, augment=False
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    label_type = getattr(task, "label_type", None)
    metadata_classes = getattr(label_type, "n_classes", None)
    if metadata_classes is not None and metadata_classes != spec.num_classes:
        raise ValueError(
            f"Registry says {spec.num_classes} classes for {dataset_id}, "
            f"but GEO-Bench metadata says {metadata_classes}."
        )
    class_names = list(getattr(label_type, "class_names", None) or [])
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        task_type=spec.task_type,
        num_classes=spec.num_classes,
        input_channels=len(band_indices),
        class_names=class_names,
        ignore_index=spec.ignore_index,
        primary_metric=spec.primary_metric,
        preprocessing={
            "input_profile": model_spec.input_profile,
            "band_indices": list(band_indices),
            "scale_divisor": spec.scale_divisor,
            "normalization": normalization,
            "image_size": config.image_size,
        },
    )
