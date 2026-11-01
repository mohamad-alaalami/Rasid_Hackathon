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
# Training samples (evenly spaced, deterministic) used for channel statistics.
STATS_MAX_SAMPLES = 256


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
        effective_resolution: tuple[int, int],
        augment: bool,
        augmentations: list[dict] | None = None,
    ):
        from .components import DEFAULT_AUGMENTATIONS

        self.dataset = dataset
        self.spec = spec
        self.band_indices = band_indices
        self.normalization = normalization
        self.effective_resolution = effective_resolution
        self.augment = augment
        # Resolved augmentation list; None keeps the historical flips.
        self.augmentations = list(DEFAULT_AUGMENTATIONS if augmentations is None else augmentations)

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
        if not isinstance(self.normalization, dict):
            image_np = np.clip(image_np / self.spec.scale_divisor, 0.0, 1.0)
        image = torch.from_numpy(np.ascontiguousarray(image_np))

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

        image, target = self._resize(image, target)
        if self.augment:
            # Photometric ops act in data units, before standardization.
            from .components import apply_augmentations

            image, target = apply_augmentations(
                image, target, self.augmentations, self.spec.task_type
            )
        if isinstance(self.normalization, dict):
            # Training-split channel statistics, stored in the checkpoint so the
            # sealed test evaluation applies exactly the same transform.
            mean = torch.tensor(self.normalization["mean"], dtype=image.dtype)[:, None, None]
            std = torch.tensor(self.normalization["std"], dtype=image.dtype)[:, None, None]
            if mean.shape[0] != image.shape[0]:
                raise ValueError("Channel statistics do not match the selected channels.")
            return (image - mean) / std, target
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

    def _resize(self, image, target):
        import torch.nn.functional as functional

        if tuple(image.shape[-2:]) != self.effective_resolution:
            image = functional.interpolate(
                image[None],
                size=self.effective_resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            if self.spec.task_type == "segmentation":
                target = functional.interpolate(
                    target[None, None].float(),
                    size=self.effective_resolution,
                    mode="nearest",
                )[0, 0].long()
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
    native_resolution: tuple[int, int]
    effective_resolution: tuple[int, int]
    preprocessing: dict[str, Any]
    # Where red/green/blue sit in the input, for first-convolution adaptation.
    rgb_positions: tuple[int, int, int] | None = (0, 1, 2)
    # Resolved pipeline and training-label counts (only when class weighting is used).
    pipeline: dict[str, Any] | None = None
    class_counts: list[int] | None = None
    dataset_fingerprint: dict[str, Any] | None = None


def inspect_sample_resolution(
    sample: Any,
    spec: DatasetSpec,
    band_indices: tuple[int, ...],
) -> tuple[int, int]:
    """Read and validate a real sample's spatial dimensions."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to inspect dataset resolution.") from exc
    bands = getattr(sample, "bands", None)
    if bands is None or max(band_indices) >= len(bands):
        raise TypeError(
            f"{spec.dataset_id} requires band indices {band_indices}; "
            f"sample has {0 if bands is None else len(bands)} bands."
        )
    shapes = [tuple(np.asarray(_band_array(bands[i])).squeeze().shape) for i in band_indices]
    if not shapes or any(len(shape) != 2 for shape in shapes):
        raise ValueError(f"Expected 2D image bands, received shapes {shapes}.")
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(f"Image bands are not spatially aligned: {shapes}.")
    resolution = shapes[0]
    if spec.task_type == "segmentation":
        mask_shape = tuple(np.asarray(_band_array(getattr(sample, "label", None))).squeeze().shape)
        if mask_shape != resolution:
            raise ValueError(
                f"Segmentation image/mask mismatch for {spec.dataset_id}: "
                f"image={resolution}, mask={mask_shape}."
            )
    return resolution


def compute_channel_statistics(dataset: Any, band_indices: tuple[int, ...],
                               max_samples: int = STATS_MAX_SAMPLES) -> dict[str, Any]:
    """Per-channel mean/std over evenly spaced training samples, in raw data units."""
    np, _ = _require_ml_dependencies()
    count = len(dataset)
    if count == 0:
        raise ValueError("Cannot compute channel statistics from an empty training split.")
    picks = sorted({int(round(value)) for value in np.linspace(0, count - 1, min(count, max_samples))})
    total = np.zeros(len(band_indices))
    total_sq = np.zeros(len(band_indices))
    pixels = 0
    for index in picks:
        bands = dataset[index].bands
        stack = np.stack([
            np.asarray(_band_array(bands[i]), dtype=np.float64).squeeze() for i in band_indices
        ])
        if not np.isfinite(stack).all():
            raise ValueError(f"Training sample {index} contains non-finite values.")
        flat = stack.reshape(len(band_indices), -1)
        total += flat.sum(axis=1)
        total_sq += np.square(flat).sum(axis=1)
        pixels += flat.shape[1]
    mean = total / pixels
    std = np.sqrt(np.maximum(total_sq / pixels - np.square(mean), 0.0))
    constant = std < 1e-6
    return {
        "type": "train_channel_stats",
        "split": "train",
        "samples": len(picks),
        "train_size": count,
        "mean": mean.tolist(),
        "std": np.where(constant, 1.0, std).tolist(),
        "constant_channels": [int(i) for i in np.flatnonzero(constant)],
    }


def compute_class_counts(dataset: Any, spec: DatasetSpec, max_samples: int = 512) -> list[int]:
    """Label (or mask-pixel) counts over evenly spaced training samples."""
    np, _ = _require_ml_dependencies()
    count = len(dataset)
    picks = sorted({int(round(value)) for value in np.linspace(0, count - 1, min(count, max_samples))})
    counts = np.zeros(spec.num_classes, dtype=np.int64)
    for index in picks:
        sample = dataset[index]
        if spec.task_type == "classification":
            counts[_classification_label(sample, spec)] += 1
            continue
        mask = np.asarray(_band_array(sample.label)).astype(np.int64).ravel()
        if spec.ignore_index is not None:
            mask = mask[mask != spec.ignore_index]
        counts += np.bincount(mask, minlength=spec.num_classes)[: spec.num_classes]
    return counts.tolist()


def _check_channel_count(sample: Any, spec: DatasetSpec) -> None:
    bands = getattr(sample, "bands", None)
    count = 0 if bands is None else len(bands)
    if count != spec.available_channels:
        raise ValueError(
            f"Registry says {spec.available_channels} channels for {spec.dataset_id}, "
            f"but the sample has {count}. Update the verified registry contract."
        )


def resolve_effective_resolution(
    spec: DatasetSpec,
    image_size: int | None,
) -> tuple[int, int]:
    """Use native dimensions unless the experiment explicitly overrides them."""
    return spec.native_resolution if image_size is None else (image_size, image_size)


def create_dataloaders(dataset_id: str, config) -> DataBundle:
    _, torch = _require_ml_dependencies()
    spec = get_dataset_spec(dataset_id)
    from .model_registry import get_model_spec

    from .components import selected_view
    from .solution import dataset_fingerprint

    model_spec = get_model_spec(config.model)
    pipeline = config.resolved_pipeline()
    view = selected_view(pipeline, model_spec)
    # Physical channel indices come from the verified registry view. For
    # sentinel2_13 this reorders storage order into SSL4EO-S12's canonical
    # B01..B08, B8A, B09..B12 order.
    band_indices = spec.profile_band_indices(view)
    use_channel_stats = (
        pipeline["normalization"] == "train_channel_stats"
        or spec.normalization == "train_channel_stats"
    )
    if not use_channel_stats:
        normalization = {
            "sentinel2_13": "none",
            "rgb": "imagenet_rgb",
            "dataset_channels": (
                "imagenet_rgb" if len(band_indices) == 3 else "imagenet_rgb_plus_nir"
            ),
        }.get(view)
        if normalization is None:
            raise ValueError(f"{dataset_id} has no fixed normalization for {view!r} inputs.")
    task = find_task(dataset_id, config.benchmark_dir)
    # Load all bands, then apply verified physical indices from the registry.
    # This is essential for datasets such as m-eurosat whose stored band names
    # are known to be wrong even though the pixel order is valid.
    train_source = task.get_dataset(split="train", band_names=None)
    val_source = task.get_dataset(split="valid", band_names=None)
    _check_channel_count(train_source[0], spec)
    if use_channel_stats:
        normalization = compute_channel_statistics(train_source, band_indices)
    class_counts = (
        compute_class_counts(train_source, spec)
        if pipeline["loss"].get("class_weighting", "none") != "none" else None
    )
    train_resolution = inspect_sample_resolution(train_source[0], spec, band_indices)
    val_resolution = inspect_sample_resolution(val_source[0], spec, band_indices)
    if train_resolution != spec.native_resolution or val_resolution != spec.native_resolution:
        raise ValueError(
            f"Registry native resolution for {dataset_id} is {spec.native_resolution}, "
            f"but actual train/validation samples are {train_resolution}/{val_resolution}. "
            "Update the verified registry contract; no implicit resize was performed."
        )
    effective_resolution = resolve_effective_resolution(spec, config.image_size)
    train_dataset = GeoBenchDataset(
        train_source, spec, band_indices, normalization, effective_resolution, augment=True,
        augmentations=pipeline["augmentations"],
    )
    val_dataset = GeoBenchDataset(
        val_source, spec, band_indices, normalization, effective_resolution, augment=False
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
    rgb_positions = spec.rgb_positions(band_indices)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        task_type=spec.task_type,
        num_classes=spec.num_classes,
        input_channels=len(band_indices),
        class_names=class_names,
        ignore_index=spec.ignore_index,
        primary_metric=spec.primary_metric,
        native_resolution=spec.native_resolution,
        effective_resolution=effective_resolution,
        rgb_positions=rgb_positions,
        pipeline=pipeline,
        class_counts=class_counts,
        dataset_fingerprint=dataset_fingerprint(task),
        preprocessing={
            "input_profile": model_spec.input_profile,
            "input_view": view,
            "band_indices": list(band_indices),
            "band_names": spec.channel_names(band_indices),
            "excluded_channels": spec.channel_names(
                [i for i in range(spec.available_channels) if i not in band_indices]
            ),
            "rgb_positions": list(rgb_positions) if rgb_positions else None,
            "scale_divisor": spec.scale_divisor,
            "normalization": normalization,
            "native_resolution": list(spec.native_resolution),
            "effective_resolution": list(effective_resolution),
            "resolution_override": config.image_size,
            "resize_applied": effective_resolution != spec.native_resolution,
            "image_interpolation": "bilinear with antialiasing",
            "mask_interpolation": (
                "nearest" if spec.task_type == "segmentation" else None
            ),
        },
    )
