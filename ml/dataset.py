"""GEO-Bench dataset adapters and verified m-eurosat preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_DATASET = "m-eurosat"
M_EUROSAT_NUM_CLASSES = 10
# GEO-Bench issue #28 / official README: physical m-eurosat order begins
# B01, B02, B03, B04, ... . RGB is therefore B04, B03, B02.
M_EUROSAT_RGB_INDICES = (3, 2, 1)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_ml_dependencies():
    try:
        import numpy as np
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "ML dependencies are missing. Install torch, torchvision, and numpy."
        ) from exc
    return np, torch, transforms


def find_task(dataset_id: str, benchmark_dir: str | None = None):
    if dataset_id != SUPPORTED_DATASET:
        raise ValueError(f"Unsupported dataset_id={dataset_id!r}; expected {SUPPORTED_DATASET!r}.")
    try:
        import geobench
    except ImportError as exc:
        raise RuntimeError("geobench is not installed.") from exc

    kwargs = (
        {"benchmark_dir": str(Path(benchmark_dir).expanduser())}
        if benchmark_dir
        else {"benchmark_name": "classification_v1.0"}
    )
    try:
        tasks = geobench.task_iterator(**kwargs)
        for task in tasks:
            if task.dataset_name == dataset_id:
                return task
    except FileNotFoundError as exc:
        location = benchmark_dir or "$GEO_BENCH_DIR/classification_v1.0"
        raise FileNotFoundError(
            f"GEO-Bench classification data was not found at {location}. "
            "Pass benchmark_dir='/absolute/path/to/classification_v1.0'."
        ) from exc
    raise ValueError(f"Dataset {dataset_id!r} was not found in the classification benchmark.")


def _band_array(band: Any):
    """Return the eagerly loaded array exposed by GEO-Bench 1.1.0."""
    data = getattr(band, "data", None)
    if data is None:
        raise TypeError(
            f"{type(band).__name__}.data is None; expected a loaded NumPy array "
            "from GEO-Bench 1.1.0."
        )
    return data


def _extract_label(sample: Any) -> int:
    value = getattr(sample, "label", None)
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, int):
        raise TypeError(f"Expected Sample.label to be int, received {type(value).__name__}.")
    if not 0 <= value < M_EUROSAT_NUM_CLASSES:
        raise ValueError(f"Label {value} is outside [0, {M_EUROSAT_NUM_CLASSES - 1}].")
    return value


def _scale_reflectance(rgb):
    """Apply the verified fixed Sentinel-2 int16 reflectance scaling."""
    np, _, _ = _require_ml_dependencies()
    rgb = np.asarray(rgb, dtype=np.float32)
    if not np.isfinite(rgb).all():
        raise ValueError("Image contains NaN or infinite values.")
    minimum = float(rgb.min())
    if minimum < 0:
        raise ValueError(f"Unexpected negative reflectance value: {minimum}")
    # SageMaker inspection confirmed GEO-Bench stores m-eurosat as int16
    # reflectance on the conventional 0..10000 scale. Apply one rule to every
    # image; never select scaling based on an individual image maximum.
    return np.clip(rgb / 10000.0, 0.0, 1.0)


class GeoBenchRGBDataset:
    """PyTorch-compatible RGB view of an m-eurosat split."""

    def __init__(self, dataset: Any, image_size: int = 64, augment: bool = False):
        np, torch, transforms = _require_ml_dependencies()
        self._np = np
        self._torch = torch
        self.dataset = dataset
        operations: list[Any] = []
        if augment:
            operations.extend(
                [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
            )
        if image_size != 64:
            operations.append(transforms.Resize((image_size, image_size), antialias=True))
        operations.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        bands = getattr(sample, "bands", None)
        if bands is None or len(bands) < 4:
            raise TypeError("Expected an m-eurosat sample with at least four bands.")
        arrays = [_band_array(bands[i]) for i in M_EUROSAT_RGB_INDICES]
        rgb = self._scale_and_stack(arrays)
        image = self._torch.from_numpy(rgb)
        image = self.transform(image)
        return image, _extract_label(sample)

    def _scale_and_stack(self, arrays: Iterable[Any]):
        rgb = self._np.stack([self._np.asarray(a).squeeze() for a in arrays], axis=0)
        if rgb.ndim != 3:
            raise ValueError(f"Expected RGB tensor [3,H,W], received {rgb.shape}.")
        return _scale_reflectance(rgb)


@dataclass(frozen=True)
class DataBundle:
    train_loader: Any
    val_loader: Any
    num_classes: int
    class_names: list[str]
    preprocessing: dict[str, Any]


def create_dataloaders(dataset_id: str, config) -> DataBundle:
    _, torch, _ = _require_ml_dependencies()
    task = find_task(dataset_id, config.benchmark_dir)
    train_source = task.get_dataset(split="train")
    val_source = task.get_dataset(split="valid")
    train_dataset = GeoBenchRGBDataset(train_source, config.image_size, augment=True)
    val_dataset = GeoBenchRGBDataset(val_source, config.image_size, augment=False)
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

    # Verified against task.label_type on the installed dataset. This is an
    # explicit dataset contract, not a silent fallback.
    num_classes = M_EUROSAT_NUM_CLASSES
    class_names = []
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        class_names=class_names,
        preprocessing={
            "input_mode": "rgb",
            "physical_bands": ["B04", "B03", "B02"],
            "band_indices": list(M_EUROSAT_RGB_INDICES),
            "reflectance_scaling": "always divide by 10000, then clip [0,1]",
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
            "image_size": config.image_size,
        },
    )

