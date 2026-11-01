"""Verified dataset contracts used by the generic GEO-Bench pipeline."""

from __future__ import annotations

from dataclasses import dataclass

# Sentinel-2 L1C order expected by SSL4EO-S12 13-band checkpoints.
SSL4EO_S2_ORDER = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A",
    "B09", "B10", "B11", "B12",
)
# Band names that identify the visible channels used to place pretrained RGB
# filters when a first convolution is adapted to more channels.
RGB_NAMES = {
    "red": ("red", "B04", "TCI_R"),
    "green": ("green", "B03", "TCI_G"),
    "blue": ("blue", "B02", "TCI_B"),
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    task_type: str
    benchmark_name: str
    num_classes: int
    available_channels: int
    # Default channel selection for models using the "dataset_channels" profile.
    band_indices: tuple[int, ...]
    scale_divisor: float | None
    primary_metric: str
    native_resolution: tuple[int, int]
    ignore_index: int | None = None
    # Verified name of every stored channel, by physical index. Channels are
    # always selected by index, never by GEO-Bench's stored band names.
    band_names: tuple[str, ...] = ()
    # Explicit input views; a model is compatible only with views listed here.
    rgb_indices: tuple[int, int, int] | None = None
    multispectral_indices: tuple[int, ...] | None = None
    sentinel2_13_indices: tuple[int, ...] | None = None
    # "scale_imagenet": divide by scale_divisor, clip, then the profile's fixed
    # normalization (the original four datasets). "train_channel_stats":
    # per-channel standardization with statistics from the training split.
    normalization: str = "scale_imagenet"
    data_dtype: str | None = None
    channel_notes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.band_names and len(self.band_names) != self.available_channels:
            raise ValueError(f"{self.dataset_id}: band_names must name every stored channel.")
        for view in (self.band_indices, self.rgb_indices, self.multispectral_indices,
                     self.sentinel2_13_indices):
            if view is not None and any(not 0 <= i < self.available_channels for i in view):
                raise ValueError(f"{self.dataset_id}: channel index outside 0..{self.available_channels - 1}.")
        if self.normalization not in ("scale_imagenet", "train_channel_stats"):
            raise ValueError(f"{self.dataset_id}: unknown normalization {self.normalization!r}.")
        if self.normalization == "scale_imagenet" and not self.scale_divisor:
            raise ValueError(f"{self.dataset_id}: scale_imagenet needs a scale_divisor.")

    @property
    def input_channels(self) -> int:
        return len(self.band_indices)

    @property
    def input_profiles(self) -> tuple[str, ...]:
        profiles = ["dataset_channels"]
        if self.rgb_indices is not None:
            profiles.append("rgb")
        if self.multispectral_indices is not None:
            profiles.append("multispectral")
        if self.sentinel2_13_indices is not None:
            profiles.append("sentinel2_13")
        return tuple(profiles)

    def profile_band_indices(self, profile: str) -> tuple[int, ...]:
        """The verified physical channels a model input profile consumes."""
        views = {
            "dataset_channels": self.band_indices,
            "rgb": self.rgb_indices,
            "multispectral": self.multispectral_indices,
            "sentinel2_13": self.sentinel2_13_indices,
        }
        if profile not in views:
            raise ValueError(f"Unknown input profile {profile!r}.")
        if views[profile] is None:
            raise ValueError(f"{self.dataset_id} has no verified {profile!r} input view.")
        return tuple(views[profile])

    def channel_names(self, indices) -> list[str]:
        if not self.band_names:
            return [f"channel_{index}" for index in indices]
        return [self.band_names[index] for index in indices]

    def rgb_positions(self, indices) -> tuple[int, int, int] | None:
        """Positions of the red, green, and blue channels within a selection."""
        names = self.channel_names(indices)
        positions = []
        for colour in ("red", "green", "blue"):
            matches = [i for i, name in enumerate(names) if name in RGB_NAMES[colour]]
            if len(matches) != 1:
                return None
            positions.append(matches[0])
        return tuple(positions)


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
        # Physical order: B8A is stored last (geo-bench issue #28, PR #30).
        band_names=("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                    "B09", "B10", "B11", "B12", "B08A"),
        rgb_indices=(3, 2, 1),
        sentinel2_13_indices=(0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 9, 10, 11),
        channel_notes=(
            "Stored GEO-Bench band names are wrong (issue #28); channels follow the "
            "corrected physical order with B8A last.",
        ),
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
        band_names=("red", "green", "blue"),
        rgb_indices=(0, 1, 2),
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
        band_names=("red", "green", "blue"),
        rgb_indices=(0, 1, 2),
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
        band_names=("red", "green", "blue", "nir"),
        rgb_indices=(0, 1, 2),
    ),
    "m-forestnet": DatasetSpec(
        dataset_id="m-forestnet",
        task_type="classification",
        benchmark_name="classification_v1.0",
        num_classes=12,
        available_channels=6,
        band_indices=(0, 1, 2, 3, 4, 5),
        scale_divisor=None,
        primary_metric="accuracy",
        native_resolution=(332, 332),
        band_names=("red", "green", "blue", "nir", "swir1", "swir2"),
        rgb_indices=(0, 1, 2),
        multispectral_indices=(0, 1, 2, 3, 4, 5),
        normalization="train_channel_stats",
        data_dtype="uint8",
        channel_notes=(
            "Landsat-8 Red, Green, Blue, NIR, SWIR1, SWIR2 stored as uint8 (0-255); "
            "standardized per channel with training-split statistics.",
        ),
    ),
    "m-brick-kiln": DatasetSpec(
        dataset_id="m-brick-kiln",
        task_type="classification",
        benchmark_name="classification_v1.0",
        num_classes=2,
        available_channels=13,
        band_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        scale_divisor=None,
        primary_metric="accuracy",
        native_resolution=(64, 64),
        # Corrected physical order from geo-bench issue #29 / PR #30 (README
        # "Known issues"); the stored names B01..B12 are wrong for most channels.
        band_names=("B01", "B02", "B03", "B04", "B05", "B07", "B08A", "B08",
                    "B11", "B12", "TCI_R", "TCI_G", "TCI_B"),
        rgb_indices=(10, 11, 12),
        multispectral_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        normalization="train_channel_stats",
        data_dtype="float64",
        channel_notes=(
            "GEO-Bench's stored band names and wavelengths are wrong (issue #29); "
            "channels are selected by corrected physical index.",
            "B06, B09, and B10 are absent from the source export.",
            "Channels 10-12 are 8-bit true-colour composites (TCI) used only as the "
            "RGB view; the multispectral view uses the ten reflectance bands.",
        ),
    ),
    "m-nz-cattle": DatasetSpec(
        dataset_id="m-nz-cattle",
        task_type="segmentation",
        benchmark_name="segmentation_v1.0",
        num_classes=2,
        available_channels=3,
        band_indices=(0, 1, 2),
        scale_divisor=None,
        primary_metric="mean_iou",
        native_resolution=(500, 500),
        band_names=("red", "green", "blue"),
        rgb_indices=(0, 1, 2),
        normalization="train_channel_stats",
        data_dtype="uint8",
        channel_notes=(
            "500x500 is not a multiple of 32; encoder-decoder models pad inputs and "
            "crop outputs back to the native size.",
        ),
    ),
    "m-cashew-plant": DatasetSpec(
        dataset_id="m-cashew-plant",
        task_type="segmentation",
        benchmark_name="segmentation_v1.0",
        num_classes=7,
        available_channels=13,
        band_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        scale_divisor=None,
        primary_metric="mean_iou",
        native_resolution=(256, 256),
        band_names=("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                    "B08A", "B09", "B11", "B12", "CLOUD_PROBABILITY"),
        rgb_indices=(3, 2, 1),
        normalization="train_channel_stats",
        data_dtype="float32",
        channel_notes=(
            "Channels 0-11 are Sentinel-2 bands (no B10); channel 12 is cloud "
            "probability, not a spectral band, and is excluded from model inputs.",
            "Classes absent from individual samples are not assumed absent globally.",
        ),
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
            "band_names": list(spec.band_names),
            "input_views": {
                profile: spec.channel_names(spec.profile_band_indices(profile))
                for profile in spec.input_profiles
            },
            "normalization": spec.normalization,
            "channel_notes": list(spec.channel_notes),
        }
        for spec in DATASET_REGISTRY.values()
    ]
