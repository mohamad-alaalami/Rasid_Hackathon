"""Factual model metadata, compatibility, and runtime eligibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata, util
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    architecture: str
    task_type: str
    datasets: tuple[str, ...]
    input_profile: str
    pretraining_source: str
    pretraining_status: str
    checkpoint_id: str
    checkpoint_url: str | None
    checkpoint_filename: str | None
    license: str
    required_package: str
    minimum_package_version: str | None
    preprocessing: str
    conservative_batch_size: int
    limitations: tuple[str, ...] = ()


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "resnet18_rgb": ModelSpec(
        model_id="resnet18_rgb",
        architecture="ResNet-18",
        task_type="classification",
        datasets=("m-eurosat", "m-pv4ger"),
        input_profile="rgb",
        pretraining_source="TorchVision ImageNet-1K",
        pretraining_status="official_verified",
        checkpoint_id="ResNet18_Weights.IMAGENET1K_V1",
        checkpoint_url=None,
        checkpoint_filename=None,
        license="BSD-3-Clause (TorchVision code; ImageNet weights subject to dataset terms)",
        required_package="torchvision",
        minimum_package_version=None,
        preprocessing="scale to [0,1], ImageNet mean/std; EuroSAT RGB=B04/B03/B02",
        conservative_batch_size=32,
    ),
    "resnet18_s2": ModelSpec(
        model_id="resnet18_s2",
        architecture="ResNet-18",
        task_type="classification",
        datasets=("m-eurosat",),
        input_profile="sentinel2_13",
        pretraining_source="SSL4EO-S12 MoCo, S2-L1C 13-band, epoch 99",
        pretraining_status="checkpoint_required",
        checkpoint_id="B13_rn18_moco_0099_ckpt.pth",
        checkpoint_url="https://drive.google.com/file/d/1iWLm7ljQ6tKZiVp47pJUPDe3Un0BUd9o/view",
        checkpoint_filename="B13_rn18_moco_0099_ckpt.pth",
        license="CC-BY-4.0 weights; Apache-2.0 source",
        required_package="torchvision",
        minimum_package_version=None,
        preprocessing="13 S2-L1C bands, divide by 10000 and clip [0,1], no ImageNet normalization",
        conservative_batch_size=16,
        limitations=("Eligible only when the exact checkpoint is present and loads successfully.",),
    ),
    "resnet50_s2": ModelSpec(
        model_id="resnet50_s2",
        architecture="ResNet-50",
        task_type="classification",
        datasets=("m-eurosat",),
        input_profile="sentinel2_13",
        pretraining_source="SSL4EO-S12 DINO, S2-L1C 13-band, epoch 99",
        pretraining_status="checkpoint_required",
        checkpoint_id="B13_rn50_dino_0099.pth",
        checkpoint_url="https://drive.google.com/file/d/1B4o_NvY7O6fJrvsOUR-7QzLYNpRL1ieA/view",
        checkpoint_filename="B13_rn50_dino_0099.pth",
        license="CC-BY-4.0 weights; Apache-2.0 source",
        required_package="torchvision",
        minimum_package_version=None,
        preprocessing="13 S2-L1C bands, divide by 10000 and clip [0,1], no ImageNet normalization",
        conservative_batch_size=8,
        limitations=("Eligible only when the exact checkpoint is present and loads successfully.",),
    ),
    "convnext_tiny_rgb": ModelSpec(
        model_id="convnext_tiny_rgb",
        architecture="ConvNeXt-Tiny",
        task_type="classification",
        datasets=("m-eurosat", "m-pv4ger"),
        input_profile="rgb",
        pretraining_source="TorchVision ImageNet-1K",
        pretraining_status="official_verified",
        checkpoint_id="ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        checkpoint_url=None,
        checkpoint_filename=None,
        license="BSD-3-Clause (TorchVision code; ImageNet weights subject to dataset terms)",
        required_package="torchvision",
        minimum_package_version=None,
        preprocessing="scale to [0,1], ImageNet mean/std; recommended crop 224",
        conservative_batch_size=16,
    ),
    "unet_r34": ModelSpec(
        model_id="unet_r34",
        architecture="U-Net with ResNet-34 encoder",
        task_type="segmentation",
        datasets=("m-pv4ger-seg", "m-chesapeake"),
        input_profile="dataset_channels",
        pretraining_source="SMP ResNet-34 ImageNet encoder",
        pretraining_status="package_and_weights_required",
        checkpoint_id="smp-hub/resnet34.imagenet@7a57b34f723329ff020b3f8bc41771163c519d0c",
        checkpoint_url="https://huggingface.co/smp-hub/resnet34.imagenet",
        checkpoint_filename=None,
        license="MIT (SMP); encoder checkpoint provenance recorded by SMP",
        required_package="segmentation-models-pytorch",
        minimum_package_version="0.5.0",
        preprocessing="ImageNet RGB normalization; Chesapeake NIR uses explicit mean-RGB weight adaptation",
        conservative_batch_size=8,
    ),
    "deeplabv3plus_r50": ModelSpec(
        model_id="deeplabv3plus_r50",
        architecture="DeepLabV3+ with ResNet-50 encoder",
        task_type="segmentation",
        datasets=("m-pv4ger-seg", "m-chesapeake"),
        input_profile="dataset_channels",
        pretraining_source="SMP ResNet-50 ImageNet encoder",
        pretraining_status="package_and_weights_required",
        checkpoint_id="smp-hub/resnet50.imagenet@00cb74e366966d59cd9a35af57e618af9f88efe9",
        checkpoint_url="https://huggingface.co/smp-hub/resnet50.imagenet",
        checkpoint_filename=None,
        license="MIT/BSD-3-Clause components (SMP); encoder checkpoint provenance recorded by SMP",
        required_package="segmentation-models-pytorch",
        minimum_package_version="0.5.0",
        preprocessing="ImageNet RGB normalization; Chesapeake NIR uses explicit mean-RGB weight adaptation",
        conservative_batch_size=4,
    ),
    # Existing implementation retained so the working segmentation pipeline is not removed.
    "fcn_resnet50": ModelSpec(
        model_id="fcn_resnet50",
        architecture="TorchVision FCN with ResNet-50 backbone",
        task_type="segmentation",
        datasets=("m-pv4ger-seg", "m-chesapeake"),
        input_profile="dataset_channels",
        pretraining_source="TorchVision ResNet50_Weights.IMAGENET1K_V1 backbone",
        pretraining_status="legacy_engine_baseline",
        checkpoint_id="ResNet50_Weights.IMAGENET1K_V1",
        checkpoint_url=None,
        checkpoint_filename=None,
        license="BSD-3-Clause (TorchVision code; ImageNet weights subject to dataset terms)",
        required_package="torchvision",
        minimum_package_version=None,
        preprocessing="ImageNet RGB normalization; fourth channel initialized from RGB mean",
        conservative_batch_size=4,
        limitations=("Preserved baseline; not one of the six requested target configurations.",),
    ),
}


ALIASES = {"resnet18": "resnet18_rgb"}


def canonical_model_id(model_id: str) -> str:
    return ALIASES.get(model_id, model_id)


def get_model_spec(model_id: str) -> ModelSpec:
    model_id = canonical_model_id(model_id)
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_id!r}; choices: {sorted(MODEL_REGISTRY)}") from exc


def _version_at_least(installed: str, minimum: str) -> bool:
    def parts(value: str) -> tuple[int, ...]:
        return tuple(int(piece) for piece in value.split(".")[:3] if piece.isdigit())

    return parts(installed) >= parts(minimum)


def model_runtime_status(model_id: str, checkpoint_dir: str | Path = "checkpoints") -> dict[str, Any]:
    spec = get_model_spec(model_id)
    import_name = spec.required_package.replace("-", "_")
    installed = util.find_spec(import_name) is not None
    installed_version = None
    reason = None
    if installed:
        try:
            installed_version = metadata.version(spec.required_package)
        except metadata.PackageNotFoundError:
            installed_version = "unknown"
    else:
        reason = f"required package {spec.required_package!r} is not installed"
    if installed and spec.minimum_package_version and installed_version != "unknown":
        if not _version_at_least(installed_version, spec.minimum_package_version):
            installed = False
            reason = (
                f"{spec.required_package} {installed_version} is older than "
                f"required {spec.minimum_package_version}"
            )
    checkpoint_path = None
    if spec.checkpoint_filename:
        checkpoint_path = Path(checkpoint_dir) / spec.checkpoint_filename
        if not checkpoint_path.is_file():
            installed = False
            reason = f"exact checkpoint is missing: {checkpoint_path}"
    return {
        "eligible": installed,
        "reason": reason,
        "installed_version": installed_version,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "measured_runtime_seconds": None,
        "measured_peak_gpu_memory_mb": None,
    }


def list_eligible_models(dataset_id: str, checkpoint_dir: str | Path = "checkpoints") -> list[dict[str, Any]]:
    from .registry import get_dataset_spec

    dataset = get_dataset_spec(dataset_id)
    results = []
    for spec in MODEL_REGISTRY.values():
        if dataset_id not in spec.datasets or spec.task_type != dataset.task_type:
            continue
        status = model_runtime_status(spec.model_id, checkpoint_dir)
        if status["eligible"]:
            results.append({**asdict(spec), **status})
    return results


def list_model_options(dataset_id: str, checkpoint_dir: str | Path = "checkpoints") -> list[dict[str, Any]]:
    from .registry import get_dataset_spec

    dataset = get_dataset_spec(dataset_id)
    return [
        {**asdict(spec), **model_runtime_status(spec.model_id, checkpoint_dir)}
        for spec in MODEL_REGISTRY.values()
        if dataset_id in spec.datasets and spec.task_type == dataset.task_type
    ]


def validate_model_for_dataset(model_id: str, dataset_id: str) -> str:
    from .registry import get_dataset_spec

    spec = get_model_spec(model_id)
    dataset = get_dataset_spec(dataset_id)
    if dataset_id not in spec.datasets or spec.task_type != dataset.task_type:
        raise ValueError(
            f"model={spec.model_id!r} is incompatible with dataset={dataset_id!r} "
            f"({dataset.task_type})."
        )
    return spec.model_id
