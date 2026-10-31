"""Model construction driven by the verified model registry."""

from __future__ import annotations

from pathlib import Path

from .model_registry import get_model_spec


def create_model(
    name: str,
    task_type: str,
    num_classes: int,
    input_channels: int,
    pretrained: bool = True,
    pretrained_checkpoint: str | None = None,
):
    spec = get_model_spec(name)
    if spec.task_type != task_type:
        raise ValueError(f"Model {spec.model_id!r} does not support {task_type!r}.")
    try:
        import torch
        from torchvision.models import (
            ConvNeXt_Tiny_Weights,
            ResNet18_Weights,
            ResNet50_Weights,
            convnext_tiny,
            resnet18,
            resnet50,
        )
        from torchvision.models.segmentation import fcn_resnet50
    except ImportError as exc:
        raise RuntimeError("torch and torchvision are required to create models.") from exc

    if spec.model_id == "resnet18_rgb":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    if spec.model_id == "convnext_tiny_rgb":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if spec.model_id == "resnet18_s2":
        path = _require_checkpoint(pretrained_checkpoint, spec.checkpoint_filename, pretrained)
        model = resnet18(weights=None)
        model.conv1 = torch.nn.Conv2d(
            13, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        if pretrained:
            _load_ssl4eo_moco(model, path)
        return model

    if spec.model_id == "resnet50_s2":
        path = _require_checkpoint(pretrained_checkpoint, spec.checkpoint_filename, pretrained)
        model = resnet50(weights=None)
        model.conv1 = torch.nn.Conv2d(
            13, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        if pretrained:
            _load_ssl4eo_dino(model, path)
        return model

    if spec.model_id in {"unet_r34", "deeplabv3plus_r50"}:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "segmentation-models-pytorch>=0.5.0 is required for this model."
            ) from exc
        encoder = "resnet34" if spec.model_id == "unet_r34" else "resnet50"
        encoder_weights = "imagenet" if pretrained else None
        if spec.model_id == "unet_r34":
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
        else:
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
        # Chesapeake's NIR channel has no ImageNet-pretrained weight. Initialize
        # it explicitly from the mean of the pretrained RGB filters.
        _adapt_first_convolution(model.encoder, "conv1", input_channels)
        return model

    if spec.model_id == "fcn_resnet50":
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = fcn_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
        )
        _adapt_first_convolution(model.backbone, "conv1", input_channels)
        return model
    raise ValueError(f"No constructor exists for model {spec.model_id!r}.")


def _require_checkpoint(
    configured_path: str | None,
    expected_filename: str | None,
    pretrained: bool,
) -> Path | None:
    if not pretrained:
        return None
    path = Path(configured_path or Path("checkpoints") / str(expected_filename)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required pretrained checkpoint not found: {path}. "
            "Download the exact registry checkpoint; random fallback is forbidden."
        )
    if expected_filename and path.name != expected_filename:
        raise ValueError(
            f"Expected checkpoint filename {expected_filename!r}, received {path.name!r}."
        )
    return path


def _load_ssl4eo_moco(model, checkpoint_path: Path) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("state_dict")
    if not isinstance(source, dict):
        raise ValueError("SSL4EO MoCo checkpoint is missing the documented 'state_dict'.")
    state = {}
    prefix = "module.encoder_q."
    for key, value in source.items():
        if key.startswith(prefix) and not key.startswith(prefix + "fc"):
            state[key[len(prefix) :]] = value
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != {"fc.weight", "fc.bias"} or result.unexpected_keys:
        raise ValueError(
            f"MoCo checkpoint structure mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}."
        )


def _load_ssl4eo_dino(model, checkpoint_path: Path) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("teacher")
    if not isinstance(source, dict):
        raise ValueError("SSL4EO DINO checkpoint is missing the documented 'teacher' state.")
    state = {}
    for key, value in source.items():
        key = key.removeprefix("module.").removeprefix("backbone.")
        if not key.startswith(("head", "fc")):
            state[key] = value
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != {"fc.weight", "fc.bias"} or result.unexpected_keys:
        raise ValueError(
            f"DINO checkpoint structure mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}."
        )


def _adapt_first_convolution(module, attribute: str, input_channels: int) -> None:
    """Retain RGB weights and initialize extra channels from their mean."""
    import torch

    old = getattr(module, attribute)
    if old.in_channels == input_channels:
        return
    new = torch.nn.Conv2d(
        input_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        copy_channels = min(3, input_channels)
        new.weight[:, :copy_channels] = old.weight[:, :copy_channels]
        if input_channels > 3:
            rgb_mean = old.weight[:, :3].mean(dim=1, keepdim=True)
            new.weight[:, 3:] = rgb_mean.repeat(1, input_channels - 3, 1, 1)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(module, attribute, new)


def model_logits(output, task_type: str):
    if task_type == "segmentation" and isinstance(output, dict):
        if "out" not in output:
            raise TypeError("Segmentation model output dictionary lacks 'out'.")
        return output["out"]
    return output
