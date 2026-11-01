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
    rgb_positions: tuple[int, int, int] | None = (0, 1, 2),
):
    """Build a registry model.

    ``rgb_positions`` gives where the verified red, green, and blue channels sit
    in the input; pretrained RGB filters are placed there when the first
    convolution is adapted to more channels. Encoder-decoder models get input
    padding hooks so any native resolution is preserved.
    """
    model = _build_model(name, task_type, num_classes, input_channels, pretrained,
                         pretrained_checkpoint, rgb_positions)
    multiple = get_model_spec(name).input_multiple
    if multiple > 1:
        apply_input_padding(model, multiple)
    return model


def _build_model(name, task_type, num_classes, input_channels, pretrained,
                 pretrained_checkpoint, rgb_positions):
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

    if spec.model_id in {"resnet18_ms", "resnet50_ms"}:
        builder, weights = (
            (resnet18, ResNet18_Weights.IMAGENET1K_V1)
            if spec.model_id == "resnet18_ms"
            else (resnet50, ResNet50_Weights.IMAGENET1K_V1)
        )
        model = builder(weights=weights if pretrained else None)
        _adapt_first_convolution(model, "conv1", input_channels, rgb_positions)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
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
        # Non-RGB channels have no ImageNet-pretrained weight. Initialize them
        # explicitly from the mean of the pretrained RGB filters.
        _adapt_first_convolution(model.encoder, "conv1", input_channels, rgb_positions)
        return model

    if spec.model_id == "fcn_resnet50":
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = fcn_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
        )
        _adapt_first_convolution(model.backbone, "conv1", input_channels, rgb_positions)
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


def _adapt_first_convolution(
    module,
    attribute: str,
    input_channels: int,
    rgb_positions: tuple[int, int, int] | None = (0, 1, 2),
) -> None:
    """Place pretrained RGB filters on the verified RGB channels.

    Every other channel is initialized from the mean of the pretrained RGB
    filters. With ``rgb_positions=(0, 1, 2)`` this is the original behavior.
    ``None`` means the input has no verified RGB channels, so every channel
    gets the mean filter.
    """
    import torch

    old = getattr(module, attribute)
    positions = tuple(rgb_positions) if rgb_positions is not None else None
    if old.in_channels == input_channels and positions == (0, 1, 2):
        return
    if positions is not None and (
        len(set(positions)) != 3 or any(not 0 <= p < input_channels for p in positions)
    ):
        raise ValueError(f"Invalid RGB positions {positions} for {input_channels} channels.")
    new = torch.nn.Conv2d(
        input_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        rgb_mean = old.weight[:, :3].mean(dim=1, keepdim=True)
        new.weight.copy_(rgb_mean.repeat(1, input_channels, 1, 1))
        for source, target in enumerate(positions or ()):
            new.weight[:, target] = old.weight[:, source]
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(module, attribute, new)


def apply_input_padding(model, multiple: int) -> None:
    """Pad inputs to a multiple of ``multiple`` and crop outputs back.

    Uses forward hooks, so no parameters or state-dict keys change. Inputs that
    are already a multiple pass through untouched. Image borders are reflected;
    the cropped-away logits never reach the loss or metrics.
    """
    import torch.nn.functional as functional

    def pad(module, args):
        images = args[0]
        height, width = images.shape[-2:]
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        module._gotrain_input_size = (height, width)
        if not pad_h and not pad_w:
            return None
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        return (functional.pad(images, (0, pad_w, 0, pad_h), mode=mode), *args[1:])

    def crop(module, args, output):
        height, width = module._gotrain_input_size
        if isinstance(output, dict):
            return {key: value[..., :height, :width] for key, value in output.items()}
        return output[..., :height, :width]

    model.register_forward_pre_hook(pad)
    model.register_forward_hook(crop)
    model.gotrain_input_padding = {"multiple": multiple, "mode": "reflect", "output": "cropped"}


def model_logits(output, task_type: str):
    if task_type == "segmentation" and isinstance(output, dict):
        if "out" not in output:
            raise TypeError("Segmentation model output dictionary lacks 'out'.")
        return output["out"]
    return output


def validate_model_output_shape(logits, targets, task_type: str) -> None:
    """Reject models that cannot preserve the requested task dimensions."""
    if task_type == "classification":
        if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Unsupported classification output/target shapes: "
                f"{tuple(logits.shape)}/{tuple(targets.shape)}."
            )
        return
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError(
            f"Unsupported segmentation output/target ranks: "
            f"{tuple(logits.shape)}/{tuple(targets.shape)}."
        )
    if logits.shape[0] != targets.shape[0] or tuple(logits.shape[-2:]) != tuple(targets.shape[-2:]):
        raise ValueError(
            "The segmentation model does not preserve the requested effective "
            f"resolution: output={tuple(logits.shape)}, target={tuple(targets.shape)}. "
            "Choose a supported model/resolution or an explicit image_size override."
        )
