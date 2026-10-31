"""Task-aware model factory for registered datasets."""

from __future__ import annotations


def create_model(
    name: str,
    task_type: str,
    num_classes: int,
    input_channels: int,
    pretrained: bool = True,
):
    try:
        import torch
        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18
        from torchvision.models.segmentation import fcn_resnet50
    except ImportError as exc:
        raise RuntimeError("torchvision is required to create models.") from exc

    if task_type == "classification" and name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        _adapt_first_convolution(model, "conv1", input_channels)
        return model
    if task_type == "segmentation" and name == "fcn_resnet50":
        backbone_weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = fcn_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
        )
        _adapt_first_convolution(model.backbone, "conv1", input_channels)
        return model
    raise ValueError(f"Model {name!r} is incompatible with task_type={task_type!r}.")


def _adapt_first_convolution(module, attribute: str, input_channels: int) -> None:
    """Adapt an ImageNet RGB stem while retaining useful pretrained weights."""
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
        if input_channels >= 3:
            new.weight[:, :3] = old.weight
            if input_channels > 3:
                mean_weight = old.weight.mean(dim=1, keepdim=True)
                new.weight[:, 3:] = mean_weight.repeat(1, input_channels - 3, 1, 1)
        else:
            new.weight.copy_(old.weight[:, :input_channels])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(module, attribute, new)


def model_logits(output, task_type: str):
    if task_type == "segmentation":
        if not isinstance(output, dict) or "out" not in output:
            raise TypeError("Segmentation model must return a dictionary containing 'out'.")
        return output["out"]
    return output
