"""Validated model factory."""

from __future__ import annotations


def create_model(name: str, num_classes: int, pretrained: bool = True):
    if name != "resnet18":
        raise ValueError(f"Unsupported model {name!r}; initial engine supports 'resnet18'.")
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("torchvision is required to create ResNet-18.") from exc
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = __import__("torch").nn.Linear(model.fc.in_features, num_classes)
    return model

