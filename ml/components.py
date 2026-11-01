"""Composable training-pipeline components with explicit compatibility rules.

A pipeline is a JSON object; every key is optional and defaults reproduce the
historical GoTrain training exactly (horizontal then vertical flips at p=0.5,
cross-entropy, AdamW, per-epoch cosine annealing, no dropout):

    {
      "input_view": "multispectral",          # which verified channels
      "normalization": "train_channel_stats",  # or "dataset_default"
      "augmentations": [{"name": "hflip", "p": 0.5}, {"name": "rot90", "p": 0.5}],
      "loss": {"name": "focal", "gamma": 2.0, "class_weighting": "inverse_frequency"},
      "optimizer": {"name": "sgd", "momentum": 0.9, "nesterov": true},
      "scheduler": {"name": "warmup_cosine", "warmup_epochs": 1},
      "dropout": 0.2
    }

Learning rate, weight decay, epochs, batch size, image size, and early
stopping remain top-level training settings. Every option here changes what
the trainer does; nothing is recorded without being applied.

Spectral safety: photometric augmentations apply one shared random factor to
all channels (a sensor gain or contrast change), so relationships between
bands are preserved. Per-channel colour jitter, hue/saturation, and channel
shuffling are deliberately not offered.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass
from typing import Any

TASKS = ("classification", "segmentation")
INPUT_VIEWS = ("rgb", "multispectral", "sentinel2_13", "dataset_channels")
NORMALIZATIONS = ("dataset_default", "train_channel_stats")
CLASS_WEIGHTING = ("none", "inverse_frequency", "inverse_sqrt_frequency")
# Input views each model input profile can consume. Profiles whose first
# convolution is adapted (see ml.models._adapt_first_convolution) may also take
# a smaller verified view; fixed-stem and exact-checkpoint models may not.
ALTERNATIVE_VIEWS = {
    "rgb": ("rgb",),
    "sentinel2_13": ("sentinel2_13",),
    "multispectral": ("multispectral", "rgb"),
    "dataset_channels": ("dataset_channels", "multispectral", "rgb"),
}


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # "float", "int", "bool", "choice"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None
    description: str = ""


@dataclass(frozen=True)
class Component:
    kind: str  # "augmentation", "loss", "optimizer", "scheduler"
    name: str
    description: str
    params: tuple[Param, ...] = ()
    tasks: tuple[str, ...] = TASKS
    effect: str | None = None  # augmentations: "spatial" or "photometric"
    constraints: tuple[str, ...] = ()


P = Param
CATALOG: dict[tuple[str, str], Component] = {
    (c.kind, c.name): c
    for c in (
        Component("augmentation", "hflip", "Horizontal flip; masks flipped identically.",
                  (P("p", "float", 0.5, 0.0, 1.0),), effect="spatial"),
        Component("augmentation", "vflip", "Vertical flip; masks flipped identically.",
                  (P("p", "float", 0.5, 0.0, 1.0),), effect="spatial"),
        Component("augmentation", "rot90", "Rotation by 90, 180, or 270 degrees; masks rotated identically.",
                  (P("p", "float", 0.5, 0.0, 1.0),), effect="spatial",
                  constraints=("requires a square effective resolution",)),
        Component("augmentation", "random_resized_crop",
                  "Random square-aspect crop of min_scale..1 of the area, resized back "
                  "(bilinear image, nearest-neighbour mask).",
                  (P("p", "float", 0.5, 0.0, 1.0), P("min_scale", "float", 0.6, 0.2, 1.0)),
                  effect="spatial"),
        Component("augmentation", "brightness",
                  "Multiply all channels by one shared factor in [1-max_delta, 1+max_delta].",
                  (P("p", "float", 0.5, 0.0, 1.0), P("max_delta", "float", 0.1, 0.0, 0.5)),
                  effect="photometric"),
        Component("augmentation", "contrast",
                  "Scale deviations from each channel's mean by one shared factor.",
                  (P("p", "float", 0.5, 0.0, 1.0), P("max_delta", "float", 0.1, 0.0, 0.5)),
                  effect="photometric"),
        Component("augmentation", "gaussian_noise",
                  "Additive noise with per-channel sigma = sigma_fraction x that channel's std.",
                  (P("p", "float", 0.5, 0.0, 1.0), P("sigma_fraction", "float", 0.03, 0.0, 0.2)),
                  effect="photometric"),
        Component("loss", "cross_entropy", "Cross-entropy (historical default).",
                  (P("label_smoothing", "float", 0.0, 0.0, 0.3),
                   P("class_weighting", "choice", "none", choices=CLASS_WEIGHTING))),
        Component("loss", "focal", "Focal loss (down-weights easy examples); gamma=0 equals cross-entropy.",
                  (P("gamma", "float", 2.0, 0.0, 5.0),
                   P("class_weighting", "choice", "none", choices=CLASS_WEIGHTING))),
        Component("loss", "dice_cross_entropy",
                  "(1-dice_weight) x cross-entropy + dice_weight x soft Dice loss.",
                  (P("dice_weight", "float", 0.5, 0.0, 1.0),
                   P("label_smoothing", "float", 0.0, 0.0, 0.3),
                   P("class_weighting", "choice", "none", choices=CLASS_WEIGHTING)),
                  tasks=("segmentation",)),
        Component("optimizer", "adamw", "AdamW (historical default); uses learning_rate and weight_decay.",
                  (P("beta1", "float", 0.9, 0.5, 0.999), P("beta2", "float", 0.999, 0.9, 0.9999))),
        Component("optimizer", "sgd", "SGD with momentum; uses learning_rate and weight_decay.",
                  (P("momentum", "float", 0.9, 0.0, 0.99), P("nesterov", "bool", True))),
        Component("scheduler", "cosine", "Per-epoch cosine annealing to zero (historical default)."),
        Component("scheduler", "constant", "Constant learning rate."),
        Component("scheduler", "step", "Multiply the learning rate by gamma every step_size epochs.",
                  (P("step_size", "int", 3, 1, 20), P("gamma", "float", 0.1, 0.01, 0.9))),
        Component("scheduler", "warmup_cosine",
                  "Linear warmup over warmup_epochs, then cosine annealing to zero.",
                  (P("warmup_epochs", "int", 1, 1, 5),),
                  constraints=("epochs must exceed warmup_epochs",)),
    )
}
DEFAULT_AUGMENTATIONS = ({"name": "hflip", "p": 0.5}, {"name": "vflip", "p": 0.5})
PIPELINE_KEYS = ("input_view", "normalization", "augmentations", "loss", "optimizer",
                 "scheduler", "dropout", "custom_components")


class PipelineError(ValueError):
    """An invalid or incompatible pipeline; ``errors`` lists every problem."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# ---------------------------------------------------------------------------
# Resolution (dataset-independent) and compatibility (dataset/model-specific)
# ---------------------------------------------------------------------------

def _resolve_params(component, raw, where, errors):
    allowed = {param.name for param in component.params}
    for key in set(raw) - allowed - {"name"}:
        errors.append(f"{where}: unsupported parameter {key!r} (allowed: {sorted(allowed)})")
    resolved = {"name": component.name}
    for param in component.params:
        value = raw.get(param.name, param.default)
        if param.kind == "bool":
            ok = isinstance(value, bool)
        elif param.kind == "choice":
            ok = value in param.choices
        elif param.kind == "int":
            ok = type(value) is int and param.minimum <= value <= param.maximum
        else:
            ok = (isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value) and param.minimum <= value <= param.maximum)
            value = float(value) if ok else value
        if not ok:
            bounds = param.choices or (param.minimum, param.maximum)
            errors.append(f"{where}.{param.name}={value!r} is invalid ({param.kind}, {bounds})")
        resolved[param.name] = value
    return resolved


def _resolve_one(kind, raw, where, errors):
    raw = {"name": raw} if isinstance(raw, str) else raw
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
        errors.append(f"{where}: must be an object with a 'name'")
        return None
    component = CATALOG.get((kind, raw["name"]))
    if component is None:
        names = sorted(name for k, name in CATALOG if k == kind)
        errors.append(f"{where}: unknown {kind} {raw['name']!r} (supported: {names})")
        return None
    return _resolve_params(component, raw, where, errors)


def resolve_pipeline(raw=None):
    """Fill defaults and validate structure; raises PipelineError."""
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise PipelineError(["pipeline must be a JSON object"])
    errors = [f"unsupported pipeline key {key!r} (allowed: {list(PIPELINE_KEYS)})"
              for key in set(raw) - set(PIPELINE_KEYS)]
    view = raw.get("input_view")
    if view is not None and view not in INPUT_VIEWS:
        errors.append(f"input_view {view!r} is not one of {list(INPUT_VIEWS)}")
    normalization = raw.get("normalization", "dataset_default")
    if normalization not in NORMALIZATIONS:
        errors.append(f"normalization {normalization!r} is not one of {list(NORMALIZATIONS)}")
    augmentations = []
    raw_augmentations = raw.get("augmentations", list(DEFAULT_AUGMENTATIONS))
    if not isinstance(raw_augmentations, list):
        errors.append("augmentations must be a list (use [] for none)")
        raw_augmentations = []
    for index, item in enumerate(raw_augmentations):
        resolved = _resolve_one("augmentation", item, f"augmentations[{index}]", errors)
        if resolved:
            augmentations.append(resolved)
    names = [item["name"] for item in augmentations]
    for name in {n for n in names if names.count(n) > 1}:
        errors.append(f"augmentation {name!r} is listed more than once")
    dropout = raw.get("dropout", 0.0)
    if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
            or not 0.0 <= dropout <= 0.5):
        errors.append(f"dropout={dropout!r} must be a number in [0, 0.5]")
    if raw.get("custom_components"):
        errors.append("custom_components are reserved for scope S2 (sandboxed code), "
                      "which is not available yet")
    resolved = {
        "input_view": view,
        "normalization": normalization,
        "augmentations": augmentations,
        "loss": _resolve_one("loss", raw.get("loss", "cross_entropy"), "loss", errors),
        "optimizer": _resolve_one("optimizer", raw.get("optimizer", "adamw"), "optimizer", errors),
        "scheduler": _resolve_one("scheduler", raw.get("scheduler", "cosine"), "scheduler", errors),
        "dropout": float(dropout) if isinstance(dropout, (int, float)) else dropout,
    }
    if errors:
        raise PipelineError(errors)
    return resolved


def selected_view(pipeline, model_spec):
    return pipeline["input_view"] or model_spec.input_profile


def compatibility_errors(pipeline, dataset_spec, model_spec, effective_resolution, epochs):
    """Every reason this resolved pipeline cannot run for the dataset and model."""
    errors = []
    task = dataset_spec.task_type
    view = selected_view(pipeline, model_spec)
    if view not in ALTERNATIVE_VIEWS[model_spec.input_profile]:
        errors.append(
            f"model {model_spec.model_id} ({model_spec.input_profile} input) cannot take the "
            f"{view!r} view; allowed: {list(ALTERNATIVE_VIEWS[model_spec.input_profile])}"
        )
    elif view not in dataset_spec.input_profiles:
        errors.append(f"{dataset_spec.dataset_id} has no verified {view!r} view; "
                      f"available: {list(dataset_spec.input_profiles)}")
    if pipeline["normalization"] == "train_channel_stats" and view == "sentinel2_13":
        errors.append("sentinel2_13 checkpoints expect their pretraining scaling; "
                      "train_channel_stats is incompatible")
    chosen = [("augmentation", item) for item in pipeline["augmentations"]] + [
        (kind, pipeline[kind]) for kind in ("loss", "optimizer", "scheduler")
    ]
    for kind, item in chosen:
        component = CATALOG[(kind, item["name"])]
        if task not in component.tasks:
            errors.append(f"{component.kind} {item['name']!r} does not support {task}")
    if any(item["name"] == "rot90" for item in pipeline["augmentations"]) and (
        effective_resolution[0] != effective_resolution[1]
    ):
        errors.append(f"rot90 needs a square image; effective resolution is {effective_resolution}")
    scheduler = pipeline["scheduler"]
    if scheduler["name"] == "warmup_cosine" and epochs <= scheduler["warmup_epochs"]:
        errors.append(f"warmup_cosine needs epochs > warmup_epochs ({scheduler['warmup_epochs']})")
    if pipeline["dropout"] > 0 and task != "classification":
        errors.append("dropout is supported only on classification heads")
    return errors


def describe_configuration_space(dataset_spec=None, model_specs=()):
    """Machine-readable catalog, optionally annotated for one dataset."""
    components = [
        {**{k: v for k, v in asdict(c).items() if k != "params"},
         "params": [asdict(p) for p in c.params]}
        for c in CATALOG.values()
    ]
    space = {
        "components": components,
        "pipeline_keys": list(PIPELINE_KEYS[:-1]),
        "normalizations": {
            "dataset_default": "the dataset's verified contract (fixed scaling for the "
                               "original four datasets, training-split statistics otherwise)",
            "train_channel_stats": "per-channel standardization from training-split statistics",
        },
        "class_weighting": "estimated from up to 512 evenly spaced training samples",
        "defaults": resolve_pipeline(None),
        "top_level_settings": ["epochs", "batch_size", "learning_rate", "weight_decay",
                               "image_size", "early_stopping", "early_stopping_patience",
                               "early_stopping_min_delta", "pretrained"],
        "not_offered": ["per-channel colour jitter", "hue/saturation", "channel shuffling",
                        "custom code (scope S2, reserved)"],
    }
    if dataset_spec is not None:
        task = dataset_spec.task_type
        space["dataset"] = {
            "dataset_id": dataset_spec.dataset_id,
            "task_type": task,
            "available_views": list(dataset_spec.input_profiles),
            "components_for_task": sorted(
                f"{c.kind}:{c.name}" for c in CATALOG.values() if task in c.tasks
            ),
            "models": {
                spec.model_id: {
                    "input_views": [
                        view for view in ALTERNATIVE_VIEWS[spec.input_profile]
                        if view in dataset_spec.input_profiles
                    ],
                    "default_view": spec.input_profile,
                    "dropout_supported": task == "classification",
                    "normalizations": (
                        ["dataset_default"] if spec.input_profile == "sentinel2_13"
                        else list(NORMALIZATIONS)
                    ),
                }
                for spec in model_specs
            },
        }
    return space


# ---------------------------------------------------------------------------
# Augmentation (train split only; applied after resizing, before normalization)
# ---------------------------------------------------------------------------

def _uniform(low, high):
    import torch

    return low + (high - low) * float(torch.rand(()))


def apply_augmentations(image, target, augmentations, task_type):
    """Apply each augmentation in order. Spatial ops move masks with the image.

    Each op draws ``torch.rand(()) < p`` first, so the default flips consume
    exactly the random numbers the historical implementation did.
    """
    import torch
    import torch.nn.functional as functional

    segmentation = task_type == "segmentation"
    for op in augmentations:
        if not float(torch.rand(())) < op["p"]:
            continue
        name = op["name"]
        if name in ("hflip", "vflip"):
            dim = -1 if name == "hflip" else -2
            image = torch.flip(image, dims=(dim,))
            if segmentation:
                target = torch.flip(target, dims=(dim,))
        elif name == "rot90":
            turns = int(torch.randint(1, 4, ()))
            image = torch.rot90(image, turns, dims=(-2, -1))
            if segmentation:
                target = torch.rot90(target, turns, dims=(-2, -1))
        elif name == "random_resized_crop":
            height, width = image.shape[-2:]
            side = math.sqrt(_uniform(op["min_scale"], 1.0))
            crop_h, crop_w = max(1, round(height * side)), max(1, round(width * side))
            top = int(torch.randint(0, height - crop_h + 1, ()))
            left = int(torch.randint(0, width - crop_w + 1, ()))
            image = functional.interpolate(
                image[None, :, top:top + crop_h, left:left + crop_w], size=(height, width),
                mode="bilinear", align_corners=False,
            )[0]
            if segmentation:
                target = functional.interpolate(
                    target[None, None, top:top + crop_h, left:left + crop_w].float(),
                    size=(height, width), mode="nearest",
                )[0, 0].long()
        elif name == "brightness":
            image = image * (1.0 + _uniform(-op["max_delta"], op["max_delta"]))
        elif name == "contrast":
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = mean + (1.0 + _uniform(-op["max_delta"], op["max_delta"])) * (image - mean)
        elif name == "gaussian_noise":
            sigma = image.std(dim=(-2, -1), keepdim=True) * op["sigma_fraction"]
            image = image + torch.randn_like(image) * sigma
        else:  # resolve_pipeline guarantees this cannot happen
            raise ValueError(f"Unknown augmentation {name!r}")
    return image, target


# ---------------------------------------------------------------------------
# Losses, optimizers, schedulers
# ---------------------------------------------------------------------------

def class_weights(counts, mode, ignore_index=None):
    """Mean-1 inverse-frequency weights; absent classes count as 1, ratio capped at 10x."""
    if mode == "none" or counts is None:
        return None
    import torch

    counts = torch.as_tensor(counts, dtype=torch.float64).clamp_min(1.0)
    weights = counts.sum() / (len(counts) * counts)
    if mode == "inverse_sqrt_frequency":
        weights = weights.sqrt()
    if ignore_index is not None and 0 <= ignore_index < len(weights):
        weights[ignore_index] = 0.0
    present = weights > 0
    weights = weights / weights[present].mean()
    floor = weights[present].min()
    weights = torch.where(present, weights.clamp(max=float(floor) * 10.0), weights)
    return (weights / weights[present].mean()).float()


@functools.lru_cache(maxsize=None)
def _loss_classes():
    """Defined lazily so validating a pipeline does not require torch."""
    import torch
    import torch.nn.functional as functional

    class FocalLoss(torch.nn.Module):
        def __init__(self, gamma, weight=None, ignore_index=None):
            super().__init__()
            self.gamma = gamma
            self.register_buffer("weight", weight)
            self.ignore_index = -100 if ignore_index is None else ignore_index

        def forward(self, logits, targets):
            ce = functional.cross_entropy(logits.float(), targets, reduction="none",
                                          ignore_index=self.ignore_index)
            loss = (1.0 - (-ce).exp()) ** self.gamma * ce
            valid = (targets != self.ignore_index).float()
            if self.weight is not None:
                valid = valid * self.weight[targets.clamp(min=0)]
            return (loss * valid).sum() / valid.sum().clamp_min(1e-12)

    class DiceCrossEntropyLoss(torch.nn.Module):
        def __init__(self, dice_weight, label_smoothing=0.0, weight=None, ignore_index=None):
            super().__init__()
            kwargs = {"label_smoothing": label_smoothing, "weight": weight}
            if ignore_index is not None:
                kwargs["ignore_index"] = ignore_index
            self.cross_entropy = torch.nn.CrossEntropyLoss(**kwargs)
            self.dice_weight = dice_weight
            self.ignore_index = ignore_index

        def forward(self, logits, targets):
            ce = self.cross_entropy(logits, targets)
            if self.dice_weight == 0:
                return ce
            probabilities = functional.softmax(logits.float(), dim=1)
            valid = targets >= 0 if self.ignore_index is None else targets != self.ignore_index
            one_hot = functional.one_hot(targets.clamp(min=0) * valid, probabilities.shape[1])
            one_hot = one_hot.permute(0, 3, 1, 2).float()
            mask = valid[:, None].float()
            intersection = (probabilities * one_hot * mask).sum(dim=(0, 2, 3))
            totals = ((probabilities + one_hot) * mask).sum(dim=(0, 2, 3))
            dice = (2 * intersection + 1.0) / (totals + 1.0)
            return (1 - self.dice_weight) * ce + self.dice_weight * (1 - dice.mean())

    return FocalLoss, DiceCrossEntropyLoss


def build_loss(pipeline, num_classes, ignore_index=None, class_counts=None, device=None):
    import torch

    spec = pipeline["loss"]
    weight = class_weights(class_counts, spec.get("class_weighting", "none"), ignore_index)
    if weight is not None and device is not None:
        weight = weight.to(device)
    if spec["name"] == "cross_entropy":
        kwargs = {}
        if ignore_index is not None:
            kwargs["ignore_index"] = ignore_index
        if spec["label_smoothing"]:
            kwargs["label_smoothing"] = spec["label_smoothing"]
        if weight is not None:
            kwargs["weight"] = weight
        return torch.nn.CrossEntropyLoss(**kwargs)
    focal, dice_cross_entropy = _loss_classes()
    if spec["name"] == "focal":
        return focal(spec["gamma"], weight, ignore_index)
    return dice_cross_entropy(spec["dice_weight"], spec["label_smoothing"], weight, ignore_index)


def build_optimizer(pipeline, parameters, learning_rate, weight_decay):
    import torch

    spec = pipeline["optimizer"]
    if spec["name"] == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=spec["momentum"],
                               nesterov=spec["nesterov"] and spec["momentum"] > 0,
                               weight_decay=weight_decay)
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay,
                             betas=(spec["beta1"], spec["beta2"]))


def build_scheduler(pipeline, optimizer, epochs):
    """Schedulers step once per epoch, as the historical cosine schedule did."""
    import torch

    spec = pipeline["scheduler"]
    if spec["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if spec["name"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    if spec["name"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=spec["step_size"],
                                               gamma=spec["gamma"])
    warmup = spec["warmup_epochs"]

    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_training_components(pipeline, model, *, num_classes, ignore_index, class_counts,
                              learning_rate, weight_decay, epochs, device=None):
    """Criterion, optimizer, and scheduler for one training run."""
    criterion = build_loss(pipeline, num_classes, ignore_index, class_counts, device)
    optimizer = build_optimizer(pipeline, model.parameters(), learning_rate, weight_decay)
    return criterion, optimizer, build_scheduler(pipeline, optimizer, epochs)
