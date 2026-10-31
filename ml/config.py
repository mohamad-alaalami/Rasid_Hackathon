"""Configuration validation for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "auto"
    pretrained: bool = True
    input_mode: str = "registry"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    mixed_precision: bool = True
    device: str = "auto"
    output_dir: str = "experiments"
    benchmark_dir: str | None = None
    image_size: int = 64

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TrainingConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unsupported configuration keys: {sorted(unknown)}")
        expected_types = {
            "model": str,
            "pretrained": bool,
            "input_mode": str,
            "epochs": int,
            "batch_size": int,
            "learning_rate": Real,
            "weight_decay": Real,
            "seed": int,
            "num_workers": int,
            "mixed_precision": bool,
            "device": str,
            "output_dir": str,
            "benchmark_dir": (str, type(None)),
            "image_size": int,
        }
        for name, value in raw.items():
            expected = expected_types[name]
            if not isinstance(value, expected) or (
                expected in {int, Real} and isinstance(value, bool)
            ):
                raise TypeError(
                    f"{name} has invalid type {type(value).__name__}; expected {expected}."
                )
        config = cls(**dict(raw))
        config.validate()
        return config

    def validate(self) -> None:
        if self.model not in {"auto", "resnet18", "fcn_resnet50"}:
            raise ValueError("model must be 'auto', 'resnet18', or 'fcn_resnet50'.")
        if self.input_mode != "registry":
            raise ValueError("input_mode must be 'registry'; dataset channels come from ml/registry.py.")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if self.benchmark_dir is not None and not Path(self.benchmark_dir).is_dir():
            raise ValueError(f"benchmark_dir does not exist: {self.benchmark_dir}")

    def validate_for_dataset(self, dataset_id: str) -> str:
        from .registry import get_dataset_spec

        spec = get_dataset_spec(dataset_id)
        selected_model = spec.default_model if self.model == "auto" else self.model
        allowed = {
            "classification": {"resnet18"},
            "segmentation": {"fcn_resnet50"},
        }[spec.task_type]
        if selected_model not in allowed:
            raise ValueError(
                f"model={selected_model!r} is incompatible with "
                f"dataset={dataset_id!r} ({spec.task_type})."
            )
        return selected_model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

