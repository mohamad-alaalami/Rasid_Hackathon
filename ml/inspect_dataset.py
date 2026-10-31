"""Inspect a registered GEO-Bench dataset without touching its test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import _band_array, find_task, inspect_sample_resolution
from .registry import get_dataset_spec


def inspect_dataset(dataset_id: str = "m-eurosat", benchmark_dir: str | None = None) -> dict:
    import numpy as np

    spec = get_dataset_spec(dataset_id)
    task = find_task(dataset_id, benchmark_dir)
    train = task.get_dataset(split="train", band_names=None)
    validation = task.get_dataset(split="valid", band_names=None)
    sample = train[0]
    actual_resolution = inspect_sample_resolution(sample, spec, spec.band_indices)
    selected = [np.asarray(_band_array(sample.bands[i])).squeeze() for i in spec.band_indices]
    if spec.task_type == "classification":
        label_summary = {"type": type(sample.label).__name__, "value": int(sample.label)}
    else:
        mask = np.asarray(_band_array(sample.label)).squeeze()
        label_summary = {
            "type": type(sample.label).__name__,
            "shape": list(mask.shape),
            "classes_present": np.unique(mask).astype(int).tolist(),
        }
    return {
        "dataset_id": dataset_id,
        "task_type": spec.task_type,
        "benchmark_name": spec.benchmark_name,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "num_classes": spec.num_classes,
        "native_resolution": list(spec.native_resolution),
        "actual_sample_resolution": list(actual_resolution),
        "supported_resolution_options": {
            "default": "native",
            "override": "explicit square image_size >= 32",
        },
        "input_band_indices": list(spec.band_indices),
        "input_shapes": [list(array.shape) for array in selected],
        "input_dtypes": [str(array.dtype) for array in selected],
        "input_min": [float(array.min()) for array in selected],
        "input_max": [float(array.max()) for array in selected],
        "label": label_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="m-eurosat")
    parser.add_argument("--benchmark-dir", help="Path to one GEO-Bench benchmark directory.")
    args = parser.parse_args()
    if args.benchmark_dir and not Path(args.benchmark_dir).is_dir():
        parser.error(f"benchmark directory does not exist: {args.benchmark_dir}")
    print(json.dumps(inspect_dataset(args.dataset_id, args.benchmark_dir), indent=2))


if __name__ == "__main__":
    main()
