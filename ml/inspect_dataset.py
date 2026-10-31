"""Inspect a registered GEO-Bench dataset without touching its test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import _band_array, find_task
from .registry import get_dataset_spec


def inspect_dataset(dataset_id: str = "m-eurosat", benchmark_dir: str | None = None) -> dict:
    import numpy as np

    spec = get_dataset_spec(dataset_id)
    task = find_task(dataset_id, benchmark_dir)
    train = task.get_dataset(split="train", band_names=None)
    validation = task.get_dataset(split="valid", band_names=None)
    sample = train[0]
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
