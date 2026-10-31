"""Inspect m-eurosat without training or touching its test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import M_EUROSAT_RGB_INDICES, _band_array, _extract_label, find_task


def inspect_dataset(benchmark_dir: str | None = None) -> dict:
    import numpy as np

    task = find_task("m-eurosat", benchmark_dir)
    train = task.get_dataset(split="train")
    validation = task.get_dataset(split="valid")
    sample = train[0]
    band_arrays = [np.asarray(_band_array(band)).squeeze() for band in sample.bands]
    return {
        "dataset_id": "m-eurosat",
        "train_samples": len(train),
        "validation_samples": len(validation),
        "sample_type": type(sample).__name__,
        "band_count": len(band_arrays),
        "band_shapes": [list(array.shape) for array in band_arrays],
        "band_dtypes": [str(array.dtype) for array in band_arrays],
        "band_min": [float(array.min()) for array in band_arrays],
        "band_max": [float(array.max()) for array in band_arrays],
        "rgb_indices": list(M_EUROSAT_RGB_INDICES),
        "label": _extract_label(sample),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        help="Absolute path to the classification_v1.0 directory.",
    )
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(args.benchmark_dir), indent=2))


if __name__ == "__main__":
    main()

