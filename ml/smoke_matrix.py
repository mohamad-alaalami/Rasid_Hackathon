"""Run one-batch checks for every runtime-eligible model/dataset combination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_registry import list_model_options
from .registry import list_dataset_specs
from .smoke_test import run_smoke_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output", default="artifacts/smoke_matrix.json")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument(
        "--image-size",
        type=int,
        help=(
            "Explicit square resolution for memory-constrained smoke testing. "
            "Omit to test every dataset at native resolution."
        ),
    )
    args = parser.parse_args()

    records = []
    for dataset in list_dataset_specs():
        dataset_id = dataset["dataset_id"]
        for option in list_model_options(dataset_id, args.checkpoint_dir):
            if option["model_id"] == "fcn_resnet50" and not args.include_legacy:
                continue
            if not option["eligible"]:
                records.append(
                    {
                        "status": "skipped",
                        "dataset_id": dataset_id,
                        "model": option["model_id"],
                        "reason": option["reason"],
                    }
                )
                continue
            batch_size = option["conservative_batch_size"]
            try:
                result = run_smoke_test(
                    dataset_id,
                    model_id=option["model_id"],
                    batch_size=batch_size,
                    checkpoint_dir=args.checkpoint_dir,
                    image_size=args.image_size,
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "dataset_id": dataset_id,
                    "model": option["model_id"],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "test_split_accessed": False,
                }
            records.append(result)
            print(json.dumps(result, indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    failures = [record for record in records if record["status"] == "failed"]
    print(json.dumps({"output": str(output), "failures": len(failures)}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
